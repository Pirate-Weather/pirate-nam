"""Proof of init-static-build on the NAM 12 km grid: the built static vs NAM's own (bgrd3d00 -> nam12_static.npz).

    python tools/check_static_build.py --built scratch/e3/nam12_built.npz --ref static/nam12_static.npz \
        --out scratch/e3/check12 [--maps]

Per field: continuous -> mean diff, mean |diff|, RMS, p99 |diff| over the cells where both are defined
(land only for the land-surface fields); categorical -> agreement fraction on cells land in both. LAND ->
agreement over all cells. Writes check.json and, with --maps, one PNG per field (NAM | built | built - NAM,
analysis charts of static fields drawn with matplotlib; no weather fields). TG and MXSNAL are compared too
(land cells).
World mode of tools/check_static_build.py: build 3 km statics at any centres and check the edge geometry.

    python tools/check_static_build.py --world 46.5,8 -41,174 70,-45 1,102 --out DIR [--date 2026-11-04T12]
        [--km 1000] [--maps]

Per centre: a nest configfile (parm/nam_configfile with NAM's CONUS-nest 3 km keys, im x jm for a km x km
box, rotated pole at the centre) and its static, built on the GPU. Checks (world.json):
  - no NaN/inf in any field; build wall under 30 s;
  - GLON in [-180, 180]; dateline crossing flagged from the GLON span;
  - GLAT strictly increasing along the centre column, unwrapped GLON strictly increasing along the centre row
    (the rotated grid has no fold or wrap seam);
  - empty_cells == 0 for land use and soil before the nearest fill (every tile the grid needs was found,
    across the dateline and near the poles);
  - named points (a lake, a sea, a summit, an ice sheet in each domain) land on the right side of the coast
    and lake masks and carry the right land use / height;
  - land use, soil class histograms on land; soil on land-ice cells.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONT = ("HGT_SFC", "SFCR", "VEG", "ALBDO", "TG", "MXSNAL")
CAT = ("VGTYP", "SOTYP")


def compare(b, r):
    out = {}
    lb, lr = b["LAND"] > 0.5, r["LAND"] > 0.5
    both = lb & lr
    out["LAND"] = {"agree": round(float((lb == lr).mean()), 5), "land_built": int(lb.sum()), "land_nam": int(lr.sum()),
                   "built_land_nam_water": int((lb & ~lr).sum()), "built_water_nam_land": int((~lb & lr).sum())}
    for k in CONT:
        if k not in b or k not in r:
            continue
        m = both if k != "HGT_SFC" else np.ones_like(both)
        d = (b[k] - r[k])[m].astype(np.float64)
        out[k] = {"cells": int(m.sum()), "mean_nam": round(float(r[k][m].mean()), 4),
                  "mean_built": round(float(b[k][m].mean()), 4), "mean_diff": round(float(d.mean()), 4),
                  "mean_abs": round(float(np.abs(d).mean()), 4), "rms": round(float(np.sqrt((d * d).mean())), 4),
                  "p99_abs": round(float(np.percentile(np.abs(d), 99)), 4),
                  "corr": round(float(np.corrcoef(b[k][m].ravel(), r[k][m].ravel())[0, 1]), 4)}
        if k == "HGT_SFC":
            dl = (b[k] - r[k])[both].astype(np.float64)
            out[k]["land_mean_abs"] = round(float(np.abs(dl).mean()), 3)
            out[k]["land_rms"] = round(float(np.sqrt((dl * dl).mean())), 3)
    for k in CAT:
        vb, vr = np.rint(b[k]).astype(int), np.rint(r[k]).astype(int)
        out[k] = {"cells": int(both.sum()), "agree": round(float((vb == vr)[both].mean()), 4)}
    return out


def maps(b, r, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = {"HGT_SFC": ("terrain", 0, 3500, 300), "LAND": ("gray", 0, 1, 1), "VGTYP": ("tab20", 0, 20, 5),
            "SOTYP": ("tab20", 0, 16, 5), "SFCR": ("viridis", 0, 3, 0.5), "VEG": ("YlGn", 0, 100, 20),
            "ALBDO": ("magma", 0, 80, 20)}
    for k, (cm, lo, hi, dd) in spec.items():
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
        for a, arr, t in ((ax[0], r[k], f"NAM 12 km static (bgrd3d00) {k}"), (ax[1], b[k], f"built (init-static-build) {k}")):
            im = a.imshow(arr, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            a.set_title(t)
            fig.colorbar(im, ax=a, shrink=0.8)
        d = b[k] - r[k]
        if k in CAT:
            d = (np.rint(b[k]) != np.rint(r[k])).astype(float)
            im = ax[2].imshow(d, origin="lower", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
            ax[2].set_title(f"{k} differs (1 = different class)")
        else:
            im = ax[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-dd, vmax=dd, interpolation="nearest")
            ax[2].set_title(f"built - NAM {k} (scale +-{dd})")
        fig.colorbar(im, ax=ax[2], shrink=0.8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.savefig(outdir / f"static12_{k}.png", dpi=70)
        plt.close(fig)


NEST3 = dict(dt_int="6", dt_num="1", dt_den="4", filt_dt_int="2", filt_dt_num="0", filt_dt_den="1",
             nrads="192", nradl="192")
DLMD3, DPHD3 = 0.126 / 4, 0.108 / 4          # NAM's CONUS nest: the 12 km parent's spacing / 4
KM_PER_DEG = 6370.0 * math.pi / 180.0

# Named points per centre: (name, lat, lon, expect) with expect one of
# "water", "land", ("hgt>", m), ("vg", class) -- what any correct static must show there.
POINTS = {
    (46.5, 8.0): [("Lake Geneva", 46.45, 6.55, "water"), ("Lake Constance", 47.63, 9.37, "water"),
                  ("Mont Blanc", 45.83, 6.86, ("hgt>", 2500.0)),
                  ("Monte Rosa", 45.94, 7.87, ("hgt>", 2500.0)), ("Ligurian Sea", 43.8, 8.6, "water"),
                  ("Po valley (Milan)", 45.46, 9.19, "land")],
    (-41.0, 174.0): [("Cook Strait", -41.45, 174.6, "water"), ("Lake Taupo", -38.8, 175.9, "water"),
                     ("Aoraki/Mt Cook", -43.6, 170.14, ("hgt>", 1500.0)), ("Wellington", -41.29, 174.78, "land"),
                     ("Tasman Sea", -40.0, 169.0, "water"), ("Ruapehu", -39.28, 175.57, ("hgt>", 1500.0))],
    (70.0, -45.0): [("Greenland ice sheet", 72.0, -40.0, ("vg", 15)), ("Ice sheet summit", 72.58, -38.46,
                                                                          ("hgt>", 2800.0)),
                    ("Disko Bay", 69.0, -52.0, "water"), ("Baffin Bay west of Disko", 69.0, -56.0, "water"),
                    ("Kangerlussuaq", 67.01, -50.7, "land")],
    (1.0, 102.0): [("Lake Toba (north basin)", 2.80, 98.65, "water"), ("Singapore", 1.35, 103.82, "land"),
                   ("South China Sea", 2.0, 105.5, "water"), ("Strait of Malacca", 2.7, 101.2, "water"),
                   ("Kerinci", -1.70, 101.26, ("hgt>", 2000.0)), ("Kuala Lumpur", 3.14, 101.69, "land")],
    (-17.5, 179.5): [("Viti Levu", -17.8, 178.0, "land"), ("Vanua Levu", -16.6, 179.2, "land"),
                     ("Koro Sea", -17.5, 179.6, "water"), ("Taveuni (180 E)", -16.85, -179.97, "land")],
}


def nest_config(lat, lon, km, out: Path) -> Path:
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from make_domain_configs import NEST_PHYSICS, render
    im = int(round(km / (DLMD3 * KM_PER_DEG))) | 1
    jm = int(round(km / (DPHD3 * KM_PER_DEG))) | 1
    tag = f"nest3km_{'n' if lat >= 0 else 's'}{abs(lat):g}_{'e' if lon >= 0 else 'w'}{abs(lon):g}".replace(".", "p")
    static = out / f"{tag}_static.npz"
    repl = dict(NEST_PHYSICS, **NEST3, tph0d=f"{lat:g}", tlm0d=f"{lon:g}", im=str(im), jm=str(jm),
                wbd=f"{-(im - 1) / 2 * DLMD3:.5f}", sbd=f"{-(jm - 1) / 2 * DPHD3:.5f}",
                static_npz=str(static), domain_name=tag)
    p = out / f"nam_configfile_{tag}"
    p.write_text(render((ROOT / "parm" / "nam_configfile").read_text(), repl, tag))
    return p


def _nearest(z, lat, lon):
    la, lo = np.radians(z["GLAT"]), np.radians(z["GLON"])
    p, q = math.radians(lat), math.radians(lon)
    d = np.sin((la - p) / 2) ** 2 + np.cos(la) * math.cos(p) * np.sin((lo - q) / 2) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return int(j), int(i), float(2 * 6370.0 * math.asin(math.sqrt(d[j, i])))


def check_one(z, rec, centre):
    out = {"grid": rec["grid"], "build_s": rec["total_s"], "stages": rec["stages"],
           "empty_cells": rec.get("empty_cells"), "landuse_tiles": rec.get("landuse_tiles")}
    bad = {k: int((~np.isfinite(z[k])).sum()) for k in z if k != "META" and z[k].dtype.kind == "f"}
    out["nonfinite"] = {k: v for k, v in bad.items() if v}
    glat, glon = z["GLAT"].astype(np.float64), z["GLON"].astype(np.float64)
    out["glon_range"] = [round(float(glon.min()), 3), round(float(glon.max()), 3)]
    out["glat_range"] = [round(float(glat.min()), 3), round(float(glat.max()), 3)]
    out["crosses_dateline"] = bool(glon.max() - glon.min() > 180.0)
    jc, ic = glat.shape[0] // 2, glat.shape[1] // 2
    row = np.unwrap(np.radians(glon[jc]))
    out["glat_monotone_centre_col"] = bool(np.all(np.diff(glat[:, ic]) > 0))
    out["glon_monotone_centre_row"] = bool(np.all(np.diff(row) > 0))
    land = z["LAND"] > 0.5
    out["land_frac"] = round(float(land.mean()), 4)
    vg, so = np.rint(z["VGTYP"]).astype(int), np.rint(z["SOTYP"]).astype(int)
    out["vgtyp_land_hist"] = {int(k): int(v) for k, v in zip(*np.unique(vg[land], return_counts=True))}
    out["sotyp_land_hist"] = {int(k): int(v) for k, v in zip(*np.unique(so[land], return_counts=True))}
    ice = land & (vg == 15)
    if ice.any():
        out["sotyp_on_ice"] = {int(k): int(v) for k, v in zip(*np.unique(so[ice], return_counts=True))}
    out["water_consistent"] = bool(np.all((vg[~land] == 0) & (so[~land] == 0)) and np.all(vg[land] > 0)
                                   and np.all(so[land] > 0))
    out["tg_land_range"] = ([round(float(z["TG"][land].min()), 2), round(float(z["TG"][land].max()), 2)]
                            if land.any() else None)
    out["hgt_range"] = [round(float(z["HGT_SFC"].min()), 1), round(float(z["HGT_SFC"].max()), 1)]
    pts, ok = [], True
    for name, la, lo, exp in POINTS.get(centre, []):
        j, i, dkm = _nearest(z, la, lo)
        inside = 0 < j < glat.shape[0] - 1 and 0 < i < glat.shape[1] - 1 and dkm < 5.0
        v = {"name": name, "lat": la, "lon": lo, "cell": [j, i], "dist_km": round(dkm, 2), "inside": inside,
             "LAND": int(land[j, i]), "VGTYP": int(vg[j, i]), "SOTYP": int(so[j, i]),
             "HGT": round(float(z["HGT_SFC"][j, i]), 1), "expect": exp}
        if exp == "water":
            good = not land[j, i]
        elif exp == "land":
            good = bool(land[j, i])
        elif exp[0] == "hgt>":
            good = float(z["HGT_SFC"][j, i]) > exp[1]
        else:
            good = int(vg[j, i]) == exp[1]
        v["pass"] = bool(good and inside)
        ok &= v["pass"]
        pts.append(v)
    out["points"] = pts
    out["pass"] = bool(ok and not out["nonfinite"] and out["build_s"] < 30.0 and out["glat_monotone_centre_col"]
                       and out["glon_monotone_centre_row"] and out["water_consistent"]
                       and all(v == 0 for v in (out["empty_cells"] or {}).values())
                       and -180.0 <= out["glon_range"][0] and out["glon_range"][1] <= 180.0)
    return out


def run_world(centres, out: Path, date: str, km: float, maps: bool):
    from namgpu import config
    from namgpu.init import static_build as sb
    out.mkdir(parents=True, exist_ok=True)
    valid = datetime.fromisoformat(date)
    res_all = {"date": date, "km": km, "domains": {}}
    for c in centres:
        lat, lon = (float(x) for x in c.split(","))
        cfgp = nest_config(lat, lon, km, out)
        cfg = config.load(cfgp)
        t = time.perf_counter()
        res, rec = sb.build(cfg, valid, log=lambda m: None)
        npz = sb.write(res, rec, Path(cfg.raw["static_npz"]), cfg)
        wall = round(time.perf_counter() - t, 2)
        z = dict(np.load(npz))
        r = check_one(z, rec, (lat, lon))
        r["wall_s_incl_write"] = wall
        r["config"] = str(cfgp)
        r["static"] = str(npz)
        res_all["domains"][cfg.raw["domain_name"]] = r
        print(f"{cfg.raw['domain_name']}: {r['grid']} build {r['build_s']} s, nonfinite {r['nonfinite']}, "
              f"glon {r['glon_range']} dateline {r['crosses_dateline']}, empty {r['empty_cells']}, "
              f"land {r['land_frac']}, points {sum(p['pass'] for p in r['points'])}/{len(r['points'])}, "
              f"PASS {r['pass']}", flush=True)
        if maps:
            import subprocess
            import sys
            subprocess.run([sys.executable, str(ROOT / "tools" / "plot_static.py"), str(npz), str(out / "maps"),
                            f"WOOF-NAM 3 km static {lat:g},{lon:g}"], check=False)
    (out / "world.json").write_text(json.dumps(res_all, indent=1))
    return res_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--maps", action="store_true")
    ap.add_argument("--world", nargs="+", metavar="LAT,LON", help="build and check 3 km statics at these centres")
    ap.add_argument("--km", type=float, default=1000.0)
    ap.add_argument("--date", default="2026-11-04T12")
    a = ap.parse_args(argv)
    if a.world:
        r = run_world(a.world, a.out, a.date, a.km, a.maps)
        return 0 if all(d["pass"] for d in r["domains"].values()) else 1
    if a.built is None or a.ref is None:
        ap.error("--built and --ref (or --world)")
    b, r = dict(np.load(a.built)), dict(np.load(a.ref))
    a.out.mkdir(parents=True, exist_ok=True)
    res = compare(b, r)
    (a.out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if a.maps:
        maps(b, r, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
