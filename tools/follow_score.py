"""Track and intensity error of WOOF-NAM nests against IBTrACS best track, every 6 h, plus a track/intensity chart.

    python tools/follow_score.py --sid 2024225N22135 --start 2024081512 --hours 48 \
        --run moving=runs/P1-ampil-follow --run static=runs/P1-ampil-static --out runs/P1-score

An independent instrument: it reads only the hourly history (wrfout_d01_*, every generation of a moving nest,
found anywhere under the run folder) and the best track. It does not use the in-run tracker's fixes.

Centre at each lead: the minimum of the history's MSLP (QUASIPOST membrane MSLP, Pa) after a 5 x 5 box mean,
searched within --search-km of the previous centre (the first guess at f00 is the best-track start, which is
also what the run was given). The central pressure is the unsmoothed minimum within 60 km of that point. A centre
within --edge-rows rows of the domain edge is flagged "edge": the storm has left (or is leaving) the nest, so
the minimum is a boundary point, not the storm. Errors: great-circle distance (km) and model minus best-track
central pressure (hPa).

Outputs in --out: score.json (every lead, every run), SCORE.md (the tables side by side), track.png (the
tracks on lon/lat with each frame's nest outline for every 6 h, and central pressure against lead). The chart
is an analysis chart (matplotlib), not a weather field.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

R_KM = 6371.0


def dist_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def box_mean(a, k=5):
    """k x k box mean with edge padding (no scipy needed)."""
    p = k // 2
    b = np.pad(a, p, mode="edge").astype(np.float64)
    c = b.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def histories(run: Path) -> dict:
    """valid time -> history file. Several files for one valid time (the generation that ended at a move hour and
    the one that started there) -> the newest file."""
    out = {}
    for f in sorted(run.rglob("wrfout_d01_*")):
        try:
            t = datetime.strptime(f.name[len("wrfout_d01_"):][:19], "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            continue
        if t not in out or f.stat().st_mtime > out[t].stat().st_mtime:
            out[t] = f
    return out


def read_frame(f: Path):
    import netCDF4
    with netCDF4.Dataset(f) as nc:
        lat = np.asarray(nc["XLAT"][0] if nc["XLAT"].ndim == 3 else nc["XLAT"][:], np.float64)
        lon = np.asarray(nc["XLONG"][0] if nc["XLONG"].ndim == 3 else nc["XLONG"][:], np.float64)
        if "MSLP" not in nc.variables:
            return lat, lon, None
        m = np.asarray(nc["MSLP"][0] if nc["MSLP"].ndim == 3 else nc["MSLP"][:], np.float64)
    return lat, lon, m / 100.0 if np.nanmedian(m) > 2000 else m


def centre(lat, lon, mslp, guess, search_km, edge_rows):
    jm, im = mslp.shape
    sm = box_mean(np.where(np.isfinite(mslp), mslp, np.nanmax(mslp)))
    d = dist_km(guess[0], guess[1], lat, lon)
    inner = np.zeros_like(sm, bool)
    inner[5:-5, 5:-5] = True
    cand = np.where((d <= search_km) & inner, sm, np.inf)
    if not np.isfinite(cand).any():
        return None
    j, i = np.unravel_index(np.argmin(cand), cand.shape)
    near = dist_km(lat[j, i], lon[j, i], lat, lon) <= 60.0
    pmin = float(np.nanmin(np.where(near, mslp, np.inf)))
    edge = min(j, i, jm - 1 - j, im - 1 - i) < edge_rows
    return {"lat": float(lat[j, i]), "lon": float(((lon[j, i] + 180) % 360) - 180), "pmin_hpa": round(pmin, 1),
            "j": int(j), "i": int(i), "edge": bool(edge), "rows_to_edge": int(min(j, i, jm - 1 - j, im - 1 - i))}


def outline(lat, lon, step=8):
    jm, im = lat.shape
    jj = list(range(0, jm, step)) + [jm - 1]
    ii = list(range(0, im, step)) + [im - 1]
    path = ([(0, i) for i in ii] + [(j, im - 1) for j in jj] + [(jm - 1, i) for i in ii[::-1]]
            + [(j, 0) for j in jj[::-1]])
    return [[float(lon[j, i]), float(lat[j, i])] for j, i in path]


def score_run(run: Path, trk: dict, start: datetime, hours: int, every_h: int, search_km, edge_rows):
    from namgpu.nest import ibtracs
    hist = histories(run)
    if not hist:
        raise SystemExit(f"follow_score: no wrfout_d01_* under {run}")
    rows, guess = [], None
    b0 = ibtracs.at(trk, start)
    guess = (b0["lat"], b0["lon"])
    for h in range(0, hours + 1):
        t = start + timedelta(hours=h)
        f = hist.get(t)
        if f is None:
            continue
        lat, lon, m = read_frame(f)
        if m is None:
            continue
        c = centre(lat, lon, m, guess, search_km, edge_rows)
        if c is None:
            continue
        guess = (c["lat"], c["lon"])
        if h % every_h:
            continue
        b = ibtracs.at(trk, t)
        row = {"lead_h": h, "valid": t.isoformat(), "file": str(f.relative_to(run)), "model": c,
               "outline": outline(lat, lon), "frame_centre": [float(lat[lat.shape[0] // 2, lat.shape[1] // 2]),
                                                              float(lon[lat.shape[0] // 2, lat.shape[1] // 2])]}
        if b:
            row["best"] = {k: b[k] for k in ("lat", "lon", "pres_hpa", "nature")}
            row["err_km"] = round(float(dist_km(b["lat"], b["lon"], c["lat"], c["lon"])), 1)
            row["dp_hpa"] = round(c["pmin_hpa"] - b["pres_hpa"], 1) if b["pres_hpa"] is not None else None
            row["best_from_frame_centre_km"] = round(float(dist_km(b["lat"], b["lon"], *row["frame_centre"])), 1)
        rows.append(row)
    ok = [r for r in rows if "err_km" in r and r["lead_h"] > 0]
    inside = [r for r in ok if not r["model"]["edge"]]
    summ = {"n": len(ok), "mean_err_km": round(float(np.mean([r["err_km"] for r in ok])), 1) if ok else None,
            "mean_abs_dp_hpa": round(float(np.mean([abs(r["dp_hpa"]) for r in ok if r.get("dp_hpa") is not None])), 1)
            if ok else None,
            "edge_leads": [r["lead_h"] for r in ok if r["model"]["edge"]],
            "mean_err_km_inside": round(float(np.mean([r["err_km"] for r in inside])), 1) if inside else None,
            "last_lead": rows[-1]["lead_h"] if rows else None}
    return {"run": str(run), "rows": rows, "summary": summ}


def chart(res: dict, trk: dict, start: datetime, hours: int, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from namgpu.nest import ibtracs
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    colours = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    bt = [ibtracs.at(trk, start + timedelta(hours=h)) for h in range(0, hours + 1, 3)]
    bt = [b for b in bt if b]
    lon0 = bt[0]["lon"]
    unwrap = lambda x: lon0 + (((np.asarray(x) - lon0) + 180) % 360 - 180)   # noqa: E731
    ax.plot(unwrap([b["lon"] for b in bt]), [b["lat"] for b in bt], "k-o", ms=3, lw=2, label=f"best track ({trk['name']}, IBTrACS)")
    for b, h in zip(bt, range(0, hours + 1, 3)):
        if h % 12 == 0:
            ax.annotate(f"{h}", (unwrap(b["lon"]), b["lat"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    for k, (label, r) in enumerate(res.items()):
        c = colours[k % len(colours)]
        rows = r["rows"]
        ax.plot(unwrap([x["model"]["lon"] for x in rows]), [x["model"]["lat"] for x in rows], "-s", color=c, ms=4,
                label=f"{label}: mean {r['summary']['mean_err_km']} km")
        for x in rows:
            if x["lead_h"] % 12 == 0:
                o = np.asarray(x["outline"])
                ax.plot(unwrap(o[:, 0]), o[:, 1], "-", color=c, lw=0.8, alpha=0.35 + 0.5 * x["lead_h"] / max(1, hours))
            if x["model"]["edge"]:
                ax.plot(unwrap(x["model"]["lon"]), x["model"]["lat"], "x", color=c, ms=10, mew=2)
        bx.plot([x["lead_h"] for x in rows], [x["model"]["pmin_hpa"] for x in rows], "-s", color=c, label=label)
    bx.plot([h for h in range(0, hours + 1, 3)][:len(bt)], [b["pres_hpa"] for b in bt], "k-o", ms=3, lw=2,
            label="best track")
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg)")
    ax.grid(alpha=0.3)
    ax.set_aspect(1.0 / max(0.2, math.cos(math.radians(np.mean([b["lat"] for b in bt])))))
    ax.legend(fontsize=8, loc="best")
    ax.set_title("track every 6 h (labels: best-track lead h; thin boxes: nest frame at f00/12/24/36/48; x: edge)",
                 fontsize=9)
    bx.set_xlabel("lead (h)")
    bx.set_ylabel("central pressure (hPa)")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8)
    bx.set_title("minimum MSLP", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--start", required=True, help="YYYYMMDDHH")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--every-h", type=int, default=6)
    ap.add_argument("--run", action="append", required=True, help="LABEL=DIR (repeat)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--search-km", type=float, default=350.0)
    ap.add_argument("--edge-rows", type=int, default=15)
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    from namgpu.nest import ibtracs
    trk = ibtracs.track(a.sid)
    start = datetime.strptime(a.start, "%Y%m%d%H")
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for spec in a.run:
        label, _, d = spec.partition("=")
        res[label] = score_run(Path(d), trk, start, a.hours, a.every_h, a.search_km, a.edge_rows)
    rec = {"sid": a.sid, "name": trk["name"], "basin": trk["basin"], "licence": trk["licence"],
           "start": start.isoformat(), "runs": res,
           "method": "5x5 box-mean MSLP minimum within search_km of the previous centre; unsmoothed min within 60 km; "
                     f"edge = within {a.edge_rows} rows of the boundary; best track linear in time between IBTrACS points"}
    (a.out / "score.json").write_text(json.dumps(rec, indent=1))
    lines = [f"# {trk['name']} ({a.sid}), start {start:%Y-%m-%d %H}Z: track and central-pressure error vs IBTrACS",
             "", f"Best track: {trk['licence']}. Centre: {rec['method']}.", "",
             "| lead | best track | " + " | ".join(f"{l}: km / hPa (model min)" for l in res) + " |",
             "|---|---|" + "---|" * len(res)]
    leads = sorted({r["lead_h"] for v in res.values() for r in v["rows"]})
    for h in leads:
        cells, best = [], ""
        for v in res.values():
            r = next((x for x in v["rows"] if x["lead_h"] == h), None)
            if r is None or "err_km" not in r:
                cells.append("-")
                continue
            best = f"{r['best']['lat']:.1f}, {r['best']['lon']:.1f}, {r['best']['pres_hpa']}"
            cells.append(f"{r['err_km']:.0f} / {r['dp_hpa']:+.0f} ({r['model']['pmin_hpa']:.0f})"
                         + (" EDGE" if r["model"]["edge"] else ""))
        lines.append(f"| f{h:02d} | {best} | " + " | ".join(cells) + " |")
    lines += ["", "| run | mean track error f06-f48, km | mean abs central-pressure error, hPa | edge leads |",
              "|---|---|---|---|"]
    for l, v in res.items():
        s = v["summary"]
        lines.append(f"| {l} | {s['mean_err_km']} | {s['mean_abs_dp_hpa']} | {s['edge_leads'] or 'none'} |")
    (a.out / "SCORE.md").write_text("\n".join(lines) + "\n")
    chart(res, trk, start, a.hours, a.out / "track.png",
          a.title or f"WOOF-NAM 3 km: {trk['name']} from {start:%Y-%m-%d %H}Z against IBTrACS best track")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
