"""Weather maps from WOOF-NAM history files with the Rust renderer rw_wrfbatch (render law: no
matplotlib for weather fields), drawn exactly like the case referee maps (same CF table, same CONUS
bounds, same colour scales), so a WOOF-NAM panel sits beside the archived NAM and MRMS panels.

Usage:  python -m namgpu.io.maps <wrfout ...> --out <dir> [--products composite_reflectivity,t2m,...]

Route: rw_wrfbatch --cf on the history's own curvilinear XLAT/XLONG (H points). The renderer's wrfout
route does not place MAP_PROJ 203 (the rotated lat-lon B grid): it rendered the 12 km parent as a band
across the top of the frame (2026-11-04, runs/init_native), so it is not used. The CF route is the one
the referee maps use for the NAM native B grid (cases/<case>/referee/make_referee.py).

Products (CF variable, from the history fields):
  composite_reflectivity  refc   COMPOSITE (dBZ); MRMS overlay where the case has the hour
  t2m                     t2m    T2 (K -> degF)
  d2m                     d2m    dewpoint from Q2 (2 m q) and PSFC (Bolton 1980 inverse), degF
  wind10                  ws10m  |U10, V10| (m/s, drawn in kt)
  wind850                 ws850  |U, V| at 850 hPa, log-p linear between the bracketing layers of PRES;
                                 blank where 850 hPa is below ground
  mslp                    mslp   MSLP (NMMB QUASIPOST membrane MSLP, Pa -> hPa)
  qpf_1h                  qpf1h  RAINNC + RAINC difference from the history one hour earlier (mm)
A product whose input is not in a frame is SKIPPED with the reason (composite reflectivity before the
microphysics has run, qpf_1h on the first frame); the caller decides what a skip means.
Host work, all cores: each drawn frame is read (only the variables its products need), derived and
written as CF in its own worker process; renderer subprocesses then run in parallel threads, one per
product and frame.
"""
from __future__ import annotations

import re
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DEFAULT_PRODUCTS = "composite_reflectivity,t2m,d2m,wind10,wind850,mslp,qpf_1h"
BOUNDS = "-126,-66,22,52"            # the referee maps' CONUS frame
RECEIPT_ENV = "NAMGPU_MAPS_RECEIPT"

# The standard 1 h precipitation ladder (in), as the CF table's "scale" (the renderer takes exactly ten edges,
# nine colour bands). Amounts under the first edge are left blank.
QPF1H_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
QPF1H_COLORS = [[139, 205, 132], [66, 170, 72], [21, 120, 36], [255, 250, 140], [255, 200, 0],
                [255, 130, 0], [230, 30, 0], [150, 0, 30], [200, 0, 200]]

# product -> (CF variable, CF table row); rows match cases/<case>/referee/cf-table.json
TABLE = {
    "composite_reflectivity": ("refc", {"names": ["refc"], "standard_names": [], "product": "composite_reflectivity",
                                        "title": "Composite reflectivity", "recipe": "composite_reflectivity",
                                        "default_units": "dBZ", "display_units": "dBZ", "unit_factors": {"dBZ": 1.0},
                                        "observation_product": "refc"}),
    "t2m": ("t2m", {"names": ["t2m"], "standard_names": [], "product": "t2m", "title": "2 m temperature",
                    "recipe": "2m_temperature", "default_units": "degF", "display_units": "degF",
                    "unit_factors": {"degF": 1.0}}),
    "d2m": ("d2m", {"names": ["d2m"], "standard_names": [], "product": "d2m", "title": "2 m dewpoint",
                    "recipe": "2m_dewpoint", "default_units": "degF", "display_units": "degF",
                    "unit_factors": {"degF": 1.0}}),
    "wind10": ("ws10m", {"names": ["ws10m"], "standard_names": [], "product": "wind10", "title": "10 m wind speed",
                         "recipe": "10m_wind_speed_and_direction", "default_units": "m s-1", "display_units": "kt",
                         "unit_factors": {"m s-1": 1.9438445}}),
    "wind850": ("ws850", {"names": ["ws850"], "standard_names": [], "product": "wind850",
                          "title": "850 hPa wind speed", "recipe": "10m_wind_speed_and_direction",
                          "default_units": "m s-1", "display_units": "kt", "unit_factors": {"m s-1": 1.9438445}}),
    "mslp": ("mslp", {"names": ["mslp"], "standard_names": [], "product": "mslp", "title": "Mean sea level pressure",
                      "recipe": "href_mean_mslp", "default_units": "hPa", "display_units": "hPa",
                      "unit_factors": {"hPa": 1.0}}),
    "qpf_1h": ("qpf1h", {"names": ["qpf1h"], "standard_names": [], "product": "qpf_1h", "title": "1 h precipitation",
                         "recipe": "total_qpf", "default_units": "kg m-2", "display_units": "in",
                         "unit_factors": {"kg m-2": 0.0393700787},
                         # the standard 1 h precipitation ladder (in), not total_qpf's run-total 0-15 in
                         # scale, which painted every real hourly amount in its first colour (judge 11:30Z)
                         "scale": {"levels": QPF1H_LEVELS, "colors": QPF1H_COLORS}}),
}

# history variables each product reads (the read takes only these, not every 3-D field of the file)
NEEDS = {"composite_reflectivity": ("COMPOSITE",), "t2m": ("T2",), "d2m": ("Q2", "PSFC"),
         "wind10": ("U10", "V10"), "wind850": ("U", "V", "PRES"), "mslp": ("MSLP",),
         "qpf_1h": ("RAINNC", "RAINC")}


def _parse(argv):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Draw weather maps from WOOF-NAM history files (wrfout_d01_*) with gpuwm's Rust "
                    "renderer: composite reflectivity, 2 m temperature and dewpoint, 10 m and 850 hPa "
                    "wind, MSLP and 1 h precipitation. A forecast draws its own maps when it ends; use "
                    "this to redraw them. Host only, no GPU.",
        epilog="example: woof-nam maps runs/2025-03-14_12z/wrfout_d01_* --out runs/2025-03-14_12z/maps",
        formatter_class=HelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="history files (wrfout_d01_<valid time>)")
    ap.add_argument("--out", type=Path, required=True, help="output folder; maps go to <out>/<product>/")
    ap.add_argument("--products", default=DEFAULT_PRODUCTS, help=f"comma list (default {DEFAULT_PRODUCTS})")
    ap.add_argument("--frames", default="all", help="which history frames to draw (default all)")
    ap.add_argument("--width", type=int, default=1200, help="image width in pixels (default 1200)")
    ap.add_argument("--height", type=int, default=900, help="image height in pixels (default 900)")
    ap.add_argument("--mrms", type=Path, default=None, help="folder of mrms_refc_YYYYmmddHH.grib2 (overlay)")
    ap.add_argument("--label", default=None, help="model name in the map titles (default: WOOF-NAM and the configured grid, e.g. WOOF-NAM 12 km)")
    a = ap.parse_args(argv)
    if a.label is None:
        from namgpu import config as _config   # the domain configfile (NAMGPU_CONFIGFILE) names the panel
        _d = _config.load().raw.get("domain_name")
        _m = re.fullmatch(r"nam(\d+)(?:km)?(band)?", _d or "")
        if _m:
            a.label = f"WOOF-NAM {_m.group(1)} km" + (" band" if _m.group(2) else "")
        else:
            a.label = f"WOOF-NAM {_d}" if _d else "WOOF-NAM 12 km"
    return a


def _bridges() -> Path:
    env = os.environ.get("GPUWM_BRIDGES")
    if env and (Path(env) / "rw_wrfbatch").is_file():
        return Path(env)
    from gpuwm import rustwx  # noqa: PLC0415
    r = rustwx.find_renderer()
    if r is None:
        raise SystemExit(rustwx.renderer_remedy())
    return Path(r).parent


def _k2f(k):
    return (k - 273.15) * 1.8 + 32.0


def _dewpoint_k(q, p):
    e = np.maximum(q * p / (0.622 + 0.378 * q), 1e-3) / 100.0          # hPa
    ln = np.log(e / 6.112)
    return 273.15 + 243.5 * ln / (17.67 - ln)


def _at_pressure(a, pres, target):
    """Log-p linear value of a [L, J, I] at pressure `target` (Pa); NaN below ground or above the top."""
    lp = np.log(pres)
    lt = np.log(target)
    above = pres <= target                                             # file is bottom-up
    k = np.argmax(above, axis=0)                                       # first level at or above target
    ok = above.any(axis=0) & (k > 0)
    k = np.clip(k, 1, a.shape[0] - 1)
    jj, ii = np.indices(k.shape)
    p0, p1 = lp[k - 1, jj, ii], lp[k, jj, ii]
    w = (lt - p0) / np.where(p1 != p0, p1 - p0, 1.0)
    out = a[k - 1, jj, ii] + w * (a[k, jj, ii] - a[k - 1, jj, ii])
    return np.where(ok, out, np.nan)


def _times(path: Path):
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        t = "".join(c.decode() for c in nc.variables["Times"][0])
        return datetime.strptime(t, "%Y-%m-%d_%H:%M:%S"), datetime.strptime(nc.START_DATE, "%Y-%m-%d_%H:%M:%S")


def _read(path: Path, names=None):
    """(valid, start, {name: array}) of one history frame; with `names`, only those variables (a frame is
    ~1 GB of 3-D fields and the maps need a handful of 2-D ones)."""
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        v = nc.variables
        t = "".join(c.decode() for c in nc.variables["Times"][0])
        valid = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S")
        start = datetime.strptime(nc.START_DATE, "%Y-%m-%d_%H:%M:%S")
        get = {n: np.asarray(v[n][0], dtype=np.float64) for n in v
               if n not in ("Times", "XTIME") and v[n].dimensions[0] == "Time" and (names is None or n in names)}
    return valid, start, get


def fields_for(prev, cur, products):
    """CF fields of one frame: {cf_var: (array, units)}; skipped: [(product, reason)]."""
    valid, start, f, _ = cur
    out, skipped = {}, []
    for prod in products:
        var = TABLE[prod][0]
        if prod == "composite_reflectivity":
            if "COMPOSITE" not in f:
                skipped.append((prod, f"{cur[3].name}: no COMPOSITE in the history (the microphysics has not written REFL_10CM yet)"))
                continue
            out[var] = (np.maximum(f["COMPOSITE"], -30.0), "dBZ")
        elif prod == "t2m":
            out[var] = (_k2f(f["T2"]), "degF")
        elif prod == "d2m":
            out[var] = (_k2f(_dewpoint_k(np.maximum(f["Q2"], 0.0), f["PSFC"])), "degF")
        elif prod == "wind10":
            out[var] = (np.hypot(f["U10"], f["V10"]), "m s-1")
        elif prod == "wind850":
            out[var] = (np.hypot(_at_pressure(f["U"], f["PRES"], 85000.0),
                                 _at_pressure(f["V"], f["PRES"], 85000.0)), "m s-1")
        elif prod == "mslp":
            if "MSLP" not in f:
                skipped.append((prod, f"{cur[3].name}: no MSLP in the history (io-quasipost-driver did not run for it)"))
                continue
            out[var] = (f["MSLP"] / 100.0, "hPa")
        elif prod == "qpf_1h":
            if prev is None or (valid - prev[0]) != timedelta(hours=1):
                skipped.append((prod, f"{cur[3].name}: no history frame one hour earlier to difference"))
                continue
            tot = f["RAINNC"] + f["RAINC"]
            q = np.maximum(tot - (prev[2]["RAINNC"] + prev[2]["RAINC"]), 0.0)
            out[var] = (np.where(q >= QPF1H_LEVELS[0] * 25.4, q, np.nan), "kg m-2")   # below 0.01 in: blank
    return out, skipped


def _write_cf(path, lat, lon, valid, fields, label):
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path, "w") as nc:
        ny, nx = lat.shape
        nc.createDimension("time", 1)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.source = label
        t = nc.createVariable("time", "f8", ("time",))
        t.units = f"hours since {valid:%Y-%m-%d %H:%M:%S}"
        t.standard_name = "time"
        t[:] = [0.0]
        a = nc.createVariable("lat", "f4", ("y", "x"))
        a.standard_name, a.units = "latitude", "degrees_north"
        a[:] = lat
        b = nc.createVariable("lon", "f4", ("y", "x"))
        b.standard_name, b.units = "longitude", "degrees_east"
        b[:] = lon
        for name, (arr, units) in fields.items():
            v = nc.createVariable(name, "f4", ("time", "y", "x"), fill_value=np.float32(np.nan))
            v.units = units
            v[0] = arr.astype(np.float32)


def _prep_frame(job):
    """One drawn frame: read the variables its products need (and the earlier frame's rain for qpf_1h),
    derive the CF fields and write the frame's CF file. Runs in a worker process.
    Returns (cf path or None, valid, start, CF variables written, skipped)."""
    path, prev_path, need, products, work, label = job
    path = Path(path)
    valid, start, f = _read(path, set(need))
    prev = None
    if prev_path is not None:
        pv, ps, pf = _read(Path(prev_path), {"RAINNC", "RAINC"})
        prev = (pv, ps, pf, Path(prev_path))
    flds, sk = fields_for(prev, (valid, start, f, path), products)
    if not flds:
        return None, valid, start, [], sk
    lat, lon = f.get("XLAT"), f.get("XLONG")
    if lat is None or lon is None:
        raise RuntimeError(f"woof-nam maps: {path.name} has no XLAT/XLONG")
    lon = (lon + 180.0) % 360.0 - 180.0
    cf = Path(work) / f"namgpu_{valid:%Y%m%d%H%M}.nc"
    _write_cf(cf, lat, lon, valid, flds,
              f"{label}, init {start:%Y-%m-%d %HZ} +{(valid - start).total_seconds() / 3600:g} h")
    return str(cf), valid, start, sorted(flds), sk


def render(argv=None) -> dict:
    """Render history files. Returns {"written": [...], "failures": [...], "skipped": [[product, reason]]}.
    Does not raise on a renderer failure or a skip; the caller decides what they mean."""
    a = _parse(list(sys.argv[1:] if argv is None else argv))
    products = [p.strip() for p in a.products.split(",") if p.strip()]
    unknown = [p for p in products if p not in TABLE]
    if unknown:
        raise SystemExit(f"woof-nam maps: unknown products {unknown} (known: {list(TABLE)})")
    bridges = _bridges()
    renderer = bridges / "rw_wrfbatch"
    a.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cf-", dir=a.out))
    table = work / "cf-table.json"
    table.write_text(json.dumps([TABLE[p][1] for p in products]))

    files = sorted(a.files)
    # frame times first (header reads only), then each drawn frame is read, derived and written as CF in its
    # own process across the cores (the old loop read every variable of every frame on one core: ~1 min per
    # 914 MB frame, 49 frames ~50 min after a 48 h run, run lead 11:05Z)
    times = [_times(f) for f in files]           # header reads; netCDF4/HDF5 is not thread-safe, keep these serial
    order = sorted(range(len(files)), key=lambda k: times[k][0])
    files = [files[k] for k in order]
    first = 0 if a.frames == "all" else max(0, len(files) - int(a.frames))
    need = {"XLAT", "XLONG"}
    for p in products:
        need.update(NEEDS[p])
    preps = []
    for k in range(first, len(files)):
        prev = files[k - 1] if k > 0 else None
        preps.append((str(files[k]), str(prev) if prev else None, sorted(need), products, str(work), a.label))
    written, failures, skipped = [], [], []
    jobs = []
    workers = max(1, min(len(preps), (os.cpu_count() or 4) // 2, 48))
    if workers == 1:
        results = [_prep_frame(p) for p in preps]
    else:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        # spawn: the caller may hold a CUDA context, which a forked child must not inherit
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_prep_frame, preps))
    for cf, valid, start, have, sk in results:
        skipped += [[p, r] for p, r in sk]
        if cf is None:
            continue
        cf = Path(cf)
        fhr = int(round((valid - start).total_seconds() / 3600))
        for prod in products:
            var = TABLE[prod][0]
            if var not in have:
                continue
            obs = None
            if prod == "composite_reflectivity" and a.mrms is not None:
                src = a.mrms / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
                if src.is_file():
                    obs = work / f"MRMS_MergedReflectivityQCComposite_00.50_{valid:%Y%m%d-%H}0000.grib2"
                    if not obs.exists():
                        obs.symlink_to(src.resolve())
            dst = a.out / prod / f"namgpu_{start:%Y%m%d_%H}z_f{fhr:03d}_{prod}.png"
            jobs.append((cf, var, valid, obs, dst, prod))

    def one(job):
        cf, var, valid, obs, dst, prod = job
        od = work / f"r_{var}_{valid:%Y%m%d%H%M}"
        cmd = [renderer, "--cf", cf, "--variable", var, "--out-dir", od, "--valid",
               valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", BOUNDS, "--cf-table", table,
               "--width", str(a.width), "--height", str(a.height)]
        if obs is not None:
            cmd += ["--observation", obs]
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
        if p.returncode or not pngs:
            return None, f"{prod} {valid:%Y%m%d%H}: exit {p.returncode}: {(p.stderr or p.stdout).strip()[-400:]}"
        # with an observation the renderer also writes the side-by-side; keep the model panel as the product
        model = [q for q in pngs if "comparison" not in q.name and "observed" not in q.name] or pngs
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(model[0]), dst)
        extra = []
        for q in pngs:
            if q is model[0] or not q.exists():
                continue
            tag = "comparison" if "comparison" in q.name else ("observed" if "observed" in q.name else "extra")
            e = dst.with_name(dst.stem + f"_{tag}.png")
            shutil.move(str(q), e)
            extra.append(str(e))
        return [str(dst), *extra], None

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), os.cpu_count() or 4, 64))) as ex:
        for w, err in ex.map(one, jobs):
            if err:
                failures.append(err)
            else:
                written += w
    shutil.rmtree(work, ignore_errors=True)
    result = {"written": written, "failures": failures, "skipped": skipped}
    receipt = os.environ.get(RECEIPT_ENV)
    if receipt:
        Path(receipt).write_text(json.dumps(result))
    return result


def main(argv=None):
    result = render(argv)
    for path in result["written"]:
        print(f"RENDERED {path}")
    for slug, reason in result["skipped"]:
        print(f"SKIPPED {slug} {reason}")
    for line in result["failures"]:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
