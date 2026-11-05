"""woof-nam columns: the any-point column store (gpuwm's rw_soundings --columns, `.wcol` per history hour).

    woof-nam columns runs/2026-11-04_18z                       (every wrfout_d01_* of a run -> <run>/columns/d01/)
    woof-nam columns runs/x/wrfout_d01_2026-11-04_19:00:00 --out runs/x/columns     (one hour, as it lands)

The store reader takes WRF's conventions, which the WOOF-NAM history does not use (io_history_fields.py): WRF's
C-grid U, V and W (one more point along x, y and z), pressure and geopotential as perturbation plus base (P + PB,
PH + PHB) and the grid-to-earth rotation (COSALPHA, SINALPHA). So each history file is first written as a WRF view
in a scratch folder (default /dev/shm), the store is written from the view, and the view is deleted:
  - U, V, W: the history's earth-relative winds and W at mass points, averaged to the cell faces (the end faces
    take the end values); the reader averages them back to mass points, so a sounding's wind is a 1-2-1 average
    of three neighbouring columns along that component's axis (U along x, V along y, W in the vertical);
  - P = PRES (full pressure), PB = 0; PHB = 0 (PH is already the full geopotential);
  - COSALPHA = 1, SINALPHA = 0 (the winds are earth-relative already);
  - every other field as written.
The output is `<out>/d01/<YYYYMMDDTHHMMSSZ>.wcol`, the layout WOOF's worker reads (it takes a store the engine
wrote beside the history as it is). Frames run in parallel processes, one per frame, with the cores shared out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FRAME = re.compile(r"^wrfout_(d\d\d)_(\d{4})-(\d\d)-(\d\d)_(\d\d):(\d\d):(\d\d)$")


def _faces(x, axis):
    """Mass-point values -> face values along `axis` (n -> n + 1): interior faces are the average of the two
    neighbours, the end faces take the end values."""
    import numpy as np
    axis %= x.ndim
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = n + 1
    y = np.empty(shape, x.dtype)

    def at(s):
        return tuple(s if i == axis else slice(None) for i in range(x.ndim))
    y[at(slice(1, n))] = 0.5 * (x[at(slice(0, n - 1))] + x[at(slice(1, n))])
    y[at(slice(0, 1))] = x[at(slice(0, 1))]
    y[at(slice(n, n + 1))] = x[at(slice(n - 1, n))]
    return y


STAGGER = {"U": ("west_east_stag", -1), "V": ("south_north_stag", -2), "W": ("bottom_top_stag", -3)}


def wrf_view(src: Path, dst: Path) -> Path:
    """Write the WRF-convention view of one WOOF-NAM history file (see the module docstring)."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w", format="NETCDF4") as b:
        for k, d in a.dimensions.items():
            b.createDimension(k, None if d.isunlimited() else len(d))
        b.createDimension("west_east_stag", len(a.dimensions["west_east"]) + 1)
        b.createDimension("south_north_stag", len(a.dimensions["south_north"]) + 1)
        if "bottom_top_stag" not in b.dimensions:
            b.createDimension("bottom_top_stag", len(a.dimensions["bottom_top"]) + 1)
        b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})

        def put(name, dims, data, like=None):
            v = b.createVariable(name, data.dtype if hasattr(data, "dtype") else "f4", dims)
            if like is not None:
                v.setncatts({k: like.getncattr(k) for k in like.ncattrs() if k != "_FillValue"})
            v[:] = data
        for name, v in a.variables.items():
            dims = list(v.dimensions)
            data = v[:]
            if name in STAGGER and v.ndim == 4:
                dim, axis = STAGGER[name]
                dims[axis] = dim
                data = _faces(np.asarray(data), axis)
            put(name, tuple(dims), data, v)
        pres = a.variables["PRES"]
        put("P", pres.dimensions, np.asarray(pres[:]), pres)
        put("PB", pres.dimensions, np.zeros(pres.shape, np.float32))
        ph = a.variables["PH"]
        put("PHB", ph.dimensions, np.zeros(ph.shape, np.float32))
        hgt = a.variables["HGT"]
        put("COSALPHA", hgt.dimensions, np.ones(hgt.shape, np.float32))
        put("SINALPHA", hgt.dimensions, np.zeros(hgt.shape, np.float32))
    return dst


def _one(frame: str, out: str, scratch: str, threads: int, exe: str) -> dict:
    """View + store for one history file, in its own process."""
    t0 = time.perf_counter()
    name = Path(frame).name
    with tempfile.TemporaryDirectory(prefix="woof-nam-columns-", dir=scratch) as tmp:
        view = wrf_view(Path(frame), Path(tmp) / name)
        t1 = time.perf_counter()
        req = Path(tmp) / "request.json"
        req.write_text(json.dumps({"sources": [{"path": str(view), "domain": FRAME.match(name)[1], "member": "det"}],
                                   "out": str(out), "tile": 16, "threads": threads}))
        r = subprocess.run([exe, "--columns", str(req)], capture_output=True, text=True)
    rec = {"frame": name, "rc": r.returncode, "view_s": round(t1 - t0, 2), "store_s": round(time.perf_counter() - t1, 2)}
    for line in r.stdout.splitlines():
        if line.startswith("COLUMNS "):
            path, *fields = line[len("COLUMNS "):].rsplit(" ", 2)
            rec.update(path=path, **dict(f.split("=", 1) for f in fields))
    if r.returncode:
        rows = [x for x in r.stderr.splitlines() if x.strip()]
        rec["error"] = (rows[-1] if rows else f"exit {r.returncode}")[:300]
    return rec


def frames_of(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.glob("wrfout_d*") if FRAME.match(f.name))
        elif FRAME.match(p.name):
            out.append(p)
    return out


def write(frames: list[Path], out: Path, workers: int | None = None, scratch: str | None = None) -> dict:
    """The store for `frames` under `out` (<out>/<domain>/<stamp>.wcol). Returns {"files", "seconds", "failed"}."""
    from gpuwm.soundings import binary
    exe = str(binary())
    t0 = time.perf_counter()
    cores = os.cpu_count() or 2
    workers = max(1, min(workers or 4, len(frames) or 1))
    threads = max(1, cores // workers)
    scratch = scratch or ("/dev/shm" if Path("/dev/shm").is_dir() else None)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(_one, [str(f) for f in frames], [str(out)] * len(frames), [scratch] * len(frames),
                           [threads] * len(frames), [exe] * len(frames)))
    failed = [r for r in recs if r["rc"]]
    return {"files": recs, "seconds": round(time.perf_counter() - t0, 2), "failed": len(failed),
            "parallelism": f"{workers} processes x {threads} threads"}


def main(argv=None) -> int:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Write the any-point column store (gpuwm rw_soundings, one .wcol file per history hour) for a "
                    "run folder or for single history files. A sounding at any point and hour is then read from the "
                    "store (gpuwm soundings). Runs on all cores; no GPU.",
        epilog="examples:\n  woof-nam columns runs/2025-03-14_12z\n"
               "  woof-nam columns runs/x/wrfout_d01_2025-03-14_13:00:00 --out runs/x/columns",
        formatter_class=HelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="a run folder (every wrfout_d01_*) or history files")
    ap.add_argument("--out", type=Path, default=None, help="store folder (default <run>/columns)")
    ap.add_argument("--workers", type=int, default=4, help="frames written at once (default 4)")
    a = ap.parse_args(argv)
    frames = frames_of(a.paths)
    if not frames:
        ap.error("no wrfout_dNN_YYYY-MM-DD_HH:MM:SS history files in " + ", ".join(map(str, a.paths)))
    out = a.out or ((a.paths[0] if a.paths[0].is_dir() else a.paths[0].parent) / "columns")
    res = write(frames, out, a.workers)
    for r in res["files"]:
        print(f"{r['frame']}: " + (f"{r.get('path')} {r.get('bytes')} bytes" if not r["rc"] else f"FAILED {r['error']}")
              + f" (view {r['view_s']} s, store {r['store_s']} s)", flush=True)
    print(f"columns: {len(frames) - res['failed']} of {len(frames)} in {res['seconds']} s ({res['parallelism']}) "
          f"-> {out}", flush=True)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
