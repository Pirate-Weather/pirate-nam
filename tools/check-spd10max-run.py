"""Real-run check of the hourly 10 m wind maximum (SPD10MAX, U10MAX, V10MAX) in the history.

    python tools/check-spd10max-run.py --case <case> --source gfs --hours 2 --out <dir>

Runs one real forecast in this process (woof-nam run, --no-maps) with a witness beside io-max-fields: on every
call that updates the 10 m extremes (a MAX_FIELDS_HR check step with NTSD > 0, the NSTEPS_PER_CHECK cadence NMMB
uses) it takes the 10 m speed of that step from the state's U10/V10 with plain CuPy, keeps the per-cell maximum
of the window, and restarts it at each hourly reset. When a history frame is written the witness maximum is
snapshotted. After the run every history file is read back and must satisfy, cell for cell:
  - SPD10MAX equals the witness maximum of the per-step speed over that history interval (m s-1, not squared);
  - hypot(U10MAX, V10MAX) equals SPD10MAX (the pair is the wind of the hour's strongest step);
  - no -999 start value anywhere (the start frame carries the speed of that instant).
Exit 0 when every frame passes. Defect: SPD10MAX was written as the squared speed (values over 3,000 on the
3 km hurricane nests, 2026-11-05).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RTOL = 2.0e-6          # float32: the kernel's u*u+v*v may contract to an FMA, the witness's does not
ATOL = 1.0e-5          # m s-1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="gfs")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case-dir", default=None)
    a = ap.parse_args(argv)

    import cupy as cp
    import numpy as np
    from namgpu.io import io_history_fields, io_max_fields
    from namgpu import run_forecast

    w = {"max2": None, "snap": {}, "updates": 0}
    orig_max, orig_hist = io_max_fields.run, io_history_fields.run

    def max_run(ctx):
        r = orig_max(ctx)
        ntsd = int(ctx.ntimestep)
        nrst = io_max_fields.nsteps_per_reset(ctx.cfg)
        nchk = io_max_fields.nsteps_per_check(ctx.cfg.dt)
        u, v = ctx.st["U10"], ctx.st["V10"]
        if w["max2"] is None or w["max2"].shape != u.shape:
            w["max2"] = cp.full(u.shape, -1.0, dtype=cp.float32)
        if ntsd == 0 or ntsd % nrst == 0:
            w["max2"].fill(-1.0)                       # the hourly reset
        if ntsd % nchk == 0 and ntsd > 0:              # the 10 m block of MAX_FIELDS_HR ran on this step
            cp.maximum(w["max2"], u * u + v * v, out=w["max2"])
            w["updates"] += 1
        return r

    def hist_run(ctx):
        out = orig_hist(ctx)
        u, v = ctx.st["U10"], ctx.st["V10"]
        m2 = w["max2"] if w["max2"] is not None else cp.full(u.shape, -1.0, dtype=cp.float32)
        m2 = cp.where(m2 < 0, u * u + v * v, m2)       # no sample yet (the start frame): the speed of that instant
        if ctx.stream is not None:
            ctx.stream.synchronize()
        w["snap"][int(ctx.ntimestep)] = cp.sqrt(m2).get()
        return out

    io_max_fields.run, io_history_fields.run = max_run, hist_run
    args = ["--case", a.case, "--source", a.source, "--hours", str(a.hours), "--out", str(a.out), "--no-maps",
            "--no-columns"]
    if a.case_dir:
        args += ["--case-dir", a.case_dir]
    rc = run_forecast.main(args)
    if rc not in (0, None):
        print(f"forecast rc {rc}", flush=True)
        return 1

    import netCDF4
    from namgpu import paths
    out = a.out if a.out.is_absolute() else paths.work_root() / a.out    # as woof-nam run resolves --out
    hist = sorted(out.glob("wrfout_d01_*")) or sorted(out.rglob("wrfout_d01_*"))
    steps = sorted(w["snap"])
    print(f"history files {len(hist)}, witness snapshots at steps {steps}, witness updates {w['updates']}", flush=True)
    if len(hist) != len(steps):
        print(f"FAIL: {len(hist)} history files vs {len(steps)} snapshots", flush=True)
        return 1
    bad = 0
    for path, step in zip(hist, steps):
        with netCDF4.Dataset(path) as d:
            spd = np.asarray(d["SPD10MAX"][0], dtype=np.float32)
            ux, vx = np.asarray(d["U10MAX"][0]), np.asarray(d["V10MAX"][0])
            u10, v10 = np.asarray(d["U10"][0]), np.asarray(d["V10"][0])
        ref = w["snap"][step]
        e1 = np.abs(spd - ref)
        ok1 = bool(np.all(e1 <= ATOL + RTOL * np.abs(ref)))
        e2 = np.abs(np.hypot(ux, vx) - spd)
        ok2 = bool(np.all(e2 <= 1.0e-4 + 1.0e-5 * spd))
        ok3 = bool(np.all(spd > -1.0) and np.all(np.abs(ux) < 900) and np.all(np.abs(vx) < 900))
        print(f"{path.name} step {step}: SPD10MAX {spd.min():.3f}..{spd.max():.3f} m/s (|U10| max "
              f"{np.hypot(u10, v10).max():.3f}); vs witness max|d| {e1.max():.2e} {'PASS' if ok1 else 'FAIL'}; "
              f"hypot(U10MAX,V10MAX) max|d| {e2.max():.2e} {'PASS' if ok2 else 'FAIL'}; "
              f"no start value {'PASS' if ok3 else 'FAIL'}", flush=True)
        bad += (not ok1) + (not ok2) + (not ok3)
    print("PASS" if bad == 0 else f"FAIL ({bad})", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
