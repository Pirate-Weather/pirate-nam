"""Run the WHOLE NAM 12 km parent forecast for N hours, all physics on, wet.

    source /work/nam-gpu/env-box.sh       # box (card 3), or ~/nam-gpu/env.sh + pool lease on node-2
    python tests/smoke_full.py --case 2025-03-14_12z --hours 24 --out runs/2025-03-14_12z

Steps: read config -> build grid -> init state from the case (native B-grid analysis or GFS, plus GFS
boundaries) -> physics init (tables) -> DFI (filter_method from the config, TDFI = 3) -> time loop to
--hours with hourly history -> maps with rw_wrfbatch beside MRMS (cases/<date>/CASE.md lists the
referee files).  Writes <out>/smoke-receipt.json: init s, forecast s, ms/step, hour lines, non-finite
census and field ranges at the end, sanity verdict.

There is no dry, adiabatic, column or reduced-size mode.  Before the GPU is touched the driver
imports every unit on the plan and reports ALL missing units in one message.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import Ctx, Driver  # noqa: E402

# Blow-up limits for the end-of-run verdict (2 m T in K, surface pressure in Pa, wind in m/s).
LIMITS = {"T2": (180.0, 340.0), "PSFC": (50000.0, 110000.0)}
WIND_MAX = 150.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="cases/<case> directory name")
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="native", help="initial state: native (NAM analysis) or gfs")
    ap.add_argument("--skip-init", default="init-fetch-case",
                    help="init stages to skip (default: the fetch, when raw/ is already on disk)")
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args(argv)

    import cupy as cp
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    case_dir = ROOT / "cases" / a.case
    start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")

    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    drv = Driver(ctx)
    ctx.case_dir, ctx.start, ctx.start_time, ctx.out_dir, ctx.driver = case_dir, start, start, out, drv
    ctx.init_source = a.source
    ctx.forecast_hours = max(6, int(a.hours + 0.999))     # boundary files cover the whole run
    ctx.dt = cfg.dt
    drv.check_ready()

    # Whole-run units (UNITS.json): init-case-driver, drv-physics-initialize, drv-time-loop (TDFI + loop).
    from namgpu.init import init_case_driver  # noqa: PLC0415
    from namgpu.steps import drv_time_loop  # noqa: PLC0415

    skip = [s for s in a.skip_init.split(",") if s]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    rec = {"case": a.case, "source": a.source, "hours": a.hours, "dt": cfg.dt, "tdfi": int(cfg.filter_method),
           "skipped_init": skip, "error": None}
    hour_lines = []

    def log(msg):
        hour_lines.append(msg)
        print(msg, flush=True)

    t0 = time.perf_counter()
    with ctx.stream:
        rec["init"] = init_case_driver.run(ctx, only=chain)   # state, boundaries, physics init (one stage)
    ctx.stream.synchronize()
    rec["init_s"] = round(time.perf_counter() - t0, 2)
    print(f"init {rec['init_s']} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    # Init temporaries (GWD terrain, table builds) stay cached in the CuPy pool (40-55 GB on card 3,
    # run lead 0945Z) while the step loop needs ~17 GB plus ~6 GB radiation: give them back.
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    t1 = time.perf_counter()
    try:
        with ctx.stream:
            drv_time_loop.run(ctx, a.hours, log=log)          # TDFI (if on), then the forecast loop
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-4000:]
        print("RUN ERROR", rec["error"], flush=True)
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    rec["steps_done"] = int(ctx.ntimestep)
    rec["forecast_s"] = round(wall, 2)
    rec["hour_lines"] = hour_lines
    bad = init_case_driver.census(ctx)
    rec["nonfinite_end"] = bad
    stats = {}
    for nm in ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "T2", "TSKIN", "PSFC", "U10", "V10",
               "ACPREC", "CUPREC"):
        v = st.fields.get(nm)
        if v is not None and v.size:
            stats[nm] = [round(float(v.min()), 5), round(float(v.max()), 5), round(float(v.mean()), 6)]
    rec["stats_end"] = stats
    problems = []
    for nm, (lo, hi) in LIMITS.items():
        if nm in stats and (stats[nm][0] < lo or stats[nm][1] > hi):
            problems.append(f"{nm} {stats[nm][0]}..{stats[nm][1]} outside {lo}..{hi}")
    for nm in ("U", "V"):
        if nm in stats and max(abs(stats[nm][0]), abs(stats[nm][1])) > WIND_MAX:
            problems.append(f"{nm} beyond {WIND_MAX} m/s: {stats[nm][0]}..{stats[nm][1]}")
    if bad:
        problems.append(f"non-finite: {bad}")
    rec["problems"] = problems
    rec["sane"] = rec["error"] is None and not problems
    print(f"forecast {a.hours} h: steps {ctx.ntimestep}, {wall:.1f} s, sane {rec['sane']} {problems}", flush=True)
    print("end stats", json.dumps(stats), flush=True)
    (out / "smoke-receipt.json").write_text(json.dumps(rec, indent=1, default=str))

    if not a.no_maps and rec["error"] is None:
        from namgpu.io import maps  # noqa: PLC0415
        hist = sorted(out.glob("wrfout_d01_*"))
        mrms = case_dir / "raw" / "mrms"
        t2 = time.perf_counter()
        maps.main([*map(str, hist), "--out", str(out / "maps"), *(["--mrms", str(mrms)] if mrms.is_dir() else [])])
        rec["maps_s"] = round(time.perf_counter() - t2, 2)
        (out / "smoke-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    return 0 if rec["sane"] else 1


if __name__ == "__main__":
    sys.exit(main())
