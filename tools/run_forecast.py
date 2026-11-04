"""Run lead: a real NAM forecast through the documented driver entry (Driver.run), history every
minutes_history, then maps (io-maps, rw_wrfbatch).  Radiation is left out (mode skip) only while
ctx.tables['rrtm_axsd'] has no producer; the receipt names it.

    source /work/nam-gpu/env-box.sh
    python tools/run_forecast.py --case 2025-03-14_12z --source native --hours 1 --out scratch/runlead/f1h
Writes <out>/forecast-receipt.json (init s, per-hour wall, ms/step, nonfinite census at the end,
peak VRAM, units left out) plus the driver's timing_receipt.json, history files and maps/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def release_card(ctx) -> dict:
    """Free this run's device memory and unlock the card lock this process inherited (rl_run.sh and the
    lane runners hold it on an inherited fd; LOCK_UN on that fd releases it for the parent too). The lock
    file is NAMGPU_CARD_LOCK (default: any open file named CARD3.lock). Returns what was done."""
    import fcntl
    import gc
    import os

    import cupy as cp
    ctx.stream.synchronize()
    ctx.st.fields.clear()
    ctx.tables.clear()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    used = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    want = os.environ.get("NAMGPU_CARD_LOCK", "")
    dropped = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            tgt = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if (want and os.path.realpath(want) == tgt) or (not want and os.path.basename(tgt) == "CARD3.lock"):
            fcntl.flock(int(fd), fcntl.LOCK_UN)
            os.close(int(fd))
            dropped.append(tgt)
    print(f"card released: pool freed, card used {used} GiB, lock dropped {dropped or 'none held'}", flush=True)
    return {"card_used_gib_after": used, "locks_dropped": dropped}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-init", default="init-fetch-case")
    ap.add_argument("--leave-out", default="", help="comma list of plan units set to mode skip (receipt lists them)")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--maps-only", action="store_true", help="internal: the post-run maps tail (no CUDA)")
    a = ap.parse_args(argv)
    if a.maps_only:
        return maps_only(a.case, a.out if a.out.is_absolute() else ROOT / a.out)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver

    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.out_dir = out
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    skip = [s for s in a.skip_init.split(",") if s]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    rec = {"case": a.case, "source": a.source, "hours": a.hours, "dt": cfg.dt, "skipped_init": skip}
    t0 = time.perf_counter()
    with ctx.stream:
        rec["init"] = init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    rec["init_s"] = round(time.perf_counter() - t0, 2)
    print(f"init done {rec['init_s']} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    # Init temporaries (GWD terrain, table builds) stay cached in the CuPy pool and showed up as 40-55 GB on
    # card 3 for a forecast whose step loop needs ~17 GB. Return them.
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    rec["after_init_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    rec["pool_after_init_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)

    leave = [u for u in a.leave_out.split(",") if u]
    for u in leave:
        drv.units[u]["mode"] = "skip"
    rec["left_out"] = leave

    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()

    hours_log = []

    def log(msg):
        hours_log.append(msg)
        print(msg, flush=True)

    t1 = time.perf_counter()
    try:
        n = drv.run(a.hours, log=log)
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        n = ctx.ntimestep
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-3000:]
        print("RUN ERROR", rec["error"], flush=True)
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    rec["steps_done"] = int(ctx.ntimestep)
    rec["run_s"] = round(wall, 2)
    rec["ms_per_step"] = round(wall / max(1, ctx.ntimestep) * 1e3, 1)
    rec["hour_lines"] = hours_log
    bad = init_case_driver.census(ctx)
    rec["nonfinite_end"] = bad
    SANE = ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "T2", "TSKIN", "ACPREC", "PREC", "CUPREC")
    sx = {}
    for nm in SANE:
        v = st.fields.get(nm)
        if v is not None:
            sx[nm] = [round(float(v.min()), 5), round(float(v.max()), 5), round(float(v.mean()), 6)]
    rec["stats_end"] = sx
    rec["pool_end_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)
    rec["card_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    print("end stats", json.dumps(sx), "nonfinite", bad, flush=True)
    rd.__exit__(None, None, None)
    if not a.no_maps and rec["error"] is None:
        # Maps are host work. Before them: drop the card lock and give back ALL of the card, so the next
        # run starts while the frames render (run lead 11:05Z: the old post-run maps held CARD3.lock and
        # 62 GB for ~50 min on one core). Freeing the pool still left 10.7 GB in unit caches, so this process
        # re-executes itself as the maps-only tail (same PID for the runner, no CUDA context left).
        from namgpu.io.io_history import drain
        drain(ctx)                                   # the last history frame is on disk
        rec["card_released"] = release_card(ctx)
        rec["exit_code"] = 1 if (rec["error"] or bad) else 0
        (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--maps-only",
                                  "--case", a.case, "--out", str(out)])
    (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {out / 'forecast-receipt.json'}", flush=True)
    return 1 if (rec["error"] or bad) else 0


def maps_only(case: str, out: Path) -> int:
    """Post-run maps in a process with no CUDA context: render every history frame of <out> not yet drawn
    (frames an hourly watcher rendered are skipped) across all cores, then finish the receipt."""
    from types import SimpleNamespace

    from namgpu.io import io_maps
    rp = out / "forecast-receipt.json"
    rec = json.loads(rp.read_text())
    ctx = SimpleNamespace(out_dir=out, case_dir=ROOT / "cases" / case,
                          start=datetime.strptime(case[:14], "%Y-%m-%d_%Hz"))
    t2 = time.perf_counter()
    try:
        rec["maps"] = str(io_maps.run(ctx))[:2000]
    except Exception as e:  # noqa: BLE001
        rec["maps_error"] = f"{type(e).__name__}: {e}"
        print("MAPS ERROR", rec["maps_error"], flush=True)
    rec["maps_s"] = round(time.perf_counter() - t2, 2)
    print(f"maps {rec.get('maps', '')} in {rec['maps_s']} s", flush=True)
    rp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {rp}", flush=True)
    return int(rec.get("exit_code", 0)) or (1 if "maps_error" in rec else 0)


if __name__ == "__main__":
    sys.exit(main())
