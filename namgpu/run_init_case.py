"""Build the real initial state for a case (init group integration run).

    woof-nam init --case 2025-03-14_12z --source native --out runs/init_native
    woof-nam init --case 2025-03-14_12z --source gfs    --out runs/init_gfs
    (same as python -m namgpu.run_init_case ...; python tools/run_init_case.py ... in a checkout)

Runs the init chain (init-case-driver) on the full 954 x 835 x 60 grid, writes the 0 h history
file through the history units and draws maps with maps.py (rw_wrfbatch).  --step runs one full
model step (first step + step 1, all physics) after init.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from namgpu import paths  # noqa: E402


def main(argv=None):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Build a case's initial state on the full NAM 12 km grid (954 x 835 x 60), write its 0 h "
                    "history file and draw its maps, without running a forecast. Uses the GPU; the memory "
                    "need is close to a forecast's.",
        epilog="example: woof-nam init --case 2025-03-14_12z --source native --out runs/init_native",
        formatter_class=HelpFormatter)
    ap.add_argument("--case", required=True,
                    help="the case folder woof-nam fetch wrote, or its bare name under <work root>/cases/")
    ap.add_argument("--source", default="native", help="native | gfs | hrrr | rrfs (init source row)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--only", nargs="*", default=None, help="run only these init units (default: the whole chain)")
    ap.add_argument("--no-maps", action="store_true", help="write the 0 h history only, no maps")
    ap.add_argument("--steps", type=int, default=0, help="model steps after init (all physics)")
    a = ap.parse_args(argv)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    cfg = config.load()
    cfg.raw["init_source"] = a.source
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = paths.case_dir(a.case)
    ctx.start = datetime.strptime(ctx.case_dir.name[:14], "%Y-%m-%d_%Hz")
    ctx.out_dir = a.out
    ctx.init_source = a.source
    a.out.mkdir(parents=True, exist_ok=True)
    from namgpu.init import init_case_driver  # noqa: PLC0415
    t0 = time.perf_counter()
    with ctx.stream:
        init_case_driver.run(ctx, only=a.only)
    ctx.stream.synchronize()
    print(f"init wall {time.perf_counter() - t0:.1f} s, state {st.nbytes() / 2**30:.2f} GiB,"
          f" peak pool {cp.get_default_memory_pool().total_bytes() / 2**30:.2f} GiB")
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    ctx.start_time = ctx.start
    with ctx.stream:
        drv.write_history()
        for _ in range(a.steps):
            drv.step()
            ctx.stream.synchronize()
            bad = init_case_driver.census(ctx)
            print(f"step {ctx.ntimestep} done, nonfinite: {bad or 'none'}")
            if bad:
                raise SystemExit(f"NaN/Inf after step {ctx.ntimestep}: {bad}")
    ctx.stream.synchronize()
    from namgpu.io import io_history  # noqa: PLC0415
    io_history.drain(ctx)
    if not a.no_maps:
        from namgpu.io import io_maps  # noqa: PLC0415
        print("maps ->", io_maps.run(ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
