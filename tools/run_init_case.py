"""Build the real initial state for a case (init group integration run).

    source env.sh
    python tools/run_init_case.py --case 2025-03-14_12z --source native --out runs/init_native
    python tools/run_init_case.py --case 2025-03-14_12z --source gfs    --out runs/init_gfs

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import Ctx, Driver  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native", help="native | gfs | hrrr | rrfs (init source row)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="model steps after init (all physics)")
    a = ap.parse_args(argv)

    import cupy as cp
    cfg = config.load()
    cfg.raw["init_source"] = a.source
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
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
