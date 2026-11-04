"""Done-rule check for bnd-bocoh-smooth (full size, explicit nsmud=1, no NaN/Inf).

    source ~/nam-gpu/env.sh
    python tools/check-bnd-bocoh-smooth.py

The strip buffers come from the real producer (bnd-boundary-init) on a full-size random State,
then bnd-bocoh-smooth runs once with nsmud=1 under one pool-v3 lease. Every strip buffer and PINT
is checked for NaN/Inf, and the count of strip cells that moved is reported (must be > 0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.boundary import bnd_boundary_init, bnd_bocoh_smooth
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    ctx.bc = {}

    kernels.module("bnd_boundary_init").compile()
    kernels.module("bnd_bocoh_smooth").compile()
    with gpu_lease(vram_mib=4096, est_s=600, cards=1, what="bnd-bocoh-smooth full-size NaN check"):
        bnd_boundary_init.run(ctx)
        stream.synchronize()
        before = {k: v[0].copy() for k, v in ctx.bc.items()}
        t0 = time.perf_counter()
        bnd_bocoh_smooth.run(ctx, nsmud=1)
        stream.synchronize()
        dt = time.perf_counter() - t0

    bad = [k for k, v in ctx.bc.items() if not bool(cp.isfinite(v).all())]
    if not bool(cp.isfinite(st.PINT).all()):
        bad.append("PINT")
    moved = {k: int(cp.count_nonzero(ctx.bc[k][0] != before[k])) for k in ctx.bc}
    total = sum(moved.values())
    print(f"im={g.im} jm={g.jm} lm={g.lm} lnsh={cfg.lnsh} buffers={len(ctx.bc)}")
    print(f"bnd-bocoh-smooth nsmud=1: wrapper {dt:.2f} s, strip cells moved {total}")
    print(f"NaN/Inf in {bad if bad else 'none'}")
    return 0 if (not bad and total > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
