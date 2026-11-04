"""Done-rule check for dyn-dht (review fix): the wrapper binds FCP/FDIV from init-grid-consts output.

    source env.sh
    python tools/check-dyn-dht.py

Runs init-grid-consts for real (publishes ctx.tables['grid']), then dyn-dht once on a full-size random
State (954 x 835 x 60). PCX/PCY/PCNE/PCNW are random stand-ins for the dyn-pgforce-pgf output, added
with st.add so the presence check passes. Checks every float array for NaN/Inf.
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
    from namgpu.init import init_grid_consts
    from namgpu.dynamics import dyn_dht

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    shape3 = (g.lm, g.jm, g.im)
    rng = cp.random.default_rng(2)
    for n in ("PCX", "PCY", "PCNE", "PCNW"):
        st.add(n, shape3)[...] = rng.random(shape3, dtype=cp.float32)

    init_grid_consts.run(ctx)
    kernels.module("dyn_dht").compile()
    t0 = time.perf_counter()
    dyn_dht.run(ctx)
    stream.synchronize()
    dt = time.perf_counter() - t0

    names = [n for n, a in st.fields.items() if a.dtype.kind == "f"]
    bad = [n for n in dict.fromkeys(names) if not bool(cp.isfinite(st.fields[n]).all())]
    print(f"global={bool(cfg.global_)} lm={g.lm} im={g.im} jm={g.jm}")
    print(f"dyn-dht: wrapper {dt:.2f} s, NaN/Inf in {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
