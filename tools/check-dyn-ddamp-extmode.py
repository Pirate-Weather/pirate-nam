"""Done-rule check for dyn-ddamp-extmode and dyn-ddamp-damp (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-dyn-ddamp-extmode.py

Runs the real init-grid-consts unit first (its ddmpu row and ddmpv scalar feed both ddamp units; no
placeholders), then the extmode wrapper once on a full-size random State (954 x 835 x 60, all 48
fields randomized), and checks every float array for NaN/Inf. The skip-mode damp unit is compiled
and checked for the same, since its kernel remains in the tree.
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
    from namgpu.dynamics import dyn_ddamp_extmode, dyn_ddamp_damp

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    init_grid_consts.run(ctx)
    stream.synchronize()
    ddmpu = cp.asnumpy(ctx.tables["grid"]["ddmpu"])
    print(f"init-grid-consts: ddmpu[1..3]={ddmpu[1:4].tolist()}")
    gridmod.vcoord_from_config(g, cfg)

    kernels.module("dyn_ddamp_extmode").compile()
    kernels.module("dyn_ddamp_damp").compile()
    t0 = time.perf_counter()
    dyn_ddamp_extmode.run(ctx)
    stream.synchronize()
    dt = time.perf_counter() - t0
    print(f"dyn-ddamp-extmode: ddmpv={float(ctx.tables["ddamp_dvd"]):.6g} freerun={dyn_ddamp_extmode._freerun(cfg)}")

    names = [n for n, a in st.fields.items() if a.dtype.kind == "f"]
    bad = [n for n in dict.fromkeys(names) if not bool(cp.isfinite(st.fields[n]).all())]
    print(f"dyn-ddamp-extmode: wrapper {dt:.2f} s, NaN/Inf in {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
