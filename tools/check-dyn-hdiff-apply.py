"""Done-rule check for dyn-hdiff-apply (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-dyn-hdiff-apply.py

Builds the configured grid, runs the real producers of the wrapper's inputs in step order:
init-grid-consts (GRID_CONSTS), init-vcoord-recover's publish_vcoord_tables (LPT2, EPSQ2), then
dyn-hdiff-coef (DEF3D, FMLX_3D, FMLY_3D), then dyn-hdiff-apply, on a full-size random State
(954 x 835 x 60). No stand-in values. Checks every float array for NaN/Inf and prints wall time
for each wrapper (CUDA events are not used: the host clock brackets a stream sync).
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
    from namgpu.dynamics import dyn_hdiff_apply, dyn_hdiff_coef
    from namgpu.init import init_grid_consts, init_vcoord_recover

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    init_grid_consts.run(ctx)
    lpt2 = init_vcoord_recover.publish_vcoord_tables(ctx)
    assert lpt2 == int(np.sum(np.abs(g.sgml2) <= init_vcoord_recover.LPT2_TOL)), "LPT2 mismatch"

    kernels.module("dyn_hdiff_coef").compile()
    kernels.module("dyn_hdiff_apply").compile()
    t0 = time.perf_counter()
    dyn_hdiff_coef.run(ctx)
    stream.synchronize()
    t_coef = time.perf_counter() - t0
    t0 = time.perf_counter()
    dyn_hdiff_apply.run(ctx)
    stream.synchronize()
    t_apply = time.perf_counter() - t0

    names = [n for n, a in st.fields.items() if a.dtype.kind == "f"]
    names += ["DEF3D", "FMLX_3D", "FMLY_3D"]
    bad = [n for n in dict.fromkeys(names)
           if not bool(cp.isfinite(st.fields[n]).all())]
    print(f"LPT2={ctx.tables['LPT2']} EPSQ2={float(ctx.tables['EPSQ2'][0])} lm={g.lm} im={g.im} jm={g.jm}")
    print(f"dyn-hdiff-coef: wrapper {t_coef:.2f} s; dyn-hdiff-apply: wrapper {t_apply:.2f} s")
    print(f"NaN/Inf in {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
