"""Done-rule check for dyn-adv2-q2-mid (full size, no NaN/Inf).

    source ~/nam-gpu/env.sh
    python tools/check-dyn-adv2-q2-mid.py

EPSQ2 is set by the real init unit (init_epsq2, the Fortran init rule), not by a stand-in. The
tracer slots come from cfg (host mirror of drv-tracer-index). The wrapper runs once on a full-size
random State (954 x 835 x 60) under one pool-v3 lease, and every float array is checked for NaN/Inf.
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
    from namgpu.dynamics import dyn_adv2_q2_mid
    from namgpu.init.init_derive_state import init_epsq2
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    init_epsq2(ctx)
    epsq2 = cp.asnumpy(ctx.tables["EPSQ2"])
    assert epsq2.shape == (g.lm,) and np.all(epsq2 == np.float32(0.02)), epsq2

    kernels.module("dyn_adv2_q2_mid").compile()
    with gpu_lease(vram_mib=2048, est_s=600, cards=1, what="dyn-adv2-q2-mid full-size NaN check"):
        t0 = time.perf_counter()
        dyn_adv2_q2_mid.run(ctx)
        stream.synchronize()
        dt = time.perf_counter() - t0

    names = [n for n, a in st.fields.items() if a.dtype.kind == "f"]
    names += ["TRACERS_SQRT", "PDOP"]
    bad = [n for n in dict.fromkeys(names) if n in st.fields
           and not bool(cp.isfinite(st.fields[n]).all())]
    print(f"lm={g.lm} im={g.im} jm={g.jm} EPSQ2 min={epsq2.min()} max={epsq2.max()}")
    print(f"dyn-adv2-q2-mid: wrapper {dt:.2f} s, NaN/Inf in {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
