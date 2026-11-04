"""Done-rule check for dyn-adv2-vert (full size, no NaN/Inf), both ADV2 calls.

    source env.sh
    python tools/check-dyn-adv2-vert.py

Runs the two ADV2 calls the way the driver step plan does, each producer then consumer:
  Q call:     dyn-adv2-q2-mid(INDX_Q..INDX_Q), dyn-adv2-vert(INDX_Q..INDX_Q, idtadt=IDTADTQ)
  tracer call: dyn-adv2-q2-mid(INDX_CW..INDX_Q2), dyn-adv2-vert(INDX_CW..INDX_Q2, idtadt=IDTADT)
Full-size random State (954 x 835 x 60) under one pool-v3 lease. Checks: every float array is finite,
and TRACERS_SQRT is positive on the range the producer wrote (the field the solve reads).
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
    from namgpu.dynamics import dyn_adv2_q2_mid, dyn_adv2_vert
    from namgpu.init.init_derive_state import init_epsq2
    from namgpu.steps.drv_gpu_lease import gpu_lease
    from namgpu.steps.drv_tracer_index import host_index

    cfg = config.load()
    g = gridmod.build(cfg)
    indx = host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))
    idtadq = int(cfg.idtad) if cfg.global_ else 1
    calls = [
        ("Q", indx["INDX_Q"], indx["INDX_Q"], idtadq),
        ("CW..Q2", indx["INDX_CW"], indx["INDX_Q2"], int(cfg.idtadt)),
    ]
    with gpu_lease(vram_mib=16384, est_s=900, cards=1, what="dyn-adv2-vert full-size NaN check"):
        st = statemod.State(g).randomize(seed=1)
        stream = cp.cuda.Stream(non_blocking=True)
        ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
        init_epsq2(ctx)
        kernels.module("dyn_adv2_q2_mid").compile()
        kernels.module("dyn_adv2_vert").compile()
        for label, kss, kse, idt in calls:
            t0 = time.perf_counter()
            dyn_adv2_q2_mid.run(ctx, kss=kss, kse=kse)
            dyn_adv2_vert.run(ctx, kss=kss, kse=kse, idtadt=idt)
            stream.synchronize()
            dt = time.perf_counter() - t0
            s1 = st["TRACERS_SQRT"][kss - 1:kse]
            pos = bool(cp.all(s1 > 0))
            print(f"call {label} {kss}..{kse} idtadt={idt}: producer+consumer {dt:.2f} s, "
                  f"TRACERS_SQRT positive on range: {pos}")
            if not pos:
                return 1

    names = [n for n, a in st.fields.items() if a.dtype.kind == "f"]
    names += ["TRACERS_SQRT", "TRACERS_TEND", "ADV2_RCMS", "ADV2_CRS", "PDOP"]
    bad = [n for n in dict.fromkeys(names) if n in st.fields
           and not bool(cp.isfinite(st.fields[n]).all())]
    print(f"lm={g.lm} im={g.im} jm={g.jm}; NaN/Inf in {bad if bad else 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
