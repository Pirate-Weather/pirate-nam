"""Done-rule check for dyn-mono-q2-back (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-dyn-mono-q2-back.py

Full-size state (954 x 835 x 60), bound the way the run binds it: TRACERS and TRACERS_TEND are the
4-D tracer arrays, EPSQ2 comes from init-derive-state's init_epsq2, and the range is the driver's
tracers-block call (KSS=INDX_CW, KSE=INDX_Q2). Compiles the kernel, runs the wrapper once, and
checks: (1) the Q2 slot of TRACERS changed and the INDX_Q slot did not, which shows the wiring
reaches the tracer array; (2) the lowest-layer TRACERS_TEND and TRACERS values of the Q2 slot, interior columns, are 0 and EPSQ2(LM); (3) no NaN/Inf in any
array the unit touches.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.dynamics import dyn_mono_q2_back
    from namgpu.init.init_derive_state import init_epsq2
    from namgpu.steps.drv_tracer_index import host_index

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names={"TRACERS", "PD", "DSG2", "EPSQ2"})
    nt = int(st.TRACERS.shape[0])
    st.add("TRACERS_TEND", (nt, g.lm, g.jm, g.im))   # dyn-mono-fix's output, pre-filled
    st.randomize(seed=1)
    if g.dsg1 is None:
        gridmod.vcoord_from_config(g, cfg)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    init_epsq2(ctx)
    indx = host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))
    kss, kse = int(indx["INDX_CW"]), int(indx["INDX_Q2"])
    q2_slot = int(indx["INDX_Q2"]) - 1
    q_slot = int(indx["INDX_Q"]) - 1
    q_before = st.TRACERS[q_slot].copy()
    q2_before = st.TRACERS[q2_slot].copy()

    kernels.module("dyn_mono_q2_back").compile()
    t0 = time.perf_counter()
    dyn_mono_q2_back.run(ctx, kss=kss, kse=kse)
    stream.synchronize()
    dt = time.perf_counter() - t0

    q2_changed = float(cp.max(cp.abs(st.TRACERS[q2_slot] - q2_before)))
    q_changed = float(cp.max(cp.abs(st.TRACERS[q_slot] - q_before)))
    # Fortran tcs(i,j,lm,indx_q2)=0 on the interior columns (lm is the lowest layer, zero-based lm-1)
    low_tend = float(cp.max(cp.abs(st.TRACERS_TEND[q2_slot, g.lm - 1, 1:-1, 1:-1])))
    low_q2_err = float(cp.max(cp.abs(st.TRACERS[q2_slot, g.lm - 1, 1:-1, 1:-1] - st["EPSQ2"][g.lm - 1])))
    bad = [n for n, a in (("TRACERS", st.TRACERS), ("TRACERS_TEND", st.TRACERS_TEND),
                          ("PD", st.PD), ("DSG2", st.DSG2), ("EPSQ2", st.EPSQ2))
           if not bool(cp.isfinite(a).all())]
    print(f"lm={g.lm} im={g.im} jm={g.jm} nt={nt} kss={kss} kse={kse} dsg1_set={g.dsg1 is not None}")
    print(f"dyn-mono-q2-back: wrapper {dt:.2f} s, Q2 slot max change {q2_changed:.4g}, "
          f"INDX_Q slot max change {q_changed:.4g}, lowest-level TRACERS_TEND Q2 max {low_tend:.4g}, "
          f"lowest-level Q2 vs EPSQ2(LM) max diff {low_q2_err:.4g}")
    print(f"NaN/Inf in {bad if bad else 'none'}")
    ok = not bad and q2_changed > 0.0 and q_changed == 0.0 and low_tend == 0.0 and low_q2_err == 0.0
    print("wiring " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
