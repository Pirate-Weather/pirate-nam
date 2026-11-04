"""Done-rule check for dyn-ddamp-damp (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-dyn-ddamp-damp.py

Compiles both kernels and runs the wrapper on a full-size random state (954 x 835 x 60), checks
every float array it touches for NaN/Inf, then changes PD/PDO and runs again: APD must follow the
new PD/PDO (Fortran lines 893-899 run on every ddamp call), i.e. it must equal the preamble formula
evaluated on the current PD and PDO over the interior i = 2..IM-2, j = 2..JM-2 (1-based).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFC = np.float32(1.533)
BFC = np.float32(1.0) - CFC


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.dynamics import dyn_ddamp_damp

    cfg = config.load()
    g = gridmod.build(cfg)
    names = {"PD", "PDO", "DSG2", "SG1", "SG2", "DIV", "U", "V"}
    st = statemod.State(g, names=names)
    st.randomize(seed=1)
    if g.dsg1 is None:
        gridmod.vcoord_from_config(g, cfg)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)

    kernels.module("dyn_ddamp_damp").compile()
    t0 = time.perf_counter()
    dyn_ddamp_damp.run(ctx)
    stream.synchronize()
    wall1 = time.perf_counter() - t0

    # second call after PD/PDO change: the defect was APD frozen at its step-1 value
    st.PD[...] = st.PD * cp.float32(1.01)
    st.PDO[...] = st.PDO * cp.float32(0.99)
    t0 = time.perf_counter()
    dyn_ddamp_damp.run(ctx)
    stream.synchronize()
    wall2 = time.perf_counter() - t0

    im, jm = g.im, g.jm
    pd, pdo, apd = st.PD, st.PDO, st.APD
    expect = cp.zeros((jm, im), dtype=cp.float32)
    s_pd = pd[1:jm - 2, 1:im - 2] + pd[1:jm - 2, 2:im - 1] + pd[2:jm - 1, 1:im - 2] + pd[2:jm - 1, 2:im - 1]
    s_pdo = pdo[1:jm - 2, 1:im - 2] + pdo[1:jm - 2, 2:im - 1] + pdo[2:jm - 1, 1:im - 2] + pdo[2:jm - 1, 2:im - 1]
    expect[1:jm - 2, 1:im - 2] = (s_pd * CFC + s_pdo * BFC) * np.float32(0.25)
    # fast numerics (fmad contraction on): compare relative to the field scale, not bitwise
    apd_ref = expect[1:jm - 2, 1:im - 2]
    apd_err = float(cp.max(cp.abs(apd[1:jm - 2, 1:im - 2] - apd_ref)) / cp.max(cp.abs(apd_ref)))

    names_chk = ["U", "V", "PD", "PDO", "DSG2", "APD"]
    bad = [n for n in names_chk if not bool(cp.isfinite(st.fields[n]).all())]
    status = "none" if not bad else str(bad)
    print(f"lm={g.lm} im={im} jm={jm} freerun={cfg.raw.get('freerun')}")
    print(f"dyn-ddamp-damp: call1 {wall1:.2f} s, call2 {wall2:.2f} s, NaN/Inf in {status}")
    print(f"APD vs preamble on current PD/PDO after PD change: max rel err {apd_err:.3e}")
    return 0 if (not bad and apd_err < 1e-6) else 1


if __name__ == "__main__":
    sys.exit(main())
