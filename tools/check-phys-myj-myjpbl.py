"""Done-rule check for phys-myj-myjpbl (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-myj-myjpbl.py

1. compiles namgpu/kernels/phys_myj_myjpbl.cu through the NVRTC loader;
2. runs the wrapper ONCE on a full-size 954 x 835 x 60 state (only the fields this unit touches,
   so it fits beside other tenants on the shared card) and asserts no NaN or Inf in any array it
   writes. The inputs are physically shaped (monotone pressure, positive layer depths, T 240-300 K,
   winds of a few m/s), built here, not from the generic randomize().
Part 3 (Fortran-to-CUDA diff review) is recorded in the kernel header.
Take the GPU pool lease first (README.md). No column test, no reduced size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNIT_NAMES_3D = [
    "DZ", "EXNER", "PHMID", "Q", "QC", "U_PHY", "V_PHY", "T", "TH", "Q2", "EXCH_H", "XLEN_MIX",
    "DUDT", "DVDT", "RTHBLTEN", "RQBLTEN", "RQCBLTEN",
]
UNIT_NAMES_PINT = ["PHINT"]
UNIT_NAMES_2D = [
    "SFCZ", "HSTDV", "TSFC", "XLAND", "SICE", "SNOW", "CHKLOWQ", "ELFLX", "AKHS", "AKMS",
    "USTAR", "UZ0", "VZ0", "THZ0", "QZ0", "QSH", "CT", "PBLH", "MIXHT",
]
UNIT_NAMES_1D = ["EPSL", "EPSQ2"]
UNIT_NAMES_INT = ["LPBL"]


def main() -> int:
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.myj import phys_myj_myjpbl as wrap

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    print(f"grid {im} x {jm} x {lm}, dt={cfg.dt}, nphs={cfg.nphs}")

    wanted = set(UNIT_NAMES_3D + UNIT_NAMES_PINT + UNIT_NAMES_2D + UNIT_NAMES_1D + UNIT_NAMES_INT)
    st = statemod.State(g, names=wanted)
    rng = cp.random.default_rng(7)
    f32 = cp.float32

    # Physically shaped fields. Layer index k = 0 is the model top (NAM convention).
    kk = cp.arange(lm, dtype=f32)[:, None, None]
    pint = cp.linspace(2.0e4, 1.0e5, lm + 1, dtype=f32)[:, None, None] * cp.ones((1, jm, im), f32)
    phmid = 0.5 * (pint[:-1] + pint[1:])
    exner = (phmid / f32(1.0e5)) ** f32(287.04 / 1004.6)

    def add(name, shape):
        if name in st.fields:
            return st.fields[name]
        return st.add(name, shape)

    add("PHINT", (lm + 1, jm, im))[...] = pint
    add("PHMID", (lm, jm, im))[...] = phmid
    add("EXNER", (lm, jm, im))[...] = exner
    add("DZ", (lm, jm, im))[...] = f32(300.0)
    t = f32(300.0) - f32(60.0) * kk / f32(lm) + 0.0 * phmid
    add("T", (lm, jm, im))[...] = t
    add("TH", (lm, jm, im))[...] = t / exner
    add("Q", (lm, jm, im))[...] = f32(0.01) * cp.ones((lm, jm, im), f32)
    add("QC", (lm, jm, im))[...] = f32(0.0)
    add("U_PHY", (lm, jm, im))[...] = f32(5.0) + f32(2.0) * rng.random((lm, jm, im), dtype=f32)
    add("V_PHY", (lm, jm, im))[...] = f32(-3.0) + f32(2.0) * rng.random((lm, jm, im), dtype=f32)
    add("Q2", (lm, jm, im))[...] = f32(0.5) + f32(0.5) * rng.random((lm, jm, im), dtype=f32)
    add("EXCH_H", (lm, jm, im))[...] = f32(0.0)
    add("XLEN_MIX", (lm, jm, im))[...] = f32(0.0)
    for n in ("DUDT", "DVDT", "RTHBLTEN", "RQBLTEN", "RQCBLTEN"):
        add(n, (lm, jm, im))[...] = f32(0.0)
    add("SFCZ", (jm, im))[...] = f32(100.0) * rng.random((jm, im), dtype=f32)
    add("HSTDV", (jm, im))[...] = f32(0.0)
    add("TSFC", (jm, im))[...] = f32(290.0) + f32(10.0) * rng.random((jm, im), dtype=f32)
    add("XLAND", (jm, im))[...] = (1.0 + (rng.random((jm, im), dtype=f32) < 0.5)).astype(f32)
    add("SICE", (jm, im))[...] = f32(0.0)
    add("SNOW", (jm, im))[...] = f32(0.0)
    add("CHKLOWQ", (jm, im))[...] = f32(1.0)
    add("ELFLX", (jm, im))[...] = f32(0.0)
    add("AKHS", (jm, im))[...] = f32(0.01) * cp.ones((jm, im), f32)
    add("AKMS", (jm, im))[...] = f32(0.01) * cp.ones((jm, im), f32)
    add("USTAR", (jm, im))[...] = f32(0.3) * cp.ones((jm, im), f32)
    add("UZ0", (jm, im))[...] = f32(5.0) * cp.ones((jm, im), f32)
    add("VZ0", (jm, im))[...] = f32(-3.0) * cp.ones((jm, im), f32)
    add("THZ0", (jm, im))[...] = f32(290.0) * cp.ones((jm, im), f32)
    add("QZ0", (jm, im))[...] = f32(0.01) * cp.ones((jm, im), f32)
    add("QSH", (jm, im))[...] = f32(0.01) * cp.ones((jm, im), f32)
    add("CT", (jm, im))[...] = f32(0.0)
    add("PBLH", (jm, im))[...] = f32(0.0)
    add("MIXHT", (jm, im))[...] = f32(0.0)
    add("EPSL", (lm,))[...] = f32(0.32)
    add("EPSQ2", (lm,))[...] = f32(0.2)
    if "LPBL" not in st.fields:
        st.add("LPBL", (jm, im), dtype="int32")

    # Tables the phys-myj-init unit writes (PTBL, QS0, SQS, RDTH, RDQ). Plausible values, positive.
    jtbl, itbl = 1201, 401
    ctx_tables = {
        "myj_ptbl": (1.0e4 + 9.0e4 * rng.random((jtbl, itbl), dtype=f32)).astype(f32),
        "myj_qs0": cp.linspace(0.0, 0.03, jtbl, dtype=f32),
        "myj_sqs": cp.full((jtbl,), 0.03, dtype=f32),
        "myj_rdth": 1.0 / (155.0 / (jtbl - 1)),
        "myj_rdq": 2.0,
    }
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.tables.update(ctx_tables)

    t0 = time.perf_counter()
    kernels.module(wrap.UNIT).compile()
    print(f"compile ok ({time.perf_counter() - t0:.2f} s)")

    t1 = time.perf_counter()
    with ctx.stream:
        wrap.run(ctx)
    ctx.stream.synchronize()
    print(f"wrapper run ok ({time.perf_counter() - t1:.2f} s)")

    written = ["Q2", "EXCH_H", "XLEN_MIX", "DUDT", "DVDT", "RTHBLTEN", "RQBLTEN", "RQCBLTEN",
               "THZ0", "QZ0", "QSH", "CT", "PBLH", "MIXHT"]
    bad = []
    for n in written:
        a = st.fields[n]
        if not bool(cp.isfinite(a).all()):
            bad.append(n)
    lp = st.fields["LPBL"]
    print("PBLH range (m):", float(st.fields["PBLH"].min()), float(st.fields["PBLH"].max()))
    print("LPBL range:", int(lp.min()), int(lp.max()))
    if bad:
        print("NaN/Inf in:", bad)
        return 1
    print("no NaN/Inf in any written array")
    return 0


if __name__ == "__main__":
    sys.exit(main())
