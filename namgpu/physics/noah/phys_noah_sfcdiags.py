"""phys-noah-sfcdiags: SFCDIAGS, 2 m T, theta and Q from the surface fluxes (NAM
nmm/module_TURBULENCE.F90:925 call; routine phys/module_LS_NOAHLSM.F90:5609-5656, nam.v4.2.11,
83df4e13c1, NMMB physics, NOAA/NCEP/EMC).

Kernel: namgpu/kernels/phys_noah_sfcdiags.cu (phys_noah_sfcdiags). One launch, one thread per H
point over the caller's B1 limits (1-based ITS_B1..ITE_B1 = 2..IM-1, JTS_B1..JTE_B1 = 2..JM-1).

Called from phys-turbl-driver after NOAHLSM on the NPHS cadence (the turbl driver creates the
scratch inputs with st.add on first use). ``run(ctx)`` is the entry point.

Actual -> dummy (module_TURBULENCE.F90:925):
    TWBS -> HFX    QWBS -> QFX    TSFC -> TSK    QSH -> QSFC    CHS2, CQS2 (same names)
    PSFC_OUT -> PSFC = PHINT(LM+1) (line 772), i.e. PHINT[LM] in the 0-based [LM+1, JM, IM] layout
    T2 -> T2 (scratch "T2")    TH2X -> TH2X    Q2X -> Q2X
    CP -> 1004.6, R_d -> 287.04, CAPPA -> R_d / CP (module_CONSTANTS, single precision)

Reads (ctx.st, must exist): TWBS, QWBS, TSFC, QSH, CHS2, CQS2, PHINT.
Writes: TH2X, Q2X (turbl driver hand-off fields, created if absent), T2 (scratch, created if absent).
Returns {"TH2X", "Q2X", "T2"}. No host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_sfcdiags"
KERNEL = "phys_noah_sfcdiags"

CP = np.float32(1004.6)        # NMMB_CP (kernels/nmmb_common.cuh, module_CONSTANTS)
R_D = np.float32(287.04)       # NMMB_R_D
CAPPA = np.float32(R_D / CP)   # ROVCP = CAPPA = R_d / cp

INPUTS_2D = ("TWBS", "QWBS", "TSFC", "QSH", "CHS2", "CQS2")


def _field2(st, name, shape):
    """The state array when it exists, otherwise a zero-filled unit scratch array."""
    if name in st.fields:
        return st.fields[name]
    return st.add(name, shape, "float32")


def run(ctx):
    """Launch SFCDIAGS over the B1 H points. Returns the output arrays by name."""
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape2 = (jm, im)

    missing = [n for n in INPUTS_2D + ("PHINT",) if n not in st.fields]
    if missing:
        raise RuntimeError(f"phys-noah-sfcdiags: state fields missing: {', '.join(missing)}")
    if tuple(st["PHINT"].shape) != (lm + 1, jm, im):
        raise RuntimeError("phys-noah-sfcdiags: PHINT must be [LM+1, JM, IM]")

    t2 = _field2(st, "T2", shape2)
    th2x = _field2(st, "TH2X", shape2)
    q2x = _field2(st, "Q2X", shape2)
    psfc = st["PHINT"][lm]         # PSFC_OUT = PHINT(I,J,LM+1), module_TURBULENCE.F90:772

    ni = im - 2 if im > 2 else 0   # B1: i in [2, IM-1] (1-based)
    nj = jm - 2 if jm > 2 else 0
    if ni == 0 or nj == 0:
        return {}
    grid = kernels.grid_2d(im, jm, 1)
    block = kernels.BLOCK_2D

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    i32 = np.int32
    f32 = np.float32
    with scope:
        k(grid, block, (
            st["TWBS"], st["QWBS"], st["TSFC"], st["QSH"], st["CHS2"], st["CQS2"], psfc,
            t2, th2x, q2x,
            f32(CP), f32(R_D), f32(CAPPA),
            i32(2), i32(im - 1), i32(2), i32(jm - 1),
            i32(im), i32(jm)))

    return {"TH2X": th2x, "Q2X": q2x, "T2": t2}
