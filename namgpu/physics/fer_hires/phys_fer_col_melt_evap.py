"""phys-fer-col-melt-evap: melting and evaporation of snow/ice, shedding, melt limits, and the
head of the rain block (module_MP_FER_HIRES.F90:1479-1587, nam.v4.2.11, 83df4e13c1).

Kernel: namgpu/kernels/phys_fer_col_melt_evap.cu (phys_fer_col_melt_evap, one launch, one thread per
(i, j, l) cell with grid_2d). The level-to-level carries (ARAIN, VRAIN1) are read as arrays; the column
kernel of phys-fer-fer-hires supplies them from the level above.

Inputs (ctx.st, named by the CONTRACT, [LM, JM, IM]; created with st.add when a producing unit has not
run yet, so the wrapper runs standalone):
    per-level locals of the FER column (from phys-fer-col-setup, -satcheck, -ice-props, -processes):
    VEL_INC, RHO, DIFFUS, DYNVIS, NLICE, QW, PCOND, WV, QSW, TC, TCC, PP, PILOSS, THERM_COND,
    THICK, QR, ARAIN, VRAIN1 (float32); INDEXS, ICE, RAIN, INDEXRmin, STRAT, DRZL (int32).
    in/out: PIEVP, PICND, PIMLT, PIACR, VRAIN2, N0r (float32); INDEXR (int32).
    outputs: PRLOSS, QTRAIN, RQR (float32).
Tables (ctx.tables["fer"], the phys-fer-init dict, float32):
    venti1, venti2   [FER_TBL_NMDI = 951]  VENTI1/VENTI2 indexed by INDEXS - 50
    massr, vrain     [FER_TBL_NMDR = 951]  MASSR/VRAIN indexed by INDEXR - 50
Scalars (ctx.tables["fer"]["scalars"]): DTPH, ESW0, CN0R0, RQR_DRMIN, RQR_DRMAX.

No host arithmetic on the step path; no host sync; scratch allocated on first call and reused.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_melt_evap"
KERNEL = "phys_fer_col_melt_evap"

FLOAT_FIELDS = (
    "VEL_INC", "RHO", "DIFFUS", "DYNVIS", "NLICE", "QW", "PCOND", "WV", "QSW", "TC", "TCC", "PP",
    "PILOSS", "THERM_COND", "THICK", "QR", "ARAIN", "VRAIN1",
)
INT_FIELDS = ("INDEXS", "ICE", "RAIN", "INDEXRmin")
INOUT_INT = ("STRAT", "DRZL", "INDEXR")
INOUT_FLOAT = ("PIEVP", "PICND", "PIMLT", "PIACR", "VRAIN2", "N0r")
OUT_FLOAT = ("PRLOSS", "QTRAIN", "RQR")


def _field(st, name, shape, dtype):
    """ctx.st[name] if the producing unit has allocated it, else allocate it once (st.add)."""
    try:
        arr = st[name]
    except (KeyError, AttributeError):
        st.add(name, shape, dtype=dtype)
        arr = st[name]
    return arr


def run(ctx):
    import cupy as cp

    st = ctx.st
    tables = ctx.tables
    lm, jm, im = st["T"].shape  # [LM, JM, IM]; T is a mass-point 3-D field
    shape = (lm, jm, im)

    arrs = {}
    for n in FLOAT_FIELDS + INOUT_FLOAT + OUT_FLOAT:
        arrs[n] = _field(st, n, shape, "float32")
    for n in INT_FIELDS + INOUT_INT:
        arrs[n] = _field(st, n, shape, "int32")

    # Tables come from phys-fer-init's one dict ctx.tables["fer"] (venti1, venti2, massr, vrain and
    # the host scalars, SCALAR_SLOTS names); built here if init has not run (standalone done-rule run).
    fer = tables.get("fer")
    if fer is None:
        from namgpu.physics.fer_hires import phys_fer_init
        phys_fer_init.run(ctx)
        fer = tables["fer"]
    ventI1, ventI2, massr, vrain = fer["venti1"], fer["venti2"], fer["massr"], fer["vrain"]
    fs = fer["scalars"]
    sc = {"DTPH": fs["DTPH"], "ESW0": fs["ESW0"], "CN0r0": fs["CN0R0"],
          "RQR_DRmin": fs["RQR_DRMIN"], "RQR_DRmax": fs["RQR_DRMAX"]}
    assert ventI1.shape == (951,) and ventI2.shape == (951,)
    assert massr.shape == (951,) and vrain.shape == (951,)

    k = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(sc["DTPH"]), np.float32(sc["ESW0"]), np.float32(sc["CN0r0"]),
        np.float32(sc["RQR_DRmin"]), np.float32(sc["RQR_DRmax"]),
        ventI1, ventI2, massr, vrain,
        arrs["VEL_INC"], arrs["RHO"], arrs["DIFFUS"], arrs["DYNVIS"], arrs["NLICE"],
        arrs["QW"], arrs["PCOND"], arrs["WV"], arrs["QSW"], arrs["TC"], arrs["TCC"], arrs["PP"],
        arrs["PILOSS"], arrs["THERM_COND"], arrs["THICK"], arrs["QR"], arrs["ARAIN"], arrs["VRAIN1"],
        arrs["INDEXS"], arrs["ICE"], arrs["RAIN"], arrs["INDEXRmin"],
        arrs["STRAT"], arrs["DRZL"],
        arrs["PIEVP"], arrs["PICND"], arrs["PIMLT"], arrs["PIACR"],
        arrs["INDEXR"], arrs["N0r"], arrs["VRAIN2"],
        arrs["PRLOSS"], arrs["QTRAIN"], arrs["RQR"],
    )
    with scope:
        k(grid, kernels.BLOCK_2D, args)
    return {n: arrs[n] for n in INOUT_INT + INOUT_FLOAT + OUT_FLOAT}
