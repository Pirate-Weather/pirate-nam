"""phys-fer-col-ice-props: FER_HIRES decision tree, time-averaged ice properties, rime limits and NLICE
bounds (module_MP_FER_HIRES.F90:1029-1320).

Kernel: namgpu/kernels/phys_fer_col_ice_props.cu, entry fer_col_ice_props_eval (one thread per column,
the level loop inside). The per-level body is the __device__ function fer_col_ice_level(), which the
production column kernel calls inside its own level loop.

Inputs (ctx.st, CONTRACT names; Fortran name in brackets):
  T          [T_col]      float32 [LM, JM, IM], Kelvin
  PMID       [P_col]      float32 [LM, JM, IM], mid-level pressure (Pa)
  RHO        [RHO]        float32 [LM, JM, IM], air density (kg m-3), from line 967 (FER_SC_RHO, phys-fer-col-satcheck)
  CLEAR      [CLEAR]      int32   [LM, JM, IM], CLEAR after the satcheck section (FER_SC_CLEAR; 1 = CYCLE big_loop)
  THICK      [THICK_col]  float32 [LM, JM, IM], layer thickness (kg m-2)
  QI         [QI_col]     float32 [LM, JM, IM]
  QR         [QR_col]     float32 [LM, JM, IM]
  ARAIN_COL  [ARAIN]      float32 [LM, JM, IM], accumulated rain from above on entry to each level
  ASNOW_COL  [ASNOW]      float32 [LM, JM, IM], accumulated snow from above on entry to each level
  F_RIMEF    [RimeF_col]  float32 [LM, JM, IM]
ctx.tables (from the init unit phys-fer-init, not from here):
  fer_massi, fer_vsnowi  float32 [951]   MASSI and VSNOWI, index m - 50
  fer_vel_rf             float32 [41, 8] VEL_RF(2:9, 0:40), C-order [IXRF, IXS-2] (phys-fer-init layout)
  fer_scalars            dict: RFmx1, RFmax, RH_NgC, RH_NgT, RQhail, AVhail, BVhail
Outputs: ctx.tables["fer_col_ice_out"] float32 [37, LM, JM, IM], slot numbers FER_OUT_* in the .cu
(logicals and integers stored as float). Allocated once and reused. No host sync in the wrapper.
DTPH (the physics step, s) comes from ctx.cfg["dtph"] or the argument.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_ice_props"
KERNEL = "fer_col_ice_props_eval"
NSLOT = 37
OUT_KEY = "fer_col_ice_out"


def run(ctx, dtph: float | None = None):
    import cupy as cp

    st = ctx.st
    T = st["T"]
    shape = tuple(T.shape)
    lm, jm, im = shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out.shape != (NSLOT,) + shape:
        out = cp.empty((NSLOT,) + shape, dtype=cp.float32)
        ctx.tables[OUT_KEY] = out
    sc = ctx.tables["fer_scalars"]
    if dtph is None:
        dtph = float(ctx.cfg["dtph"])
    n = im * jm
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (T, st["PMID"], st["FER_SC_RHO"], st["THICK"], st["QI"], st["QR"], st["ARAIN_COL"], st["ASNOW_COL"],
           st["F_RIMEF"], st["FER_SC_CLEAR"], ctx.tables["fer_massi"], ctx.tables["fer_vsnowi"], ctx.tables["fer_vel_rf"],
           np.float32(sc["RFmx1"]), np.float32(sc["RFmax"]), np.float32(sc["RH_NgC"]),
           np.float32(sc["RH_NgT"]), np.float32(sc["RQhail"]), np.float32(sc["AVhail"]),
           np.float32(sc["BVhail"]), np.float32(dtph), out,
           np.int32(im), np.int32(jm), np.int32(lm)))
    assert n == im * jm
    return out
