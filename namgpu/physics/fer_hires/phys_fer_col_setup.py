"""phys-fer-col-setup: EGCP01COLUMN_hr column setup and top-down level setup, Ferrier-Aligo microphysics.

Ported from phys/module_MP_FER_HIRES.F90:650-910 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE
EGCP01COLUMN_hr. Kernel: namgpu/kernels/phys_fer_col_setup.cu (phys_fer_col_setup), one launch,
one thread per column (grid_cols), no host sync, no per-step allocation after the first call.

Reads:
    T             float32 [LM, JM, IM]   model temperature (state table, Fortran T_col)
    FER_LSFC      int32   [JM, IM]       LSFC per column (Fortran LSFC = KTE - LOWLYR + 1). Set by the
                                         column prep of phys-fer-fer-hires; this unit only reads it.
                                         The unit refuses to run if the prep has not produced it.
Writes (scratch, added with st.add; Fortran per-level arrays of KTS:KTE):
    pcond1d pidep1d piacw1d piacwi1d piacwr1d piacr1d picnd1d pievp1d pimlt1d praut1d pracw1d prevp1d
    pisub1d pevap1d vsnow1d vrain11d vrain21d vci1d NSmICE1d INDEXR1d INDEXS1d RFlag1d
    DBZ_col NR_col NS_col                  float32 [LM, JM, IM]   per level
    TK_col TC_col                          float32 [LM, JM, IM]   TK and TC of each level (K, C)
    CLEAR_col                              float32 [LM, JM, IM]   CLEAR flag per level (1.0 clear, 0.0 not)
    FER_COL_CARRY float32 [9, JM, IM]      column carry-state after the setup loop, rows:
                                           0 ARAIN, 1 ASNOW, 2 VRAIN1, 3 VSNOW1, 4 TIMLT,
                                           5 INDEXS0C, 6 STRAT (0/1), 7 DRZL (0/1), 8 RHOX0C
    FER_LSFC_BAD  int32   [1]              count of columns whose LSFC was outside [0, LM]; must be 0.
                                           Read by the caller's check (not read on the step path).
Levels 1..LSFC are the column. Levels LSFC+1..LM are written with defined defaults on every call
(zero, DBZ_col = DBZmin, CLEAR_col = 0), so nothing stale is left from an earlier call.

Returns the dict of written arrays (device arrays, no host copy).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_setup"
KERNEL = "phys_fer_col_setup"
PROBE_KERNEL = "phys_fer_col_setup_probe"
LSFC_NAME = "FER_LSFC"
CARRY_NAME = "FER_COL_CARRY"
BAD_NAME = "FER_LSFC_BAD"
NCARRY = 9

# Kernel argument order after T and lsfc (matches the .cu signature exactly).
LEVEL_ARRAYS = (
    "pcond1d", "pidep1d", "piacw1d", "piacwi1d", "piacwr1d", "piacr1d", "picnd1d", "pievp1d",
    "pimlt1d", "praut1d", "pracw1d", "prevp1d", "pisub1d", "pevap1d", "vsnow1d", "vrain11d",
    "vrain21d", "vci1d", "NSmICE1d", "INDEXR1d", "INDEXS1d", "RFlag1d",
    "DBZ_col", "NR_col", "NS_col",
)
OUT_LEVEL_ARRAYS = ("TK_col", "TC_col", "CLEAR_col")
BLOCK = 128


def run(ctx):
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    if LSFC_NAME not in st.fields:
        raise RuntimeError(
            f"{LSFC_NAME} is not set: the column prep of phys-fer-fer-hires must run before "
            "phys-fer-col-setup. Without it every column has LSFC=0 and the unit writes nothing.")

    t = st["T"]
    lsfc = st[LSFC_NAME]
    outs = [st.add(name, shape3) for name in LEVEL_ARRAYS]
    tk_tc_clear = [st.add(name, shape3) for name in OUT_LEVEL_ARRAYS]
    carry = st.add(CARRY_NAME, (NCARRY, jm, im))
    bad = st.add(BAD_NAME, (1,), "int32")
    bad.fill(0)

    k = kernels.get(UNIT, KERNEL)
    grid = ((im * jm + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (t, lsfc, *outs, *tk_tc_clear, carry, bad,
           np.int32(im), np.int32(jm), np.int32(lm)))

    written = dict(zip(LEVEL_ARRAYS, outs))
    written.update(zip(OUT_LEVEL_ARRAYS, tk_tc_clear))
    written[CARRY_NAME] = carry
    written[BAD_NAME] = bad
    return written


def run_branch_probe(cases):
    """Test entry: run one level of the setup on injected carry states (rain and snow branches).

    cases: float32 array [NCASE, 5] = ARAIN, ASNOW, VRAIN1, VSNOW1, TK (K).
    Returns float32 [NCASE, 6] = CLEAR (1/0), ARAIN, ASNOW, VRAIN1, VSNOW1 after the level, TC (C).
    """
    import cupy as cp

    cases = cp.ascontiguousarray(cp.asarray(cases, dtype=cp.float32))
    n = int(cases.shape[0])
    out = cp.zeros((n, 6), dtype=cp.float32)
    scratch = cp.zeros(32, dtype=cp.float32)
    k = kernels.get(UNIT, PROBE_KERNEL)
    k((1, 1, 1), (1, 1, 1), (cases, out, scratch, np.int32(n)))
    return out
