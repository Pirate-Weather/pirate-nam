# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb03: TAUGB03, LW band-3 optical depths and Planck fractions (RRTM group).

NOAA / NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE; header kept in the NAM clone). Ported from
phys/radlw_main_nmmb.f:3792-4094 (nam.v4.2.11, 83df4e13c1), SUBROUTINE taugb03. Kernel:
namgpu/kernels/phys_rrtm_lw_taugb03.cu (phys_rrtm_lw_taugb03, one thread per column, one launch).

Tables (uploaded once, ctx.tables["rrtm_lw_kgb03"]): namgpu/data/rrtm_lw/kgb03.npz, extracted by
tools/extract_rrtm_lw_kgb03.py from radlw_datatb.f (module_radlw_kgb03 and module_radlw_ref chi_mls).
The taugb03 routine reads module_radlw_kgb03 (not module_radlw_kgb03_nmmb), and the table layout
matches that module: absa(ig, id) with 585 entries, not the 650-entry nmmb layout.

Inputs: the per-column coefficients that phys-rrtm-lw-setcoef produces (setcoef is not yet ported, so
the names below are the contract for it). All are ctx.st fields, created by alloc() on first use:
  LW_LAYTROP  int32[JM, IM]           laytrop(iplon): last lower-atmosphere layer (1-based Fortran value)
  LW_JP       int32[LMK, JM, IM]      jp(k)   (1-based)
  LW_JT       int32[LMK, JM, IM]      jt(k)   (1-based)
  LW_JT1      int32[LMK, JM, IM]      jt1(k)  (1-based)
  LW_COLAMT   float32[6, LMK, JM, IM] colamt(k, m), molecule m = 1..6 (n0 slowest, CONTRACT IDX4)
  LW_RFRATE   float32[2, LMK, JM, IM] rfrate(k, 1, n), n = 1..2 (only gas index 1 is used here)
  LW_COLDRY   float32[LMK, JM, IM]    coldry(k)
  LW_FAC00/FAC01/FAC10/FAC11          float32[LMK, JM, IM]  fac00(k) .. fac11(k)
  LW_SELFFAC, LW_SELFFRAC, LW_FORFAC, LW_FORFRAC, LW_MINORFRAC   float32[LMK, JM, IM]
  LW_INDSELF, LW_INDFOR, LW_INDMINOR  int32[LMK, JM, IM]  (1-based Fortran values)
Outputs (band-3 rows only; the caller maps them to rows NS03+1..NS03+16 = 23..38 of the 140-row arrays):
  LW_TAUG_B3  float32[16, LMK, JM, IM]   taug(NS03 + ig, k), ig0 = ig-1
  LW_FRACS_B3 float32[16, LMK, JM, IM]   fracs(NS03 + ig, k)

LMK = LM + 1: nlay is LMK (grrad_nmmb.f:210 LTP = 1; lwrad is called with LMK, grrad_nmmb.f:1992-1996).
Every per-layer array here has LMK layers; the top layer is computed like the others.
Reads: ctx.grid (im, jm, lm), ctx.st (the inputs above), ctx.tables.
Writes: ctx.st outputs above; ctx.tables["rrtm_lw_kgb03"] once.
Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation after the first call.
Single-card run, no MPI. The Fortran print/STOP paths do not exist in taugb03.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb03"
KERNEL = "phys_rrtm_lw_taugb03"
TABLE_KEY = "rrtm_lw_kgb03"
TABLE_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb03.npz"
NG03 = 16
NS03 = 22          # radlw_param_nmmb.f:160 (0-based offset of band 3 in the 140 g-point arrays)
MAXGAS = 6         # colamt(MAXLAY, MAXGAS)

# name -> (dtype, shape kind). Shape kinds: "col" [JM, IM]; "lay" [LMK, JM, IM]; "colamt" [6, LMK, JM, IM];
# "rfrate" [2, LMK, JM, IM]; "g16" [16, LMK, JM, IM].
INPUTS_INT_COL = ("LW_LAYTROP",)
INPUTS_INT_LAY = ("LW_JP", "LW_JT", "LW_JT1", "LW_INDSELF", "LW_INDFOR", "LW_INDMINOR")
INPUTS_FLOAT_LAY = ("LW_COLDRY", "LW_FAC00", "LW_FAC01", "LW_FAC10", "LW_FAC11",
                    "LW_SELFFAC", "LW_SELFFRAC", "LW_FORFAC", "LW_FORFRAC", "LW_MINORFRAC")
OUTPUTS = ("LW_TAUG_B3", "LW_FRACS_B3")
TABLE_NAMES = ("absa", "absb", "selfref", "forref", "fracrefa", "fracrefb", "ka_mn2o", "kb_mn2o", "chi_mls")


def alloc(ctx):
    """Create (once) every input and output field in ctx.st; returns {name: device array}."""
    st = ctx.st
    im, jm, lm = ctx.grid.im, ctx.grid.jm, ctx.grid.lm
    lmk = lm + 1                    # LMK, the nlay the kernel runs over
    out = {}
    for n in INPUTS_INT_COL:
        out[n] = st.add(n, (jm, im), "int32")
    for n in INPUTS_INT_LAY:
        out[n] = st.add(n, (lmk, jm, im), "int32")
    for n in INPUTS_FLOAT_LAY:
        out[n] = st.add(n, (lmk, jm, im), "float32")
    out["LW_COLAMT"] = st.add("LW_COLAMT", (MAXGAS, lmk, jm, im), "float32")
    out["LW_RFRATE"] = st.add("LW_RFRATE", (2, lmk, jm, im), "float32")
    for n in OUTPUTS:
        out[n] = st.add(n, (NG03, lmk, jm, im), "float32")
    return out


def _tables(ctx):
    """Upload the band-3 k-tables once (C-order arrays, CONTRACT section 1)."""
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    tbl = ctx.tables.get(TABLE_KEY)
    if tbl is None:
        z = np.load(TABLE_NPZ)
        tbl = {}
        for n in TABLE_NAMES:
            tbl[n] = cp.asarray(np.ascontiguousarray(z[n], dtype=np.float32))
        ctx.tables[TABLE_KEY] = tbl
    return tbl


def run(ctx):
    """TAUGB03 for the whole domain: writes LW_TAUG_B3 and LW_FRACS_B3 from the inputs in ctx.st."""
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    im, jm, lm = ctx.grid.im, ctx.grid.jm, ctx.grid.lm
    lmk = lm + 1
    io = alloc(ctx)
    tb = _tables(ctx)
    k = kernels.get(UNIT, KERNEL)
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lmk),
            tb["chi_mls"], tb["absa"], tb["absb"], tb["selfref"], tb["forref"],
            tb["fracrefa"], tb["fracrefb"], tb["ka_mn2o"], tb["kb_mn2o"],
            io["LW_LAYTROP"], io["LW_JP"], io["LW_JT"], io["LW_JT1"],
            io["LW_COLAMT"], io["LW_RFRATE"], io["LW_COLDRY"],
            io["LW_FAC00"], io["LW_FAC01"], io["LW_FAC10"], io["LW_FAC11"],
            io["LW_SELFFAC"], io["LW_SELFFRAC"], io["LW_FORFAC"], io["LW_FORFRAC"],
            io["LW_MINORFRAC"], io["LW_INDSELF"], io["LW_INDFOR"], io["LW_INDMINOR"],
            io["LW_TAUG_B3"], io["LW_FRACS_B3"],
        ))
    return io
