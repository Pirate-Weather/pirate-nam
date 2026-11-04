# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb05: LW band 5 optical depths and Planck fractions (RRTM group, NMMB longwave).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:4341-4598 (SUBROUTINE taugb05, internal to
taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the notice is kept in the kernel file
(namgpu/kernels/phys_rrtm_lw_taugb05.cu). Kernel: phys_rrtm_lw_taugb05, one launch, one thread per column.

Table source (a decision to check, not a silent choice): taugb05 has `use module_radlw_kgb05`. That module is
defined in phys/radlw_datatb.f:15209-21676 (GFS-layout tables), not in radlw_datatb_nmmb.f, which defines
module_radlw_kgb05_nmmb with a different layout and different numbers (absa(1,1): 2.06e-6 GFS-named vs 1.23e-6
NMMB-named). The NAM build compiles both, and taugb05 binds to the GFS-named module, so that is the one
extracted: namgpu/data/rrtm_lw_kgb05.npz (tools/extract_rrtm_lw_kgb05.py). chi_mls (radlw_datatb.f:733-905) is in
the same npz.

Inputs read from ctx.st (device arrays, CONTRACT layout; the Fortran column arrays are per layer, LM = grid.lm):
  colamt      float32 [3, LM, JM, IM]   colamt(k, g) for g = 1 (h2o), 2 (co2), 3 (o3)      [phys-rrtm-lw-setcoef]
  rfrate      float32 [4, LM, JM, IM]   slot 0 rfrate(k,1,1), 1 rfrate(k,1,2), 2 rfrate(k,6,1), 3 rfrate(k,6,2)
  wx_ccl4     float32 [LM, JM, IM]      wx(k,1), CCL4 amount                                     [phys-rrtm-lw-setcoef]
  fac00, fac01, fac10, fac11            float32 [LM, JM, IM]                                     [phys-rrtm-lw-setcoef]
  selffac, selffrac, forfac, forfrac, minorfrac    float32 [LM, JM, IM]                          [phys-rrtm-lw-setcoef]
  jp, jt, jt1, indself, indfor, indminor           int32   [LM, JM, IM]                          [phys-rrtm-lw-setcoef]
  laytrop     int32   [JM, IM]          number of lower-atmosphere layers (1-based count)         [phys-rrtm-lw-setcoef]
  All of these are written by phys-rrtm-lw-setcoef (not landed when this unit was written); a missing name
  raises, it is never filled with a default.

Outputs added to ctx.st (float32, allocated once):
  taug_b05    [16, LM, JM, IM]   taug_b05[ig0, l0, j0, i0]  = taug(NS05 + ig, k), NS05 = 52, ig0 = ig - 1
  fracs_b05   [16, LM, JM, IM]   fracs_b05[ig0, l0, j0, i0] = fracs(NS05 + ig, k)
  The caller places band 5 into its taug(NGPTLW=140, LM) slots 52..67 (0-based). Memory: 2 x 3.06 GB at the
  full 954 x 835 x 60 grid, so a caller that runs the LW on column blocks should size these to the block.

Constants: refrat_planck_a = chi_mls(1,5)/chi_mls(2,5), refrat_planck_b = chi_mls(3,43)/chi_mls(2,43),
refrat_m_a = chi_mls(1,7)/chi_mls(2,7) (radlw_main_nmmb.f:4363-4365), computed here in float32. nspa(5) = 9,
nspb(5) = 5 (radlw_main_nmmb.f:284-285) and NG05 = 16 are compiled into the kernel.

Tables (uploaded once into ctx.tables["rrtm_lw_kgb05"]): absa, absb, selfref, forref, fracrefa, fracrefb,
ka_mo3 (transposed to C order, index order documented in the kernel file), ccl4.
Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation. Single card, no MPI.

Declared divergences (see the kernel header): physics is float32 here while native NAM's Linux.gnu, Linux.intel
and wcoss2 configs run kind_phys=8 (no -DSINGLE_PREC); tables and refrat constants are rounded to float32 once;
out-of-range table indices and laytrop outside [0, LM] give NaN for that layer or column, not a read past the
table; nlay must equal grid.lm.
Not yet verified against real inputs: phys-rrtm-lw-setcoef (todo) is not landed, so colamt, rfrate, fac*, jp/jt/jt1,
ind* and laytrop are random-input tested only. The scatter of taug_b05/fracs_b05 into taug(NGPTLW, LM) slots 52..67
belongs to the taumol caller (phys-rrtm-lw-taumol, todo) and is not landed.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.kernels import BLOCK_COL, grid_cols

UNIT = "phys_rrtm_lw_taugb05"
KERNEL = "phys_rrtm_lw_taugb05"
KEY = "rrtm_lw_kgb05"
NG05 = 16
NS05 = 52
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw_kgb05.npz"

IN_FLOAT_4 = ("colamt", "rfrate")
IN_FLOAT_3 = ("wx_ccl4", "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac",
              "forfac", "forfrac", "minorfrac")
IN_INT_3 = ("jp", "jt", "jt1", "indself", "indfor", "indminor")
IN_INT_2 = ("laytrop",)
OUT = ("taug_b05", "fracs_b05")


def load_tables():
    """Host read of the npz, converted to the C-order layouts the kernel indexes. Cached on the module."""
    import cupy as cp

    z = np.load(TABLE_FILE)
    chi = z["chi_mls"]                                  # Fortran chi_mls(i, j) at chi[i-1, j-1]
    # refrat ratios in float64 (native kind_phys=8), rounded to float32 once
    refrat_planck_a = np.float32(chi[0, 4] / chi[1, 4])     # chi_mls(1,5)/chi_mls(2,5)
    refrat_planck_b = np.float32(chi[2, 42] / chi[1, 42])   # chi_mls(3,43)/chi_mls(2,43)
    refrat_m_a = np.float32(chi[0, 6] / chi[1, 6])          # chi_mls(1,7)/chi_mls(2,7)
    tbl = {
        "absa": cp.asarray(np.ascontiguousarray(z["absa"].T, dtype=np.float32)),        # [585, 16]
        "absb": cp.asarray(np.ascontiguousarray(z["absb"].T, dtype=np.float32)),        # [1175, 16]
        "selfref": cp.asarray(np.ascontiguousarray(z["selfref"].T, dtype=np.float32)),  # [10, 16]
        "forref": cp.asarray(np.ascontiguousarray(z["forref"].T, dtype=np.float32)),    # [4, 16]
        "fracrefa": cp.asarray(np.ascontiguousarray(z["fracrefa"].T, dtype=np.float32)),  # [9, 16]
        "fracrefb": cp.asarray(np.ascontiguousarray(z["fracrefb"].T, dtype=np.float32)),  # [5, 16]
        "ka_mo3": cp.asarray(np.ascontiguousarray(z["ka_mo3"].transpose(2, 1, 0), dtype=np.float32)),  # [19, 9, 16]
        "ccl4": cp.asarray(np.ascontiguousarray(z["ccl4"], dtype=np.float32)),          # [16]
        "refrat_planck_a": float(refrat_planck_a),
        "refrat_planck_b": float(refrat_planck_b),
        "refrat_m_a": float(refrat_m_a),
    }
    return tbl


def _tables(ctx):
    if ctx.tables is None:
        ctx.tables = {}
    tbl = ctx.tables.get(KEY)
    if tbl is None:
        tbl = load_tables()
        ctx.tables[KEY] = tbl
    return tbl


_EXPECT_DTYPE = {n: np.float32 for n in IN_FLOAT_4 + IN_FLOAT_3} | {n: np.int32 for n in IN_INT_3 + IN_INT_2}


def _check_inputs(arrs, IM, JM, LM):
    """Host check on shapes and dtypes only (no device sync)."""
    for n, a in arrs.items():
        if n == "colamt":
            want = (3, LM, JM, IM)
        elif n == "rfrate":
            want = (4, LM, JM, IM)
        elif n == "laytrop":
            want = (JM, IM)
        else:
            want = (LM, JM, IM)
        if tuple(a.shape) != want:
            raise RuntimeError(f"{UNIT}: {n} shape {tuple(a.shape)} != {want} (phys-rrtm-lw-setcoef layout)")
        if np.dtype(a.dtype) != np.dtype(_EXPECT_DTYPE[n]):
            raise RuntimeError(f"{UNIT}: {n} dtype {a.dtype} != {np.dtype(_EXPECT_DTYPE[n])}")


def run(ctx, nlay=None):
    """Band-5 LW optical depths and Planck fractions for every column and layer, one launch."""
    import cupy as cp  # noqa: F401  (device arrays come from ctx.st)

    st = ctx.st
    grid = st.grid
    IM, JM, LM = int(grid.im), int(grid.jm), int(grid.lm)

    def get(name):
        try:
            return st[name]
        except KeyError as e:
            raise RuntimeError(f"{UNIT} needs ctx.st['{name}'] from phys-rrtm-lw-setcoef (not present)") from e

    # Fortran's upper loop runs to nlay (dummy argument); the kernel runs to LM. Refuse any mismatch.
    if nlay is not None and int(nlay) != LM:
        raise RuntimeError(f"{UNIT}: nlay={nlay} != grid.lm={LM}; the kernel runs the upper loop to LM")

    arrs = {}
    for n in IN_FLOAT_4 + IN_FLOAT_3 + IN_INT_3 + IN_INT_2:
        arrs[n] = get(n)
    _check_inputs(arrs, IM, JM, LM)

    taug = st.add(OUT[0], (NG05, LM, JM, IM), dtype="float32")
    fracs = st.add(OUT[1], (NG05, LM, JM, IM), dtype="float32")

    tbl = _tables(ctx)
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    args = (
        arrs["colamt"], arrs["rfrate"], arrs["wx_ccl4"],
        arrs["fac00"], arrs["fac01"], arrs["fac10"], arrs["fac11"],
        arrs["selffac"], arrs["selffrac"], arrs["forfac"], arrs["forfrac"], arrs["minorfrac"],
        arrs["jp"], arrs["jt"], arrs["jt1"], arrs["indself"], arrs["indfor"], arrs["indminor"],
        arrs["laytrop"],
        tbl["absa"], tbl["absb"], tbl["selfref"], tbl["forref"], tbl["fracrefa"], tbl["fracrefb"],
        tbl["ka_mo3"], tbl["ccl4"],
        np.float32(tbl["refrat_planck_a"]), np.float32(tbl["refrat_planck_b"]), np.float32(tbl["refrat_m_a"]),
        np.int32(IM), np.int32(JM), np.int32(LM),
        taug, fracs,
    )
    with scope:
        k(grid_cols(IM, JM), BLOCK_COL, args)
    return taug, fracs
