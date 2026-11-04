# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb01: TAUGB01, NMMB RRTM longwave band-1 optical depths and Planck fractions.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:3610-3712 (subroutine taugb01, inside taumol).
Kernel: namgpu/kernels/phys_rrtm_lw_taugb01.cu (taugb01_cols, one thread per column, one launch).
The Fortran NOTICE/author header is kept in the kernel file.

Which tables: the k-tables are module_radlw_kgb01 from the NAM build (phys/radlw_datatb.f), extracted to
namgpu/data/rrtm_lw/kgb01.npz by tools/extract_rrtm_lw_kgb01.py. NG01 = 10 in this build (see the kernel header).

In (ctx.st, Fortran names; shapes from the layer state [LM, JM, IM], set by the setcoef / taumol units):
  laytrop              int32  [JM, IM]     tropopause layer index (1-based)
  pavel, scaleminorn2, selffac, selffrac, forfac, forfrac, minorfrac,
  fac00, fac01, fac10, fac11                  float32 [LM, JM, IM]
  indself, indfor, indminor, jp, jt, jt1      int32   [LM, JM, IM]  (1-based, as in Fortran)
  colamt               float32 [6, LM, JM, IM]  Fortran colamt(MAXLAY,MAXGAS) convention (gas index n = 1..6,
                       H2O is index 0 here). TAUGB01 reads only colamt(:,1) = colamt[0].
colbrd is a caller input: lwrad computes colbrd = coldry - summol (radlw_main_nmmb.f:891, state name
lwrad_colbrd) and setcoef takes the same input as colbrd. The seam maps lwrad_colbrd to colbrd.
Out (ctx.st, scratch added here, State.add is idempotent):
  taug_b01             float32 [LM, NG01, JM, IM]   taug(ig,k) for g = 1..NG01
  fracs_b01            float32 [LM, NG01, JM, IM]   fracs(ig,k), = fracrefa(ig) (lower) or fracrefb(ig) (upper)
Tables: ctx.tables["rrtm_lw_kgb01_<name>"], uploaded once (float32, flat, Fortran column-major with ig fastest).
Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation after the first call.

Not done here: TAUGB01 is called from taumol by the taumol unit; this wrapper is the one-band kernel only.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb01"
KERNEL = "taugb01_cols"
NG01 = 10
MAXGAS = 6
KGB01 = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb01.npz"
TABLES = ("absa", "absb", "selfref", "forref", "ka_mn2", "kb_mn2", "fracrefa", "fracrefb")


def _tables(ctx):
    import cupy as cp

    t = ctx.tables
    if "rrtm_lw_kgb01_absa" not in t:
        z = np.load(KGB01)
        for name in TABLES:
            arr = np.ascontiguousarray(z[name], dtype=np.float32).reshape(-1)
            t["rrtm_lw_kgb01_" + name] = cp.asarray(arr)
    return t


def run(ctx):
    import cupy as cp  # noqa: F401  (device arrays come from ctx.st)

    st = ctx.st
    t = _tables(ctx)
    lm, jm, im = st["pavel"].shape
    ncol = jm * im

    colamt = st["colamt"]
    if tuple(colamt.shape) != (MAXGAS, lm, jm, im):
        raise ValueError(f"taugb01: colamt {tuple(colamt.shape)} != ({MAXGAS}, {lm}, {jm}, {im})")
    colamt_h2o = colamt[0]  # Fortran colamt(:,1), H2O

    st.add("taug_b01", (lm, NG01, jm, im), "float32")
    st.add("fracs_b01", (lm, NG01, jm, im), "float32")

    def lay(name):
        return st[name].reshape(lm, ncol)

    k = kernels.get(UNIT, KERNEL)
    block = kernels.BLOCK_COL[0]
    grid = ((ncol + block - 1) // block, 1, 1)
    args = (
        st["laytrop"].reshape(ncol),
        np.int32(lm),
        np.int32(ncol),
        lay("pavel"), colamt_h2o.reshape(lm, ncol), lay("colbrd"), lay("scaleminorn2"),
        lay("selffac"), lay("selffrac"), lay("indself"),
        lay("forfac"), lay("forfrac"), lay("indfor"),
        lay("minorfrac"), lay("indminor"),
        lay("fac00"), lay("fac01"), lay("fac10"), lay("fac11"),
        lay("jp"), lay("jt"), lay("jt1"),
        t["rrtm_lw_kgb01_absa"], t["rrtm_lw_kgb01_absb"],
        t["rrtm_lw_kgb01_selfref"], t["rrtm_lw_kgb01_forref"],
        t["rrtm_lw_kgb01_ka_mn2"], t["rrtm_lw_kgb01_kb_mn2"],
        t["rrtm_lw_kgb01_fracrefa"], t["rrtm_lw_kgb01_fracrefb"],
        st["taug_b01"].reshape(lm, NG01, ncol),
        st["fracs_b01"].reshape(lm, NG01, ncol),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (block, 1, 1), args)
    return st
