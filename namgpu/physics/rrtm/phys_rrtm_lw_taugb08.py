# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb08: LW band 8 optical depths and Planck fractions (phys/radlw_main_nmmb.f:4946-5065).

NOAA/NCEP NAM (nam.v4.2.11, 83df4e13c1) source; header notice kept in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_lw_taugb08.cu, phys_rrtm_lw_taugb08. One thread per column
(kernels.grid_cols), the layer loop and the g loop inside the thread. One launch per tile, no host
sync, no per-step allocation (taug and fracs tile buffers are scratch in ctx.tables).

Inputs: the per-layer setcoef outputs of the same column set, passed as a dict (setcoef is a
separate unit; its outputs keep the Fortran names and the Fortran 1-based integer values):
    laytrop          int32  [JM, IM]          Fortran laytrop per column
    jp, jt, jt1      int32  [LM, JM, IM]      1-based pressure/temperature reference indices
    indself, indfor, indminor  int32 [LM, JM, IM]  1-based reference-profile indices
    coldry           f32    [LM, JM, IM]
    colamt           f32    [6, LM, JM, IM]    colamt(k,n) at [n-1, l, j, i]
    wx               f32    [4, LM, JM, IM]    wx(k,n) at [n-1, l, j, i]
    selffac, selffrac, forfac, forfrac, minorfrac, fac00, fac10, fac01, fac11  f32 [LM, JM, IM]
The k-tables come from namgpu/data/rrtm_lw/kgb08.npz (tools/extract_rrtm_lw_kgb08.py), uploaded
once into ctx.tables.

Outputs (tile-local, columns c0 .. c0+ncol-1 of the flat column index j*IM + i):
    taug   f32 [140, LM, ncol]   optical depth; only g = 88..95 (band 8) written, rest zero
    fracs  f32 [140, LM, ncol]   Planck fractions; only g = 88..95 written, rest zero
Index contract (kernel guards, NaN on violation): lower-atmosphere layers (laytrop) need
jp 1..11, upper layers need jp 13..57; jt, jt1 1..5; indself 1..9; indfor 1..3; indminor 1..18.
setcoef produces these; a violation shows up as NaN in the output, never as a stale read.
With c0 = 0 and ncol = JM*IM this is the CONTRACT layout [140, LM, JM, IM] reshaped.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb08"
KERNEL = "phys_rrtm_lw_taugb08"
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb08.npz"
NGPTLW = 140
NG08 = 8

_TABLE_KEYS = ("absa", "absb", "selfref", "forref", "ka_mco2", "kb_mco2", "ka_mo3", "ka_mn2o",
               "kb_mn2o", "fracrefa", "fracrefb", "cfc12", "cfc22adj", "chi_mls_co2")


def tables(ctx):
    import cupy as cp
    t = ctx.tables
    if "rrtm_lw_kgb08" not in t:
        z = np.load(TABLE_FILE)
        t["rrtm_lw_kgb08"] = {k: cp.ascontiguousarray(cp.asarray(z[k], dtype=cp.float32))
                              for k in _TABLE_KEYS}
    return t["rrtm_lw_kgb08"]


def _scratch(ctx, name, shape):
    import cupy as cp
    buf = ctx.tables.get(name)
    if buf is None or tuple(buf.shape) != shape:
        # zero-filled once: g-points outside band 8 are never written by the kernel and must read 0
        buf = cp.zeros(shape, dtype=cp.float32)
        ctx.tables[name] = buf
    return buf


def run(ctx, inputs: dict, *, c0: int = 0, ncol: int | None = None):
    """Compute taug and fracs for columns c0 .. c0+ncol-1 (default: all columns).

    inputs: dict of CuPy arrays with the names and shapes in the module docstring.
    Returns (taug, fracs), each [140, LM, ncol] float32 tile-local scratch (valid until the next
    call with the same shape).
    """
    import cupy as cp

    lm, jm, im = (int(x) for x in inputs["coldry"].shape)
    nc = jm * im
    if ncol is None:
        ncol = nc - c0
    assert 0 <= c0 and c0 + ncol <= nc, (c0, ncol, nc)
    tab = tables(ctx)

    taug = _scratch(ctx, "rrtm_lw_taugb08_taug", (NGPTLW, lm, ncol))
    fracs = _scratch(ctx, "rrtm_lw_taugb08_fracs", (NGPTLW, lm, ncol))

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(ncol, 1), kernels.BLOCK_COL,
          (inputs["laytrop"].reshape(-1),
           inputs["jp"].reshape(-1), inputs["jt"].reshape(-1), inputs["jt1"].reshape(-1),
           inputs["indself"].reshape(-1), inputs["indfor"].reshape(-1),
           inputs["indminor"].reshape(-1),
           inputs["coldry"].reshape(-1), inputs["colamt"].reshape(-1), inputs["wx"].reshape(-1),
           inputs["selffac"].reshape(-1), inputs["selffrac"].reshape(-1),
           inputs["forfac"].reshape(-1), inputs["forfrac"].reshape(-1),
           inputs["minorfrac"].reshape(-1),
           inputs["fac00"].reshape(-1), inputs["fac10"].reshape(-1),
           inputs["fac01"].reshape(-1), inputs["fac11"].reshape(-1),
           tab["chi_mls_co2"], tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
           tab["ka_mco2"], tab["kb_mco2"], tab["ka_mo3"], tab["ka_mn2o"], tab["kb_mn2o"],
           tab["fracrefa"], tab["fracrefb"], tab["cfc12"], tab["cfc22adj"],
           taug, fracs,
           np.int32(lm), np.int32(nc), np.int32(c0), np.int32(ncol)))
    return taug, fracs
