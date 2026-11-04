# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb09: LW band 9 optical depths and Planck fractions (phys/radlw_main_nmmb.f:5069-5313).

NOAA/NCEP NAM (nam.v4.2.11, 83df4e13c1) source; header notice kept in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_lw_taugb09.cu, phys_rrtm_lw_taugb09. One thread per column
(kernels.grid_cols), the layer loop and the g loop inside the thread. One launch per tile, no host
sync, no per-step allocation (taug and fracs tile buffers are scratch in ctx.tables).

Inputs: the per-layer setcoef outputs of the same column set, passed as a dict (setcoef is a separate
unit; its outputs keep the Fortran names and the Fortran 1-based integer values):
    laytrop                     int32  [JM, IM]          Fortran laytrop per column
    jp, jt, jt1                 int32  [LM, JM, IM]      1-based reference indices
    indself, indfor, indminor   int32  [LM, JM, IM]      1-based reference-profile indices
    coldry                      f32    [LM, JM, IM]
    colamt                      f32    [6, LM, JM, IM]   colamt(k,n) at [n-1, l, j, i]
    rfrate                      f32    [2, 6, LM, JM, IM] rfrate(k,m,n) at [n-1, m-1, l, j, i]
    selffac, selffrac, forfac, forfrac, minorfrac, fac00, fac10, fac01, fac11   f32 [LM, JM, IM]
The k-tables and chi_mls come from namgpu/data/rrtm_lw/kgb09.npz (tools/extract_rrtm_lw_kgb09.py),
uploaded once into ctx.tables. The table source is module_radlw_kgb09 (radlw_datatb.f), the module
the routine uses; the _nmmb table copy is not what taugb09 calls.

LAYER ORDER: every per-layer input and every per-layer output (the LM axis below) is RRTM INTERNAL
order, SURFACE FIRST: index 0 = the LOWEST layer, index LM-1 = the model top. This is the radlw
exception in CONTRACT section 1, not the NMMB top-first order. laytrop counts the lower-atmosphere
layers from index 0 (Fortran k = 1 .. laytrop). A caller holding NMMB top-first arrays must reverse
them along the layer axis before run() and reverse taug and fracs back after.

Outputs (tile-local, columns c0 .. c0+ncol-1 of the flat column index j*IM + i):
    taug   f32 [140, LM, ncol]   optical depth; only g = 96..107 (band 9) written; surface-first
    fracs  f32 [140, LM, ncol]   Planck fractions; only g = 96..107 written; surface-first
With c0 = 0 and ncol = JM*IM this is [140, LM, JM, IM] in the surface-first layer order.

Fortran-side constants: nspa(9) = 9, nspb(9) = 1 (radlw_main_nmmb.f:284-285), NG09 = 12,
NS09 = 96 (radlw_param.f:106-115), oneminus = 1 - 1e-6 (radlw_main_nmmb.f:267-268).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb09"
KERNEL = "phys_rrtm_lw_taugb09"
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb09.npz"
NGPTLW = 140
NG09 = 12

_TABLE_KEYS = ("absa", "absb", "selfref", "forref", "ka_mn2o", "kb_mn2o", "fracrefa", "fracrefb",
               "chi_mls")


def tables(ctx):
    import cupy as cp
    t = ctx.tables
    if "rrtm_lw_kgb09" not in t:
        z = np.load(TABLE_FILE)
        t["rrtm_lw_kgb09"] = {k: cp.ascontiguousarray(cp.asarray(z[k], dtype=cp.float32))
                              for k in _TABLE_KEYS}
    return t["rrtm_lw_kgb09"]


def _scratch(ctx, name, shape):
    import cupy as cp
    buf = ctx.tables.get(name)
    if buf is None or tuple(buf.shape) != shape:
        buf = cp.empty(shape, dtype=cp.float32)
        ctx.tables[name] = buf
    return buf


_INT_KEYS = ("laytrop", "jp", "jt", "jt1", "indself", "indfor", "indminor")
_F32_CELL_KEYS = ("coldry", "selffac", "selffrac", "forfac", "forfrac", "minorfrac",
                  "fac00", "fac10", "fac01", "fac11")


def _check_inputs(inputs: dict, lm: int, jm: int, im: int):
    """Dtype, shape and C-contiguity of every input (host metadata only)."""
    import cupy as cp
    cell = (lm, jm, im)
    want = {k: (jm, im) if k == "laytrop" else cell for k in _INT_KEYS}
    want.update({k: cell for k in _F32_CELL_KEYS})
    want.update({"colamt": (6,) + cell, "rfrate": (2, 6) + cell})
    dtypes = {k: cp.int32 for k in _INT_KEYS}
    dtypes.update({k: cp.float32 for k in _F32_CELL_KEYS + ("colamt", "rfrate")})
    for k, shape in want.items():
        a = inputs[k]
        if not isinstance(a, cp.ndarray):
            raise TypeError(f"phys_rrtm_lw_taugb09: {k} must be a CuPy array, got {type(a).__name__}")
        if a.dtype != dtypes[k]:
            raise TypeError(f"phys_rrtm_lw_taugb09: {k} dtype {a.dtype}, need {dtypes[k]}")
        if tuple(a.shape) != shape:
            raise ValueError(f"phys_rrtm_lw_taugb09: {k} shape {tuple(a.shape)}, need {shape}")
        if not a.flags.c_contiguous:
            raise ValueError(f"phys_rrtm_lw_taugb09: {k} must be C-contiguous")


def check_ranges(inputs: dict) -> dict:
    """Value-range check of the caller obligations (device sync; debug and selftest only).

    Returns the number of out-of-range values per key; raises ValueError if any is nonzero.
    Bounds: the table row limits of the kernel (absa 585 rows, absb 235, selfref 10, forref 4,
    ka_mn2o 19, kb_mn2o 19) plus setcoef's clamps (jt 1..5, jt1 1..4); jp 1..58 (Fortran
    reference-pressure index); inds+1 <= 10 and indf+1 <= 4 and indm+1 <= 19 so the +1 partners
    stay in the tables.
    """
    import cupy as cp
    bounds = {
        "jp": (1, 58), "jt": (1, 5), "jt1": (1, 4),
        "indself": (1, 9), "indfor": (1, 3), "indminor": (1, 18),
        "laytrop": (0, int(inputs["coldry"].shape[0])),
        "selffrac": (0.0, 1.0), "forfrac": (0.0, 1.0), "minorfrac": (0.0, 1.0),
        "fac00": (0.0, 1.0), "fac10": (0.0, 1.0), "fac01": (0.0, 1.0), "fac11": (0.0, 1.0),
    }
    bad = {}
    for k, (lo, hi) in bounds.items():
        a = inputs[k]
        bad[k] = int(cp.count_nonzero((a < lo) | (a > hi)))
    bad["coldry"] = int(cp.count_nonzero(~(inputs["coldry"] > 0)))
    if any(bad.values()):
        raise ValueError(f"phys_rrtm_lw_taugb09: out-of-range inputs {bad}")
    return bad


def run(ctx, inputs: dict, *, c0: int = 0, ncol: int | None = None):
    """Compute taug and fracs for columns c0 .. c0+ncol-1 (default: all columns).

    inputs: dict of CuPy arrays with the names, dtypes and shapes in the module docstring. Dtype,
    shape and C-contiguity are checked here (host metadata only, no device sync). The value
    ranges (jp 1..58, jt 1..5, jt1 1..4, indself 1..9, indfor 1..3, indminor 1..18, laytrop 0..LM)
    are caller obligations; check_ranges() verifies them with a device sync and is not called on
    the step path.
    Returns (taug, fracs), each [140, LM, ncol] float32 tile-local scratch (valid until the next
    call with the same shape).
    """
    lm, jm, im = (int(x) for x in inputs["coldry"].shape)
    nc = jm * im
    _check_inputs(inputs, lm, jm, im)
    if ncol is None:
        ncol = nc - c0
    assert 0 <= c0 and c0 + ncol <= nc, (c0, ncol, nc)
    tab = tables(ctx)

    taug = _scratch(ctx, "rrtm_lw_taugb09_taug", (NGPTLW, lm, ncol))
    fracs = _scratch(ctx, "rrtm_lw_taugb09_fracs", (NGPTLW, lm, ncol))

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(ncol, 1), kernels.BLOCK_COL,
          (inputs["laytrop"].reshape(-1),
           inputs["jp"].reshape(-1), inputs["jt"].reshape(-1), inputs["jt1"].reshape(-1),
           inputs["indself"].reshape(-1), inputs["indfor"].reshape(-1),
           inputs["indminor"].reshape(-1),
           inputs["coldry"].reshape(-1), inputs["colamt"].reshape(-1), inputs["rfrate"].reshape(-1),
           inputs["selffac"].reshape(-1), inputs["selffrac"].reshape(-1),
           inputs["forfac"].reshape(-1), inputs["forfrac"].reshape(-1),
           inputs["minorfrac"].reshape(-1),
           inputs["fac00"].reshape(-1), inputs["fac10"].reshape(-1),
           inputs["fac01"].reshape(-1), inputs["fac11"].reshape(-1),
           tab["chi_mls"], tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
           tab["ka_mn2o"], tab["kb_mn2o"], tab["fracrefa"], tab["fracrefb"],
           taug, fracs,
           np.int32(lm), np.int32(nc), np.int32(c0), np.int32(ncol)))
    return taug, fracs


def _random_inputs(lm: int, jm: int, im: int, seed: int = 1):
    """Plausible full-size random setcoef outputs: lower layers have jp 1..12, upper jp 13..58,
    binary-species and minor-species indices within the table bounds, positive column amounts."""
    import cupy as cp
    rs = cp.random.RandomState(seed)
    nc = jm * im
    cell = (lm, jm, im)
    lt = rs.randint(20, 41, size=(jm, im)).astype(cp.int32)            # laytrop per column
    layer = cp.arange(lm, dtype=cp.int32).reshape(lm, 1, 1)
    lower = layer < lt[None]                                            # lower-atmosphere mask
    jp = cp.where(lower, rs.randint(1, 13, size=cell), rs.randint(13, 59, size=cell)).astype(cp.int32)
    jt = rs.randint(1, 6, size=cell).astype(cp.int32)                  # 1..5 (setcoef clamp)
    jt1 = rs.randint(1, 5, size=cell).astype(cp.int32)                 # 1..4 (setcoef clamps jt1 to 4)
    indself = rs.randint(1, 10, size=cell).astype(cp.int32)            # selfref rows 1..10 (inds+1 <= 10)
    indfor = rs.randint(1, 4, size=cell).astype(cp.int32)              # forref rows 1..4
    indminor = rs.randint(1, 19, size=cell).astype(cp.int32)           # ka/kb_mn2o rows 1..19
    coldry = rs.uniform(1.0e23, 2.0e23, cell).astype(cp.float32)
    colamt = rs.uniform(1.0e18, 1.0e22, (6,) + cell).astype(cp.float32)
    rfrate = rs.uniform(0.5, 2.0, (2, 6) + cell).astype(cp.float32)
    return dict(
        laytrop=lt,
        jp=jp, jt=jt, jt1=jt1,
        indself=indself, indfor=indfor, indminor=indminor,
        coldry=coldry, colamt=colamt, rfrate=rfrate,
        selffac=rs.uniform(0.0, 1.0e-2, cell).astype(cp.float32),
        selffrac=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        forfac=rs.uniform(0.0, 1.0e-2, cell).astype(cp.float32),
        forfrac=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        minorfrac=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        fac00=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        fac10=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        fac01=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
        fac11=rs.uniform(0.0, 1.0, cell).astype(cp.float32),
    )


def selftest(ctx, lm: int = 60, jm: int = 835, im: int = 954, tile: int = 8192, seed: int = 1):
    """Run the wrapper over every column of a full-size random state (954 x 835 x 60) and return
    the NaN/Inf counts of the band-9 outputs.

    The columns are independent, so each tile of `tile` columns gets its own random state of
    shape (LM, 1, tile) and is run with c0 = 0. The total computed is the full domain; the
    whole-domain input and taug arrays (several GB) are never resident at once."""
    import cupy as cp
    nc = jm * im
    bad = 0
    total = 0
    for n, c0 in enumerate(range(0, nc, tile)):
        ncol = min(tile, nc - c0)
        inputs = _random_inputs(lm, 1, ncol, seed + n)
        check_ranges(inputs)
        taug, fracs = run(ctx, inputs, c0=0, ncol=ncol)
        band = slice(96, 108)
        for arr in (taug[band], fracs[band]):
            bad += int(cp.count_nonzero(~cp.isfinite(arr)))
            total += int(arr.size)
    return {"nonfinite": bad, "checked": total, "columns": nc}
