# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb10: TAUGB10, RRTM longwave band 10 (1390-1480 cm-1) optical depths and Planck fractions.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5317-5387 (subroutine taugb10).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the lw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_lw_taugb10.cu (rrtm_lw_taugb10, one thread per (i, j) column, one launch).

What it computes, per column: for layers 1..laytrop (lower atmosphere) and laytrop+1..nlay (upper
atmosphere), the six band-10 g-point optical depths taug(108+ig, k) from the h2o key-species k-table
interpolated at the setcoef pressure/temperature cell (self and foreign continuum added in the lower
atmosphere only), and the Planck fractions fracs(108+ig, k) = fracrefa or fracrefb.

Inputs (device, CONTRACT layout, Fortran (i,j,l) -> [l, j, i]; the h2o slab only of colamt):
  laytrop                               [JM, IM] int32
  jp, jt, jt1, indself, indfor          [nlay, JM, IM] int32 (Fortran 1-based, from setcoef)
  fac00, fac01, fac10, fac11            [nlay, JM, IM] float32
  selffac, selffrac, forfac, forfrac    [nlay, JM, IM] float32
  colamt                                [MAXGAS=6, nlay, JM, IM] float32 (colamt(k,1) = slab 0 is read)
Tables (device float32, uploaded once with load_tables):
  absa [65, 6], absb [235, 6], selfref [10, 6], forref [4, 6], fracrefa [6], fracrefb [6]
  from namgpu/data/rrtm_lw/kgb10.npz (module_radlw_kgb10, radlw_datatb.f:27968-28421).
Outputs (device float32): taug, fracs  [nlay, NGPTLW=140, JM, IM]. Only g-points 108..113 (0-based)
are written by this unit; the rest belong to other bands and are left as allocated.

Caller: the LW driver (phys-rrtm-lw-lwrad) calls this after setcoef, once per column chunk. It puts
the inputs under ctx.tables["rrtm_lw_taugb10_in"], the k-tables under
ctx.tables["rrtm_lw_taugb10_tables"], and the outputs are allocated here under
ctx.tables["rrtm_lw_taugb10_out"]. The column axis is the flat (j, i) domain; the chunk size in
JM is the caller's choice (the outputs are about 2 * nlay * 140 * JM * IM * 4 bytes).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb10"
KERNEL = "rrtm_lw_taugb10"
NG10 = 6
NGPTLW = 140
NS10 = 108
MAXGAS = 6
MSA10, MSB10, MSF10, MFR10 = 65, 235, 10, 4
KGB10_PATH = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb10.npz"

IN_KEY = "rrtm_lw_taugb10_in"
TAB_KEY = "rrtm_lw_taugb10_tables"
OUT_KEY = "rrtm_lw_taugb10_out"

_INT_IN = ("jp", "jt", "jt1", "indself", "indfor")
_FLOAT_IN = ("fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac")
_TAB_SHAPES = {
    "absa": (MSA10, NG10), "absb": (MSB10, NG10), "selfref": (MSF10, NG10),
    "forref": (MFR10, NG10), "fracrefa": (NG10,), "fracrefb": (NG10,),
}


def load_tables(path: Path = KGB10_PATH) -> dict:
    """Upload the band-10 k-tables once. Shapes are checked against the Fortran declarations."""
    import cupy as cp

    npz = np.load(path)
    tab = {}
    for name, shape in _TAB_SHAPES.items():
        arr = npz[name].astype(np.float32)
        if arr.shape != shape:
            raise ValueError(f"taugb10 table {name} shape {arr.shape} != {shape} in {path}")
        tab[name] = cp.asarray(arr)
    return tab


def alloc_outputs(nlay: int, jm: int, im: int):
    """taug and fracs for one column chunk, allocated once (zeros, so unwritten g-points are finite)."""
    import cupy as cp

    return {"taug": cp.zeros((nlay, NGPTLW, jm, im), dtype=cp.float32),
            "fracs": cp.zeros((nlay, NGPTLW, jm, im), dtype=cp.float32)}


def _check(inp: dict, tab: dict, nlay: int, jm: int, im: int):
    import cupy as cp

    if inp["laytrop"].shape != (jm, im) or inp["laytrop"].dtype != cp.int32:
        raise ValueError("taugb10: laytrop must be int32 [JM, IM]")
    for name in _INT_IN:
        if inp[name].shape != (nlay, jm, im) or inp[name].dtype != cp.int32:
            raise ValueError(f"taugb10: {name} must be int32 {(nlay, jm, im)}")
    for name in _FLOAT_IN:
        if inp[name].shape != (nlay, jm, im) or inp[name].dtype != cp.float32:
            raise ValueError(f"taugb10: {name} must be float32 {(nlay, jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im) or inp["colamt"].dtype != cp.float32:
        raise ValueError(f"taugb10: colamt must be float32 {(MAXGAS, nlay, jm, im)}")
    for name, shape in _TAB_SHAPES.items():
        if tab[name].shape != shape or tab[name].dtype != cp.float32:
            raise ValueError(f"taugb10: table {name} must be float32 {shape}")


def taugb10(inp: dict, tab: dict, out: dict, stream=None):
    """One launch over every (i, j) column of this chunk. All arrays are device arrays in CONTRACT layout."""
    nlay, jm, im = inp["fac00"].shape
    _check(inp, tab, nlay, jm, im)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(nlay),
            inp["laytrop"],
            inp["jp"], inp["jt"], inp["jt1"],
            inp["colamt"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
            tab["fracrefa"], tab["fracrefb"],
            out["taug"], out["fracs"],
        ))
    return out


def run(ctx):
    """Driver entry. The LW driver puts the inputs and tables in ctx.tables before calling this."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb10 needs ctx.tables[{IN_KEY!r}] from the LW driver (phys-rrtm-lw-lwrad)")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["fac00"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug"].shape != (nlay, NGPTLW, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taugb10(inp, tab, out, stream=getattr(ctx, "stream", None))
