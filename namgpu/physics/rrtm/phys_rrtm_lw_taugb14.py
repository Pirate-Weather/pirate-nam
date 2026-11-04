# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb14: RRTM LW band 14 optical depths and Planck fractions (taumol group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:5912-5976, subroutine taugb14
(internal to taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_lw_taugb14.cu). Kernel: phys_rrtm_lw_taugb14, one thread per column.

Band 14 (2250-2380 cm-1): lower layers use the H2O self and foreign continuum and the CO2 major
absorption (colamt(k,2)); upper layers use the CO2 absorption only. The band's 2 g-points are rows
NS14 + ig of the taug / fracs arrays (NS14 = 134). This wrapper writes the 2-row band slab
taug14[ig][k][col] and fracs14[ig][k][col]; the caller places the slab at rows 134..135.

Tables: module_radlw_kgb14 from phys/radlw_datatb.f (the module the routine USEs at
radlw_main_nmmb.f:5919), extracted by tools/extract-rrtm-lw-kgb14.py to namgpu/data/rrtm_lw/kgb14.npz.
The module_radlw_kgb14_nmmb block of radlw_datatb_nmmb.f is not used by this routine and is not loaded.

Reads ctx.tables:
  "rrtm_lw_taumol_in"  dict of device inputs (CONTRACT layouts, see the kernel header):
      laytrop [jm, im] int32; jp, jt, jt1, indself, indfor [nlay, jm, im] int32;
      colamt [6, nlay, jm, im]; selffac, selffrac, forfac, forfrac, fac00, fac01, fac10, fac11 [nlay, jm, im] float32.
      Indices are the Fortran 1-based values; these are the keys phys-rrtm-lw-setcoef writes.
Output: run() returns {"taug": [140, nlay, ntile], "fracs": [140, nlay, ntile]} for the column tile
c0..c0+ntile-1: the full g-point arrays the Fortran writes (taug(ns14+ig,k), fracs(ns14+ig,k), NS14 = 134).
Only rows 134 and 135 are written by this unit; the other rows stay zero (zero-filled once per buffer), exactly
like the taugb08 scratch convention. Full-size domains run tile by tile (the full 140-row array for
954x835x60 is 26.7 GB).

Precision: float32. The NAM production configure (gate0/nmmb/src/conf/configure.nems.Wcoss_C.intel_ESMF_630r_gsm,
CPPFLAGS -DSINGLE_PREC) defines SINGLE_PREC, so machine.f sets kind_phys = 4 (the NAM build the port follows).

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb14"
KERNEL = "phys_rrtm_lw_taugb14"
IN_KEY = "rrtm_lw_taumol_in"
OUT_KEY = "rrtm_lw_taugb14_out"
KGB_KEY = "rrtm_lw_kgb14"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb14.npz"

NG14 = 2        # module_radlw_parameters NG14 (radlw_param.f:109)
NGPTLW = 140    # module_radlw_parameters ngptlw: full g-point count of the taug/fracs arrays
NS14 = 134      # module_radlw_parameters NS14 (radlw_param.f:116)
MAXGAS = 6      # module_radlw_parameters maxgas


def load_kgb14():
    """Band-14 k-tables as device float32 arrays (uploaded once; host read is at init only)."""
    import cupy as cp

    with np.load(NPZ) as d:
        return {
            "absa": cp.asarray(d["absa"].astype(np.float32)),          # [65, 2]
            "absb": cp.asarray(d["absb"].astype(np.float32)),          # [235, 2]
            "selfref": cp.asarray(d["selfref"].astype(np.float32)),    # [10, 2]
            "forref": cp.asarray(d["forref"].astype(np.float32)),      # [4, 2]
            "fracrefa": cp.asarray(d["fracrefa"].astype(np.float32)),  # [2]
            "fracrefb": cp.asarray(d["fracrefb"].astype(np.float32)),  # [2]
        }


_INT_INPUTS = ("laytrop", "jp", "jt", "jt1", "indself", "indfor")
_F32_INPUTS = ("selffac", "selffrac", "forfac", "forfrac", "fac00", "fac01", "fac10", "fac11", "colamt")


def _check_arr(name, arr, shape, dtype):
    """Shape, dtype and C-contiguity: the kernel indexes flat, so all three must hold."""
    import cupy as cp

    if not isinstance(arr, cp.ndarray):
        raise TypeError(f"taugb14: {name} must be a CuPy array, got {type(arr).__name__}")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"taugb14: {name} shape {tuple(arr.shape)} != {tuple(shape)}")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(f"taugb14: {name} dtype {arr.dtype} != {np.dtype(dtype)}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"taugb14: {name} is not C-contiguous")


def _check_shapes(inp, jm, im, nlay):
    _check_arr("laytrop", inp["laytrop"], (jm, im), np.int32)
    for name in _INT_INPUTS[1:]:
        _check_arr(name, inp[name], (nlay, jm, im), np.int32)
    for name in _F32_INPUTS:
        shape = (MAXGAS, nlay, jm, im) if name == "colamt" else (nlay, jm, im)
        _check_arr(name, inp[name], shape, np.float32)


def taugb14(inp: dict, kgb: dict, out: dict, nlay: int, jm: int, im: int, c0: int, ntile: int, stream=None):
    """One launch over tile columns c0..c0+ntile-1 of the jm*im column set. Device arrays, CONTRACT layout."""
    _check_shapes(inp, jm, im, nlay)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    ncol = im * jm
    assert 0 <= c0 and c0 + ntile <= ncol, (c0, ntile, ncol)
    with scope:
        kern(kernels.grid_cols(ntile, 1), kernels.BLOCK_COL, (
            np.int32(nlay), np.int32(ncol), np.int32(c0), np.int32(ntile),
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"], inp["indself"], inp["indfor"],
            inp["colamt"],
            inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            kgb["absa"], kgb["absb"], kgb["selfref"], kgb["forref"],
            kgb["fracrefa"], kgb["fracrefb"],
            out["taug"], out["fracs"],
        ))
    return out


def _scratch(ctx, name, shape):
    import cupy as cp
    buf = ctx.tables.get(name)
    if buf is None or tuple(buf.shape) != shape:
        buf = cp.zeros(shape, dtype=cp.float32)   # zero-filled once: rows other than 134, 135 must read 0
        ctx.tables[name] = buf
    return buf


def run(ctx, stream=None, *, c0: int = 0, ncol: int | None = None):
    """Band-14 optical depths for the current taumol inputs, columns c0..c0+ncol-1 (default: all).

    Returns ctx.tables[OUT_KEY] = {"taug", "fracs"}, each [NGPTLW, nlay, ntile] tile-local scratch.
    """
    import cupy as cp

    if ctx.tables is None:
        ctx.tables = {}
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_lw_taugb14 needs ctx.tables[{IN_KEY!r}] from the taumol driver")

    nlay, jm, im = inp["jp"].shape
    kgb = ctx.tables.get(KGB_KEY)
    if kgb is None:
        kgb = load_kgb14()
        ctx.tables[KGB_KEY] = kgb

    if ncol is None:
        ncol = jm * im - c0
    shape = (NGPTLW, nlay, ncol)
    out = {"taug": _scratch(ctx, OUT_KEY + "_taug", shape),
           "fracs": _scratch(ctx, OUT_KEY + "_fracs", shape)}
    ctx.tables[OUT_KEY] = out
    return taugb14(inp, kgb, out, nlay, jm, im, c0, ncol, stream=stream)
