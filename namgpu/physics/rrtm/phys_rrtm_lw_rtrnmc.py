# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-rtrnmc: McICA longwave radiative transfer (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:3059-3418, subroutine rtrnmc.
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the NOAA/AER header is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_lw_rtrnmc.cu). Kernel: rrtm_lw_rtrnmc, one thread per column, one launch
per column chunk.

Callers: the LW driver (lwrad, radlw_main_nmmb.f) calls rtrnmc once per column, after the gas and
cloud optical depths and the McICA sub-column fractions are built for that column. This wrapper is
that call for a chunk of columns.

Why chunks: the McICA inputs are per g-point and per layer (140 g-points x 60 layers per column).
Over the full 954 x 835 domain cldfmc, tautot and fracs alone are about 27 GB each, which does not fit
the card. The Fortran already works in column chunks (CHNK_RRTM); the chunk size here is the caller's
choice, and every chunk runs the same single launch. Chunks do not change the numbers.

Column index: the flat column axis c is j0*IM + i0 (CONTRACT section 1, IDX3 ordering) for the
Fortran column (i, j). A chunk is a contiguous run of c.

Arguments (CuPy float32 unless noted; the column axis is LAST and flat over the chunk):
  inputs (dict):
    semiss  [NBANDS, NCOL]           surface emissivity per band
    secdif  [NBANDS, NCOL]           secant of the diffusivity angle per band
    delp    [NLAY, NCOL]             layer pressure thickness (mb)
    cldfmc  [NLAY, NGPTLW, NCOL]     McICA sub-column cloud fraction
    taucld  [NLAY, NBANDS, NCOL]     cloud optical depth per band
    tautot  [NLAY, NGPTLW, NCOL]     total (gas + aerosol) optical depth per g-point
    pklay   [NLAY+1, NBANDS, NCOL]   Planck function at layer temperature (index 0..nlay)
    pklev   [NLAY+1, NBANDS, NCOL]   Planck function at level temperature (index 0..nlay)
    fracs   [NLAY, NGPTLW, NCOL]     Planck fractions per g-point
    ngb     [NGPTLW] int32           band index of each g-point, 1-based (1..NBANDS), checked once
    tau_tbl, exp_tbl, tfn_tbl [NTBL+1]  look-up tables; build_tables() is the rlwinit transcription
  outputs (dict, written in full):
    totuflux, totdflux, totuclfl, totdclfl  [NLAY+1, NCOL]  fluxes (w/m2), index 0..nlay
    htr, htrcl                              [NLAY, NCOL]    heating rates (heatfac units)
    htrb                                    [NBANDS, NLAY, NCOL]  band heating rates (heatfac units)
  htrcl is zero when lhlw0 is off and htrb is zero when lhlwb is off (the kernel writes the zeros, so
  no output is ever left as uninitialised device memory).

Scalars:
  nlay            1..MAXLAY
  ilwrate         heating-rate unit switch, physparam.f:71 (NAM = 2, k/s). heatfac follows it:
                  radlw_main_nmmb.f:1290-1295, heatfac = con_g*864/con_cp (ilwrate 1, k/day) or
                  con_g*1e-2/con_cp (k/s). heatfac=None binds the live value for ilwrate.
  fluxfac         pi*2e4 in single precision (radlw_main_nmmb.f:1287), default below.
  lhlw0, lhlwb    module flags (lhlw0 / lhlwb in radlw_main_nmmb.f).

Input contiguity: inputs and outputs may be any CuPy view. Non-contiguous inputs are copied in;
non-contiguous outputs are computed into a dense buffer and copied back. The kernel assumes dense
C order with the chunk width as the trailing stride.

Band table: the g-point to band map is module_radlw_parameters NGB in radlw_param.f (the module
radlw_main_nmmb.f USEs): 10*1, 12*2, 16*3, 14*4, 16*5, 8*6, 12*7, 8*8, 12*9, 6*10, 8*11, 8*12, 4*13,
2*14, 2*15, 2*16. The check builds ngb from NG01..NG16 of that module.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_rtrnmc"
NBANDS = 16        # radlw_param.f:94
NGPTLW = 140       # radlw_param.f:95
NTBL = 10000       # radlw_param.f:96
MAXLAY = 128       # kernel scratch limit (RRTM_LW_MAXLAY)
ILWRATE_NAM = 2    # physparam.f:71 (1 = k/day, 2 = k/s)
CON_G = np.float32(9.80665)     # physcons.f:46
CON_CP = np.float32(1.0046e3)   # physcons.f:59
FLUXFAC_DEFAULT = float(np.float32(math.pi) * np.float32(2.0e4))   # radlw_main_nmmb.f:1287

_IN_SHAPES = {
    "semiss": lambda n, nc: (NBANDS, nc),
    "secdif": lambda n, nc: (NBANDS, nc),
    "delp": lambda n, nc: (n, nc),
    "cldfmc": lambda n, nc: (n, NGPTLW, nc),
    "taucld": lambda n, nc: (n, NBANDS, nc),
    "tautot": lambda n, nc: (n, NGPTLW, nc),
    "pklay": lambda n, nc: (n + 1, NBANDS, nc),
    "pklev": lambda n, nc: (n + 1, NBANDS, nc),
    "fracs": lambda n, nc: (n, NGPTLW, nc),
}
_OUT_SHAPES = {
    "totuflux": lambda n, nc: (n + 1, nc),
    "totdflux": lambda n, nc: (n + 1, nc),
    "totuclfl": lambda n, nc: (n + 1, nc),
    "totdclfl": lambda n, nc: (n + 1, nc),
    "htr": lambda n, nc: (n, nc),
    "htrcl": lambda n, nc: (n, nc),
    "htrb": lambda n, nc: (NBANDS, n, nc),
}
_TBL_SHAPES = (("tau_tbl", NTBL + 1), ("exp_tbl", NTBL + 1), ("tfn_tbl", NTBL + 1))

# ngb tables already range-checked, keyed by device pointer and shape. The array is kept alive in
# the value so its pointer cannot be reused by another array. One host copy per table, not per call.
_NGB_CHECKED: dict = {}


def heatfac_for(ilwrate: int = ILWRATE_NAM) -> float:
    """heatfac as radlw_main_nmmb.f:1290-1295 sets it (single precision, Fortran evaluation order)."""
    if ilwrate == 1:
        val = (CON_G * np.float32(864.0)) / CON_CP
    else:
        val = (CON_G * np.float32(1.0e-2)) / CON_CP
    return float(np.float32(val))


def build_tables():
    """tau_tbl, exp_tbl, tfn_tbl: transcription of rlwinit (radlw_main_nmmb.f:1313-1339), single.

    Returns host numpy float32 arrays of length NTBL+1. The caller uploads them to the device.
    """
    f32 = np.float32
    ntbl = NTBL
    bpade = f32(1.0 / 0.278)                 # radlw_main_nmmb.f:270
    expeps = f32(1.0e-20)                    # radlw_main_nmmb.f:1220
    explimit = f32(math.floor(-math.log(float(np.finfo(np.float32).tiny))))  # aint(-log(tiny))
    tau = np.zeros(ntbl + 1, dtype=f32)
    exp_t = np.zeros(ntbl + 1, dtype=f32)
    tfn = np.zeros(ntbl + 1, dtype=f32)
    exp_t[0] = f32(1.0)
    tau[ntbl] = f32(1.0e10)
    exp_t[ntbl] = expeps
    tfn[ntbl] = f32(1.0)
    i = np.arange(1, ntbl, dtype=np.int64)
    tfn_i = i.astype(f32) / (ntbl - i).astype(f32)
    tau_i = (bpade * tfn_i).astype(f32)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exp_i = np.where(tau_i >= explimit, expeps, np.exp(-tau_i)).astype(f32)
        tfn_big = (f32(1.0) - f32(2.0) * ((f32(1.0) / tau_i) - (exp_i / (f32(1.0) - exp_i)))).astype(f32)
    tfn_i = np.where(tau_i < f32(0.06), tau_i / f32(6.0), tfn_big).astype(f32)
    tau[1:ntbl] = tau_i
    exp_t[1:ntbl] = exp_i
    tfn[1:ntbl] = tfn_i
    return {"tau_tbl": tau, "exp_tbl": exp_t, "tfn_tbl": tfn}


def _check_ngb(cp, ngb):
    key = (int(ngb.data.ptr), tuple(ngb.shape))
    if key in _NGB_CHECKED:
        return
    host = cp.asnumpy(ngb)
    if host.min() < 1 or host.max() > NBANDS:
        raise ValueError(f"phys-rrtm-lw-rtrnmc: ngb must be 1-based band indices 1..{NBANDS}, "
                         f"got min {int(host.min())} max {int(host.max())}")
    _NGB_CHECKED[key] = (ngb, True)


def run(ctx, *, inputs: dict, outputs: dict, nlay: int, heatfac: float | None = None,
        lhlw0: bool = False, lhlwb: bool = False, fluxfac: float = FLUXFAC_DEFAULT,
        ilwrate: int = ILWRATE_NAM):
    """One rtrnmc launch over the columns of this chunk. Writes into `outputs`, returns it."""
    import cupy as cp

    if not 1 <= nlay <= MAXLAY:
        raise ValueError(f"phys-rrtm-lw-rtrnmc: nlay must be in 1..{MAXLAY}, got {nlay}")
    if heatfac is None:
        heatfac = heatfac_for(ilwrate)
    nc = int(inputs["delp"].shape[-1])
    args_in = {}
    for name, shape_of in _IN_SHAPES.items():
        arr = inputs[name]
        if tuple(arr.shape) != shape_of(nlay, nc) or arr.dtype != cp.float32:
            raise ValueError(f"phys-rrtm-lw-rtrnmc: {name} must be float32 {shape_of(nlay, nc)}, "
                             f"got {arr.dtype} {tuple(arr.shape)}")
        args_in[name] = arr if arr.flags.c_contiguous else cp.ascontiguousarray(arr)
    args_out = {}
    for name, shape_of in _OUT_SHAPES.items():
        arr = outputs[name]
        if tuple(arr.shape) != shape_of(nlay, nc) or arr.dtype != cp.float32:
            raise ValueError(f"phys-rrtm-lw-rtrnmc: output {name} must be float32 {shape_of(nlay, nc)}")
        args_out[name] = arr if arr.flags.c_contiguous else cp.empty(arr.shape, dtype=cp.float32)
    for name, n in (("ngb", NGPTLW),) + _TBL_SHAPES:
        arr = inputs[name]
        if tuple(arr.shape) != (n,):
            raise ValueError(f"phys-rrtm-lw-rtrnmc: {name} must have shape ({n},)")
    if inputs["ngb"].dtype != cp.int32:
        raise ValueError("phys-rrtm-lw-rtrnmc: ngb must be int32")
    for name, _ in _TBL_SHAPES:
        if inputs[name].dtype != cp.float32:
            raise ValueError(f"phys-rrtm-lw-rtrnmc: {name} must be float32")
    _check_ngb(cp, inputs["ngb"])
    args_in["ngb"] = inputs["ngb"]
    for name, _ in _TBL_SHAPES:
        args_in[name] = inputs[name]

    grid = kernels.grid_cols(nc, 1)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        kern = kernels.get(UNIT, "rrtm_lw_rtrnmc")
        kern(grid, kernels.BLOCK_COL,
             (np.int32(nc), np.int32(nlay), np.int32(bool(lhlw0)), np.int32(bool(lhlwb)),
              np.float32(fluxfac), np.float32(heatfac),
              args_in["semiss"], args_in["delp"], args_in["cldfmc"], args_in["taucld"],
              args_in["tautot"], args_in["pklay"], args_in["pklev"], args_in["fracs"],
              args_in["secdif"], args_in["ngb"], args_in["tau_tbl"], args_in["exp_tbl"],
              args_in["tfn_tbl"],
              args_out["totuflux"], args_out["totdflux"], args_out["htr"],
              args_out["totuclfl"], args_out["totdclfl"], args_out["htrcl"], args_out["htrb"]))
        for name in _OUT_SHAPES:
            if args_out[name] is not outputs[name]:
                outputs[name][...] = args_out[name]
    return outputs
