# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-rtrnmr: rtrnmr, maximum-random overlap LW flux and heating-rate transfer (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:2489-3054, subroutine rtrnmr.
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE), no licence file; the routine header is kept as the notice in the kernel
file namgpu/kernels/phys_rrtm_lw_rtrnmr.cu. Kernel: phys_rrtm_lw_rtrnmr, one thread per column,
one launch over the whole (IM, JM) grid. No host sync, no per-step allocation.

Callers: the LW driver (lwrad, phys/radlw_main_nmmb.f) passes the per-column layer and cloud state
and gets back the fluxes and heating rates. This wrapper is that call.

Level order: NAM slot order, the same convention as phys-rrtm-lw-rtrn (slot 0 = model top, the
CONTRACT order). The kernel maps the Fortran indices itself, so the caller passes the arrays lwrad
builds (k1 = nlp1 - k, radlw_main_nmmb.f ivflip == 0):
  Fortran layer k (1..nlay)        -> slot nlay - k
  Fortran interface m (0..nlay)    -> slot nlay - m     (pklay, pklev, totuflux, ...)
  Fortran cldfrc(m), m = 0..nlay+1 -> slot nlay + 1 - m
No flip is applied by the caller and none is needed; the outputs come back in the same slot order.

Band order: the g-point band map is radlw_param.f module_radlw_parameters (NG01=10, NGB data), the
table radlw_main_nmmb.f rtrnmr reads (ib = ngb(ig)). It is the same layout the gas-optics units produce
(taugb01 has NG01 = 10). The wrapper asserts the kernel band table against the NGB data statement at
import.

Arguments (CuPy float32, C order; the column is the last two axes, [JM, IM]):
  semiss   [NBANDS, JM, IM]           Fortran semiss(nbands)
  secdif   [NBANDS, JM, IM]           Fortran secdif(nbands)
  delp     [NLAY, JM, IM]             slot s holds Fortran delp(k), s = nlay - k
  cldfrc   [NLAY+2, JM, IM]           slot s holds Fortran cldfrc(m), s = nlay + 1 - m
  taucld   [NBANDS, NLAY, JM, IM]     slot s holds Fortran layer k, s = nlay - k
  tautot   [NGPTLW, NLAY, JM, IM]     slot s holds Fortran layer k, s = nlay - k
  fracs    [NGPTLW, NLAY, JM, IM]     slot s holds Fortran layer k, s = nlay - k
  pklay    [NBANDS, NLAY+1, JM, IM]   slot s holds Fortran pklay(nbands, m), s = nlay - m
  pklev    [NBANDS, NLAY+1, JM, IM]   slot s holds Fortran pklev(nbands, m), s = nlay - m
  tables   dict with exp_tbl, tfn_tbl, tau_tbl, each float32 [NTBL+1] (see lw_tables)
  outputs (written in place, float32, same slot order):
    totuflux, totdflux, totuclfl, totdclfl   [NLAY+1, JM, IM]   slot nlay - m
    htr, htrcl                               [NLAY, JM, IM]     slot nlay - k
    htrb                                     [NBANDS, NLAY, JM, IM]  slot nlay - k
  heatfac  float, con_g*1e-2/con_cp (ilwrate = 2, k/second; radlw_param_nmmb.f:67, radlw_main_nmmb.f:1295)
  fluxfac  float, pi*2e4 (radlw_main_nmmb.f:1287)
  lhlw0, lhlwb  bool, the optional clear-sky and per-band outputs (radlw_main_nmmb.f:621-622)

Table constants: NTBL = 10000 is module_radlw_parameters NTBL (radlw_param.f:96), the table size that
radlw_main_nmmb.f uses. lw_tables() builds exp_tbl, tfn_tbl and tau_tbl as rlwinit does
(radlw_main_nmmb.f:1309-1335), vectorised, in single precision.

run(None) is the done-rule entry: a random, plausible state at the full 954 x 835 x 60 grid
(nlay = 61), every column computed, every output finite. The full state does not fit beside the
other tenants on the shared card, so run(None) walks the grid in row slabs; each slab is one launch
of the same kernel and is independent, so the result is the full-domain result.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_rtrnmr"
KERNEL = "phys_rrtm_lw_rtrnmr"
NBANDS = 16
NGPTLW = 140
NTBL = 10000        # module_radlw_parameters NTBL, radlw_param.f:96
MAXL = 64           # kernel LW_MAXL: largest nlay the kernel handles
LTP = 1             # extra top layer (grrad_nmmb.f:210), nlay = LM + LTP
ILWRATE = 2         # radlw_param_nmmb.f:67, lw heating rate in k/second
CON_G = np.float32(9.80665)      # physcons.f:46
CON_CP = np.float32(1.0046e3)    # physcons.f:59
ROW_SLAB = 64       # run(None) row-slab height

# radlw_param.f:120-122 NGB(:) data statement: band index of each g-point (1-based), NG01 = 10.
NGB_FORTRAN = np.array(
    [1] * 10 + [2] * 12 + [3] * 16 + [4] * 14 + [5] * 16 + [6] * 8 + [7] * 12 + [8] * 8
    + [9] * 12 + [10] * 6 + [11] * 8 + [12] * 8 + [13] * 4 + [14] * 2 + [15] * 2 + [16] * 2,
    dtype=np.int32)


def _band_table_matches_fortran() -> None:
    """The kernel's LW_NS / LW_NG constants must reproduce NGB_FORTRAN exactly."""
    ns = [0, 10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138]   # kernel LW_NS
    ng = [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2]                 # kernel LW_NG
    ngb = np.zeros(NGPTLW, dtype=np.int32)
    for ib in range(NBANDS):
        ngb[ns[ib]:ns[ib] + ng[ib]] = ib + 1
    if ngb.shape != NGB_FORTRAN.shape or not np.array_equal(ngb, NGB_FORTRAN):
        raise AssertionError("phys-rrtm-lw-rtrnmr: kernel band table does not match radlw_param.f NGB")


_band_table_matches_fortran()


def heatfac_value() -> np.float32:
    """heatfac = con_g * 1.0e-2 / con_cp, evaluated left to right in single precision."""
    return np.float32(CON_G * np.float32(1.0e-2) / CON_CP)


def fluxfac_value() -> np.float32:
    """fluxfac = pival * 2.0d4, pival = 2*asin(1) (radlw_main_nmmb.f:1286-1287)."""
    pival = np.float32(2.0 * np.arcsin(1.0))
    return np.float32(float(pival) * 2.0e4)


def lw_tables(ntbl: int = NTBL):
    """exp_tbl, tfn_tbl, tau_tbl as rlwinit builds them (radlw_main_nmmb.f:1309-1335).

    Returns float32 numpy arrays of length ntbl+1. Index 0 and ntbl are the Fortran end values.
    """
    f32 = np.float32
    bpade = f32(f32(1.0) / f32(0.278))
    expeps = f32(1.0e-20)
    explimit = f32(np.trunc(-np.log(np.float32(np.finfo(np.float32).tiny))))  # aint(-log(tiny))
    i = np.arange(1, ntbl, dtype=np.int64)
    # tfn = real(i)/real(ntbl-i), a single-precision division as the Fortran does it
    tfn = (i.astype(f32) / (np.int64(ntbl) - i).astype(f32)).astype(f32)
    tau = (bpade * tfn).astype(f32)                          # tau_tbl(i) = bpade*tfn
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.where(tau >= explimit, expeps, np.exp(-tau.astype(np.float64))).astype(f32)
        tfn_t = np.where(tau < f32(0.06), tau / f32(6.0),
                         f32(1.0) - f32(2.0) * ((f32(1.0) / tau) - (e / (f32(1.0) - e)))).astype(f32)
    exp_tbl = np.empty(ntbl + 1, dtype=f32)
    tau_tbl = np.empty(ntbl + 1, dtype=f32)
    tfn_tbl = np.empty(ntbl + 1, dtype=f32)
    exp_tbl[0], tau_tbl[0], tfn_tbl[0] = f32(1.0), f32(0.0), f32(0.0)
    exp_tbl[1:ntbl], tau_tbl[1:ntbl], tfn_tbl[1:ntbl] = e, tau, tfn_t
    exp_tbl[ntbl], tau_tbl[ntbl], tfn_tbl[ntbl] = expeps, f32(1.0e10), f32(1.0)
    return {"exp_tbl": exp_tbl, "tfn_tbl": tfn_tbl, "tau_tbl": tau_tbl}


def _check(name, arr, shape):
    import cupy as cp
    if tuple(arr.shape) != tuple(shape) or arr.dtype != cp.float32:
        raise ValueError(f"phys-rrtm-lw-rtrnmr: {name} must be float32 {tuple(shape)}, "
                         f"got {arr.dtype} {tuple(arr.shape)}")


def rtrnmr(*, semiss, secdif, delp, cldfrc, taucld, tautot, fracs, pklay, pklev,
           tables, out, nlay: int, heatfac: float, fluxfac: float,
           lhlw0: bool, lhlwb: bool, stream=None, surface_first: bool = False):
    """One launch over every column. Writes the outputs dict `out` in place and returns it.
    All arrays are in NAM slot order (see the module docstring), or with surface_first=True in the radlw
    internal order the LW seam holds (slot 0 = the surface layer, every layer array and output; pklay/pklev
    [NLAY+1, NBANDS, JM, IM] as setcoef writes them): kernel phys_rrtm_lw_rtrnmr_sf, the same body with the
    other slot map, so the values are the same and the seam makes no reversed copies."""
    import cupy as cp

    jm, im = int(semiss.shape[1]), int(semiss.shape[2])
    nlay = int(nlay)
    if nlay > MAXL:
        raise ValueError(f"phys-rrtm-lw-rtrnmr: nlay {nlay} exceeds the kernel limit {MAXL}")
    nlp1 = nlay + 1
    _check("semiss", semiss, (NBANDS, jm, im))
    _check("secdif", secdif, (NBANDS, jm, im))
    _check("delp", delp, (nlay, jm, im))
    _check("cldfrc", cldfrc, (nlay + 2, jm, im))
    _check("taucld", taucld, (NBANDS, nlay, jm, im))
    _check("tautot", tautot, (NGPTLW, nlay, jm, im))
    _check("fracs", fracs, (NGPTLW, nlay, jm, im))
    pk_shape = (nlp1, NBANDS, jm, im) if surface_first else (NBANDS, nlp1, jm, im)
    _check("pklay", pklay, pk_shape)
    _check("pklev", pklev, pk_shape)
    for name in ("totuflux", "totdflux", "totuclfl", "totdclfl"):
        _check(name, out[name], (nlp1, jm, im))
    _check("htr", out["htr"], (nlay, jm, im))
    _check("htrcl", out["htrcl"], (nlay, jm, im))
    _check("htrb", out["htrb"], (NBANDS, nlay, jm, im))
    for k in ("exp_tbl", "tfn_tbl", "tau_tbl"):
        if tuple(tables[k].shape) != (NTBL + 1,) or tables[k].dtype != cp.float32:
            raise ValueError(f"phys-rrtm-lw-rtrnmr: tables[{k!r}] must be float32 [{NTBL + 1}]")

    kern = kernels.get(UNIT, KERNEL + "_sf" if surface_first else KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(nlay),
            np.int32(1 if lhlw0 else 0), np.int32(1 if lhlwb else 0),
            np.float32(heatfac), np.float32(fluxfac),
            tables["exp_tbl"], tables["tfn_tbl"], tables["tau_tbl"],
            semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs, secdif,
            out["totuflux"], out["totdflux"], out["htr"],
            out["totuclfl"], out["totdclfl"], out["htrcl"], out["htrb"],
        ))
    return out


def _device_tables(ctx=None):
    import cupy as cp
    host = lw_tables(NTBL)
    if ctx is not None and getattr(ctx, "tables", None) is not None:
        dev = ctx.tables.get("rrtm_lw_tables")
        if dev is None:
            dev = {k: cp.asarray(v) for k, v in host.items()}
            ctx.tables["rrtm_lw_tables"] = dev
        return dev
    return {k: cp.asarray(v) for k, v in host.items()}


def run(ctx=None):
    """Done-rule entry. With no ctx: the full 954 x 835 grid, nlay = 61, random plausible state,
    walked in row slabs; returns the summary dict. With ctx, the inputs are read from ctx.st under
    the Fortran names (SEMISS, SECDIF, DELP, CLDFRC, TAUCLD, TAUTOT, FRACS, PKLAY, PKLEV) in NAM slot
    order, and the outputs are added to ctx.st (TOTUFLUX, TOTDFLUX, HTR, TOTUCLFL, TOTDCLFL, HTRCL, HTRB)."""
    import cupy as cp

    heatfac = float(heatfac_value())
    fluxfac = float(fluxfac_value())
    if ctx is None:
        im, jm, lm = 954, 835, 60
        nlay = lm + LTP
        rng = cp.random
        tables = _device_tables(None)
        finite = {}
        for j0 in range(0, jm, ROW_SLAB):
            jn = min(ROW_SLAB, jm - j0)
            semiss = (0.9 + 0.1 * rng.random((NBANDS, jn, im), dtype=cp.float32))
            secdif = (1.66 + 0.02 * rng.random((NBANDS, jn, im), dtype=cp.float32))
            delp = (50.0 + 400.0 * rng.random((nlay, jn, im), dtype=cp.float32))
            cldfrc = rng.random((nlay + 2, jn, im), dtype=cp.float32)
            taucld = 2.0 * rng.random((NBANDS, nlay, jn, im), dtype=cp.float32)
            tautot = 3.0 * rng.random((NGPTLW, nlay, jn, im), dtype=cp.float32)
            fracs = rng.random((NGPTLW, nlay, jn, im), dtype=cp.float32)
            pklay = 150.0 + 150.0 * rng.random((NBANDS, nlay + 1, jn, im), dtype=cp.float32)
            pklev = 150.0 + 150.0 * rng.random((NBANDS, nlay + 1, jn, im), dtype=cp.float32)
            out = {
                "totuflux": cp.empty((nlay + 1, jn, im), cp.float32),
                "totdflux": cp.empty((nlay + 1, jn, im), cp.float32),
                "totuclfl": cp.empty((nlay + 1, jn, im), cp.float32),
                "totdclfl": cp.empty((nlay + 1, jn, im), cp.float32),
                "htr": cp.empty((nlay, jn, im), cp.float32),
                "htrcl": cp.empty((nlay, jn, im), cp.float32),
                "htrb": cp.empty((NBANDS, nlay, jn, im), cp.float32),
            }
            rtrnmr(semiss=semiss, secdif=secdif, delp=delp, cldfrc=cldfrc, taucld=taucld,
                   tautot=tautot, fracs=fracs, pklay=pklay, pklev=pklev, tables=tables,
                   out=out, nlay=nlay, heatfac=heatfac, fluxfac=fluxfac, lhlw0=True, lhlwb=True)
            for name, arr in out.items():
                ok = bool(cp.isfinite(arr).all())
                finite[name] = finite.get(name, True) and ok
            del semiss, secdif, delp, cldfrc, taucld, tautot, fracs, pklay, pklev, out
        return {"finite": finite, "shape": (im, jm, nlay)}

    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    nlay = lm + LTP
    tables = _device_tables(ctx)
    out = {
        "totuflux": st.add("TOTUFLUX", (nlay + 1, jm, im)),
        "totdflux": st.add("TOTDFLUX", (nlay + 1, jm, im)),
        "totuclfl": st.add("TOTUCLFL", (nlay + 1, jm, im)),
        "totdclfl": st.add("TOTDCLFL", (nlay + 1, jm, im)),
        "htr": st.add("HTR", (nlay, jm, im)),
        "htrcl": st.add("HTRCL", (nlay, jm, im)),
        "htrb": st.add("HTRB", (NBANDS, nlay, jm, im)),
    }
    rtrnmr(semiss=st.fields["SEMISS"], secdif=st.fields["SECDIF"], delp=st.fields["DELP"],
           cldfrc=st.fields["CLDFRC"], taucld=st.fields["TAUCLD"], tautot=st.fields["TAUTOT"],
           fracs=st.fields["FRACS"], pklay=st.fields["PKLAY"], pklev=st.fields["PKLEV"],
           tables=tables, out=out, nlay=nlay, heatfac=heatfac, fluxfac=fluxfac,
           lhlw0=True, lhlwb=True, stream=ctx.stream)
    return out
