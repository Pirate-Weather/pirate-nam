# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol23: band 23 (8050-12850 cm-1) gas and Rayleigh optical depths, NMMB RRTM shortwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:4206-4272 (SUBROUTINE taumol23, internal
to taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 package notice is kept in the kernel file
(namgpu/kernels/phys_rrtm_sw_taumol23.cu). Kernel: rrtm_sw_taumol23, one thread per column, one launch per
column chunk. The k-table is module_radsw_kgb23_nmmb (phys/radsw_datatb_nmmb.f:15195-15436), extracted by
tools/extract_rrtm_sw_kgb23.py to namgpu/data/rrtm_sw/kgb23.npz and uploaded once by load_tables().

What it computes, per column and layer k (Fortran k = 1..nlay):
  taur[j, k]  = colmol(k) * rayl(j)                                  for every layer and the 10 g-intervals
  taug[j, k]  = colamt_h2o(k) * ( givfac * (f00 absa(ind01) + f10 absa(ind02) + f01 absa(ind11) + f11 absa(ind12))
              + selffac * (selfref(inds) + selffrac (selfref(inds+1) - selfref(inds)))
              + forfac  * (forref(indf)  + forfrac  (forref(indf+1)  - forref(indf))) )
              for k <= laytrop, zero above (the Fortran second WHERE loop; see the kernel notes for the
              MINVAL(laytrop) detail at k == laytrop).
Band 23 g-points are 56..65 of the 112-point spectrum (NS23 = 56).

Inputs (device, CONTRACT layout; a dict in ctx.tables[IN_KEY] built by the SW driver phys-rrtm-sw-swrad):
  colamt  [MAXGAS=6, nlay, JM, IM] float32 (plane 0 = h2o; other planes are not read here)
  colmol, fac00, fac01, fac10, fac11, forfac, forfrac, selffac, selffrac    [nlay, JM, IM] float32
  jp, jt, jt1, indfor, indself                                             [nlay, JM, IM] int32
  laytrop [JM, IM] int32
Outputs: the consumer sink, called once per column chunk with (col0, ncol, taug, taur), each
[NG23=10, nlay, ncol] float32 for the chunk of flat columns col0 .. col0+ncol-1 (c = j*IM + i).
Chunking is needed for memory: full-domain taug is about 21 GB.

Consumers: the SW taumol dispatcher and the swflux/spcvrtm path (taug and taur summed into the band-23
optical depth), which the driver supplies as the sink.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol23"
KERNEL = "rrtm_sw_taumol23"
IN_KEY = "rrtm_sw_taumol23_in"
TAB_KEY = "rrtm_sw_taumol23_tables"

NG23 = 10          # NG23 (radsw_param_nmmb.f NG table, band 23)
NGPT = 112         # total g-points
NS23 = 56          # first 0-based g-point of band 23 (NS table, band 23)
MSA23, MSF23, MFR23 = 65, 10, 3
MAXGAS = 6
COLS_PER_CHUNK = 8192
CHK = 8            # CHNK_RRTM (configure.nems.*, -DCHNK_RRTM=8): the Fortran laytrop MINVAL tile
KTABLE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb23.npz"


def load_tables():
    """Upload the band-23 k-table once. Flat, Fortran-natural C order: absa[ind*NG23 + j], etc."""
    import cupy as cp

    d = np.load(KTABLE)
    absa, forref, selfref, rayl = d["absa"], d["forref"], d["selfref"], d["rayl"]
    assert absa.shape == (MSA23, NG23), absa.shape
    assert forref.shape == (MFR23, NG23), forref.shape
    assert selfref.shape == (MSF23, NG23), selfref.shape
    assert rayl.shape == (NG23,), rayl.shape
    return {
        "absa": cp.asarray(absa, dtype=cp.float32).ravel(),
        "forref": cp.asarray(forref, dtype=cp.float32).ravel(),
        "selfref": cp.asarray(selfref, dtype=cp.float32).ravel(),
        "rayl": cp.asarray(rayl, dtype=cp.float32).ravel(),
    }


def _check_inputs(inp: dict):
    nlay, jm, im = inp["colmol"].shape
    for name in ("fac00", "fac01", "fac10", "fac11", "forfac", "forfrac", "selffac", "selffrac"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taumol23: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    for name in ("jp", "jt", "jt1", "indfor", "indself"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"taumol23: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["colamt"].shape[1:] != (nlay, jm, im) or inp["colamt"].shape[0] < 1:
        raise ValueError(f"taumol23: colamt shape {inp['colamt'].shape} must be [MAXGAS, nlay, JM, IM]")
    if inp["laytrop"].shape != (jm, im):
        raise ValueError(f"taumol23: laytrop shape {inp['laytrop'].shape} != {(jm, im)}")
    return nlay, jm, im


def taumol23_chunk(inp: dict, tab: dict, kmin, out: dict, col0: int, ncol: int,
                   nlay: int, jm: int, im: int, tiles_per_row: int, stream=None):
    """One launch over the flat columns col0 .. col0+ncol-1. out['taug'], out['taur'] are
    [NG23, nlay, ncol] device float32 buffers; kmin is the [JM * tiles_per_row] int32 device array of
    MINVAL(laytrop) per CHK-column tile (Fortran MINVAL(laytrop(1:CHK)) per caller tile)."""
    kern = kernels.get(UNIT, KERNEL)
    grid = ((ncol + kernels.BLOCK_COL[0] - 1) // kernels.BLOCK_COL[0], 1, 1)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, (
            inp["colamt"], inp["colmol"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["jp"], inp["jt"], inp["jt1"], inp["laytrop"], kmin,
            inp["forfac"], inp["forfrac"], inp["indfor"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            tab["absa"], tab["forref"], tab["selfref"], tab["rayl"],
            out["taug"], out["taur"],
            np.int32(nlay), np.int32(im), np.int32(jm), np.int32(MAXGAS),
            np.int32(col0), np.int32(ncol), np.int32(tiles_per_row),
        ))


def taumol23(inp: dict, tab: dict, sink, stream=None, chunk: int = COLS_PER_CHUNK):
    """Every column of the domain, in chunks. sink(col0, ncol, taug, taur) consumes each chunk
    (enqueued work only; no host sync here). Returns the number of chunks launched."""
    import cupy as cp

    nlay, jm, im = _check_inputs(inp)
    ncol_dom = jm * im
    # Fortran MINVAL(laytrop(1:CHK)) is per caller tile: one row, tiles of CHK consecutive i from i = 0.
    # Padded entries take a large sentinel (the Fortran pad repeats an existing tile value). Device only.
    tiles_per_row = -(-im // CHK)
    lt = cp.pad(inp["laytrop"], ((0, 0), (0, tiles_per_row * CHK - im)),
                constant_values=int(np.iinfo(np.int32).max))
    kmin = lt.reshape(jm, tiles_per_row, CHK).min(axis=2).astype(cp.int32).ravel()
    bufs = {}
    nchunk = 0
    for col0 in range(0, ncol_dom, chunk):
        ncol = min(chunk, ncol_dom - col0)
        if ncol not in bufs:
            bufs[ncol] = {
                "taug": cp.zeros((NG23, nlay, ncol), dtype=cp.float32),
                "taur": cp.zeros((NG23, nlay, ncol), dtype=cp.float32),
            }
        out = bufs[ncol]
        taumol23_chunk(inp, tab, kmin, out, col0, ncol, nlay, jm, im, tiles_per_row, stream=stream)
        sink(col0, ncol, out["taug"], out["taur"])
        nchunk += 1
    return nchunk


def run(ctx):
    """Driver entry. The SW driver fills ctx.tables[IN_KEY] (the inputs above) and ctx.tables["rrtm_sw_taumol23_sink"]
    (the consumer). The k-table is uploaded once into ctx.tables[TAB_KEY]."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_sw_taumol23 needs ctx.tables[{IN_KEY!r}] from the SW driver (phys-rrtm-sw-swrad)")
    sink = ctx.tables.get("rrtm_sw_taumol23_sink")
    if sink is None:
        raise RuntimeError("phys_rrtm_sw_taumol23 needs ctx.tables['rrtm_sw_taumol23_sink'] from the SW driver")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    return taumol23(inp, tab, sink, stream=ctx.stream)


def tile_check(seed: int = 2):
    """Defect check for the laytrop tile minimum (review gap 1). k == laytrop keeps taug only where laytrop equals
    the MINVAL of its CHK-column tile. Small shape; every absorber factor and table entry is positive, so taug > 0
    exactly where the kernel computes. Returns the number of mismatching (layer, column) entries (must be 0)."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    nlay, jm, im = 12, 3, 21
    lt = rng.integers(3, 10, size=(jm, im)).astype(np.int32)
    shp = (nlay, jm, im)

    def one():
        return cp.ones(shp, dtype=cp.float32)

    def ones_i():
        return cp.ones(shp, dtype=cp.int32)

    inp = {
        "colamt": cp.ones((MAXGAS,) + shp, dtype=cp.float32), "colmol": one(),
        "fac00": one(), "fac01": one(), "fac10": one(), "fac11": one(),
        "jp": ones_i(), "jt": ones_i(), "jt1": ones_i(),
        "laytrop": cp.asarray(lt),
        "forfac": one(), "forfrac": one(), "indfor": ones_i(),
        "selffac": one(), "selffrac": one(), "indself": ones_i(),
    }
    tab = {"absa": cp.ones(MSA23 * NG23, dtype=cp.float32), "forref": cp.ones(MFR23 * NG23, dtype=cp.float32),
           "selfref": cp.ones(MSF23 * NG23, dtype=cp.float32), "rayl": cp.ones(NG23, dtype=cp.float32)}
    got = {}

    def sink(col0, ncol, taug, taur):
        got["nz"] = cp.asnumpy(taug[0] != 0)          # [nlay, ncol], band g-interval 0

    taumol23(inp, tab, sink, chunk=jm * im)
    # reference: Fortran rule per caller tile (row j, i in [8m, 8m+7])
    exp = np.zeros((nlay, jm * im), dtype=bool)
    for j in range(jm):
        for i in range(im):
            m0 = (i // CHK) * CHK
            tmin = int(lt[j, m0:min(m0 + CHK, im)].min())
            ltc = int(lt[j, i])
            for k in range(1, nlay + 1):
                exp[k - 1, j * im + i] = (k <= ltc) and not ((k >= tmin + 1) and (k >= ltc))
    return int(np.count_nonzero(got["nz"] != exp))


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """Done-rule run: one pass over the full 954 x 835 x 60 domain on random but index-valid inputs
    (jp 1..58, jt and jt1 1..4, indself 1..9, indfor 1..2, laytrop 15..45). Asserts no NaN/Inf in
    every band-23 output chunk. The k-table is the real extracted table."""
    import cupy as cp

    rng = cp.random.default_rng(seed)
    f32 = cp.float32

    def u(shape, lo, hi):
        return (lo + (hi - lo) * rng.random(shape, dtype=f32)).astype(f32)

    def ri(shape, lo, hi):
        return rng.integers(lo, hi + 1, size=shape, dtype=cp.int32)

    shape3 = (nlay, jm, im)
    inp = {
        "colamt": u((MAXGAS,) + shape3, 0.0, 1.0e22),
        "colmol": u(shape3, 1.0e20, 1.0e25),
        "fac00": u(shape3, 0.0, 1.0), "fac01": u(shape3, 0.0, 1.0),
        "fac10": u(shape3, 0.0, 1.0), "fac11": u(shape3, 0.0, 1.0),
        "jp": ri(shape3, 1, 58), "jt": ri(shape3, 1, 4), "jt1": ri(shape3, 1, 4),
        "laytrop": ri((jm, im), 15, 45),
        "forfac": u(shape3, 0.0, 1.0e-2), "forfrac": u(shape3, 0.0, 1.0),
        "indfor": ri(shape3, 1, 2),
        "selffac": u(shape3, 0.0, 1.0e-2), "selffrac": u(shape3, 0.0, 1.0),
        "indself": ri(shape3, 1, 9),
    }
    tab = load_tables()
    bad = {"n": 0, "chunks": 0, "maxabs_taug": 0.0, "maxabs_taur": 0.0}

    def sink(col0, ncol, taug, taur):
        bad["chunks"] += 1
        for name, arr in (("taug", taug), ("taur", taur)):
            if not bool(cp.isfinite(arr).all()):
                bad["n"] += 1
        bad["maxabs_taug"] = max(bad["maxabs_taug"], float(cp.abs(taug).max()))
        bad["maxabs_taur"] = max(bad["maxabs_taur"], float(cp.abs(taur).max()))

    nchunk = taumol23(inp, tab, sink)
    cp.cuda.Stream.null.synchronize()
    assert nchunk == bad["chunks"] and bad["n"] == 0, bad
    return bad
