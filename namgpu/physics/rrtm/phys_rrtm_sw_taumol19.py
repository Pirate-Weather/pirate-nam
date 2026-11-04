# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol19: TAUMOL19, NMMB RRTM shortwave band 19 optical depths (4650-5150 cm-1).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3743-3847 (subroutine taumol19).
The sw-rrtm3 header of that file is the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_sw_taumol19.cu, rrtm_sw_taumol19 (one thread per column, one launch
per chunk or tile). The k-table module_radsw_kgb19_nmmb is read from namgpu/data/rrtm_sw/kgb19.npz
(tools/extract_rrtm_sw_kgb19.py), uploaded once and cached.

Layer order: layer index 0 is the LOWEST layer (surface), the radiation-internal order of swrad with
ivflip = 0 (the NAM value). swrad_prep does the flip from NMMB top-first input. laytrop is the count of
lower layers from the surface, as setcoef writes it. See the kernel header for the Fortran references.

Chunk minimum: the Fortran MINVAL(laytrop(1:CHK)) is taken over each group of CHK = 8 consecutive day
columns (CHNK_RRTM = 8). The kernel computes that minimum per group on the device, so the result does
not depend on how the launch is tiled. The chunk path needs cstart and nc to be multiples of 8.

Entries:
  run_chunk(ctx, ch, cstart, nc)  driver seam (phys-rrtm-sw-swrad._seam). ch holds [nc, nlay] and
                                   [nc, nlay, 6] / [nc, nlay, NGPTSW] scratch. SW_NDAY gates the
                                   live columns on the device.
  taumol19(inp, out)              full-tile launch, [nlay, ncol] layout, used by selftest.

Tiles, full-tile layout: colamt [6, NLAY, ncol]; colmol, fac00..fac11, selffac, selffrac, forfac,
forfrac, indself, indfor, jp, jt, jt1 [NLAY, ncol]; laytrop [ncol]. Outputs taug, taur [112, NLAY, ncol].
Only g = 27..34 (band 19, Fortran NS19+j with NS19 = 26) are written.

Tables: absa [585, 8], absb [235, 8], selfref [10, 8], forref [3, 8] float32, uploaded once.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol19"
KERNEL = "rrtm_sw_taumol19"
NGPTSW = 112
NG19 = 8
NS19 = 26
CHK = 8                 # CHNK_RRTM (configure.nems -DCHNK_RRTM=8)
MAXGAS = 6
SW_NDAY_KEY = "SW_NDAY"
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb19.npz"

_TABLES = None


def load_tables():
    """absa [585, 8], absb [235, 8], selfref [10, 8], forref [3, 8] float32, uploaded once."""
    global _TABLES
    if _TABLES is None:
        import cupy as cp

        z = np.load(TABLE_FILE)
        _TABLES = {
            "absa": cp.asarray(np.ascontiguousarray(z["absa"], dtype=np.float32)),
            "absb": cp.asarray(np.ascontiguousarray(z["absb"], dtype=np.float32)),
            "selfref": cp.asarray(np.ascontiguousarray(z["selfref"], dtype=np.float32)),
            "forref": cp.asarray(np.ascontiguousarray(z["forref"], dtype=np.float32)),
        }
        for k, a in _TABLES.items():
            if a.dtype != cp.float32:
                raise ValueError(f"taumol19 table {k} is not float32")
    return _TABLES


def alloc_outputs(nlay: int, ncol: int):
    """Full-tile outputs [112, NLAY, ncol]."""
    import cupy as cp

    return {
        "taug": cp.zeros((NGPTSW, nlay, ncol), dtype=cp.float32),
        "taur": cp.zeros((NGPTSW, nlay, ncol), dtype=cp.float32),
    }


def _launch(inp: dict, out: dict, tab: dict, *, ncols: int, nlay: int, lc: int, lk: int,
            ca: tuple, tg: tuple, cstart: int, ndayp, nday_host: int, stream=None):
    """One launch over ncols columns. ca = (ca_c, ca_k, ca_m) for colamt, tg = (tg_c, tg_k, tg_g) for
    taug and taur, lc/lk the per (column, layer) strides. ndayp is a device int32 array gating the live
    columns (chunk path); when None the first nday_host columns of this launch are live."""
    import cupy as cp

    kern = kernels.get(UNIT, KERNEL)
    gate = ndayp is not None
    dummy = inp["laytrop"]          # valid device pointer; read only when gate is set
    args = (
        inp["colamt"], inp["colmol"],
        inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
        inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"],
        inp["indself"], inp["indfor"], inp["jp"], inp["jt"], inp["jt1"],
        inp["laytrop"],
        tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
        ndayp if gate else dummy, np.int32(1 if gate else 0), np.int32(nday_host), np.int32(cstart),
        np.int32(ncols), np.int32(nlay),
        np.int64(lc), np.int64(lk),
        np.int64(ca[0]), np.int64(ca[1]), np.int64(ca[2]),
        np.int64(tg[0]), np.int64(tg[1]), np.int64(tg[2]),
        out["taug"], out["taur"],
    )
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(ncols, 1), kernels.BLOCK_COL, args)
    return out


def taumol19(inp: dict, out: dict, stream=None):
    """Full-tile launch over [nlay, ncol] arrays (c fastest). Every column is live. The tile starts at
    a multiple of CHK columns, so the Fortran chunk groups fall on tile-local multiples of CHK."""
    nlay, ncol = inp["colmol"].shape
    if inp["colamt"].shape != (6, nlay, ncol):
        raise ValueError(f"taumol19: colamt shape {inp['colamt'].shape} != {(6, nlay, ncol)}")
    if out["taug"].shape != (NGPTSW, nlay, ncol) or out["taur"].shape != (NGPTSW, nlay, ncol):
        raise ValueError("taumol19: taug/taur must be [112, NLAY, ncol]")
    tab = load_tables()
    return _launch(inp, out, tab, ncols=ncol, nlay=nlay, lc=1, lk=ncol,
                   ca=(1, ncol, nlay * ncol), tg=(1, ncol, nlay * ncol),
                   cstart=0, ndayp=None, nday_host=ncol, stream=stream)


def run_chunk(ctx, ch: dict, cstart: int, nc: int):
    """Driver seam entry (phys-rrtm-sw-swrad._seam): taumol19 for one chunk of day columns. Reads ch
    colamt [nc, nlay, 6], colmol, fac00..fac11, selffac, selffrac, forfac, forfrac, jp, jt, jt1,
    indself, indfor [nc, nlay] and laytrop [nc] (setcoef outputs). Writes ch taug and taur
    [nc, nlay, NGPTSW] for planes NS19..NS19+7. The chunk path does all its columns in one launch; the
    device SW_NDAY gates the live columns, so the last chunk may run past NDAY."""
    if cstart % CHK != 0 or nc % CHK != 0:
        raise ValueError(
            f"taumol19 chunk: cstart={cstart} and nc={nc} must be multiples of CHK={CHK}; the Fortran "
            "MINVAL(laytrop) groups are CHK day columns, so another partition gives another minimum")
    tab = load_tables()
    colmol = ch["colmol"]
    if colmol.shape[0] != nc:
        raise ValueError(f"taumol19 chunk: colmol {colmol.shape} does not match nc={nc}")
    nlay = int(colmol.shape[1])
    if ch["taug"].shape != (nc, nlay, NGPTSW) or ch["taur"].shape != (nc, nlay, NGPTSW):
        raise ValueError(f"taumol19 chunk: taug/taur must be [nc, nlay, NGPTSW] = {(nc, nlay, NGPTSW)}")
    ndayp = ctx.st[SW_NDAY_KEY]
    stream = getattr(ctx, "stream", None)
    return _launch(ch, ch, load_tables(), ncols=nc, nlay=nlay, lc=nlay, lk=1,
                   ca=(nlay * MAXGAS, MAXGAS, 1), tg=(nlay * NGPTSW, NGPTSW, 1),
                   cstart=cstart, ndayp=ndayp, nday_host=0, stream=stream)


def _random_tile(nlay: int, ncol: int, rng):
    """Plausible device state for one tile: lower layers 1..laytrop carry the setcoef lower-atmosphere
    indices (indself 1..9, indfor 1..2); upper layers carry indself 0, indfor 3 as setcoef writes."""
    import cupy as cp

    lay = cp.arange(1, nlay + 1, dtype=cp.int32)[:, None]
    laytrop = rng.integers(5, nlay - 5, size=ncol, dtype=np.int32)
    laytrop = cp.asarray(laytrop)
    lower = lay <= laytrop[None, :]

    def f(lo=0.0, hi=1.0):
        return rng.random((nlay, ncol), dtype=np.float32) * np.float32(hi - lo) + np.float32(lo)

    def ints(lo, hi):
        return cp.asarray(rng.integers(lo, hi + 1, size=(nlay, ncol), dtype=np.int32))

    indself = cp.where(lower, ints(1, 9), cp.int32(0)).astype(cp.int32)
    indfor = cp.where(lower, ints(1, 2), cp.int32(3)).astype(cp.int32)
    colamt = cp.asarray(rng.random((6, nlay, ncol), dtype=np.float32) * np.float32(2.0e-2))
    colamt[0] += np.float32(1.0e-4)     # H2O, keeps speccomb > 0
    colamt[1] += np.float32(1.0e-4)     # CO2
    return {
        "colamt": colamt,
        "colmol": cp.asarray(f(1.0e-4, 5.0e4)),
        "fac00": cp.asarray(f()), "fac01": cp.asarray(f()),
        "fac10": cp.asarray(f()), "fac11": cp.asarray(f()),
        "selffac": cp.asarray(f()), "selffrac": cp.asarray(f()),
        "forfac": cp.asarray(f()), "forfrac": cp.asarray(f()),
        "indself": indself, "indfor": indfor,
        "jp": ints(1, 58), "jt": ints(1, 4), "jt1": ints(1, 4),
        "laytrop": laytrop,
    }


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, tile: int = 16384, seed: int = 1):
    """Done-rule run: the full 954 x 835 x 60 state, one launch per tile, output checked for NaN/Inf
    in the band-19 slices the kernel writes. Tiles start at multiples of CHK (16384 is a multiple of 8)."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    ncol_total = im * jm
    out = alloc_outputs(nlay, tile)
    n_written = 0
    for c0 in range(0, ncol_total, tile):
        ncol = min(tile, ncol_total - c0)
        inp = _random_tile(nlay, ncol, rng)
        if ncol != tile:
            out = alloc_outputs(nlay, ncol)
        taumol19(inp, out)
        assert bool(cp.isfinite(out["taug"][NS19:NS19 + NG19]).all()), f"NaN/Inf taug tile at {c0}"
        assert bool(cp.isfinite(out["taur"][NS19:NS19 + NG19]).all()), f"NaN/Inf taur tile at {c0}"
        n_written += ncol
    cp.cuda.Stream.null.synchronize()
    band = out["taug"][NS19:NS19 + NG19]
    assert bool(cp.isfinite(band).all()), "NaN/Inf in taug band 19"
    assert bool(cp.isfinite(out["taur"][NS19:NS19 + NG19]).all()), "NaN/Inf in taur band 19"
    return {"columns": n_written, "taug_mean": float(band.mean()), "taur_mean": float(out["taur"][NS19:NS19 + NG19].mean())}
