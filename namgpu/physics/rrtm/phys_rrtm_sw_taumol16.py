# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol16: TAUMOL16, SW band 16 (2600-3250 cm-1) optical depths and Rayleigh scattering.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3384-3490 (internal subroutine
taumol16 of taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header of that file is kept as the notice in
the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol16.cu (rrtm_sw_taumol16, one thread per column, one launch).

What it computes, per column and layer k (1-based Fortran k): taur(g, k) = colmol(k) * rayl for the six
band-16 g-points; for k <= laytrop the lower-atmosphere band-16 optical depth from the H2O/CH4 binary
species interpolation of absa, plus the self and foreign continuum (selfref, forref); for k >= laytrop the
upper-atmosphere form CH4 * absb. Where both blocks reach the same layer (k == laytrop), the upper form is
the value kept, as in the Fortran when laytrop exceeds its chunk minimum. That overlap is a declared
divergence: see the kernel header. No domain-wide minimum is used.

Two entry points, one kernel:
  taumol16(inp, out, tab)       full domain, [nlay, JM, IM] inputs, outputs taug/taur [6, nlay, JM, IM].
                                Planes 0..5 only: this unit writes nothing else, so the allocation is
                                6 planes (about 1.1 GB each at 954 x 835 x 60), not the 112-plane driver array.
  run_chunk(ctx, ch, cstart, nc) driver seam (phys_rrtm_sw_swrad._seam). ch is the chunk scratch
                                 ([nc, nlay] inputs, taug/taur [nc, nlay, NGPTSW]); chunk column t is day
                                 column cstart + t; live columns are cstart + t < SW_NDAY (device). Writes
                                 planes 0..5 of ch taug and taur. This is the path the driver runs.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol16"
KERNEL = "rrtm_sw_taumol16"
NG16 = 6
NS16 = 0
NGPTSW = 112
MAXGAS = 6
KGB16_NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb16.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
COLAMT_KEY = "rrtm_sw_colamt"
COLMOL_KEY = "rrtm_sw_colmol"
TAB_KEY = "rrtm_sw_taumol16_tables"
OUT_KEY = "rrtm_sw_taumol16_out"
SW_NDAY_KEY = "SW_NDAY"


def load_tables():
    """kgb16.npz -> device float32 tables plus the rayl and strrat(16) scalars. Uploaded once."""
    import cupy as cp

    d = np.load(KGB16_NPZ)
    tab = {n: cp.asarray(np.ascontiguousarray(d[n], dtype=np.float32))
           for n in ("absa", "absb", "selfref", "forref")}
    if tab["absa"].shape != (NG16, 585) or tab["absb"].shape != (NG16, 235) \
            or tab["selfref"].shape != (NG16, 10) or tab["forref"].shape != (NG16, 3):
        raise ValueError("taumol16: kgb16.npz shapes do not match module_radsw_kgb16_nmmb")
    tab["rayl"] = np.float32(d["rayl"])
    tab["strrat16"] = np.float32(d["strrat16"])
    return tab


def alloc_outputs(nlay: int, jm: int, im: int, nplanes: int = NG16):
    """Full-domain taug and taur. nplanes defaults to NG16 = 6: this unit writes planes 0..5 only, and the
    kernel indexes from plane 0. The 112-plane driver array (21.4 GB each at 954 x 835 x 60) is not
    allocated here; the driver path is run_chunk, whose chunk scratch is [nc, nlay, NGPTSW]."""
    import cupy as cp

    return {
        "taug": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((nplanes, nlay, jm, im), dtype=cp.float32),
    }


def _launch(inp: dict, out: dict, tab: dict, *, ncols: int, nlay: int, lc: int, lk: int,
            ca: tuple, tg: tuple, cstart: int, ndayp, nday_host: int, stream=None):
    """One launch over ncols columns. ca = (ca_c, ca_k, ca_m) for colamt, tg = (tg_c, tg_k, tg_g) for
    taug and taur. ndayp is a device int32 array gating the live columns (chunk path); when None the
    first nday_host columns of this launch are live (full domain)."""
    import cupy as cp

    kern = kernels.get(UNIT, KERNEL)
    gate = ndayp is not None
    dummy = inp["laytrop"]          # valid device pointer; read only when gate is set
    args = (
        inp["colamt"], inp["colmol"],
        inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
        inp["laytrop"],
        inp["forfac"], inp["forfrac"], inp["indfor"],
        inp["selffac"], inp["selffrac"], inp["indself"],
        inp["jp"], inp["jt"], inp["jt1"],
        tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
        tab["rayl"], tab["strrat16"],
        ndayp if gate else dummy, np.int32(1 if gate else 0), np.int32(nday_host), np.int32(cstart),
        np.int32(ncols), np.int32(nlay),
        np.int64(lc), np.int64(lk),
        np.int64(ca[0]), np.int64(ca[1]), np.int64(ca[2]),
        np.int64(tg[0]), np.int64(tg[1]), np.int64(tg[2]),
        out["taug"], out["taur"],
    )
    block = kernels.BLOCK_COL
    grid = ((ncols + block[0] - 1) // block[0], 1, 1)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
    return out


def taumol16(inp: dict, out: dict, tab: dict, stream=None):
    """Full domain: one launch over every (i, j) column, all columns live. inp holds device arrays in
    [nlay, JM, IM] (colamt [6, nlay, JM, IM]) and laytrop [JM, IM]; out holds taug and taur
    [NG16, nlay, JM, IM]."""
    nlay, jm, im = inp["colmol"].shape
    jmim = jm * im
    return _launch(inp, out, tab, ncols=jmim, nlay=nlay, lc=1, lk=jmim,
                   ca=(1, jmim, nlay * jmim), tg=(1, jmim, nlay * jmim),
                   cstart=0, ndayp=None, nday_host=jmim, stream=stream)


def run_chunk(ctx, ch: dict, cstart: int, nc: int):
    """Driver seam entry (phys-rrtm-sw-swrad._seam): taumol16 for one chunk of day columns. Reads ch colamt
    [nc, nlay, 6], colmol, fac00..fac11, selffac, selffrac, forfac, forfrac, jp, jt, jt1, indself, indfor
    [nc, nlay] and laytrop [nc] (setcoef outputs, phys-rrtm-sw-setcoef run_chunk). Writes planes 0..5 of
    ch taug and taur [nc, nlay, NGPTSW]. The chunk path does all its columns in one launch; the device
    SW_NDAY gates the live columns, so the last chunk may run past NDAY."""
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    colmol = ch["colmol"]
    if colmol.shape[0] != nc:
        raise ValueError(f"taumol16 chunk: colmol {colmol.shape} does not match nc={nc}")
    nlay = int(colmol.shape[1])
    if ch["taug"].shape != (nc, nlay, NGPTSW) or ch["taur"].shape != (nc, nlay, NGPTSW):
        raise ValueError(f"taumol16 chunk: taug/taur must be [nc, nlay, NGPTSW] = {(nc, nlay, NGPTSW)}")
    ndayp = ctx.st[SW_NDAY_KEY]
    stream = getattr(ctx, "stream", None)
    return _launch(ch, ch, tab, ncols=nc, nlay=nlay, lc=nlay, lk=1,
                   ca=(nlay * MAXGAS, MAXGAS, 1), tg=(nlay * NGPTSW, NGPTSW, 1),
                   cstart=cstart, ndayp=ndayp, nday_host=0, stream=stream)


def run(ctx):
    """Full-domain driver entry, kept for the full-domain check. Reads colamt, colmol (ctx.tables) and the
    setcoef outputs; writes taug and taur (planes 0..5) into ctx.tables[OUT_KEY]. The driver runs
    run_chunk instead."""
    sout = ctx.tables.get(SETCOEF_OUT)
    colamt = ctx.tables.get(COLAMT_KEY)
    colmol = ctx.tables.get(COLMOL_KEY)
    if sout is None or colamt is None or colmol is None:
        raise RuntimeError("phys_rrtm_sw_taumol16 needs setcoef outputs, rrtm_sw_colamt and rrtm_sw_colmol in ctx.tables")
    inp = {
        "colamt": colamt, "colmol": colmol,
        "laytrop": sout["laytrop"],
        "jp": sout["jp"], "jt": sout["jt"], "jt1": sout["jt1"],
        "fac00": sout["fac00"], "fac01": sout["fac01"], "fac10": sout["fac10"], "fac11": sout["fac11"],
        "selffac": sout["selffac"], "selffrac": sout["selffrac"], "indself": sout["indself"],
        "forfac": sout["forfac"], "forfrac": sout["forfrac"], "indfor": sout["indfor"],
    }
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["colmol"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None:
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol16(inp, out, tab, stream=getattr(ctx, "stream", None))
