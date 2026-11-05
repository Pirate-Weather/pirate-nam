# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""lw-core: the scheme calls inside lwrad (cldprop, setcoef, taumol with taugb01..16, rtrnmr), glued.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:924-1054 (the lwrad call sequence)
and 3423-3609 (taumol: the band dispatch and tautot). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); each unit's kernel
keeps the NOAA/AER header as its notice. This module adds two tiny kernels (tautot and the h2ovmr
recovery) as CuPy elementwise ops; everything else is the units' own kernels.

The band units were written one at a time and each takes its inputs in its own layout. This module
holds ONE canonical set, in radlw internal order (index 0 = the SURFACE layer, Fortran k = 1), and
hands every unit the view or copy it expects:
  setcoef in   pavel, tavel, coldry, colbrd, h2ovmr [nlay, JC, IM]; tz [nlp1]; stemp [JC, IM];
               colamt [6, nlay]; wx [4, nlay] (lwrad_prep scratch, surface first)
  setcoef out  laytrop [JC, IM]; jp, jt, jt1, indself, indfor, indminor [nlay]; fac00..fac11,
               selffac, selffrac, forfac, forfrac, minorfrac, scaleminor, scaleminorn2 [nlay];
               rfrate [2, 6, nlay] (Fortran rfrate(k, rate, side) at [side, rate, k]);
               pklay, pklev [nlp1, 16]
  taumol out   taug, fracs [140, nlay] (g-point major, Fortran g = NSnn + ig at plane g - 1)
               tautot = taug + tauaer(ngb(g)) (radlw_main_nmmb.f taumol, 3597-3606)
rtrnmr runs as phys_rrtm_lw_rtrnmr_sf: the same kernel body as the NAM slot-order kernel, reading and writing
every layer array in this surface-first order and pklay/pklev in setcoef's own [nlp1, 16] layout, so no
reversed or transposed copies are made.

Memory (VRAM audit 2026-11-04): the chunk holds ONE pair of full 140-g-point arrays, taug and fracs. Every
band unit writes its own g-point planes straight into them (the units that allocate full-g outputs of their
own are handed the pair, in their own layout: g-first units get it as is, the layer-first units 2, 4 and 10
get it through their out_gfirst stride argument; the band-local units 6, 13 and 15 get their plane slices).
tautot = taug + tauaer(ngb) is formed in place in taug. Before: 18 such arrays per chunk (each unit's own
pair, the canonical pair, the reversed rtrnmr pair), 1.56 GB at 333 x 8 columns, 4.7 GB at 954 x 8.
Every band writes every element of its planes on every call (checked: NAMGPU_LW_FILLCHECK=1 NaN-fills the
pair before the bands and counts NaN after), so what the pair held before the bands is never read.

Called by phys-rrtm-lw-lwrad.run for one row chunk that covers the whole (chunk) domain:
  seam(ctx, s, nlay, nlp1, jc, im) -> t  (the dict lwrad_post reads)
"""
from __future__ import annotations

import os
import types
from functools import lru_cache

import numpy as np

NBANDS = 16
NGPTLW = 140
NG = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
NS = tuple(int(x) for x in np.concatenate([[0], np.cumsum(NG)[:-1]]))
NGB = np.repeat(np.arange(NBANDS, dtype=np.int32), NG)          # 0-based band of each g-point


class _St:
    """A minimal State stand-in for the units that read ctx.st by name."""

    def __init__(self, xp, grid, fields):
        self.xp = xp
        self.grid = grid
        self.fields = dict(fields)

    def __getitem__(self, k):
        return self.fields[k]

    def __contains__(self, k):
        return k in self.fields

    def add(self, name, shape, dtype="float32"):
        a = self.fields.get(name)
        if a is None or tuple(a.shape) != tuple(shape):
            a = self.xp.zeros(shape, dtype=dtype)
            self.fields[name] = a
        return a


def _proxy(ctx, fields=None, lm=None, tables=None):
    import cupy as cp
    g = types.SimpleNamespace(im=ctx.grid.im, jm=ctx.grid.jm, lm=ctx.grid.lm if lm is None else lm)
    return types.SimpleNamespace(grid=g, st=_St(cp, g, fields or {}), stream=ctx.stream,
                                 tables=ctx.tables if tables is None else tables, cfg=ctx.cfg)


@lru_cache(maxsize=None)
def _tautot_kernel():
    """taug[g, k, c] += tauaer[ngb[g], k, c], in place (radlw_main_nmmb.f 3597-3606 tautot): the same one float
    add the out-of-place taug + tauaer[ngb] made, with no [140, nlay, ...] temporaries."""
    import cupy as cp
    return cp.ElementwiseKernel(
        "raw float32 tauaer, raw int32 ngb, int64 lp", "float32 taug",
        "taug = taug + tauaer[(long long)ngb[i / lp] * lp + i % lp]", "lw_core_tautot_inplace")


def _scr(ctx, name, shape, dtype="float32"):
    import cupy as cp
    key = "lw_core:" + name
    a = ctx.tables.get(key)
    if a is None or tuple(a.shape) != tuple(shape):
        a = cp.zeros(shape, dtype=dtype)
        ctx.tables[key] = a
    return a


def _bands(ctx, sin, sc, wx, taug, fracs, nlay, jc, im):
    """taumol: every band unit, its outputs scattered into taug/fracs [140, nlay, JC, IM]."""
    import cupy as cp
    from namgpu.physics.rrtm import (phys_rrtm_lw_taugb01 as b01, phys_rrtm_lw_taugb02 as b02,
                                     phys_rrtm_lw_taugb03 as b03, phys_rrtm_lw_taugb04 as b04,
                                     phys_rrtm_lw_taugb05 as b05, phys_rrtm_lw_taugb06 as b06,
                                     phys_rrtm_lw_taugb07 as b07, phys_rrtm_lw_taugb08 as b08,
                                     phys_rrtm_lw_taugb09 as b09, phys_rrtm_lw_taugb10 as b10,
                                     phys_rrtm_lw_taugb11 as b11, phys_rrtm_lw_taugb12 as b12,
                                     phys_rrtm_lw_taugb13 as b13, phys_rrtm_lw_taugb14 as b14,
                                     phys_rrtm_lw_taugb15 as b15, phys_rrtm_lw_taugb16 as b16)
    ncol = jc * im
    colamt6 = sin["colamt"]
    lay = {k: sc[k] for k in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "fac00", "fac01",
                              "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac",
                              "minorfrac", "scaleminor", "scaleminorn2")}
    union = dict(lay, laytrop=sc["laytrop"], rfrate=sc["rfrate"], colamt=colamt6, wx=wx,
                 coldry=sin["coldry"], colbrd=sin["colbrd"], pavel=sin["pavel"])

    def put(b, tg, fr):          # tg, fr: [NG, nlay, JC, IM] band-local
        s0 = NS[b - 1]
        taug[s0:s0 + NG[b - 1]] = tg
        fracs[s0:s0 + NG[b - 1]] = fr

    def planes(b):                   # band b's g-point planes of the pair: contiguous [NG, nlay, JC, IM] views
        s0 = NS[b - 1]
        return taug[s0:s0 + NG[b - 1]], fracs[s0:s0 + NG[b - 1]]

    t = ctx.tables
    gc = (NGPTLW, nlay, ncol)        # the pair as the flat-column g-first units see it
    pair = (taug, fracs)
    # The units that write full-g or band-local outputs of their own write into the pair instead (see the
    # module docstring): their output slots in ctx.tables are the pair, or its band planes.
    t["rrtm_lw_taugb08_taug"], t["rrtm_lw_taugb08_fracs"] = taug.reshape(gc), fracs.reshape(gc)
    t["rrtm_lw_taugb09_taug"], t["rrtm_lw_taugb09_fracs"] = taug.reshape(gc), fracs.reshape(gc)
    t[b12.OUT_KEY] = {"taug": taug, "fracs": fracs}
    t[b14.OUT_KEY + "_taug"], t[b14.OUT_KEY + "_fracs"] = taug.reshape(gc), fracs.reshape(gc)
    t[b16.OUT_KEY] = {"taug": taug, "fracs": fracs}
    tg6, fr6 = planes(6)
    t[b06.OUT_KEY] = {"taug6": tg6, "fracs6": fr6}
    tg15, fr15 = planes(15)
    t[b15.OUT_KEY] = {"taug15": tg15, "fracs15": fr15}
    # setcoef-keyed units read these tables entries
    t["rrtm_lw_setcoef_in"] = sin
    t["rrtm_lw_setcoef_out"] = sc
    t["rrtm_lw_taumol_in"] = union
    t["rrtm_lw_taugb10_in"] = union

    # band 1: ctx.st lowercase names, [lm, ...] from pavel's shape
    p = _proxy(ctx, dict(union, colamt=colamt6, laytrop=sc["laytrop"]))
    b01.run(p)
    put(1, p.st["taug_b01"].transpose(1, 0, 2, 3), p.st["fracs_b01"].transpose(1, 0, 2, 3))
    # band 2: ctx.st uppercase names, lm = grid.lm, COLAMT = the H2O plane
    up = {k.upper(): v for k, v in lay.items()}
    up.update(LAYTROP=sc["laytrop"], PAVEL=sin["pavel"], COLAMT=cp.ascontiguousarray(colamt6[0]))
    p = _proxy(ctx, up, lm=nlay)
    b02.init_tables(p)
    b02.run(p, out_gfirst=pair)
    # band 3: ctx.st LW_* names, kernel nlay = grid.lm + 1; rfrate = rate 1, both sides
    lw = {"LW_" + k.upper(): v for k, v in lay.items()}
    lw.update(LW_LAYTROP=sc["laytrop"], LW_COLAMT=colamt6, LW_COLDRY=sin["coldry"],
              LW_RFRATE=cp.ascontiguousarray(sc["rfrate"][:, 0]))
    p = _proxy(ctx, lw, lm=nlay - 1)
    io = b03.run(p)
    put(3, io["LW_TAUG_B3"], io["LW_FRACS_B3"])
    # band 4: tables keyed (setcoef in/out)
    b04.run(ctx, out_gfirst=pair)
    # band 5: ctx.st lowercase, colamt = (h2o, co2, o3), rfrate = (r11, r12, r61, r62), wx_ccl4 = wx(k,1)
    rf = sc["rfrate"]
    p = _proxy(ctx, dict(lay, laytrop=sc["laytrop"], colamt=cp.ascontiguousarray(colamt6[:3]),
                         rfrate=cp.ascontiguousarray(cp.stack([rf[0, 0], rf[1, 0], rf[0, 5], rf[1, 5]])),
                         wx_ccl4=cp.ascontiguousarray(wx[0])), lm=nlay)
    tg, fr = b05.run(p)
    put(5, tg, fr)
    # band 6: tables keyed (taumol_in), into its planes
    b06.run(ctx, stream=ctx.stream)
    # band 7: [ncol, nlay] column-major inputs; rfrate(k, 2, side) as [ncol, nlay, 2]
    def cl(a):
        return cp.ascontiguousarray(a.reshape(nlay, ncol).T)
    inp7 = {k: cl(lay[k]) for k in ("jp", "jt", "jt1", "indself", "indfor", "indminor", "selffac",
                                     "selffrac", "forfac", "forfrac", "minorfrac", "fac00", "fac01",
                                     "fac10", "fac11")}
    inp7.update(colamt_h2o=cl(colamt6[0]), colamt_co2=cl(colamt6[1]), colamt_o3=cl(colamt6[2]),
                coldry=cl(sin["coldry"]),
                rfrate_h2oo3=cp.ascontiguousarray(rf[:, 1].reshape(2, nlay, ncol).transpose(2, 1, 0)),
                laytrop=cp.ascontiguousarray(sc["laytrop"].reshape(ncol)))
    tg7, fr7 = b07.run(ctx, inp7, nlay)
    put(7, tg7.transpose(2, 1, 0).reshape(NG[6], nlay, jc, im),
        fr7.transpose(2, 1, 0).reshape(NG[6], nlay, jc, im))
    # bands 8, 9, 14: input dicts, outputs [140, nlay, ncol] = the pair
    b08.run(ctx, union)
    b09.run(ctx, union)
    # band 10: tables keyed, into the pair (g-first strides)
    b10.run(ctx, out_gfirst=pair)
    # band 11: writes straight into the canonical arrays at row0 = NS11
    t[b11.OUT_KEY] = {"taug": taug, "fracs": fracs, "row0": NS[10]}
    b11.run(ctx)
    # bands 12, 16: outputs [140, nlay, ...] = the pair
    b12.run(ctx)
    # band 13: no driver entry; taugb13(inp, chi_mls, tab, out) with colamt of 7 gases, into its planes
    colamt7 = sin["colamt7"]
    tg13, fr13 = planes(13)
    b13.taugb13(dict(union, colamt=colamt7), t["rrtm_lw_setcoef_tables"]["chi_mls"], b13.run(ctx),
                {"taug13": tg13, "fracs13": fr13}, stream=ctx.stream)
    b14.run(ctx, stream=ctx.stream)
    b15.run(ctx, stream=ctx.stream)
    b16.run(ctx)


def pair_floats(nlay: int, jc: int, im: int) -> int:
    """Words of the taug/fracs pair of one chunk (what seam() carves from the shared workspace)."""
    return 2 * NGPTLW * nlay * jc * im


def seam(ctx, s, nlay: int, nlp1: int, jc: int, im: int, lhlw0=False, lhlwb=False):
    """cldprop, setcoef, taumol, rtrnmr for one chunk whose rows are the whole (chunk) domain."""
    import cupy as cp
    from namgpu.physics.rrtm import phys_rrtm_lw_cldprop as cldprop
    from namgpu.physics.rrtm import phys_rrtm_lw_rtrnmr as rtrnmr
    from namgpu.physics.rrtm import phys_rrtm_lw_setcoef as setcoef

    if int(ctx.grid.jm) != jc:
        raise RuntimeError("lw-core: the chunk must cover the whole (chunk) domain; run RRTM through "
                           "rrtm_call (row chunks of at most phys_rrtm_lw_lwrad.JC_DEFAULT rows)")
    # 1. cldprop (lcf1 columns; zero elsewhere), taucld [16, nlay] surface first
    taucld = cldprop.run(ctx)
    if isinstance(taucld, dict):
        taucld = taucld.get("taucld", ctx.tables.get("rrtm_lw_cldprop_taucld"))
    if taucld is None or not hasattr(taucld, "shape"):
        taucld = ctx.tables["rrtm_lw_cldprop_taucld"]
    # 2. setcoef
    colamt7 = s["lwrad_colamt"].reshape(7, nlay, jc, im)
    wx = s["lwrad_wx"].reshape(4, nlay, jc, im)
    coldry = s["lwrad_coldry"]
    h2ovmr = _scr(ctx, "h2ovmr", (nlay, jc, im))
    cp.divide(colamt7[0], cp.maximum(coldry, cp.float32(1e-30)), out=h2ovmr)
    sin = {"pavel": s["lwrad_pavel"], "tavel": s["lwrad_tavel"], "tz": s["lwrad_tz"],
           "stemp": s["lwrad_stemp"].reshape(jc, im), "h2ovmr": h2ovmr,
           "colamt": cp.ascontiguousarray(colamt7[:6]), "colamt7": colamt7,
           "coldry": coldry, "colbrd": s["lwrad_colbrd"]}
    tab = ctx.tables.get("rrtm_lw_setcoef_tables")
    if tab is None:
        tab = setcoef.load_tables()
        ctx.tables["rrtm_lw_setcoef_tables"] = tab
    ctx.tables["rrtm_lw_setcoef_in"] = sin
    sc = setcoef.run(ctx)
    # 3. taumol: every band into the one pair
    from namgpu.physics.rrtm import rrtm_chunk
    ws = rrtm_chunk.workspace(ctx)
    npair = NGPTLW * nlay * jc * im
    if ws is not None and ws.size >= 2 * npair:      # the chunk's shared g-point workspace (rrtm_chunk.WS_KEY)
        taug = ws[:npair].reshape(NGPTLW, nlay, jc, im)
        fracs = ws[npair:2 * npair].reshape(NGPTLW, nlay, jc, im)
    else:
        taug = _scr(ctx, "taug", (NGPTLW, nlay, jc, im))
        fracs = _scr(ctx, "fracs", (NGPTLW, nlay, jc, im))
    check = os.environ.get("NAMGPU_LW_FILLCHECK", "") == "1"
    if check:                        # proof aid, off the step path: what the pair held must not survive the bands
        taug.fill(cp.nan)
        fracs.fill(cp.nan)
    _bands(ctx, sin, sc, wx, taug, fracs, nlay, jc, im)
    if check:
        bad = int(cp.isnan(taug).sum()) + int(cp.isnan(fracs).sum())
        if bad:
            raise RuntimeError(f"lw-core: {bad} taug/fracs elements not written by any band unit")
    ngb = ctx.tables.get("lw_core:ngb")
    if ngb is None:
        ngb = cp.asarray(NGB)
        ctx.tables["lw_core:ngb"] = ngb
    tauaer = s["lwrad_tauaer"].reshape(NBANDS, nlay, jc, im)
    # 4. tautot = taug + tauaer(ngb(g)) (radlw_main_nmmb.f 3597-3606), in place: taug is not read again
    _tautot_kernel()(tauaer, ngb, np.int64(nlay * jc * im), taug)
    # 5. rtrnmr in this (surface-first) order: no reversed copies in, outputs already in lwrad_post's order
    t = {n: _scr(ctx, "pt_" + n, (nlp1, jc, im)) for n in ("totuflux", "totdflux", "totuclfl", "totdclfl")}
    t["htr"] = _scr(ctx, "pt_htr", (nlay, jc, im))
    t["htrcl"] = _scr(ctx, "pt_htrcl", (nlay, jc, im))
    t["htrb"] = _scr(ctx, "pt_htrb", (NBANDS, nlay, jc, im))
    rtrnmr.rtrnmr(semiss=s["lwrad_semiss"], secdif=s["lwrad_secdiff"], delp=s["lwrad_delp"],
                  cldfrc=s["lwrad_cldfrc"], taucld=cp.ascontiguousarray(taucld.reshape(NBANDS, nlay, jc, im)),
                  tautot=taug, fracs=fracs, pklay=sc["pklay"], pklev=sc["pklev"],
                  tables=rtrnmr._device_tables(ctx), out=t, nlay=nlay,
                  heatfac=float(rtrnmr.heatfac_value()), fluxfac=float(rtrnmr.fluxfac_value()),
                  lhlw0=lhlw0, lhlwb=lhlwb, stream=ctx.stream, surface_first=True)
    return t
