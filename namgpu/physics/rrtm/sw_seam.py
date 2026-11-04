"""sw-seam: the scheme calls inside swrad (cldprop, setcoef, taumol, taumol16..29, spcvrtc) on one chunk of
day columns, glued.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:1003-1110 (the swrad call sequence),
3107-3383 (taumol: sfluxzen and the band dispatch) and 3384-4767 (taumol16..29). NOAA US government work;
each unit's kernel keeps the NOAA/AER header as its notice. This module has no kernel of its own: it moves
arrays between the swrad chunk scratch and the layouts the band units were written in.

phys-rrtm-sw-swrad holds ONE canonical chunk scratch ch (phys_rrtm_sw_swrad.chunk_scratch): per day column
t (0..nc-1, column cstart + t of the device day list), radsw internal layer order (index 0 = the LOWEST
layer, CONTRACT section 1 SW exception):
    pavel .. forfrac [nc, nlay]; colamt [nc, nlay, 6]; taug, taur [nc, nlay, 112]; tauae, taucw .. [nc, nlay, 14];
    fxupc .. [nc, nlp1, 14]; laytrop, cosz1, zcf1, ftoauc .. [nc]; albbm, sfbmc .. [nc, 2]; sfluxzen [nc, 112].
The units that already read ch directly (setcoef, taumol16, 19, 20, 22: run_chunk, gated on the device
SW_NDAY) are called as they are. Every other unit takes the CONTRACT layout [.., NLAY, JM, IM]; here JM = 1
and IM = nc, so the per-chunk copies are layer-major [.., nlay, 1, nc] and the outputs are scattered back
to ch planes NSnn + g (Fortran taug(:, k, NSnn + g)).

Columns past the device day count (cstart + t >= SW_NDAY) are not written by swrad_post. The ungated units
still run over them, so their layer-major index inputs are made safe first (jp = jt = jt1 = indself =
indfor = 1, laytrop = nlay, so every index stays inside the k-tables and the per-chunk MINVAL(laytrop) of
taumol23/25/28 is the minimum over live columns only).

Declared divergence (as the band units' own headers state): MINVAL(laytrop) groups. The Fortran takes it
over each CHK = 8 day-column tile; taumol23 does that here, taumol25 and taumol28 take it over the chunk.
Not bitwise; the per-column result differs only in layers between the two minima.
"""
from __future__ import annotations

import numpy as np

NGPTSW = 112
NBDSW = 14
MAXGAS = 6
CHK = 8
# NS16..NS29 (radsw_param_nmmb.f:170-177), 0-based first plane of each band.
NS = {16: 0, 17: 6, 18: 18, 19: 26, 20: 34, 21: 44, 22: 54, 23: 56, 24: 66, 25: 74, 26: 80, 27: 86,
      28: 94, 29: 100}
NG = {16: 6, 17: 12, 18: 8, 19: 8, 20: 10, 21: 10, 22: 2, 23: 10, 24: 8, 25: 6, 26: 6, 27: 8, 28: 6, 29: 12}

_F2 = ("colmol", "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac")
_I2 = ("jp", "jt", "jt1", "indself", "indfor")
KEY_L = "sw_seam:layer_major"


def _mod(name):
    import importlib
    return importlib.import_module(f"namgpu.physics.rrtm.{name}")


def _tab(ctx, key, loader):
    t = ctx.tables.get(key)
    if t is None:
        t = loader()
        ctx.tables[key] = t
    return t


def _live(ctx, cstart, nc):
    """[nc] bool, True for live day columns (device compare against SW_NDAY, no host read)."""
    import cupy as cp
    return (cp.arange(nc, dtype=cp.int32) + np.int32(cstart)) < ctx.st["SW_NDAY"][0]


def layer_major(ctx, ch, cstart, nc):
    """The setcoef outputs and gas columns of ch as [nlay, 1, nc] (colamt [6, nlay, 1, nc], laytrop [1, nc]),
    dead columns made safe. Built once per chunk, after setcoef."""
    import cupy as cp
    nlay = int(ch["pavel"].shape[1])
    L = {}
    live = _live(ctx, cstart, nc)
    for n in _F2:
        L[n] = cp.ascontiguousarray(ch[n].T).reshape(nlay, 1, nc)
    for n in _I2:
        L[n] = cp.ascontiguousarray(cp.where(live[None, :], ch[n].T, cp.int32(1))).reshape(nlay, 1, nc)
    L["colamt"] = cp.ascontiguousarray(ch["colamt"].transpose(2, 1, 0)).reshape(MAXGAS, nlay, 1, nc)
    L["laytrop"] = cp.ascontiguousarray(cp.where(live, ch["laytrop"], cp.int32(nlay))).reshape(1, nc)
    L["nlay"], L["nc"], L["live"] = nlay, nc, live
    ctx.tables[KEY_L] = L
    return L


def _put(ch, band, taug, taur):
    """Scatter band planes taug [ng, nlay, 1, nc] (taur the same, or [nlay, 1, nc] shared by every g)."""
    ns, ng = NS[band], NG[band]
    nlay, nc = ch["taug"].shape[1], ch["taug"].shape[0]
    g = taug.reshape(ng, nlay, nc).transpose(2, 1, 0)
    ch["taug"][:, :, ns:ns + ng] = g
    if taur.size == nlay * nc:                                # one Rayleigh plane for the band
        ch["taur"][:, :, ns:ns + ng] = taur.reshape(nlay, nc).T[:, :, None]
    else:
        ch["taur"][:, :, ns:ns + ng] = taur.reshape(ng, nlay, nc).transpose(2, 1, 0)


# --- cldprop (radsw_main_nmmb.f:1003-1006, cloudy columns: the kernel gates zcf1 > 0) ----------------------
def cldprop(ctx, ch, cstart, nc):
    import cupy as cp
    from namgpu import kernels
    m = _mod("phys_rrtm_sw_cldprop")
    nlay = int(ch["pavel"].shape[1])
    tab = m.tables(ctx)
    lm = lambda n: cp.ascontiguousarray(ch[n].T).reshape(nlay, 1, nc)   # noqa: E731
    cf1 = cp.ascontiguousarray(ch["zcf1"]).reshape(1, nc)
    taucw = cp.empty((NBDSW, nlay, 1, nc), dtype=cp.float32)
    ssacw = cp.empty_like(taucw)
    asycw = cp.empty_like(taucw)
    cldfrc = cp.empty((nlay, 1, nc), dtype=cp.float32)
    dummy = cp.empty(1, dtype=cp.float32)
    k = kernels.get(m.UNIT, m.KERNEL)
    with ctx.stream:
        k(kernels.grid_cols(nc, 1), kernels.BLOCK_COL,
          (lm("cfrac"), lm("cliqp"), lm("reliq"), lm("cicep"), lm("reice"),
           lm("cdat1"), lm("cdat2"), lm("cdat3"), lm("cdat4"), cf1,
           tab["extliq1"], tab["ssaliq1"], tab["asyliq1"],
           tab["extice2"], tab["ssaice2"], tab["asyice2"],
           tab["extice3"], tab["ssaice3"], tab["asyice3"],
           taucw, ssacw, asycw, cldfrc, dummy,
           np.int32(nlay), np.int32(1), np.int32(3), np.int32(0), np.int32(nc), np.int32(1)))
        for n, a in (("taucw", taucw), ("ssacw", ssacw), ("asycw", asycw)):
            ch[n][...] = a.reshape(NBDSW, nlay, nc).transpose(2, 1, 0)
        ch["cldfrc"][...] = cldfrc.reshape(nlay, nc).T


# --- taumol: sfluxzen (radsw_main_nmmb.f:3240-3330) ---------------------------------------------------------
def taumol(ctx, ch, cstart, nc):
    import cupy as cp
    with ctx.stream:
        L = layer_major(ctx, ch, cstart, nc)
        sfl = _mod("phys_rrtm_sw_table_sflux")
        sfl.run(ctx)
        p = sfl.PREFIX
        tm = _mod("phys_rrtm_sw_taumol")
        sz = cp.empty((NGPTSW, 1, nc), dtype=cp.float32)
        tm.taumol_sfluxzen(L["colamt"], L["jp"], L["laytrop"], ctx.tables[p + "sfluxref01"],
                           ctx.tables[p + "sfluxref02"], ctx.tables[p + "sfluxref03"], sz, stream=ctx.stream)
        ch["sfluxzen"][...] = sz.reshape(NGPTSW, nc).T


def _inp(L, names):
    return {n: L[n] for n in names}


_ALL = ("colamt",) + _F2 + _I2 + ("laytrop",)


def band17(ctx, ch, cstart, nc):
    m = _mod("phys_rrtm_sw_taumol17")
    L = ctx.tables[KEY_L]
    out = m.alloc_outputs(L["nlay"], 1, nc)
    m.taumol17(_inp(L, _ALL), _tab(ctx, m.KGB_KEY, m.load_kgb17), out, L["nlay"], 1, nc, stream=ctx.stream)
    with ctx.stream:
        _put(ch, 17, out["taug17"], out["taur17"])


def band18(ctx, ch, cstart, nc):
    m = _mod("phys_rrtm_sw_taumol18")
    L = ctx.tables[KEY_L]
    out = m.alloc_outputs(L["nlay"], 1, nc)
    m.taumol18(_inp(L, _ALL), _tab(ctx, m.TAB_KEY, m.load_tables), out, stream=ctx.stream)
    with ctx.stream:
        _put(ch, 18, out["taug"], out["taur"])


def _band_plane0(b):
    def f(ctx, ch, cstart, nc):
        m = _mod(f"phys_rrtm_sw_taumol{b}")
        L = ctx.tables[KEY_L]
        out = m.alloc_outputs(L["nlay"], 1, nc)
        getattr(m, f"taumol{b}")(_inp(L, _ALL), out, _tab(ctx, m.TAB_KEY, m.load_tables), plane0=0,
                                 stream=ctx.stream)
        with ctx.stream:
            _put(ch, b, out["taug"], out["taur"])
    return f


def band23(ctx, ch, cstart, nc):
    import cupy as cp
    m = _mod("phys_rrtm_sw_taumol23")
    L = ctx.tables[KEY_L]
    nlay = L["nlay"]
    with ctx.stream:
        tpr = -(-nc // CHK)
        lt = cp.pad(L["laytrop"], ((0, 0), (0, tpr * CHK - nc)), constant_values=int(np.iinfo(np.int32).max))
        kmin = cp.ascontiguousarray(lt.reshape(1, tpr, CHK).min(axis=2).astype(cp.int32).ravel())
        out = {"taug": cp.zeros((NG[23], nlay, nc), dtype=cp.float32),
               "taur": cp.zeros((NG[23], nlay, nc), dtype=cp.float32)}
    m.taumol23_chunk(_inp(L, _ALL), _tab(ctx, m.TAB_KEY, m.load_tables), kmin, out, 0, nc, nlay, 1, nc, tpr,
                     stream=ctx.stream)
    with ctx.stream:
        _put(ch, 23, out["taug"], out["taur"])


def band24(ctx, ch, cstart, nc):
    m = _mod("phys_rrtm_sw_taumol24")
    L = ctx.tables[KEY_L]
    nlay = L["nlay"]
    out = m.alloc_outputs(nlay, 1, nc)                        # all 112 planes; band 24 writes 66..73
    m.taumol24(_inp(L, _ALL), out, _tab(ctx, m.TAB_KEY, m.load_tables), stream=ctx.stream)
    ns, ng = NS[24], NG[24]
    with ctx.stream:
        _put(ch, 24, out["taug"][ns:ns + ng], out["taur"][ns:ns + ng])


def band25(ctx, ch, cstart, nc):
    import cupy as cp
    m = _mod("phys_rrtm_sw_taumol25")
    L = ctx.tables[KEY_L]
    out = m.alloc_outputs(L["nlay"], 1, nc)
    kmin = cp.zeros((1,), dtype=cp.int32)
    m.taumol25(_inp(L, _ALL), out, _tab(ctx, m.TAB_KEY, m.load_tables), kmin, stream=ctx.stream)
    with ctx.stream:
        _put(ch, 25, out["taug"], out["taur"])


def band26(ctx, ch, cstart, nc):
    m = _mod("phys_rrtm_sw_taumol26")
    L = ctx.tables[KEY_L]
    out = m.alloc_outputs(L["nlay"], 1, nc)
    m.taumol26({"colmol": L["colmol"]}, _tab(ctx, m.TAB_KEY, m.load_tables), out, stream=ctx.stream)
    with ctx.stream:
        _put(ch, 26, out["taug26"], out["taur26"])


def band28(ctx, ch, cstart, nc):
    m = _mod("phys_rrtm_sw_taumol28")
    L = ctx.tables[KEY_L]
    out = m.alloc_outputs(L["nlay"], 1, nc)
    m.taumol28(_inp(L, _ALL), out, _tab(ctx, m.TAB_KEY, m.load_tables), stream=ctx.stream)
    with ctx.stream:
        _put(ch, 28, out["taug"], out["taur"])


def band29(ctx, ch, cstart, nc):
    """id0/id1 of band 29 (nspa = nspb = 1, radsw_main_nmmb.f:3246-3256) are built here, as the dispatcher."""
    import cupy as cp
    m = _mod("phys_rrtm_sw_taumol29")
    L = ctx.tables[KEY_L]
    nlay = L["nlay"]
    with ctx.stream:
        k1 = cp.arange(1, nlay + 1, dtype=cp.int32).reshape(nlay, 1, 1)
        low = k1 <= L["laytrop"][None]
        jp, jt, jt1 = L["jp"], L["jt"], L["jt1"]
        id0 = cp.where(low, (jp - 1) * 5 + (jt - 1), (jp - 13) * 5 + (jt - 1)).astype(cp.int32)
        id1 = cp.where(low, jp * 5 + (jt1 - 1), (jp - 12) * 5 + (jt1 - 1)).astype(cp.int32)
        inp = _inp(L, _ALL)
        inp["id0"], inp["id1"] = cp.ascontiguousarray(id0), cp.ascontiguousarray(id1)
    out = m.alloc_outputs(nlay, 1, nc)
    m.taumol29(inp, _tab(ctx, m.TAB_KEY, m.load_tables), out, stream=ctx.stream)
    with ctx.stream:
        _put(ch, 29, out["taug29"], out["taur29"])


# --- spcvrtc (radsw_main_nmmb.f:1079-1085, isubcsw <= 0) ----------------------------------------------------
def spcvrtc(ctx, ch, cstart, nc):
    import cupy as cp
    m = _mod("phys_rrtm_sw_spcvrtc")
    nlay = int(ch["pavel"].shape[1])
    nlp1 = nlay + 1
    with ctx.stream:
        v1 = lambda n: cp.ascontiguousarray(ch[n]).reshape(1, nc)                       # noqa: E731
        inp = {"cosz": v1("cosz1"), "sntz": v1("sntz1"), "ssolar": v1("ssolar"), "cf1": v1("zcf1"),
               "cf0": v1("zcf0"),
               "albbm": cp.ascontiguousarray(ch["albbm"].T).reshape(2, 1, nc),
               "albdf": cp.ascontiguousarray(ch["albdf"].T).reshape(2, 1, nc),
               "cldfrc": cp.ascontiguousarray(ch["cldfrc"].T).reshape(nlay, 1, nc)}
        tg = cp.ascontiguousarray(ch["taug"].transpose(2, 1, 0)).reshape(NGPTSW, nlay, 1, nc)
        tr = cp.ascontiguousarray(ch["taur"].transpose(2, 1, 0)).reshape(NGPTSW, nlay, 1, nc)
        opt = {n: cp.ascontiguousarray(ch[n].transpose(2, 1, 0)).reshape(NBDSW, nlay, 1, nc)
               for n in ("tauae", "ssaae", "asyae", "taucw", "ssacw", "asycw")}
        sz = cp.ascontiguousarray(ch["sfluxzen"].T).reshape(NGPTSW, 1, nc)
        exp_tbl = ctx.tables.get("rrtm_sw_exp_tbl")
        if exp_tbl is None:
            exp_tbl = cp.asarray(m.build_exp_tbl())
            ctx.tables["rrtm_sw_exp_tbl"] = exp_tbl
        out = m.alloc_outputs(nlay, 1, nc)
    m.spcvrtc(inp, out, lambda jg: (tg[jg - 1], tr[jg - 1]), lambda ib: {n: a[ib - 1] for n, a in opt.items()},
              exp_tbl, sz, stream=ctx.stream)
    with ctx.stream:
        for n in ("fxupc", "fxdnc", "fxup0", "fxdn0"):
            ch[n][...] = out[n].reshape(NBDSW, nlp1, nc).transpose(2, 1, 0)
        for n in ("ftoauc", "ftoau0", "ftoadc", "fsfcuc", "fsfcu0", "fsfcdc", "fsfcd0", "suvbfc", "suvbf0"):
            ch[n][...] = out[n].reshape(nc)
        for n in ("sfbmc", "sfdfc", "sfbm0", "sfdf0"):
            ch[n][...] = out[n].reshape(2, nc).T


# Seam module -> adapter, for the units that take the CONTRACT layout (no run_chunk of their own).
ADAPTERS = {
    "namgpu.physics.rrtm.phys_rrtm_sw_cldprop": cldprop,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol": taumol,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol17": band17,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol18": band18,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol21": _band_plane0(21),
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol23": band23,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol24": band24,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol25": band25,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol26": band26,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol27": _band_plane0(27),
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol28": band28,
    "namgpu.physics.rrtm.phys_rrtm_sw_taumol29": band29,
    "namgpu.physics.rrtm.phys_rrtm_sw_spcvrtc": spcvrtc,
}
