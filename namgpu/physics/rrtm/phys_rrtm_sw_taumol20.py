# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol20: TAUMOL20, SW band 20 (5150-6150 cm-1) optical depths, Rayleigh optical depths
and the band-20 top-of-atmosphere solar source.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3852-3943 (SUBROUTINE taumol20).
The band-20 index setup is the driver WHERE blocks at radsw_main_nmmb.f:3250-3257; the band-20 solar
source is the taumol driver case block at radsw_main_nmmb.f:3267-3273 (case (16, 20, 23, 25, 26, 29),
sfluxzen(:, ns+j) = sfluxref01(j, 1, ibd), ns = NS20 = 34).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_taumol20.cu (rrtm_sw_taumol20, one thread per (i, j) column,
one launch for the full domain, one launch per chunk on the driver path).

What it computes, per column:
  - taur(NS20+g, k) = colmol(k) * rayl for every layer k and every band-20 g-point g = 1..10;
  - taug(NS20+g, k), lower atmosphere (k <= laytrop): colamt(h2o) times the absa table (binary
    species interpolation over jp, jt, jt1), the self-continuum (selfref, selffac, selffrac, indself),
    the foreign continuum (forref, forfac, forfrac, indfor), plus colamt(ch4) * absch4;
  - taug(NS20+g, k), upper atmosphere (k > laytrop): colamt(h2o) times the absb table with the
    foreign continuum, plus colamt(ch4) * absch4;
  - sfluxzen(NS20+g) = sfluxref01(g, 1, ibx(20)) on the full-domain path. The taumol unit writes the
    same band-20 solar source for every column already, so the chunk path does not write it.
NS20 = 34 and NGPTSW = 112 (radsw_param_nmmb.f:140, 150, 173). Planes outside NS20..NS20+9 are not
written here.

Inputs (device, CONTRACT layout, Fortran (i,j,l) -> [l, j, i]):
  full domain, ctx.tables["rrtm_sw_taumol20_in"]: colamt [MAXGAS=6, nlay, JM, IM] float32 (gas 0 = h2o,
      gas 4 = ch4), colmol [nlay, JM, IM] float32; setcoef outputs in ctx.tables["rrtm_sw_setcoef_out"]:
      laytrop [JM, IM] int32; jp, jt, jt1, indself, indfor [nlay, JM, IM] int32; fac00..fac11, selffac,
      selffrac, forfac, forfrac [nlay, JM, IM] float32.
  chunk path, run_chunk(ctx, ch, cstart, nc): ch holds colamt [nc, nlay, 6], colmol, laytrop [nc],
      jp..indfor and fac/self/for arrays [nc, nlay], and taug, taur [nc, nlay, NGPTSW]. Chunk column t is
      day column cstart + t. The device SW_NDAY (ctx.st["SW_NDAY"]) gates the live columns, so columns at or
      past NDAY are not written, matching the day-list compaction of phys-rrtm-sw-swrad.
Tables (built once from namgpu/data/rrtm_sw/kgb20.npz, the module_radsw_kgb20_nmmb extract, and
namgpu/data/rrtm_sw/sflux.npz for sfluxref01 and ibx): absa [NG20, 65], absb [NG20, 235],
selfref [NG20, 10], forref [NG20, 4], absch4 [NG20], sflux20 [NG20]. Stored with the CONTRACT reversal
(Fortran (m, ig) -> [ig, m]). Uploaded once into ctx.tables["rrtm_sw_taumol20_tables"].

DECLARED DIVERGENCE (pending a decision by the project lead, not resolved here): the Fortran upper pass
is WHERE(k .GE. laytrop) over k = MINVAL(laytrop(1:CHK))+1 .. NLAY. In the driver that also rewrites layer
k == laytrop(column) with the upper formula, indexed by the lower-branch id0/id1, whenever laytrop(column)
exceeds the chunk minimum. The result therefore depends on chunking (CHK is a CPU detail that CONTRACT
section 2 drops). This unit uses the per-column rule k > laytrop, which the scalar RRTM taumol20 states and
which does not depend on chunking. The Fortran is not reproduced as written.

Run `python -m namgpu.physics.rrtm.phys_rrtm_sw_taumol20` for the value check (numpy reference on a small
random domain, lower and upper branches, chunk day gating) and the full-size 954 x 835 x 60 no-NaN pass.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol20"
KERNEL = "rrtm_sw_taumol20"
NG20 = 10
NS20 = 34
NGPTSW = 112
MAXGAS = 6
MSA20 = 65
MSB20 = 235
MSF20 = 10
MFR20 = 4
BAND = 20
BAND0 = 16          # first SW band (radsw_param_nmmb.f NBLOW); ibx and sflux arrays index from here
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw"
KGB20_NPZ = DATA_DIR / "kgb20.npz"
SFLUX_NPZ = DATA_DIR / "sflux.npz"

SETCOEF_OUT = "rrtm_sw_setcoef_out"
IN_KEY = "rrtm_sw_taumol20_in"
TAB_KEY = "rrtm_sw_taumol20_tables"
OUT_KEY = "rrtm_sw_taumol20_out"
SW_NDAY_KEY = "SW_NDAY"


def load_tables():
    """kgb20.npz and sflux.npz -> device float32 arrays. Uploaded once."""
    import cupy as cp

    d = np.load(KGB20_NPZ)
    s = np.load(SFLUX_NPZ)
    ibd = int(s["ibx"][BAND - BAND0])                  # ibx(20) = 2 (radsw_datatb_nmmb.f:1119)
    sflux20 = np.ascontiguousarray(s["sfluxref01"][ibd - 1, 0, :NG20], dtype=np.float32)
    tab = {
        "absa": cp.asarray(np.ascontiguousarray(d["absa"], dtype=np.float32)),
        "absb": cp.asarray(np.ascontiguousarray(d["absb"], dtype=np.float32)),
        "selfref": cp.asarray(np.ascontiguousarray(d["selfref"], dtype=np.float32)),
        "forref": cp.asarray(np.ascontiguousarray(d["forref"], dtype=np.float32)),
        "absch4": cp.asarray(np.ascontiguousarray(d["absch4"], dtype=np.float32)),
        "sflux20": cp.asarray(sflux20),
        "rayl": float(d["rayl"]),
    }
    if tab["absa"].shape != (NG20, MSA20) or tab["absb"].shape != (NG20, MSB20) \
            or tab["selfref"].shape != (NG20, MSF20) or tab["forref"].shape != (NG20, MFR20) \
            or tab["absch4"].shape != (NG20,) or tab["sflux20"].shape != (NG20,):
        raise ValueError("taumol20: kgb20.npz / sflux.npz shapes do not match module_radsw_kgb20_nmmb")
    return tab


def alloc_outputs(nlay: int, jm: int, im: int):
    """taug, taur, sfluxzen allocated once; this unit writes planes NS20..NS20+9 of each."""
    import cupy as cp

    return {
        "taug": cp.zeros((NGPTSW, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((NGPTSW, nlay, jm, im), dtype=cp.float32),
        "sfluxzen": cp.zeros((NGPTSW, jm, im), dtype=cp.float32),
    }


def _launch(inp: dict, taug, taur, sfluxzen, tab: dict, *, ncols: int, nlay: int, lc: int, lk: int,
            ca: tuple, tg: tuple, cstart: int, ndayp, nday_host: int, sf, stream=None):
    """One launch over ncols columns. Strides as in phys_rrtm_sw_taumol16: ca = (c, k, m) for colamt,
    tg = (c, k, g) for taug and taur, sf = (c, g) for sfluxzen (None: no solar write). taug, taur and
    sfluxzen are passed already offset to plane NS20."""
    import cupy as cp

    kern = kernels.get(UNIT, KERNEL)
    gate = ndayp is not None
    dummy = inp["laytrop"]          # valid device pointer; read only when gate is set
    do_sf = sf is not None
    sfc, sfg = sf if do_sf else (0, 0)
    args = (
        inp["colamt"], inp["colmol"], inp["laytrop"],
        inp["jp"], inp["jt"], inp["jt1"],
        inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
        inp["selffac"], inp["selffrac"], inp["indself"],
        inp["forfac"], inp["forfrac"], inp["indfor"],
        tab["absa"], tab["absb"], tab["selfref"], tab["forref"], tab["absch4"], tab["sflux20"],
        np.float32(tab["rayl"]),
        ndayp if gate else dummy, np.int32(1 if gate else 0), np.int32(nday_host), np.int32(cstart),
        np.int32(ncols), np.int32(nlay),
        np.int64(lc), np.int64(lk),
        np.int64(ca[0]), np.int64(ca[1]), np.int64(ca[2]),
        np.int64(tg[0]), np.int64(tg[1]), np.int64(tg[2]),
        np.int32(1 if do_sf else 0), np.int64(sfc), np.int64(sfg),
        taug, taur, sfluxzen if do_sf else taug,
    )
    block = kernels.BLOCK_COL
    grid = ((ncols + block[0] - 1) // block[0], 1, 1)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)


def taumol20(inp: dict, out: dict, tab: dict, stream=None):
    """Full domain: one launch over every (i, j) column, all columns live. inp holds device arrays in
    [nlay, JM, IM] (colamt [6, nlay, JM, IM]) and laytrop [JM, IM]; out holds taug, taur [112, nlay, JM, IM]
    and sfluxzen [112, JM, IM]."""
    nlay, jm, im = inp["fac00"].shape
    jmim = jm * im
    _launch(inp, out["taug"][NS20:], out["taur"][NS20:], out["sfluxzen"][NS20:], tab,
            ncols=jmim, nlay=nlay, lc=1, lk=jmim, ca=(1, jmim, nlay * jmim),
            tg=(1, jmim, nlay * jmim), cstart=0, ndayp=None, nday_host=jmim,
            sf=(1, jmim), stream=stream)
    return out


def run_chunk(ctx, ch: dict, cstart: int, nc: int):
    """Driver seam entry (phys-rrtm-sw-swrad._seam): taumol20 for one chunk of day columns. Reads ch colamt
    [nc, nlay, 6], colmol, fac00..fac11, selffac, selffrac, forfac, forfrac, jp, jt, jt1, indself, indfor
    [nc, nlay] and laytrop [nc]. Writes planes NS20..NS20+9 of ch taug and taur [nc, nlay, NGPTSW]. Live
    columns are cstart + t < SW_NDAY (device); the last chunk may run past NDAY and those columns are left
    as they are."""
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    colmol = ch["colmol"]
    if colmol.shape[0] != nc:
        raise ValueError(f"taumol20 chunk: colmol {colmol.shape} does not match nc={nc}")
    nlay = int(colmol.shape[1])
    if ch["taug"].shape != (nc, nlay, NGPTSW) or ch["taur"].shape != (nc, nlay, NGPTSW):
        raise ValueError(f"taumol20 chunk: taug/taur must be [nc, nlay, NGPTSW] = {(nc, nlay, NGPTSW)}")
    ndayp = ctx.st[SW_NDAY_KEY]
    stream = getattr(ctx, "stream", None)
    _launch(ch, ch["taug"][..., NS20:], ch["taur"][..., NS20:], None, tab,
            ncols=nc, nlay=nlay, lc=nlay, lk=1, ca=(nlay * MAXGAS, MAXGAS, 1),
            tg=(nlay * NGPTSW, NGPTSW, 1), cstart=cstart, ndayp=ndayp, nday_host=0, sf=None,
            stream=stream)
    return ch


def run(ctx):
    """Full-domain driver entry. Reads colamt, colmol (ctx.tables[IN_KEY]) and the setcoef outputs
    (ctx.tables[SETCOEF_OUT]); writes taug, taur and sfluxzen into ctx.tables[OUT_KEY]."""
    tin = ctx.tables.get(IN_KEY)
    sout = ctx.tables.get(SETCOEF_OUT)
    if tin is None or sout is None:
        raise RuntimeError("phys_rrtm_sw_taumol20 needs ctx.tables[IN_KEY] and the setcoef outputs")
    inp = {
        "colamt": tin["colamt"], "colmol": tin["colmol"],
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
    nlay, jm, im = inp["fac00"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug"].shape != (NGPTSW, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol20(inp, out, tab, stream=ctx.stream)


def reference(colamt, colmol, laytrop, jp, jt, jt1, f00, f01, f10, f11, sfac, sfrac, inds, ffac, ffrac,
              indf, tab_np, rayl):
    """Independent float64 reference of the per-column rule. Per-layer arrays are [nlay, N]; colamt is
    [6, nlay, N]; laytrop is [N]. Returns taug, taur as [NG20, nlay, N]."""
    absa, absb = tab_np["absa"].astype(np.float64), tab_np["absb"].astype(np.float64)
    selfref, forref = tab_np["selfref"].astype(np.float64), tab_np["forref"].astype(np.float64)
    absch4 = tab_np["absch4"].astype(np.float64)
    nlay, n = colmol.shape
    taug = np.zeros((NG20, nlay, n))
    taur = np.zeros((NG20, nlay, n))
    for c in range(n):
        lt = int(laytrop[c])
        for k0 in range(nlay):
            k = k0 + 1
            taur[:, k0, c] = colmol[k0, c] * rayl
            c2 = colamt[0, k0, c]
            c4 = colamt[4, k0, c]
            if k <= lt:
                id0 = (int(jp[k0, c]) - 1) * 5 + (int(jt[k0, c]) - 1)
                id1 = int(jp[k0, c]) * 5 + (int(jt1[k0, c]) - 1)
                i01 = max(1, min(MSA20, id0 + 1))
                i02 = min(MSA20, i01 + 1)
                i11 = max(1, min(MSA20, id1 + 1))
                i12 = min(MSA20, i11 + 1)
                s = int(inds[k0, c])
                f = int(indf[k0, c])
                for g in range(NG20):
                    ab = (f00[k0, c] * absa[g, i01 - 1] + f10[k0, c] * absa[g, i02 - 1]
                          + f01[k0, c] * absa[g, i11 - 1] + f11[k0, c] * absa[g, i12 - 1])
                    se = selfref[g, s - 1] + sfrac[k0, c] * (selfref[g, s] - selfref[g, s - 1])
                    fo = forref[g, f - 1] + ffrac[k0, c] * (forref[g, f] - forref[g, f - 1])
                    taug[g, k0, c] = c2 * (ab + sfac[k0, c] * se + ffac[k0, c] * fo) + c4 * absch4[g]
            else:
                id0 = (int(jp[k0, c]) - 13) * 5 + (int(jt[k0, c]) - 1)
                id1 = (int(jp[k0, c]) - 12) * 5 + (int(jt1[k0, c]) - 1)
                i01 = max(1, min(MSB20, id0 + 1))
                i02 = min(MSB20, i01 + 1)
                i11 = max(1, min(MSB20, id1 + 1))
                i12 = min(MSB20, i11 + 1)
                f = int(indf[k0, c])
                for g in range(NG20):
                    ab = (f00[k0, c] * absb[g, i01 - 1] + f10[k0, c] * absb[g, i02 - 1]
                          + f01[k0, c] * absb[g, i11 - 1] + f11[k0, c] * absb[g, i12 - 1])
                    fo = forref[g, f - 1] + ffrac[k0, c] * (forref[g, f] - forref[g, f - 1])
                    taug[g, k0, c] = c2 * (ab + ffac[k0, c] * fo) + c4 * absch4[g]
    return taug, taur


def _random_state(rng, nlay: int, shape_n: int):
    """Setcoef-like random state over N columns (flattened). Lower layers get indself 1..9 and jp 1..12;
    upper layers get jp 13..58 and indself 0 (setcoef, radsw_main_nmmb.f:2116 and 2129)."""
    f32 = np.float32
    laytrop = rng.integers(1, nlay + 1, (shape_n,)).astype(np.int32)
    kk = np.arange(1, nlay + 1)[:, None]
    lower = kk <= laytrop[None, :]
    jp = np.where(lower, rng.integers(1, 13, (nlay, shape_n)), rng.integers(13, 59, (nlay, shape_n))).astype(np.int32)
    jt = rng.integers(1, 5, (nlay, shape_n)).astype(np.int32)
    jt1 = rng.integers(1, 5, (nlay, shape_n)).astype(np.int32)
    inds = np.where(lower, rng.integers(1, 10, (nlay, shape_n)), 0).astype(np.int32)
    indf = rng.integers(1, 4, (nlay, shape_n)).astype(np.int32)
    st = {
        "laytrop": laytrop, "jp": jp, "jt": jt, "jt1": jt1, "indself": inds, "indfor": indf,
        "colamt": rng.uniform(1.0e18, 3.0e23, (MAXGAS, nlay, shape_n)).astype(f32),
        "colmol": rng.uniform(1.0e18, 3.0e25, (nlay, shape_n)).astype(f32),
        "fac00": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "fac01": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "fac10": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "fac11": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "selffac": rng.uniform(0.0, 1.0e-2, (nlay, shape_n)).astype(f32),
        "selffrac": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "forfac": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
        "forfrac": rng.uniform(0.0, 1.0, (nlay, shape_n)).astype(f32),
    }
    return st


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, rows: int = 8):
    """1. Value check: full-domain launch on a small 5 x 4 x 7 random state against the numpy reference
    (lower and upper branches both run, indself 1..9 on the lower branch so the kernel never reads
    selfref[-1]). 2. Day gating: a chunk of 6 columns with SW_NDAY = 4; live columns match the reference,
    columns 4..5 are not written (stay zero). 3. Full-size 954 x 835 x 60 pass in row bands: no NaN/Inf
    in the band planes, and the band planes are written (nonzero)."""
    import time
    import types

    import cupy as cp

    kernels.module(UNIT).compile()
    tab = load_tables()
    tab_np = {"absa": cp.asnumpy(tab["absa"]), "absb": cp.asnumpy(tab["absb"]),
              "selfref": cp.asnumpy(tab["selfref"]), "forref": cp.asnumpy(tab["forref"]),
              "absch4": cp.asnumpy(tab["absch4"])}
    rayl = tab["rayl"]
    rng = np.random.default_rng(seed)

    # 1. full-domain value check on a small grid (nlay 7, JM 4, IM 5).
    nl, jsm, ism = 7, 4, 5
    st = _random_state(rng, nl, jsm * ism)
    st = {k: v.reshape((MAXGAS, nl, jsm, ism) if v.ndim == 3 and v.shape[0] == MAXGAS
                       else (nl, jsm, ism) if v.ndim == 2 and v.shape[0] == nl
                       else (jsm, ism)) for k, v in st.items()}
    inp = {k: cp.asarray(v) for k, v in st.items() if k != "colmol" and k != "colamt"}
    inp["colamt"] = cp.asarray(st["colamt"])
    inp["colmol"] = cp.asarray(st["colmol"])
    out = alloc_outputs(nl, jsm, ism)
    taumol20(inp, out, tab)
    cp.cuda.Stream.null.synchronize()
    ref_g, ref_r = reference(
        st["colamt"].reshape(MAXGAS, nl, -1), st["colmol"].reshape(nl, -1), st["laytrop"].reshape(-1),
        st["jp"].reshape(nl, -1), st["jt"].reshape(nl, -1), st["jt1"].reshape(nl, -1),
        st["fac00"].reshape(nl, -1), st["fac01"].reshape(nl, -1), st["fac10"].reshape(nl, -1),
        st["fac11"].reshape(nl, -1), st["selffac"].reshape(nl, -1), st["selffrac"].reshape(nl, -1),
        st["indself"].reshape(nl, -1), st["forfac"].reshape(nl, -1), st["forfrac"].reshape(nl, -1),
        st["indfor"].reshape(nl, -1), tab_np, rayl)
    got_g = cp.asnumpy(out["taug"][NS20:NS20 + NG20]).reshape(NG20, nl, -1).astype(np.float64)
    got_r = cp.asnumpy(out["taur"][NS20:NS20 + NG20]).reshape(NG20, nl, -1).astype(np.float64)
    err_full = max(float(np.max(np.abs(got_g - ref_g) / (np.abs(ref_g) + 1e-30))),
                   float(np.max(np.abs(got_r - ref_r) / (np.abs(ref_r) + 1e-30))))
    assert err_full < 1e-4, f"taumol20 full-domain value check failed: max rel err {err_full:.3e}"

    # 2. chunk day gating: nc = 6 columns, SW_NDAY = 4, cstart = 0.
    nc = 6
    cs = _random_state(rng, nl, nc)
    ch = {
        "colamt": cp.asarray(cs["colamt"].transpose(2, 1, 0).copy()),
        "colmol": cp.asarray(cs["colmol"].T.copy()),
        "laytrop": cp.asarray(cs["laytrop"]),
        "taug": cp.zeros((nc, nl, NGPTSW), dtype=cp.float32),
        "taur": cp.zeros((nc, nl, NGPTSW), dtype=cp.float32),
    }
    for key in ("jp", "jt", "jt1", "indself", "indfor", "fac00", "fac01", "fac10", "fac11",
                "selffac", "selffrac", "forfac", "forfrac"):
        ch[key] = cp.asarray(cs[key].T.copy())
    ctx = types.SimpleNamespace(tables={TAB_KEY: tab}, st={SW_NDAY_KEY: cp.asarray(np.array([4], np.int32))},
                                stream=None)
    run_chunk(ctx, ch, 0, nc)
    cp.cuda.Stream.null.synchronize()
    cg, cr = ch["taug"], ch["taur"]
    rg, rr = reference(
        cs["colamt"], cs["colmol"], cs["laytrop"], cs["jp"], cs["jt"], cs["jt1"], cs["fac00"], cs["fac01"],
        cs["fac10"], cs["fac11"], cs["selffac"], cs["selffrac"], cs["indself"], cs["forfac"], cs["forfrac"],
        cs["indfor"], tab_np, rayl)
    got_cg = cp.asnumpy(cg[:4, :, NS20:NS20 + NG20]).transpose(2, 1, 0).astype(np.float64)
    got_cr = cp.asnumpy(cr[:4, :, NS20:NS20 + NG20]).transpose(2, 1, 0).astype(np.float64)
    err_chunk = max(float(np.max(np.abs(got_cg - rg[:, :, :4]) / (np.abs(rg[:, :, :4]) + 1e-30))),
                    float(np.max(np.abs(got_cr - rr[:, :, :4]) / (np.abs(rr[:, :, :4]) + 1e-30))))
    assert err_chunk < 1e-4, f"taumol20 chunk value check failed: max rel err {err_chunk:.3e}"
    dead = int(cp.count_nonzero(cg[4:])) + int(cp.count_nonzero(cr[4:]))
    assert dead == 0, f"taumol20 chunk wrote {dead} values past SW_NDAY"

    # 3. full-size no-NaN pass in row bands.
    t0 = time.perf_counter()
    nbad = 0
    nzero = 0
    for j0 in range(0, jm, rows):
        jc = min(rows, jm - j0)
        bst = _random_state(rng, nlay, jc * im)
        binp = {k: cp.asarray(v.reshape((MAXGAS, nlay, jc, im) if v.ndim == 3 and v.shape[0] == MAXGAS
                                       else (nlay, jc, im) if v.ndim == 2 and v.shape[0] == nlay
                                       else (jc, im)))
                for k, v in bst.items()}
        bout = alloc_outputs(nlay, jc, im)
        taumol20(binp, bout, tab)
        cp.cuda.Stream.null.synchronize()
        band = bout["taug"][NS20:NS20 + NG20]
        bandr = bout["taur"][NS20:NS20 + NG20]
        nbad += int(cp.count_nonzero(~cp.isfinite(band))) + int(cp.count_nonzero(~cp.isfinite(bandr)))
        nbad += int(cp.count_nonzero(~cp.isfinite(bout["sfluxzen"][NS20:NS20 + NG20])))
        nzero += int(cp.count_nonzero(band == 0.0))
    dt = time.perf_counter() - t0
    assert nbad == 0, f"NaN/Inf in taumol20 band planes: {nbad}"
    assert nzero == 0, f"taumol20 wrote {nzero} zero optical depths (band planes must be filled)"
    return {"full_max_rel_err": err_full, "chunk_max_rel_err": err_chunk, "chunk_dead_writes": dead,
            "nbad": nbad, "nzero": nzero, "seconds": round(dt, 2)}


if __name__ == "__main__":
    print(selftest())
