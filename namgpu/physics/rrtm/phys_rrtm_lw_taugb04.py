# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb04: TAUGB04, NMMB RRTM longwave band 4 optical depths and Planck fractions.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:4098-4337 (SUBROUTINE taugb04).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the module header is kept as the notice in
namgpu/kernels/phys_rrtm_lw_taugb04.cu (kernel rrtm_lw_taugb04, one launch, one thread per (i, j, layer)).

Band-4 k-tables: module_radlw_kgb04 of phys/radlw_datatb.f (lines 10030-15204), which taugb04 USEs at
radlw_main_nmmb.f:4105. Extracted by tools/extract_rrtm_lw_kgb04.py into namgpu/data/rrtm_lw/kgb04.npz
(absa [585,14], absb [1175,14], selfref [10,14], forref [4,14], fracrefa [9,14], fracrefb [5,14];
C-order [index, g]). Reference gas amounts chi_mls (module_radlw_ref, radlw_datatb.f) are in
namgpu/data/rrtm_lw/ref.npz.

Inputs: the per-layer interpolation state from phys-rrtm-lw-setcoef, in ctx.tables["rrtm_lw_setcoef_out"]
  (laytrop, jp, jt, jt1, rfrate, fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac,
  indfor), and the column amounts colamt in ctx.tables["rrtm_lw_setcoef_in"]. Index arrays hold the
  Fortran 1-based values (setcoef's convention, Fortran jp(k) used as jp-1 / jp-12 / jp-13 here).
Outputs: ctx.tables["rrtm_lw_taugb04_out"] = {"taug04": [nlay, 140, JM, IM], "fracs04": same}, float32.
  These are the full LW g-point arrays of the Fortran taumol (NGPTLW = 140, radlw_param_nmmb.f:141).
  The kernel writes only rows NS04..NS04+13 (NS04 = 38, radlw_param_nmmb.f:160), i.e. the Fortran
  taug(ns04+ig,k) and fracs(ns04+ig,k) placement. Other rows are left as allocated (zero).
Refrat for the Planck fraction is computed once on the host from chi_mls (two scalars, cached).

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation (outputs allocated once).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_lw_setcoef import IN_KEY, OUT_KEY as SETCOEF_OUT_KEY

UNIT = "phys_rrtm_lw_taugb04"
KERNEL = "rrtm_lw_taugb04"
NG04 = 14            # NG04, radlw_param_nmmb.f:153
NGPT = 140           # NGPTLW, radlw_param_nmmb.f:141
NS04 = 38            # NS04, radlw_param_nmmb.f:160 (zero-based row offset of band 4)
NRATES = 6
MAXGAS = 6
NSPA4 = 9            # nspa(4), radlw_main_nmmb.f:284
NSPB4 = 5            # nspb(4), radlw_main_nmmb.f:285
ONEMINUS = np.float32(1.0 - 1.0e-6)   # radlw_main_nmmb.f:267-268
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw"

TAB_KEY = "rrtm_lw_kgb04"
REF_KEY = "rrtm_lw_taugb04_refrat"
OUT_KEY = "rrtm_lw_taugb04_out"

# Table shapes as C-order [index, g] (Fortran (ig, index)); checked on load.
TAB_SHAPES = {"absa": (585, NG04), "absb": (1175, NG04), "selfref": (10, NG04), "forref": (4, NG04),
              "fracrefa": (9, NG04), "fracrefb": (5, NG04)}


def _require(name, arr, shape, dtype):
    """Shape, dtype and C-contiguity guard (the kernel reads raw pointers; a mismatch is silent)."""
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"phys_rrtm_lw_taugb04: {name} shape {tuple(arr.shape)} != {tuple(shape)}")
    if np.dtype(arr.dtype) != np.dtype(dtype):
        raise ValueError(f"phys_rrtm_lw_taugb04: {name} dtype {np.dtype(arr.dtype).name} != {np.dtype(dtype).name}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"phys_rrtm_lw_taugb04: {name} is not C-contiguous")


def _tables(ctx):
    """Band-4 k-tables on the device, uploaded once (C-order [index, g] float32)."""
    import cupy as cp

    t = ctx.tables.get(TAB_KEY)
    if t is None:
        t = {}
        with np.load(DATA_DIR / "kgb04.npz") as d:
            for k, shape in TAB_SHAPES.items():
                host = d[k]
                if tuple(host.shape) != shape:
                    raise ValueError(f"phys_rrtm_lw_taugb04: kgb04.npz {k} shape {host.shape} != {shape}")
                t[k] = cp.asarray(host.astype(np.float32))
        ctx.tables[TAB_KEY] = t
    return t


def _refrat(ctx):
    """refrat_planck_a = chi_mls(1,11)/chi_mls(2,11), refrat_planck_b = chi_mls(3,13)/chi_mls(2,13).
    chi_mls is stored C-order [ref, gas] (Fortran chi_mls(gas, ref)), so the indices below are
    [ref-1, gas-1]. Computed once on the host in float32, as the Fortran does in single precision."""
    r = ctx.tables.get(REF_KEY)
    if r is None:
        with np.load(DATA_DIR / "ref.npz") as d:
            chi = d["chi_mls"].astype(np.float32)
        if chi.shape[0] < 13 or chi.shape[1] < 3:
            raise ValueError(f"phys_rrtm_lw_taugb04: ref.npz chi_mls shape {chi.shape} too small")
        a = np.float32(chi[10, 0] / chi[10, 1])
        b = np.float32(chi[12, 2] / chi[12, 1])
        r = (float(a), float(b))
        ctx.tables[REF_KEY] = r
    return r


def alloc_outputs(nlay: int, jm: int, im: int):
    import cupy as cp

    return {"taug04": cp.zeros((nlay, NGPT, jm, im), dtype=cp.float32),
            "fracs04": cp.zeros((nlay, NGPT, jm, im), dtype=cp.float32)}


def check_inputs(inp: dict, nlay: int, jm: int, im: int):
    """Shapes and dtypes of every device input the kernel reads through raw pointers."""
    f32, i32 = np.float32, np.int32
    _require("laytrop", inp["laytrop"], (jm, im), i32)
    _require("colamt", inp["colamt"], (MAXGAS, nlay, jm, im), f32)
    _require("rfrate", inp["rfrate"], (2, NRATES, nlay, jm, im), f32)
    for name in ("jp", "jt", "jt1", "indself", "indfor"):
        _require(name, inp[name], (nlay, jm, im), i32)
    for name in ("fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac"):
        _require(name, inp[name], (nlay, jm, im), f32)


def taugb04(inp: dict, tab: dict, refrat: tuple, out: dict, stream=None):
    """One launch over every (i, j, layer). inp holds the device arrays named as in the kernel."""
    nlay, jm, im = inp["fac00"].shape
    check_inputs(inp, nlay, jm, im)
    for name, shape in (("absa", TAB_SHAPES["absa"]), ("absb", TAB_SHAPES["absb"]),
                        ("selfref", TAB_SHAPES["selfref"]), ("forref", TAB_SHAPES["forref"]),
                        ("fracrefa", TAB_SHAPES["fracrefa"]), ("fracrefb", TAB_SHAPES["fracrefb"])):
        _require(name, tab[name], shape, np.float32)
    for name in ("taug04", "fracs04"):
        _require(name, out[name], (nlay, NGPT, jm, im), np.float32)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_2d(im, jm, nlay), kernels.BLOCK_2D, (
            inp["laytrop"], inp["colamt"], inp["rfrate"], inp["jp"], inp["jt"], inp["jt1"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"], tab["fracrefa"], tab["fracrefb"],
            np.float32(refrat[0]), np.float32(refrat[1]),
            np.int32(im), np.int32(jm), np.int32(nlay),
            out["taug04"], out["fracs04"]))
    return out


def run(ctx):
    """Driver entry. Reads the setcoef state from ctx.tables; returns the band-4 outputs."""
    inp_sc = ctx.tables.get(IN_KEY)
    sc = ctx.tables.get(SETCOEF_OUT_KEY)
    if inp_sc is None or sc is None:
        raise RuntimeError("phys_rrtm_lw_taugb04 needs the phys-rrtm-lw-setcoef inputs and outputs in ctx.tables")
    nlay, jm, im = inp_sc["pavel"].shape
    inp = {"laytrop": sc["laytrop"], "colamt": inp_sc["colamt"], "rfrate": sc["rfrate"],
           "jp": sc["jp"], "jt": sc["jt"], "jt1": sc["jt1"],
           "fac00": sc["fac00"], "fac01": sc["fac01"], "fac10": sc["fac10"], "fac11": sc["fac11"],
           "selffac": sc["selffac"], "selffrac": sc["selffrac"], "indself": sc["indself"],
           "forfac": sc["forfac"], "forfrac": sc["forfrac"], "indfor": sc["indfor"]}
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug04"].shape != (nlay, NGPT, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taugb04(inp, _tables(ctx), _refrat(ctx), out, stream=ctx.stream)


def _random_band(rng, nlay: int, jm: int, im: int):
    """Random but plausible setcoef-style state for a band of jm rows at full width im and nlay layers."""
    f32 = np.float32
    shp3 = (nlay, jm, im)
    lt = rng.integers(20, 50, (jm, im)).astype(np.int32)
    jp = np.where(np.arange(nlay)[:, None, None] < lt[None], rng.integers(1, 12, shp3),
                  rng.integers(13, 50, shp3)).astype(np.int32)
    jt = rng.integers(1, 5, shp3).astype(np.int32)
    jt1 = rng.integers(1, 5, shp3).astype(np.int32)
    colamt = rng.uniform(1.0e18, 1.0e24, (MAXGAS, nlay, jm, im)).astype(f32)
    rfrate = rng.uniform(0.5, 2.0, (2, NRATES, nlay, jm, im)).astype(f32)
    fr = lambda: rng.uniform(0.0, 1.0, shp3).astype(f32)  # noqa: E731
    fac00, fac01, fac10, fac11 = fr(), fr(), fr(), fr()
    selffac, selffrac, forfac, forfrac = fr(), fr(), fr(), fr()
    indself = rng.integers(1, 10, shp3).astype(np.int32)   # Fortran inds in 1..9 (indsp <= 10)
    indfor = rng.integers(1, 4, shp3).astype(np.int32)     # Fortran indf in 1..3 (indfp <= 4)
    return {"laytrop": lt, "colamt": colamt, "rfrate": rfrate, "jp": jp, "jt": jt, "jt1": jt1,
            "fac00": fac00, "fac01": fac01, "fac10": fac10, "fac11": fac11,
            "selffac": selffac, "selffrac": selffrac, "indself": indself,
            "forfac": forfac, "forfrac": forfrac, "indfor": indfor}


# CO2 stratospheric-cooling factors, Fortran taug(ns04+8..14,k), 1-based g 8..14 -> zero-based 7..13.
_CO2_SCALE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.88, 1.07, 1.1, 0.99, 0.88, 0.943)


def _fortran_select(specparm, fs, ind, eight_branch: bool):
    """Index and fraction selection of the Fortran lower loop (eight_branch=True, radlw_main_nmmb.f:4140-4190)
    and the upper loop has no selection (eight_branch=False: fk0=1-fs, fk1=fs, ids as written).
    Returns (fk0, fk1, fk2, ids) with ids in Fortran order (id000, id010, id100, id110, id200, id210)."""
    if not eight_branch:
        return 1.0 - fs, fs, 0.0, (ind, ind + 5, ind + 1, ind + 6)
    if specparm < 0.125:
        p = fs - np.float32(1.0)
        p4 = p ** 4
        return p4, np.float32(1.0) - p - np.float32(2.0) * p4, p + p4, (
            ind, ind + 9, ind + 1, ind + 10, ind + 2, ind + 11)
    if specparm > 0.875:
        p = -fs
        p4 = p ** 4
        return p4, np.float32(1.0) - p - np.float32(2.0) * p4, p + p4, (
            ind + 1, ind + 10, ind, ind + 9, ind - 1, ind + 8)
    return np.float32(1.0) - fs, fs, np.float32(0.0), (ind, ind + 9, ind + 1, ind + 10, ind, ind)


def reference_column(h: dict, j: int, i: int, tab: dict, refrat: tuple):
    """Direct transcription of the Fortran taugb04 loops (radlw_main_nmmb.f:4098-4337) for one column,
    in float32 numpy. Returns (taug, fracs), each [nlay, NG04], Fortran g order (band-local).
    Each k-table row access tab[name][idx-1] is the Fortran (ig, idx) column for all 14 g at once."""
    f = np.float32
    nlay = h["jp"].shape[0]
    lt = int(h["laytrop"][j, i])
    ra, rb = f(refrat[0]), f(refrat[1])
    taug = np.zeros((nlay, NG04), dtype=f)
    fracs = np.zeros((nlay, NG04), dtype=f)
    absa, absb = tab["absa"], tab["absb"]
    selfref, forref = tab["selfref"], tab["forref"]
    fra, frb = tab["fracrefa"], tab["fracrefb"]
    for k in range(nlay):
        kf = k + 1
        c1 = f(h["colamt"][0, k, j, i])
        c2 = f(h["colamt"][1, k, j, i])
        c3 = f(h["colamt"][2, k, j, i])
        rf = h["rfrate"][:, :, k, j, i]          # [2, 6]: rfrate(k, n, m) = rf[m-1, n-1]
        jp = int(h["jp"][k, j, i])
        jt = int(h["jt"][k, j, i])
        jt1 = int(h["jt1"][k, j, i])
        f00, f01 = f(h["fac00"][k, j, i]), f(h["fac01"][k, j, i])
        f10, f11 = f(h["fac10"][k, j, i]), f(h["fac11"][k, j, i])
        sf, sfr = f(h["selffac"][k, j, i]), f(h["selffrac"][k, j, i])
        ff, ffr = f(h["forfac"][k, j, i]), f(h["forfrac"][k, j, i])
        inds = int(h["indself"][k, j, i])
        indf = int(h["indfor"][k, j, i])
        indsp, indfp = inds + 1, indf + 1
        if kf <= lt:
            # lower loop: rfrate(k,1,1), rfrate(k,1,2), refrat_planck_a, multiplier 8.0, nspa(4)
            sc = c1 + f(rf[0, 0]) * c2
            sp = c1 / sc
            sm = f(8.0) * min(sp, ONEMINUS)
            js = 1 + int(sm)
            fs = sm - f(int(sm))
            ind0 = ((jp - 1) * 5 + (jt - 1)) * NSPA4 + js
            sc1 = c1 + f(rf[1, 0]) * c2
            sp1 = c1 / sc1
            sm1 = f(8.0) * min(sp1, ONEMINUS)
            js1 = 1 + int(sm1)
            fs1 = sm1 - f(int(sm1))
            ind1 = (jp * 5 + (jt1 - 1)) * NSPA4 + js1
            scp = c1 + ra * c2
            spp = c1 / scp
            smp = f(8.0) * min(spp, ONEMINUS)
            jpl = 1 + int(smp)
            fpl = smp - f(int(smp))
            fk0, fk1, fk2, ids0 = _fortran_select(sp, fs, ind0, True)
            g0, g1, g2, ids1 = _fortran_select(sp1, fs1, ind1, True)
            # fac000=fk0*fac00, fac100=fk1*fac00, fac200=fk2*fac00, fac010=fk0*fac10, ... (id order m)
            fk = (f(fk0), f(fk1), f(fk2))
            fk_1 = (f(g0), f(g1), f(g2))
            tau_major = np.zeros(NG04, dtype=f)
            tau_major1 = np.zeros(NG04, dtype=f)
            for m in range(6):
                tau_major = tau_major + (fk[m // 2] * (f00 if m % 2 == 0 else f10)) * absa[ids0[m] - 1]
                tau_major1 = tau_major1 + (fk_1[m // 2] * (f01 if m % 2 == 0 else f11)) * absa[ids1[m] - 1]
            tau_major = sc * tau_major
            tau_major1 = sc1 * tau_major1
            tauself = sf * (selfref[inds - 1] + sfr * (selfref[indsp - 1] - selfref[inds - 1]))
            taufor = ff * (forref[indf - 1] + ffr * (forref[indfp - 1] - forref[indf - 1]))
            taug[k] = tau_major + tau_major1 + tauself + taufor
            fracs[k] = fra[jpl - 1] + f(fpl) * (fra[jpl] - fra[jpl - 1])
        else:
            # upper loop: rfrate(k,6,1), rfrate(k,6,2), refrat_planck_b, multiplier 4.0, nspb(4)
            sc = c3 + f(rf[0, 5]) * c2
            sp = c3 / sc
            sm = f(4.0) * min(sp, ONEMINUS)
            js = 1 + int(sm)
            fs = sm - f(int(sm))
            ind0 = ((jp - 13) * 5 + (jt - 1)) * NSPB4 + js
            sc1 = c3 + f(rf[1, 5]) * c2
            sp1 = c3 / sc1
            sm1 = f(4.0) * min(sp1, ONEMINUS)
            js1 = 1 + int(sm1)
            fs1 = sm1 - f(int(sm1))
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * NSPB4 + js1
            scp = c3 + rb * c2
            spp = c3 / scp
            smp = f(4.0) * min(spp, ONEMINUS)
            jpl = 1 + int(smp)
            fpl = smp - f(int(smp))
            _, _, _, ids0 = _fortran_select(sp, fs, ind0, False)
            _, _, _, ids1 = _fortran_select(sp1, fs1, ind1, False)
            fk = (f(1.0) - f(fs), f(fs))
            fk_1 = (f(1.0) - f(fs1), f(fs1))
            tau_major = np.zeros(NG04, dtype=f)
            tau_major1 = np.zeros(NG04, dtype=f)
            for m in range(4):
                tau_major = tau_major + (fk[m // 2] * (f00 if m % 2 == 0 else f10)) * absb[ids0[m] - 1]
                tau_major1 = tau_major1 + (fk_1[m // 2] * (f01 if m % 2 == 0 else f11)) * absb[ids1[m] - 1]
            tg = tau_major * sc + tau_major1 * sc1
            for g0, factor in enumerate(_CO2_SCALE):
                tg[g0] = tg[g0] * f(factor)
            taug[k] = tg
            fracs[k] = frb[jpl - 1] + f(fpl) * (frb[jpl] - frb[jpl - 1])
    return taug, fracs


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, band: int = 32, seed: int = 1, ref_cols: int = 3):
    """One pass over a full-size random but plausible state (954 x 835 x 60), in row bands of `band`
    rows at full width and full layer count (the card is shared; a full-domain resident state does not
    fit beside the other tenants). Each cell is computed exactly as in a full-domain launch: the kernel
    only sees the band's JM for its strides. Asserts no NaN/Inf in every output, and that band-4 rows of
    a few columns per band match a float32 transcription of the Fortran loops (reference_column) to 1e-4
    of the column's peak value. Uses the real band-4 tables and refrat.
    Returns (min/max per output, refrat, worst relative error against the Fortran transcription)."""
    import cupy as cp

    rng = np.random.default_rng(seed)

    class _Ctx:  # minimal stand-in for the table cache
        tables: dict = {}

    tab = _tables(_Ctx)
    refrat = _refrat(_Ctx)
    tab_np = {k: cp.asnumpy(v) for k, v in tab.items()}
    lo = {k: np.inf for k in ("taug04", "fracs04")}
    hi = {k: -np.inf for k in lo}
    worst = 0.0
    for j0 in range(0, jm, band):
        jb = min(band, jm - j0)
        host = _random_band(rng, nlay, jb, im)
        inp = {k: cp.asarray(v) for k, v in host.items()}
        out = alloc_outputs(nlay, jb, im)
        taugb04(inp, tab, refrat, out)
        cp.cuda.Stream.null.synchronize()
        for k, a in out.items():
            assert bool(cp.isfinite(a).all()), f"NaN/Inf in taugb04 output {k} rows {j0}..{j0 + jb - 1}"
            lo[k] = min(lo[k], float(a.min()))
            hi[k] = max(hi[k], float(a.max()))
        cols = [(0, 0), (jb - 1, im - 1), (jb // 2, im // 2)][:ref_cols]
        for j, i in cols:
            ref_t, ref_f = reference_column(host, j, i, tab_np, refrat)
            for name, ref in (("taug04", ref_t), ("fracs04", ref_f)):
                kern = cp.asnumpy(out[name][:, NS04:NS04 + NG04, j, i])
                scale = float(np.max(np.abs(ref)))
                err = float(np.max(np.abs(kern - ref)))
                assert err <= 1e-4 * scale, (
                    f"taugb04 {name} band-row {j0 + j},{i}: max |kernel-Fortran| {err:.6g} > 1e-4*{scale:.6g}")
                worst = max(worst, err / scale)
    return {k: (lo[k], hi[k]) for k in lo}, refrat, worst
