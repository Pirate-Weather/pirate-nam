# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol18: NMMB RRTM shortwave band 18 optical depths (taumol18) and Rayleigh depth.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3632-3738 (subroutine taumol18).
The sw-rrtm3 / AER header of that file is the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_sw_taumol18.cu, phys_rrtm_sw_taumol18 (one thread per column, one launch).
Band-local outputs (the global g-points are NS18 + g = 18..25, 0-based 18..25 in the 112-point list):
  taug  float32 [8, NLAY, JM, IM]   taug(k, g) for band 18, Fortran taug(:, k, NS18+g) with g = 0..7
  taur  float32 [NLAY, JM, IM]      Rayleigh optical depth; the Fortran writes the same value to all 8 g-points
  band_gpoint(out, jg)              the Fortran g-point planes for jg = 19..26 (NS18 = 18): taug[jg-19] and the
                                    shared taur array (same values, not copied per g-point)
Tables: namgpu/data/rrtm_sw/kgb18.npz (phys-rrtm-sw-kgb18, from tools/extract-rrtm-sw-kgb18.py):
  absa [585, 8], absb [235, 8], selfref [10, 8], forref [3, 8], rayl (scalar). Band constant strrat(18).

ctx.tables inputs (written by the upstream units, layouts in the kernel header):
  rrtm_sw_taumol18_in   dict: colamt [6,NLAY,JM,IM], colmol [NLAY,JM,IM], laytrop [JM,IM] int32,
                        jp, jt, jt1 [NLAY,JM,IM] int32, fac00, fac01, fac10, fac11 [NLAY,JM,IM] float32,
                        selffac, selffrac, forfac, forfrac [NLAY,JM,IM] float32,
                        indself, indfor [NLAY,JM,IM] int32   (setcoef outputs; gas and layer layout as CONTRACT)
  rrtm_sw_taumol18_tables  (optional) dict of device tables; loaded from kgb18.npz if absent.
Outputs: ctx.tables["rrtm_sw_taumol18_out"] = {"taug", "taur"}, allocated once and reused.

Fortran lines kept on purpose (see the kernel header): lower and upper loops overlap at k == laytrop in
the Fortran; this port takes k <= laytrop for the lower table and k > laytrop for the upper table.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol18"
KERNEL = "phys_rrtm_sw_taumol18"
NG18 = 8
MSA18, MSB18, MSF18, MFR18 = 585, 235, 10, 3
STRRAT18 = np.float32(3.89589e+1)        # strrat(18), radsw_datatb_nmmb.f:1108-1110 (single precision)
NS18 = 18                                # radsw_param_nmmb.f: band 18 planes are Fortran g-points NS18+1..NS18+8
KTAB = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb18.npz"

IN_KEY = "rrtm_sw_taumol18_in"
TAB_KEY = "rrtm_sw_taumol18_tables"
OUT_KEY = "rrtm_sw_taumol18_out"

_IN_FIELDS = ("fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac")
_IN_INT = ("jp", "jt", "jt1", "indself", "indfor")


def load_tables():
    """Device tables for band 18, from namgpu/data/rrtm_sw/kgb18.npz (float32, [ind-1, g-1] layout)."""
    import cupy as cp

    d = np.load(KTAB)
    absa, absb = d["absa"], d["absb"]
    selfref, forref = d["selfref"], d["forref"]
    if absa.shape != (MSA18, NG18) or absb.shape != (MSB18, NG18):
        raise ValueError(f"kgb18.npz: absa {absa.shape}, absb {absb.shape}")
    if selfref.shape != (MSF18, NG18) or forref.shape != (MFR18, NG18):
        raise ValueError(f"kgb18.npz: selfref {selfref.shape}, forref {forref.shape}")
    return {
        "absa": cp.asarray(absa.astype(np.float32)),
        "absb": cp.asarray(absb.astype(np.float32)),
        "selfref": cp.asarray(selfref.astype(np.float32)),
        "forref": cp.asarray(forref.astype(np.float32)),
        "rayl": np.float32(d["rayl"]),
    }


def alloc_outputs(nlay: int, jm: int, im: int):
    import cupy as cp

    return {
        "taug": cp.zeros((NG18, nlay, jm, im), dtype=cp.float32),
        "taur": cp.zeros((nlay, jm, im), dtype=cp.float32),
    }


def _check_shapes(inp: dict, tab: dict):
    nlay, jm, im = inp["colmol"].shape
    if tuple(inp["colamt"].shape) != (6, nlay, jm, im):
        raise ValueError(f"taumol18: colamt {tuple(inp['colamt'].shape)} != (6, {nlay}, {jm}, {im})")
    for name in _IN_FIELDS + _IN_INT:
        if tuple(inp[name].shape) != (nlay, jm, im):
            raise ValueError(f"taumol18: {name} shape {tuple(inp[name].shape)} != {(nlay, jm, im)}")
    if tuple(inp["laytrop"].shape) != (jm, im):
        raise ValueError("taumol18: laytrop must be [JM, IM]")
    if tab["absa"].shape != (MSA18, NG18) or tab["absb"].shape != (MSB18, NG18):
        raise ValueError("taumol18: k-table shapes")
    return nlay, jm, im


def taumol18(inp: dict, tab: dict, out: dict, stream=None):
    """One launch over every (i, j) column. All arrays are device arrays in CONTRACT layout."""
    nlay, jm, im = _check_shapes(inp, tab)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["colamt"], inp["colmol"], inp["laytrop"],
            inp["jp"], inp["jt"], inp["jt1"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["indself"],
            inp["forfac"], inp["forfrac"], inp["indfor"],
            tab["absa"], tab["absb"], tab["selfref"], tab["forref"],
            np.float32(tab["rayl"]), STRRAT18,
            out["taug"], out["taur"],
            np.int32(nlay), np.int32(jm), np.int32(im),
        ))
    return out


def band_gpoint(out: dict, jg: int):
    """Fortran g-point jg (1-based, 19..26) as this band's (taug, taur) planes, for the SW spectral sum."""
    g0 = jg - NS18 - 1
    if not 0 <= g0 < NG18:
        raise ValueError(f"taumol18: g-point {jg} is not in band 18 (19..26)")
    return out["taug"][g0], out["taur"]


def check_setcoef_ranges(inp: dict):
    """Host-side check, not on the step path. The kernel indexes selfref/forref without clamping, so the lower
    layers (k <= laytrop) must satisfy setcoef's range: indself in 1..9, indfor in 1..2."""
    import cupy as cp

    nlay = inp["colmol"].shape[0]
    kk = cp.arange(1, nlay + 1, dtype=cp.int32)[:, None, None]
    low = kk <= inp["laytrop"][None, :, :]
    s = inp["indself"][low]
    f = inp["indfor"][low]
    if s.size and (int(s.min()) < 1 or int(s.max()) > MSF18 - 1):
        raise ValueError("taumol18: indself outside 1..9 in lower layers")
    if f.size and (int(f.min()) < 1 or int(f.max()) > MFR18 - 1):
        raise ValueError("taumol18: indfor outside 1..2 in lower layers")


def run(ctx):
    """Driver entry. Upstream units fill ctx.tables[IN_KEY] (setcoef, gas and cldprop outputs)."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_sw_taumol18 needs ctx.tables[{IN_KEY!r}] from the SW driver")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        tab = load_tables()
        ctx.tables[TAB_KEY] = tab
    nlay, jm, im = inp["colmol"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or tuple(out["taug"].shape) != (NG18, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol18(inp, tab, out, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """One launch on a full-size plausible state (954 x 835 x 60). Asserts no NaN/Inf and that the
    Rayleigh depth equals colmol * rayl everywhere."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    f32 = np.float32
    shape3 = (nlay, jm, im)
    inp = {
        "colamt": cp.asarray(rng.uniform(1e-6, 1e3, (6, nlay, jm, im)).astype(f32)),
        "colmol": cp.asarray(rng.uniform(1e18, 1e25, shape3).astype(f32)),
        "laytrop": cp.asarray(rng.integers(1, nlay + 1, (jm, im)).astype(np.int32)),
        "jp": cp.asarray(rng.integers(1, 59, shape3).astype(np.int32)),
        "jt": cp.asarray(rng.integers(1, 5, shape3).astype(np.int32)),
        "jt1": cp.asarray(rng.integers(1, 5, shape3).astype(np.int32)),
        "fac00": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "fac01": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "fac10": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "fac11": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "selffac": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "selffrac": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "indself": cp.asarray(rng.integers(1, 10, shape3).astype(np.int32)),
        "forfac": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "forfrac": cp.asarray(rng.uniform(0, 1, shape3).astype(f32)),
        "indfor": cp.asarray(rng.integers(1, 3, shape3).astype(np.int32)),
    }
    # H2O = 0 and CH4 = 0 in a 16 x 16 column patch: speccomb = 0 in the lower layers there (the NaN case).
    inp["colamt"][0, :, :16, :16] = 0.0
    inp["colamt"][4, :, :16, :16] = 0.0
    check_setcoef_ranges(inp)
    tab = load_tables()
    out = alloc_outputs(nlay, jm, im)
    taumol18(inp, tab, out)
    cp.cuda.Stream.null.synchronize()
    for name in ("taug", "taur"):
        assert bool(cp.isfinite(out[name]).all()), f"NaN/Inf in taumol18 output {name}"
    kk = cp.arange(1, nlay + 1, dtype=cp.int32)[:, None, None]
    zero_lower = (inp["colamt"][0] == 0) & (inp["colamt"][4] == 0) & (kk <= inp["laytrop"][None, :, :])
    assert bool(zero_lower.any()), "selftest patch has no zero-gas lower layers"
    assert bool((out["taug"][:, zero_lower] == 0).all()), "taug != 0 where H2O and CH4 are both zero"
    tg, tr = band_gpoint(out, 26)
    assert bool(cp.array_equal(tg, out["taug"][7])) and tr is out["taur"], "band_gpoint offset"
    want_r = inp["colmol"] * tab["rayl"]
    assert bool(cp.allclose(out["taur"], want_r, rtol=1e-6)), "taur != colmol * rayl"
    return {"taug_mean": float(out["taug"].mean()), "taur_mean": float(out["taur"].mean())}
