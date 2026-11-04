# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-rtrn: rtrn, the clear and cloudy RRTM longwave transfer (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:2127-2484 (subroutine rtrn).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the NOAA header of radlw_main_nmmb.f is kept as the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_lw_rtrn.cu, kernel phys_rrtm_lw_rtrn. One launch, one thread per
(i, j) column (kernels.grid_cols). Band order, layer mapping and array layouts are documented in the
kernel header; in short, Fortran layer k (RRTM order, 1 = surface) is NAM slot LM - k, the mapping
lwrad uses for its ivflip == 0 input (radlw_main_nmmb.f, k1 = nlp1 - k).

Caller: lwrad (phys-rrtm-lw-lwrad) calls rtrn when isubclw <= 0 and iovrlw <= 0 (the NAM setting).
rtrnmr and rtrnmc are separate units. The tables tau_tbl, exp_tbl, tfn_tbl and the band map ngb come from
phys-rrtm-lw-rlwinit and phys-rrtm-lw-params; run(ctx) reads them from ctx.tables under the names
below and fails with a named error if they have not landed.

Arrays (ctx.st.fields, CONTRACT names, device float32 unless noted):
  inputs   rlw_semiss [NBANDS, JM, IM]          rlw_delp [LM, JM, IM]  (layer pressure thickness, MB)
           rlw_cldfrc [LM, JM, IM]              rlw_taucld [NBANDS, LM, JM, IM]
           rlw_tautot [NGPTLW, LM, JM, IM]      rlw_pklay, rlw_pklev [NBANDS, LM+1, JM, IM]
           rlw_fracs [NGPTLW, LM, JM, IM]       rlw_secdif [NBANDS, JM, IM]
  tables   ctx.tables["rrtm_lw_ngb"] int32 [NGPTLW] (1-based band index of each g-point), written by
           phys-rrtm-lw-params (its run() fills it; the declared dependency of this unit)
           ctx.tables["rrtm_lw_tau_tbl"], ["rrtm_lw_exp_tbl"], ["rrtm_lw_tfn_tbl"] [NTBL+1]
  outputs  rlw_totuflux, rlw_totdflux, rlw_totuclfl, rlw_totdclfl [LM+1, JM, IM]
           rlw_htr, rlw_htrcl [LM, JM, IM]      rlw_htrb [NBANDS, LM, JM, IM] (only when lhlwb)
Config: cfg.lhlw0, cfg.lhlwb (bools), cfg.ilwrate (int; 1 = K per day, else K per second).
Constants: fluxfac = pi * 2e4 (radlw_main_nmmb.f:1286-1287); heatfac from rlwinit (lines 1290-1295)
with con_g = 9.80665 and con_cp = 1.0046e3 (physcons.f:46, :59).

Units. rlw_delp is in MB (hPa): the unit of the Fortran rtrn delp (radlw_main_nmmb.f:2148 and the
rfdelp = heatfac / delp line, :2456). heatfac = g*1e-2/cp (g*864/cp for K/day) carries the mb-to-Pa
factor. The whole LW chain carries mb: lwrad's plvl = 10*prsi (grrad_nmmb.f:1253, :1290, prsi in cb),
lwrad's delp is a plvl difference (radlw_main_nmmb.f:678, :786), and phys-rrtm-lw-rtrnmc takes delp in mb.
No conversion sits between lwrad and rtrn. A delp in Pa would make htr 100x too small, in cb 10x too
small. run() passes delp through unchanged; selftest draws it from 0.5 to 30 mb, a layer-thickness range
only mb produces.

selftest() runs the full-size 954 x 835 x 60 case on random inputs once and checks for NaN/Inf.
The ngb partition in the selftest is the RRTM-LW g-point count per band (10, 12, 16, 14, 16, 8, 12, 8,
12, 6, 8, 8, 4, 2, 2, 2 = 140), a stand-in until phys-rrtm-lw-params lands the real table.
"""
from __future__ import annotations


import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_rtrn"
KERNEL = "phys_rrtm_lw_rtrn"
NBANDS = 16                 # radlw_param_nmmb.f:140
NGPTLW = 140                # radlw_param_nmmb.f:141
NTBL = 10000                # radlw_param.f:96
LMAX = 60                   # kernel per-thread band scratch; NAM 12 km parent LM = 60
CON_G = 9.80665             # physcons.f:46
CON_CP = 1.0046e3           # physcons.f:59

TABLE_NAMES = ("rrtm_lw_ngb", "rrtm_lw_tau_tbl", "rrtm_lw_exp_tbl", "rrtm_lw_tfn_tbl")
INPUT_NAMES = ("rlw_semiss", "rlw_delp", "rlw_cldfrc", "rlw_taucld", "rlw_tautot",
               "rlw_pklay", "rlw_pklev", "rlw_fracs", "rlw_secdif")
OUTPUT_NAMES = ("rlw_totuflux", "rlw_totdflux", "rlw_totuclfl", "rlw_totdclfl",
                "rlw_htr", "rlw_htrcl", "rlw_htrb")


def fluxfac() -> float:
    """pi * 2e4 (radlw_main_nmmb.f:1286-1287)."""
    return float(2.0 * np.arcsin(1.0) * 2.0e4)


def heatfac(ilwrate: int) -> float:
    """radlw_main_nmmb.f:1290-1295: K per day when ilwrate == 1, else K per second."""
    if ilwrate == 1:
        return CON_G * 864.0 / CON_CP
    return CON_G * 1.0e-2 / CON_CP


def rtrn(semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs, secdif, ngb,
         tau_tbl, exp_tbl, tfn_tbl, totuflux, totdflux, totuclfl, totdclfl, htr, htrcl, htrb,
         fluxfac_v: float, heatfac_v: float, lhlw0: bool, lhlwb: bool, stream=None):
    """Launch phys_rrtm_lw_rtrn on device arrays with the shapes documented above.

    delp is layer pressure thickness in mb (module docstring, Units). The sizes come from delp:
    LM = delp.shape[0], JM, IM from its last two axes. htrb is only
    written when lhlwb; pass any float32 array of the documented shape otherwise.
    """
    lm, jm, im = (int(s) for s in delp.shape)
    if lm > LMAX:
        raise ValueError(f"phys-rrtm-lw-rtrn: LM = {lm} exceeds the kernel scratch limit {LMAX}")
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else _NullScope()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs, secdif, ngb,
           tau_tbl, exp_tbl, tfn_tbl,
           totuflux, totdflux, totuclfl, totdclfl, htr, htrcl, htrb,
           np.float32(fluxfac_v), np.float32(heatfac_v),
           np.int32(int(lhlw0)), np.int32(int(lhlwb)),
           np.int32(im), np.int32(jm), np.int32(lm)))


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _out(st, name, shape):
    """Output array: allocated once on first use, then reused every call (no per-step allocation)."""
    a = st.fields.get(name)
    if a is None:
        a = st.add(name, shape)
    return a


def run(ctx):
    """Driver entry: reads the inputs named above from ctx.st, writes the outputs back into ctx.st.

    Raises RuntimeError naming the missing array or table when a dependency unit has not landed.
    """
    st = ctx.st
    grid = ctx.grid
    cfg = ctx.cfg
    lm = int(grid.LM)
    jm = int(grid.JM)
    im = int(grid.IM)
    lmk = lm + 1
    lhlw0 = bool(getattr(cfg, "lhlw0", False))
    lhlwb = bool(getattr(cfg, "lhlwb", False))
    ilwrate = int(getattr(cfg, "ilwrate", 1))

    inp = {}
    for n in INPUT_NAMES:
        a = st.fields.get(n)
        if a is None:
            raise RuntimeError(f"phys-rrtm-lw-rtrn needs ctx.st.fields[{n!r}] (unit not landed)")
        inp[n] = a
    tab = {}
    for n in TABLE_NAMES:
        a = ctx.tables.get(n)
        if a is None:
            raise RuntimeError(f"phys-rrtm-lw-rtrn needs ctx.tables[{n!r}] (phys-rrtm-lw-params or "
                               f"phys-rrtm-lw-rlwinit not landed)")
        tab[n] = a

    outs = {
        "rlw_totuflux": _out(st, "rlw_totuflux", (lmk, jm, im)),
        "rlw_totdflux": _out(st, "rlw_totdflux", (lmk, jm, im)),
        "rlw_totuclfl": _out(st, "rlw_totuclfl", (lmk, jm, im)),
        "rlw_totdclfl": _out(st, "rlw_totdclfl", (lmk, jm, im)),
        "rlw_htr": _out(st, "rlw_htr", (lm, jm, im)),
        "rlw_htrcl": _out(st, "rlw_htrcl", (lm, jm, im)),
        "rlw_htrb": _out(st, "rlw_htrb", (NBANDS, lm, jm, im)),
    }
    rtrn(inp["rlw_semiss"], inp["rlw_delp"], inp["rlw_cldfrc"], inp["rlw_taucld"],
         inp["rlw_tautot"], inp["rlw_pklay"], inp["rlw_pklev"], inp["rlw_fracs"],
         inp["rlw_secdif"], tab["rrtm_lw_ngb"], tab["rrtm_lw_tau_tbl"], tab["rrtm_lw_exp_tbl"],
         tab["rrtm_lw_tfn_tbl"],
         outs["rlw_totuflux"], outs["rlw_totdflux"], outs["rlw_totuclfl"], outs["rlw_totdclfl"],
         outs["rlw_htr"], outs["rlw_htrcl"], outs["rlw_htrb"],
         fluxfac(), heatfac(ilwrate), lhlw0, lhlwb, stream=ctx.stream)
    return outs


def selftest(seed: int = 1, rows_per_chunk: int = 84):
    """Full-size 954 x 835 x 60 domain on random plausible inputs, NaN/Inf check.

    The g-resolved arrays (tautot, fracs: 140 x 60 x 835 x 954 float32 = 26.7 GB each) do not fit a
    shared 32 GB card at full domain, so the domain is covered in row chunks of ``rows_per_chunk``
    (835 = 9 x 84 + 79). rtrn is column-local, so each chunk is the same kernel on the same columns
    as a full-domain launch; every one of the 954 x 835 columns is computed. Inputs are generated on
    the GPU in float32 (no host float64 copies).
    """
    import cupy as cp

    lm, jm_full, im = 60, 835, 954
    lmk = lm + 1
    cp.random.seed(seed)

    def f32(shape, lo, hi):
        a = cp.random.random(shape, dtype=cp.float32)
        return (a * cp.float32(hi - lo) + cp.float32(lo)).astype(cp.float32)

    tables = {
        "tau_tbl": f32((NTBL + 1,), 0.0, 20.0),
        "exp_tbl": f32((NTBL + 1,), 0.0, 1.0),
        "tfn_tbl": f32((NTBL + 1,), 0.0, 1.0),
    }
    band_counts = [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2]
    ngb = cp.asarray(np.repeat(np.arange(1, NBANDS + 1, dtype=np.int32), band_counts))
    assert ngb.shape[0] == NGPTLW

    bad_all = []
    maxima = {}
    for j0 in range(0, jm_full, rows_per_chunk):
        jm = min(rows_per_chunk, jm_full - j0)
        semiss = f32((NBANDS, jm, im), 0.90, 1.0)
        delp = f32((lm, jm, im), 0.5, 30.0)
        cldfrc = f32((lm, jm, im), 0.0, 1.0)
        cldfrc = cp.where(cldfrc < 0.5, cp.float32(0.0), cldfrc)
        taucld = f32((NBANDS, lm, jm, im), 0.0, 5.0)
        tautot = f32((NGPTLW, lm, jm, im), 0.0, 3.0)
        pklay = f32((NBANDS, lmk, jm, im), 50.0, 500.0)
        pklev = f32((NBANDS, lmk, jm, im), 50.0, 500.0)
        fracs = f32((NGPTLW, lm, jm, im), 0.0, 1.0)
        secdif = f32((NBANDS, jm, im), 1.5, 1.8)

        z = lambda shape: cp.zeros(shape, dtype=cp.float32)  # noqa: E731
        totuflux, totdflux = z((lmk, jm, im)), z((lmk, jm, im))
        totuclfl, totdclfl = z((lmk, jm, im)), z((lmk, jm, im))
        htr, htrcl = z((lm, jm, im)), z((lm, jm, im))
        htrb = z((NBANDS, lm, jm, im))
        rtrn(semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs, secdif, ngb,
             tables["tau_tbl"], tables["exp_tbl"], tables["tfn_tbl"],
             totuflux, totdflux, totuclfl, totdclfl, htr, htrcl, htrb,
             fluxfac(), heatfac(1), True, True)
        cp.cuda.Stream.null.synchronize()
        out = {"totuflux": totuflux, "totdflux": totdflux, "totuclfl": totuclfl,
               "totdclfl": totdclfl, "htr": htr, "htrcl": htrcl, "htrb": htrb}
        for n, a in out.items():
            if not bool(cp.isfinite(a).all()):
                bad_all.append((j0, n))
            maxima[n] = max(maxima.get(n, -np.inf), float(cp.nanmax(cp.abs(a))))
        del semiss, delp, cldfrc, taucld, tautot, pklay, pklev, fracs, secdif, out
        del totuflux, totdflux, totuclfl, totdclfl, htr, htrcl, htrb
    if bad_all:
        raise RuntimeError(f"phys-rrtm-lw-rtrn selftest: NaN/Inf (row0, field) {bad_all}")
    return maxima
