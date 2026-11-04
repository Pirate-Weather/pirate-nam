# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-lwrad: the LW driver per column (lwrad) of the NMMB RRTM LW package.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:352-1139 (subroutine lwrad). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the package header and the AER notice are kept in the kernel file
namgpu/kernels/phys_rrtm_lw_lwrad.cu.

Kernels (namgpu/kernels/phys_rrtm_lw_lwrad.cu, one launch each, one thread per (i, j) column):
  lwrad_prep   the lwrad statements before the scheme calls (ivflip branches, absorber amounts,
               rare gases, tauaer, cloud unpack, cldfrc padding, pwvcm, colbrd, secdiff, lcf1,
               semiss, stemp, ipseed, and the lcf1 == 0 zeroing of taucld and cldfmc).
  lwrad_post   the lwrad statements after the scheme calls (topflw, sfcflw, flxprf, hlwc, hlw0, hlwb).

The scheme calls between them are separate units and are NOT here. lwrad calls, in order:
  cldprop  (lcf1 true only)      unit phys-rrtm-lw-cldprop      namgpu.physics.rrtm.phys_rrtm_lw_cldprop
           The cldprop unit is launched for every column and must leave lcf1 == 0 columns untouched
           (lwrad_prep already zeroed them).
  setcoef                        unit phys-rrtm-lw-setcoef      namgpu.physics.rrtm.phys_rrtm_lw_setcoef
  taumol                         unit phys-rrtm-lw-taumol       namgpu.physics.rrtm.phys_rrtm_lw_taumol
  transfer (rtrn_unit)           rtrn, rtrnmr or rtrnmc (Fortran lwrad dispatch, radlw_main_nmmb.f:1020-1054):
                                   isubclw <= 0, iovrlw <= 0: phys_rrtm_lw_rtrn   (cldfrc)
                                   isubclw <= 0, iovrlw >  0: phys_rrtm_lw_rtrnmr (cldfrc)
                                   isubclw >  0:              phys_rrtm_lw_rtrnmc (cldfmc)
They read the scratch written by lwrad_prep and write totuflux, totdflux, totuclfl, totdclfl [nlp1],
htr, htrcl [nlay], htrb [NBDLW, nlay] per column, which lwrad_post reads.

Scratch (chunk-local, allocated once per chunk shape in ctx.tables): see SCRATCH below.

Chunking: the domain is processed in row chunks of JC rows (default 64). Inputs and outputs are full
domain arrays (global strides); the kernel gets jstart and jc, so scratch is about 2.6 KB per column
(about 0.65 GB for one 64-row chunk at 954 columns). cldfmc [NGPTLW, nlay] is allocated only when
isubclw > 0 (it is read only by rtrnmc), so the NAM route (isubclw = 0) does not hold it.

Inputs (ctx.st names; Fortran lwrad arguments, grrad_nmmb call site 1992 and 2031):
  plyr, tlyr, qlyr, olyr   [LMK, JM, IM]         layer mean pressure (mb), temperature, q, ozone
  plvl, tlvl               [LMP, JM, IM]         interface pressure (mb), temperature
  gasvmr                   [NF_VGAS=10, LMK, JM, IM]   producer phys-rrtm-getgases
  clouds                   [NF_CLDS=9, LMK, JM, IM]    producer phys-rrtm-grrad-clouds
  sfcemis                  [JM, IM]              producer phys-rrtm-setemis
  tsfg                     [JM, IM]              grrad caller (grrad.f:1094-1101, tsfg = tsfc)
  icsdlw                   [JM, IM] int32        grrad caller (grrad.f:634), used when isubclw == 2
  faerlw                   [NF_AELW=3, NBDLW=16, LMK, JM, IM]  ctx.tables["rrtm_faerlw"], producer
                                                  phys-rrtm-setaer
A missing input is refused by name with its producer, never zero-filled.
Outputs written by lwrad_post (ctx.st names, grrad_nmmb uses these): htlwc [LMK, JM, IM],
htlwb [NBDLW, LMK, JM, IM] (when hlwb wanted), topflw_upfxc, topflw_upfx0,
sfcflw_upfxc, sfcflw_upfx0, sfcflw_dnfxc, sfcflw_dnfx0 [JM, IM], htlw0 [LMK, JM, IM] and
flwprf [4, LMP, JM, IM] when those are wanted.

Constants (NAM runtime and NMMB defaults, physparam.f): ivflip = 0 (TOA->surface input, NAM,
module_SOLVER_GRID_COMP.F90:10714), ilwrgas = 1 (physparam.f:73), ilwcliq = 1 (physparam.f:75),
isubclw = 0 (physparam.f:166), iovrlw = 1 (physparam.f:144), ipsdlw0 = ngptlw = 140
(radlw_param_nmmb.f:141).
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_lwrad"
NBDLW = 16
NF_CLDS = 9
NF_AELW = 3
NF_VGAS = 10
MAXGAS = 7
MAXXSEC = 4
NGPTLW = 140        # NGPTLW (radlw_param_nmmb.f:141)
IVFLIP = 0          # NAM runtime: module_SOLVER_GRID_COMP.F90:10714 IFLIP=0 (TOA->sfc input)
ILWRGAS = 1         # physparam.f:73
ILWCLIQ = 1         # physparam.f:75
ISUBCLW = 0         # physparam.f:166
IOVRLW = 1          # physparam.f:144
JC_DEFAULT = 64     # rows per chunk

# Seam units run in this order by run(); the transfer unit is chosen by rtrn_unit().
SEAM_BEFORE = (
    "namgpu.physics.rrtm.phys_rrtm_lw_cldprop",
    "namgpu.physics.rrtm.phys_rrtm_lw_setcoef",
    "namgpu.physics.rrtm.phys_rrtm_lw_taumol",
)

# Chunk scratch names and per-column shapes (level counts are nlay or nlp1; see the kernel header).
SCRATCH = (
    ("lwrad_pavel", "nlay"), ("lwrad_tavel", "nlay"), ("lwrad_delp", "nlay"),
    ("lwrad_tz", "nlp1"), ("lwrad_coldry", "nlay"), ("lwrad_colbrd", "nlay"),
    ("lwrad_colamt", "7*nlay"), ("lwrad_wx", "4*nlay"), ("lwrad_tauaer", "16*nlay"),
    ("lwrad_cldfrc", "nlp1+1"), ("lwrad_cldin", "8*nlay"), ("lwrad_taucld", "16*nlay"),
    ("lwrad_secdiff", "16"), ("lwrad_semiss", "16"),
    ("lwrad_pwvcm", "1"), ("lwrad_stemp", "1"),
    ("lwrad_ipseed", "1i"), ("lwrad_lcf1", "1i"),
)
# Read only by rtrnmc (isubclw > 0); not allocated on the NAM route.
SCRATCH_SUBCOL = (("lwrad_cldfmc", "140*nlay"),)


def rtrn_unit(isubclw: int = ISUBCLW, iovrlw: int = IOVRLW):
    """Fortran lwrad transfer dispatch (radlw_main_nmmb.f:1020-1054).

    Returns (module, cloud array the unit reads). isubclw <= 0 passes cldfrc to rtrn (iovrlw <= 0) or
    rtrnmr (iovrlw > 0); isubclw > 0 passes cldfmc to rtrnmc.
    """
    if isubclw <= 0:
        mod = "phys_rrtm_lw_rtrn" if iovrlw <= 0 else "phys_rrtm_lw_rtrnmr"
        return f"namgpu.physics.rrtm.{mod}", "cldfrc"
    return "namgpu.physics.rrtm.phys_rrtm_lw_rtrnmc", "cldfmc"


def _dep(module: str):
    """A seam unit that is not landed fails loudly by name, it is never skipped silently."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(
            f"phys-rrtm-lw-lwrad needs {module} (unit not landed): {e}") from e


def _chunk_scratch(xp, nlay: int, nlp1: int, jc: int, im: int, isubclw: int = ISUBCLW):
    """Chunk scratch, allocated once per shape. cldfmc is allocated only when isubclw > 0."""
    sizes = {"nlay": nlay, "nlp1": nlp1}
    specs = list(SCRATCH) + (list(SCRATCH_SUBCOL) if isubclw > 0 else [])
    out = {}
    for name, spec in specs:
        if spec.endswith("i"):
            n = int(spec[:-1])
            out[name] = xp.zeros((n, jc, im), dtype=xp.int32)
            continue
        if "*" in spec:
            m, lev = spec.split("*")
            per = int(m) * sizes[lev]
            out[name] = xp.zeros((per, jc, im), dtype=xp.float32)
        elif spec in sizes:
            out[name] = xp.zeros((sizes[spec], jc, im), dtype=xp.float32)
        elif spec.endswith("+1"):
            out[name] = xp.zeros((sizes[spec[:-2]] + 1, jc, im), dtype=xp.float32)
        else:
            out[name] = xp.zeros((int(spec), jc, im), dtype=xp.float32)
    return out


def prep(a: dict, s: dict, nlay: int, nlp1: int, jm: int, im: int, jstart: int, jc: int,
         stream=None, isubclw: int = ISUBCLW):
    """lwrad statements before the scheme calls, chunk rows [jstart, jstart + jc).

    Includes the lcf1 == 0 zeroing of taucld and cldfmc (Fortran radlw_main_nmmb.f:942-944)."""
    import cupy as cp

    k = kernels.get(UNIT, "lwrad_prep")
    scope = stream if stream is not None else contextlib.nullcontext()
    n = im * jc
    grid = ((n + 127) // 128, 1, 1)
    cldfmc = s["lwrad_cldfmc"] if isubclw > 0 else cp.zeros(1, dtype=cp.float32)
    with scope:
        k(grid, (128, 1, 1),
          (a["plyr"], a["plvl"], a["tlyr"], a["tlvl"], a["qlyr"], a["olyr"],
           a["gasvmr"], a["clouds"], a["faerlw"], a["sfcemis"], a["tsfg"], a["icsdlw"],
           s["lwrad_pavel"], s["lwrad_tavel"], s["lwrad_delp"], s["lwrad_tz"],
           s["lwrad_coldry"], s["lwrad_colbrd"], s["lwrad_colamt"], s["lwrad_wx"], s["lwrad_tauaer"],
           s["lwrad_cldfrc"], s["lwrad_cldin"], s["lwrad_taucld"], cldfmc,
           s["lwrad_secdiff"], s["lwrad_semiss"],
           s["lwrad_pwvcm"], s["lwrad_stemp"], s["lwrad_ipseed"], s["lwrad_lcf1"],
           np.int32(nlay), np.int32(nlp1), np.int32(jm), np.int32(im),
           np.int32(jstart), np.int32(jc),
           np.int32(IVFLIP), np.int32(ILWRGAS), np.int32(ILWCLIQ), np.int32(isubclw),
           np.int32(NGPTLW)))


def post(t: dict, o: dict, nlay: int, nlp1: int, jm: int, im: int, jstart: int, jc: int,
         lhlw0: bool, lhlwb: bool, lflxprf: bool, stream=None):
    """lwrad statements after the scheme calls. t: transfer-unit scratch (totuflux, totdflux,
    totuclfl, totdclfl [nlp1, JC, IM]; htr, htrcl [nlay, JC, IM]; htrb [NBDLW, nlay, JC, IM]).
    o: global outputs (topflw_upfxc, topflw_upfx0, sfcflw_upfxc, sfcflw_upfx0, sfcflw_dnfxc,
    sfcflw_dnfx0 [JM, IM]; hlwc [nlay, JM, IM]; hlw0 [nlay, JM, IM] or None; hlwb [NBDLW, nlay, JM, IM]
    or None; flxprf [4, nlp1, JM, IM] or None)."""
    import cupy as cp

    k = kernels.get(UNIT, "lwrad_post")
    scope = stream if stream is not None else contextlib.nullcontext()
    one = cp.zeros(1, dtype=cp.float32)
    n = im * jc
    grid = ((n + 127) // 128, 1, 1)
    with scope:
        k(grid, (128, 1, 1),
          (t["totuflux"], t["totdflux"], t["totuclfl"], t["totdclfl"],
           t["htr"], t["htrcl"], t["htrb"],
           o["topflw_upfxc"], o["topflw_upfx0"], o["sfcflw_upfxc"], o["sfcflw_upfx0"],
           o["sfcflw_dnfxc"], o["sfcflw_dnfx0"],
           o["hlwc"], o["hlw0"] if lhlw0 else one, o["hlwb"] if lhlwb else one,
           o["flxprf"] if lflxprf else one,
           np.int32(nlay), np.int32(nlp1), np.int32(jm), np.int32(im),
           np.int32(jstart), np.int32(jc),
           np.int32(IVFLIP), np.int32(int(lhlw0)), np.int32(int(lhlwb)), np.int32(int(lflxprf))))


def _need(st, name: str, shape, producer: str, dtype=None):
    """A producer's array, checked by name, shape and dtype. Missing is a named error, never a zero fill."""
    arr = st.fields.get(name)
    if arr is None:
        raise KeyError(f"phys-rrtm-lw-lwrad: ctx.st['{name}'] is missing (written by {producer})")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"phys-rrtm-lw-lwrad: ctx.st['{name}'] shape {tuple(arr.shape)}, expected {tuple(shape)}")
    if dtype is not None and str(arr.dtype) != dtype:
        raise TypeError(f"phys-rrtm-lw-lwrad: ctx.st['{name}'] dtype {arr.dtype}, expected {dtype}")
    return arr


def _inputs(ctx, lmk: int, lmp: int, jm: int, im: int) -> dict:
    """Bind every lwrad input by name with its producer; refuse by name when one is missing."""
    st = ctx.st
    shapes = {
        "plyr": (lmk, jm, im), "tlyr": (lmk, jm, im), "qlyr": (lmk, jm, im), "olyr": (lmk, jm, im),
        "plvl": (lmp, jm, im), "tlvl": (lmp, jm, im),
        "gasvmr": (NF_VGAS, lmk, jm, im), "clouds": (NF_CLDS, lmk, jm, im),
        "sfcemis": (jm, im), "tsfg": (jm, im),
    }
    producers = {
        "plyr": "grrad caller (lwrad argument plyr)", "tlyr": "grrad caller (lwrad argument tlyr)",
        "qlyr": "grrad caller (lwrad argument qlyr)", "olyr": "grrad caller (lwrad argument olyr)",
        "plvl": "grrad caller (lwrad argument plvl)", "tlvl": "grrad caller (lwrad argument tlvl)",
        "gasvmr": "phys-rrtm-getgases", "clouds": "phys-rrtm-grrad-clouds",
        "sfcemis": "phys-rrtm-setemis", "tsfg": "grrad caller (grrad.f:1094-1101)",
    }
    a = {n: _need(st, n, shapes[n], producers[n]) for n in shapes}
    a["icsdlw"] = _need(st, "icsdlw", (jm, im), "grrad caller (grrad.f:634)", dtype="int32")
    faerlw = ctx.tables.get("rrtm_faerlw") if hasattr(ctx, "tables") else None
    if faerlw is None:
        raise KeyError("phys-rrtm-lw-lwrad: ctx.tables['rrtm_faerlw'] is missing (written by phys-rrtm-setaer)")
    want = (NF_AELW, NBDLW, lmk, jm, im)
    if tuple(faerlw.shape) != want:
        raise ValueError(f"phys-rrtm-lw-lwrad: rrtm_faerlw shape {tuple(faerlw.shape)}, expected {want}")
    a["faerlw"] = faerlw
    return a


# Producer names of the lwrad inputs (grrad_nmmb.f:1985-2001 call site), bound by run().
INPUT_SOURCES = {
    "plyr": "RAD_PLYR", "plvl": "RAD_PLVL", "tlyr": "RAD_TLYR", "tlvl": "TLVL", "qlyr": "QLYR",
    "olyr": "RAD_OLYR", "gasvmr": "GASVMR", "clouds": "RAD_CLOUDS", "tsfg": "RAD_TSFG",
}


def run(ctx):
    """Driver entry: lwrad_prep, the seam (lw_core: cldprop, setcoef, taumol with taugb01..16,
    rtrnmr; isubclw = 0, iovrlw = 1 on the NAM route), then lwrad_post.

    The rows of ctx.grid must fit one chunk (JC_DEFAULT): rrtm_call runs RRTM in row chunks, so
    the whole chunk domain is one lwrad chunk and every seam unit sees it as its domain."""
    import cupy as cp

    from namgpu.physics.rrtm import lw_core

    g = ctx.grid
    lm = int(g.lm)
    nlay = lm + 1                  # LMK (LTP = 1)
    nlp1 = nlay + 1                # LMP
    jm = int(g.jm)
    im = int(g.im)
    if jm > JC_DEFAULT:
        raise RuntimeError(f"phys-rrtm-lw-lwrad: {jm} rows > JC_DEFAULT {JC_DEFAULT}; call through rrtm_call")
    if ISUBCLW > 0 or IOVRLW <= 0:
        raise NotImplementedError("phys-rrtm-lw-lwrad: only the NAM route (isubclw 0, iovrlw 1, rtrnmr) is glued")
    st = ctx.st
    for k, src in INPUT_SOURCES.items():
        if k not in st.fields and src in st.fields:
            st.fields[k] = st.fields[src]
    if "icsdlw" not in st.fields:
        st.add("icsdlw", (jm, im), "int32")   # permutation seeds, read only when isubclw == 2
    if "rrtm_faerlw" not in ctx.tables:
        for key in ("faerlw", "RAD_FAERLW"):
            if key in st.fields:
                ctx.tables["rrtm_faerlw"] = st.fields[key]
    a = _inputs(ctx, nlay, nlp1, jm, im)
    key = "rrtm_lw_lwrad:scratch"
    s = ctx.tables.get(key)
    if s is None or s["lwrad_pavel"].shape != (nlay, jm, im):
        s = _chunk_scratch(cp, nlay, nlp1, jm, im, ISUBCLW)
        ctx.tables[key] = s
    prep(a, s, nlay, nlp1, jm, im, 0, jm, stream=ctx.stream, isubclw=ISUBCLW)
    t = lw_core.seam(ctx, s, nlay, nlp1, jm, im)
    o = {n: st.add(n, (jm, im)) for n in ("topflw_upfxc", "topflw_upfx0", "sfcflw_upfxc",
                                          "sfcflw_upfx0", "sfcflw_dnfxc", "sfcflw_dnfx0")}
    o["hlwc"] = st.add("htlwc", (nlay, jm, im))
    o["hlw0"] = o["hlwb"] = o["flxprf"] = None
    post(t, o, nlay, nlp1, jm, im, 0, jm, False, False, False, stream=ctx.stream)
    return o


def selftest(seed: int = 1, jc: int = JC_DEFAULT):
    """Full-domain 954 x 835 x 60 run of lwrad_prep and lwrad_post on random plausible inputs.

    The domain is run in row chunks, each chunk with its own random inputs, so that the whole domain
    is covered without holding the full aerosol array (9.3 GB) on the card. The transfer outputs
    (totuflux ...) are random stand-ins for the seam. Checks: outputs finite; delp > 0; the
    surface/TOA flux picks are the right levels; hlwc equals htr with the ivflip level flip;
    lcf1 == 0 columns have taucld == 0 (cldprop not called, Fortran 942-944); the transfer dispatch
    picks the Fortran unit; on the first chunk with isubclw = 1, cldfmc is zero on lcf1 == 0 columns.
    """
    import cupy as cp

    lm, jm, im = 60, 835, 954
    nlay, nlp1 = lm + 1, lm + 2
    rng = np.random.default_rng(seed)
    bad = []
    total_cols = 0
    zero_cols = 0
    s_cache = None
    assert rtrn_unit(0, 0) == ("namgpu.physics.rrtm.phys_rrtm_lw_rtrn", "cldfrc")
    assert rtrn_unit(0, 1) == ("namgpu.physics.rrtm.phys_rrtm_lw_rtrnmr", "cldfrc")
    assert rtrn_unit(1, 1) == ("namgpu.physics.rrtm.phys_rrtm_lw_rtrnmc", "cldfmc")
    for jstart in range(0, jm, jc):
        rows = min(jc, jm - jstart)
        if rows != jc:
            s_cache = None
        if s_cache is None:
            s_cache = _chunk_scratch(cp, nlay, nlp1, rows, im, ISUBCLW)

        def rnd(shape, lo, hi):
            return cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))

        # pressures: TOA first (ivflip 0), increasing with level index
        lev = np.arange(nlp1, dtype=np.float32)
        plvl_np = (1000.0 * (0.955 ** lev))[:, None, None] * np.ones((nlp1, rows, im), np.float32)
        plvl_np = plvl_np * (1.0 + 1.0e-3 * rng.random((nlp1, rows, im), dtype=np.float32))
        plvl_np = np.sort(plvl_np, axis=0).copy()   # ivflip 0: input levels run TOA -> surface
        plvl = cp.asarray(plvl_np)
        plyr = 0.5 * (plvl[:-1] + plvl[1:])
        a = {
            "plyr": plyr, "plvl": plvl,
            "tlyr": rnd((nlay, rows, im), 200.0, 300.0), "tlvl": rnd((nlp1, rows, im), 200.0, 300.0),
            "qlyr": rnd((nlay, rows, im), 1.0e-6, 2.0e-2), "olyr": rnd((nlay, rows, im), 1.0e-7, 1.0e-5),
            "gasvmr": rnd((NF_VGAS, nlay, rows, im), 1.0e-9, 4.0e-4),
            "clouds": rnd((NF_CLDS, nlay, rows, im), 0.0, 1.0),
            "faerlw": rnd((NF_AELW, NBDLW, nlay, rows, im), 0.0, 0.1),
            "sfcemis": rnd((rows, im), 0.9, 1.0), "tsfg": rnd((rows, im), 250.0, 310.0),
            "icsdlw": cp.asarray(rng.integers(0, 1 << 20, size=(rows, im)).astype(np.int32)),
        }
        # ssa in [0, 1) for band optical properties
        a["faerlw"][1] = cp.asarray(rng.uniform(0.0, 0.99, size=(NBDLW, nlay, rows, im)).astype(np.float32))
        # clouds: fraction in [0,1], paths and radii plausible
        a["clouds"][0] = cp.clip(a["clouds"][0], 0.0, 1.0)
        for n_ in (1, 3, 5):
            a["clouds"][n_] = a["clouds"][n_] * 50.0
        a["clouds"][2] = cp.asarray(rng.uniform(5.0, 30.0, size=(nlay, rows, im)).astype(np.float32))
        a["clouds"][4] = cp.asarray(rng.uniform(20.0, 80.0, size=(nlay, rows, im)).astype(np.float32))
        # every 4th column is clear (no layer above eps), so the lcf1 == 0 branch is exercised
        a["clouds"][0][:, :, ::4] = 0.0

        prep(a, s_cache, nlay, nlp1, rows, im, 0, rows, isubclw=ISUBCLW)
        cp.cuda.Stream.null.synchronize()
        # the prep arrays must be finite and delp must be positive
        for name, arr in s_cache.items():
            if arr.dtype.kind == "f" and not bool(cp.isfinite(arr).all()):
                bad.append((jstart, name))
        if not bool(cp.all(s_cache["lwrad_delp"] > 0)):
            bad.append((jstart, "delp<=0"))
        # Fortran 942-944: cldprop not called on lcf1 == 0 columns, so taucld must be zero there
        clear = s_cache["lwrad_lcf1"][0] == 0                    # (rows, im)
        nclear = int(clear.sum())
        zero_cols += nclear
        if nclear == 0:
            bad.append((jstart, "no lcf1==0 columns in chunk"))
        elif not bool(cp.all(s_cache["lwrad_taucld"][..., clear] == 0.0)):
            bad.append((jstart, "taucld nonzero on lcf1==0 columns"))

        # stand-in transfer outputs for the seam (random, plausible magnitudes)
        t = {
            "totuflux": rnd((nlp1, rows, im), 100.0, 400.0), "totdflux": rnd((nlp1, rows, im), 100.0, 400.0),
            "totuclfl": rnd((nlp1, rows, im), 100.0, 400.0), "totdclfl": rnd((nlp1, rows, im), 100.0, 400.0),
            "htr": rnd((nlay, rows, im), -1.0e-4, 1.0e-4), "htrcl": rnd((nlay, rows, im), -1.0e-4, 1.0e-4),
            "htrb": rnd((NBDLW, nlay, rows, im), -1.0e-4, 1.0e-4),
        }
        o = {
            "topflw_upfxc": cp.zeros((rows, im), cp.float32), "topflw_upfx0": cp.zeros((rows, im), cp.float32),
            "sfcflw_upfxc": cp.zeros((rows, im), cp.float32), "sfcflw_upfx0": cp.zeros((rows, im), cp.float32),
            "sfcflw_dnfxc": cp.zeros((rows, im), cp.float32), "sfcflw_dnfx0": cp.zeros((rows, im), cp.float32),
            "hlwc": cp.zeros((nlay, rows, im), cp.float32),
            "hlw0": cp.zeros((nlay, rows, im), cp.float32),
            "hlwb": cp.zeros((NBDLW, nlay, rows, im), cp.float32),
            "flxprf": cp.zeros((4, nlp1, rows, im), cp.float32),
        }
        post(t, o, nlay, nlp1, rows, im, 0, rows, lhlw0=True, lhlwb=True, lflxprf=True)
        cp.cuda.Stream.null.synchronize()
        sl = slice(0, rows)
        for name in ("topflw_upfxc", "sfcflw_dnfxc", "hlwc", "hlwb", "flxprf"):
            if not bool(cp.isfinite(o[name][..., sl, :]).all()):
                bad.append((jstart, name))
        # identities (ivflip 0): surface flux = index 0, TOA flux = index nlay, hlwc level flipped vs htr
        assert bool(cp.array_equal(o["sfcflw_dnfxc"][sl], t["totdflux"][0])), "sfc dn flux"
        assert bool(cp.array_equal(o["topflw_upfxc"][sl], t["totuflux"][nlay])), "toa up flux"
        assert bool(cp.array_equal(o["hlwc"][:, sl], t["htr"][::-1])), "hlwc identity (ivflip 0, level flip)"
        total_cols += rows * im
        if jstart == 0:
            # isubclw = 1 path on this chunk: cldfmc is zeroed on lcf1 == 0 columns by the same kernel
            s_sub = _chunk_scratch(cp, nlay, nlp1, rows, im, isubclw=1)
            prep(a, s_sub, nlay, nlp1, rows, im, 0, rows, isubclw=1)
            cp.cuda.Stream.null.synchronize()
            clear0 = s_sub["lwrad_lcf1"][0] == 0
            if not bool(cp.all(s_sub["lwrad_cldfmc"][..., clear0] == 0.0)):
                bad.append((jstart, "cldfmc nonzero on lcf1==0 columns (isubclw=1)"))
            del s_sub
    assert total_cols == jm * im
    return {"columns": total_cols, "lcf1_zero_columns": zero_cols, "bad": bad}
