# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-swrad: the SW driver per column (swrad) of the NMMB RRTM shortwave package.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:374-1239 (subroutine swrad).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header and the AER notice are kept in the kernel file
namgpu/kernels/phys_rrtm_sw_swrad.cu.

Kernels (namgpu/kernels/phys_rrtm_sw_swrad.cu, one launch each, one thread per DAY column):
  swrad_zero_cld  clear columns (zcf1 <= 0) get zero cloud fraction and optics before the seam.
  swrad_prep      swrad statements before the scheme calls (ipseed, ivflip branches, profile, absorber
                  amounts, rare gases, aerosol and cloud unpack, cloud overlap, cosz1, sntz1, ssolar,
                  albbm, albdf).
  swrad_post      swrad statements after the scheme calls (band sums, topflx, sfcflx, fdncmp, hswc,
                  hsw0, hswb, flxprf).

The scheme calls between prep and post are separate units, NOT here, in the Fortran order:
  cldprop       (cloudy columns only)     namgpu.physics.rrtm.phys_rrtm_sw_cldprop
  setcoef                                 namgpu.physics.rrtm.phys_rrtm_sw_setcoef
  taumol                                  namgpu.physics.rrtm.phys_rrtm_sw_taumol
  taumol16..taumol29 (launched by the driver after taumol, radsw_main_nmmb.f:3384-4767)
  spcvrtc       (isubcsw <= 0)            namgpu.physics.rrtm.phys_rrtm_sw_spcvrtc
Each seam unit exposes run_chunk(ctx, chunk, cstart, nc) over the chunk scratch described below.
A seam unit that is not landed fails by name (RuntimeError), it is never skipped silently.

Day list (replaces the Fortran idxday/NDAY compaction; NDAY never leaves the device):
  st.SW_IDXDAY int32[JM*IM]  flat grid index g = j0*IM + i0 of each day column, first SW_NDAY valid
  st.SW_NDAY   int32[1]      number of day columns
  Written by phys-rrtm-grrad-sw (its scatter kernel). Every chunk launch reads *SW_NDAY on the device;
  threads at or past NDAY return. The host loop runs ceil(JM*IM / NC) chunks, not a per-cell loop.

Inputs (ctx.st names; NMMB/grrad names where they exist, the rest are mapped in INPUT_NAMES):
  plyr, tlyr, qlyr, olyr   [LMK, JM, IM]         layer pressure (mb), temperature, q, ozone
  plvl                     [LMK+1, JM, IM]       interface pressure (mb); tlvl is not in use
  GASVMR                   [10, LMK, JM, IM]     fields 1..9 (phys-rrtm-getgases)
  clouds                   [9, LMK, JM, IM]      cloud fields (prognostic if iswcliq > 0)
  faersw                   ctx.tables["rrtm_faersw"], [42, LMK, JM, IM] (3 x 14 bands, flattened)
  SFCALB                   [4, JM, IM]           setalb output
  CZEN                     [JM, IM]              cosine of solar zenith angle
  icsdsw                   [JM, IM] int32        permutation seeds (isubcsw == 2 only)
  solcon                   ctx.cfg.solcon (W/m2), default 1368.22 (NMMB internal s0)

Outputs (ctx.st, full domain, zero-filled each call, scratch via st.add):
  HTSWC    [LMK, JM, IM]           total-sky SW heating rate (k/s; iswrate = 2)
  TOPFLX   [3, JM, IM]             upfxc, dnfxc, upfx0 at TOA
  SFCFLX   [4, JM, IM]             upfxc, dnfxc, upfx0, dnfx0 at the surface
  HTSWB    [14, LMK, JM, IM]       spectral band heating rates (has_htrswb)
  HSW0     [LMK, JM, IM]           clear-sky heating (ctx.cfg.lhsw0)
  FLXPRF   [4, LMK+1, JM, IM]      level fluxes (ctx.cfg.lflxprf)
  FDNCMP   [6, JM, IM]             uvbf0, uvbfc, nirbm, nirdf, visbm, visdf (ctx.cfg.lfdncmp)

Chunk scratch (phys-rrtm-sw-swrad owns it, allocated once per shape, chunk-local index t):
  per-column float arrays, shape [NC, ...] in the names below. They are NOT full-domain: the full
  domain for the g-point arrays (cldfmc, taug, taur, 112 g-points x 61 levels) is ~22 GB and does not
  fit beside the State on a 32 GB card, so every per-column scratch is chunk-local.

Constants (NAM values): ivflip = 0 (iflip of rad_initialize, module_SOLVER_GRID_COMP.F90:10714), iswrgas = 1, iswcliq = 1, iswcice = 3,
iswmode = 2, iovrsw = 1, isubcsw = 0, iswrate = 2, ipsdsw0 = 1, NBDSW = 14, NGPTSW = 112, MAXGAS = 6.
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_swrad"
NBDSW = 14            # radsw_param_nmmb.f:156
NGPTSW = 112          # radsw_param_nmmb.f:150
MAXGAS = 6            # radsw_param_nmmb.f:152
NF_GAS = 10           # GASVMR fields (grrad_nmmb NF_VGAS); 1..9 used
NF_CLDS = 9           # clouds fields (NF_CLDS)
LTP = 1               # grrad_nmmb.f:210, extra top layer (LMK = LM + 1)
S0 = np.float32(1368.22)
SOLCON_DEFAULT = 1368.22
IVFLIP = 0            # NAM iflip = 0 (module_SOLVER_GRID_COMP.F90:10714, rad_initialize_nmmb.f:183): input top->surface
IOVRSW = 1            # physparam.f:143
ISWRGAS = 1           # physparam.f:51
ISWCLIQ = 1           # physparam.f:53
ISUBCSW = 0           # physparam.f:165
IPSDSW0 = 1           # radsw_main_nmmb.f:349
ISWRATE = 2           # physparam.f:49 (heating rate in k/s)
NC_DEFAULT = 32768    # day columns per chunk (one SW chunk per 32-row rrtm-call chunk; 4096 left spcvrtc launch-bound)

# ctx.st / ctx.tables names for the swrad inputs. The grrad names are NMMB/grrad names; the ones marked
# UNVERIFIED are the names this unit expects and must match the producers (phys-rrtm-grrad-setup,
# phys-rrtm-grrad-clouds) when they are wired. Change them here, nowhere else.
INPUT_NAMES = {
    "plyr": "RAD_PLYR",       # UNVERIFIED producer name
    "plvl": "RAD_PLVL",       # UNVERIFIED producer name (grrad_setup.py:19)
    "tlyr": "RAD_TLYR",       # UNVERIFIED producer name (grrad_setup.py:19)
    "qlyr": "RAD_QLYR",       # UNVERIFIED producer name
    "olyr": "RAD_OLYR",       # UNVERIFIED producer name
    "gasvmr": "GASVMR",       # phys-rrtm-getgases
    "clouds": "RAD_CLOUDS",   # UNVERIFIED producer name (grrad_clouds)
    "icsdsw": "RAD_ICSDSW",   # UNVERIFIED producer name
    "cosz": "SW_COSZEN",      # coszen_loc, the SW-interval mean cosine (grrad_nmmb.f:1878, 1904), written by phys-rrtm-grrad-sw
    "sfcalb": "SFCALB",       # phys-rrtm-setalb
}
FAERSW_TABLE = "rrtm_faersw"  # phys-rrtm-grrad-aerosols (ctx.tables)

# Seam units, in the Fortran order of swrad (cldprop, setcoef, taumol, the taumol16..29 bands driven
# after taumol, spcvrtc). Each must provide run_chunk(ctx, chunk, cstart, nc).
SEAM_CLDPROP = "namgpu.physics.rrtm.phys_rrtm_sw_cldprop"
SEAM_SETCOEF = "namgpu.physics.rrtm.phys_rrtm_sw_setcoef"
SEAM_TAUMOL = "namgpu.physics.rrtm.phys_rrtm_sw_taumol"
SEAM_TAUMOL_BANDS = tuple(f"namgpu.physics.rrtm.phys_rrtm_sw_taumol{b}" for b in range(16, 30))
SEAM_SPCVRTC = "namgpu.physics.rrtm.phys_rrtm_sw_spcvrtc"
SEAM_ORDER = (SEAM_CLDPROP, SEAM_SETCOEF, SEAM_TAUMOL) + SEAM_TAUMOL_BANDS + (SEAM_SPCVRTC,)


def _dep(module_name: str):
    """A seam unit that is not landed is a named gap."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(f"phys-rrtm-sw-swrad needs {module_name} (unit not landed): {e}") from e


def _heatfac():
    """heatfac for iswrate = 2: con_g * 1.0e-2 / con_cp, double arithmetic (kind_phys = 8 in the
    reference build, radsw_main_nmmb.f:1374), stored float32 as the swrad consumer reads it."""
    return np.float32(9.80665 * 1.0e-2 / 1.0046e3)


_WS_ALIGN = 64      # floats: every array carved from a workspace starts on a 256-byte boundary


def _scratch_spec(nlay: int, nlp1: int) -> list:
    """(name, per-column shape, 'f' | 'i') of every chunk-scratch array, in chunk_scratch's order."""
    spec = []
    f = lambda *s: ("f", s)   # noqa: E731
    i = lambda *s: ("i", s)   # noqa: E731
    for name, (kind, shp) in _scratch_items(f, i, nlay, nlp1).items():
        spec.append((name, shp, kind))
    return spec


def chunk_scratch_floats(nc: int, nlay: int, nlp1: int) -> int:
    """Workspace size (4-byte words) chunk_scratch(..., ws=) carves for nc columns."""
    n = 0
    for _, shp, _ in _scratch_spec(nlay, nlp1):
        n += -(-(nc * int(np.prod(shp, dtype=np.int64))) // _WS_ALIGN) * _WS_ALIGN
    return n


def sw_ncap(n: int, nc: int | None = None) -> int:
    """Columns of the one fixed-size fit-path scratch block of a row chunk of n columns (see run())."""
    return min(NC_DEFAULT if nc is None else nc, -(-n // 8) * 8)


def chunk_scratch(xp, nc: int, nlay: int, nlp1: int, ws=None) -> dict:
    """Chunk scratch, allocated once per shape. Per-column blocks, index [t, ...].

    ws: a flat float32 workspace (rrtm_chunk.workspace) of at least chunk_scratch_floats(nc, ...) words: the
    arrays are carved from it (int32 ones as int32 views of the same words) and hold whatever it held, so
    the caller zero-fills them before use (run() does, for every SW chunk). Without ws: fresh zeros."""
    if ws is None:
        f = lambda *s: xp.zeros((nc,) + s, dtype=xp.float32)   # noqa: E731
        i = lambda *s: xp.zeros((nc,) + s, dtype=xp.int32)     # noqa: E731
        return _scratch_items(f, i, nlay, nlp1)
    need = chunk_scratch_floats(nc, nlay, nlp1)
    if ws.dtype != xp.float32 or ws.ndim != 1 or ws.size < need:
        raise ValueError(f"phys-rrtm-sw-swrad: workspace too small ({ws.size} < {need} words)")
    out, off = {}, 0
    for name, shp, kind in _scratch_spec(nlay, nlp1):
        cnt = nc * int(np.prod(shp, dtype=np.int64))
        v = ws[off:off + cnt].reshape((nc,) + tuple(shp))
        out[name] = v.view(xp.int32) if kind == "i" else v
        off += -(-cnt // _WS_ALIGN) * _WS_ALIGN
    return out


def _scratch_items(f, i, nlay: int, nlp1: int) -> dict:
    """The chunk-scratch table: name -> f(*per-column shape) or i(*per-column shape)."""
    return {
        # swrad_prep outputs
        "pavel": f(nlay), "tavel": f(nlay), "delp": f(nlay), "coldry": f(nlay), "colmol": f(nlay),
        "h2ovmr": f(nlay), "o3vmr": f(nlay), "temcol": f(nlay), "colamt": f(nlay, MAXGAS),
        "cfrac": f(nlay), "cliqp": f(nlay), "reliq": f(nlay), "cicep": f(nlay), "reice": f(nlay),
        "cdat1": f(nlay), "cdat2": f(nlay), "cdat3": f(nlay), "cdat4": f(nlay),
        "tauae": f(nlay, NBDSW), "ssaae": f(nlay, NBDSW), "asyae": f(nlay, NBDSW),
        "zcf0": f(), "zcf1": f(), "cosz1": f(), "sntz1": f(), "ssolar": f(),
        "albbm": f(2), "albdf": f(2), "ipseed": i(),
        # cldprop / spcvrtc / taumol working set (seam units read and write these)
        "cldfrc": f(nlay), "cldfmc": f(nlay, NGPTSW), "taucw": f(nlay, NBDSW), "ssacw": f(nlay, NBDSW),
        "asycw": f(nlay, NBDSW), "taug": f(nlay, NGPTSW), "taur": f(nlay, NGPTSW),
        "sfluxzen": f(NGPTSW),
        "laytrop": i(), "jp": i(nlay), "jt": i(nlay), "jt1": i(nlay), "indself": i(nlay), "indfor": i(nlay),
        "fac00": f(nlay), "fac01": f(nlay), "fac10": f(nlay), "fac11": f(nlay),
        "selffac": f(nlay), "selffrac": f(nlay), "forfac": f(nlay), "forfrac": f(nlay),
        # spcvrtc outputs (swrad_post inputs)
        "fxupc": f(nlp1, NBDSW), "fxdnc": f(nlp1, NBDSW), "fxup0": f(nlp1, NBDSW), "fxdn0": f(nlp1, NBDSW),
        "ftoauc": f(), "ftoau0": f(), "ftoadc": f(), "fsfcuc": f(), "fsfcu0": f(), "fsfcdc": f(), "fsfcd0": f(),
        "suvbfc": f(), "suvbf0": f(), "sfbmc": f(2), "sfdfc": f(2), "sfbm0": f(2), "sfdf0": f(2),
    }


def _flag(ctx, name: str, default: bool) -> bool:
    return bool(getattr(getattr(ctx, "cfg", None), name, default))


def _prep_chunk(inp: dict, ch: dict, ndayp, idxday, cstart: int, nc: int, nlay: int, nlp1: int,
                jm: int, im: int, solcon: float, stream=None):
    """swrad_prep and swrad_zero_cld for one chunk. inp: full-domain inputs by kernel name."""
    import cupy as cp

    k = kernels.get(UNIT, "swrad_prep")
    scope = stream if stream is not None else contextlib.nullcontext()
    grid = ((nc + 127) // 128, 1, 1)
    with scope:
        k(grid, (128, 1, 1), (
            inp["plyr"], inp["plvl"], inp["tlyr"], inp["qlyr"], inp["olyr"], inp["gasvmr"], inp["clouds"],
            inp["faersw"], inp["cosz"], inp["sfcalb"], inp["icsdsw"],
            idxday, ndayp,
            np.int32(cstart), np.int32(nc), np.int32(nlay), np.int32(nlp1), np.int32(im), np.int32(jm),
            np.int32(IVFLIP), np.int32(ISWRGAS), np.int32(ISWCLIQ), np.int32(IOVRSW), np.int32(ISUBCSW),
            np.int32(IPSDSW0), np.float32(solcon),
            ch["pavel"], ch["tavel"], ch["delp"], ch["coldry"], ch["colmol"], ch["h2ovmr"], ch["o3vmr"],
            ch["temcol"], ch["colamt"], ch["cfrac"], ch["cliqp"], ch["reliq"], ch["cicep"], ch["reice"],
            ch["cdat1"], ch["cdat2"], ch["cdat3"], ch["cdat4"], ch["tauae"], ch["ssaae"], ch["asyae"],
            ch["zcf0"], ch["zcf1"], ch["cosz1"], ch["sntz1"], ch["ssolar"], ch["albbm"], ch["albdf"],
            ch["ipseed"]))
        kz = kernels.get(UNIT, "swrad_zero_cld")
        kz(grid, (128, 1, 1), (
            ch["zcf1"], ndayp, np.int32(cstart), np.int32(nc), np.int32(nlay),
            ch["cldfrc"], ch["cldfmc"], ch["taucw"], ch["ssacw"], ch["asycw"]))


def _post_chunk(ch: dict, out: dict, ndayp, idxday, cstart: int, nc: int, nlay: int, nlp1: int,
                jm: int, im: int, heatfac, lhsw0: bool, lhswb: bool, lflxprf: bool, lfdncmp: bool,
                stream=None):
    """swrad_post for one chunk; writes the full-domain outputs for its day columns."""
    import cupy as cp

    k = kernels.get(UNIT, "swrad_post")
    scope = stream if stream is not None else contextlib.nullcontext()
    grid = ((nc + 127) // 128, 1, 1)
    one = out["_dummy"]
    with scope:
        k(grid, (128, 1, 1), (
            ch["delp"], ch["fxupc"], ch["fxdnc"], ch["fxup0"], ch["fxdn0"],
            ch["ftoauc"], ch["ftoau0"], ch["ftoadc"], ch["fsfcuc"], ch["fsfcu0"], ch["fsfcdc"], ch["fsfcd0"],
            ch["suvbfc"], ch["suvbf0"], ch["sfbmc"], ch["sfdfc"],
            idxday, ndayp,
            np.int32(cstart), np.int32(nc), np.int32(nlay), np.int32(nlp1), np.int32(im), np.int32(jm),
            np.int32(IVFLIP), np.float32(heatfac), np.int32(int(lhsw0)), np.int32(int(lhswb)),
            np.int32(int(lflxprf)), np.int32(int(lfdncmp)),
            out["HTSWC"], out["TOPFLX"], out["SFCFLX"],
            out["HSW0"] if lhsw0 else one, out["HTSWB"] if lhswb else one,
            out["FLXPRF"] if lflxprf else one, out["FDNCMP"] if lfdncmp else one))


def _seam(ctx, ch: dict, cstart: int, nc: int) -> None:
    """The scheme calls between prep and post, in the Fortran order. A unit with its own run_chunk reads ch
    directly; the others run through the layout glue in namgpu.physics.rrtm.sw_seam (ADAPTERS)."""
    from namgpu.physics.rrtm import sw_seam
    for mod_name in SEAM_ORDER:
        mod = _dep(mod_name)
        fn = getattr(mod, "run_chunk", None) or sw_seam.ADAPTERS.get(mod_name)
        if fn is None:
            raise RuntimeError(f"phys-rrtm-sw-swrad: seam unit {mod_name} has no run_chunk and no sw_seam adapter")
        fn(ctx, ch, cstart, nc)


def run(ctx, has_htrswb: bool = False, nc: int = NC_DEFAULT):
    """Driver entry, one call per SW radiation step (grrad_nmmb.f:1872-1916). Reads SW_IDXDAY and
    SW_NDAY on the device. Writes HTSWC, TOPFLX, SFCFLX and, when requested, the optional outputs."""
    import cupy as cp

    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    nlay = lm + LTP                 # LMK
    nlp1 = nlay + 1                 # LMP
    n = jm * im
    lhsw0 = _flag(ctx, "lhsw0", False)
    lhswb = bool(has_htrswb)
    lflxprf = _flag(ctx, "lflxprf", False)
    lfdncmp = _flag(ctx, "lfdncmp", False)
    # solcon of THIS call: radupdate's sun-earth-distance-corrected value, handed to grrad_nmmb as SOLCON
    # (module_RA_RRTM.F90 GRRAD call; rrtm-call records it). cfg.solcon / 1368.22 only outside the chain.
    flags = ctx.tables.get("rrtm_call_flags") if getattr(ctx, "tables", None) is not None else None
    if flags is not None and flags.get("solcon") is not None:
        solcon = float(flags["solcon"])
    else:
        solcon = float(getattr(getattr(ctx, "cfg", None), "solcon", SOLCON_DEFAULT))
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()

    inp = {k: st[v] for k, v in INPUT_NAMES.items() if k != "icsdsw"}
    # icsdsw is read only when isubcsw == 2; with ISUBCSW = 0 a zero array stands in when absent
    inp["icsdsw"] = st.fields.get(INPUT_NAMES["icsdsw"]) if INPUT_NAMES["icsdsw"] in st.fields \
        else cp.zeros((jm, im), dtype=cp.int32)
    inp["faersw"] = ctx.tables[FAERSW_TABLE]
    idxday = st["SW_IDXDAY"]
    ndayp = st["SW_NDAY"]

    out = {
        "HTSWC": st.add("HTSWC", (nlay, jm, im)),
        "TOPFLX": st.add("TOPFLX", (3, jm, im)),
        "SFCFLX": st.add("SFCFLX", (4, jm, im)),
        "_dummy": st.add("SWRAD_DUMMY", (1,)),
    }
    if lhswb:
        out["HTSWB"] = st.add("HTSWB", (NBDSW, nlay, jm, im))
    if lhsw0:
        out["HSW0"] = st.add("HSW0", (nlay, jm, im))
    if lflxprf:
        out["FLXPRF"] = st.add("FLXPRF", (4, nlp1, jm, im))
    if lfdncmp:
        out["FDNCMP"] = st.add("FDNCMP", (6, jm, im))

    with scope:
        # swrad zeroes hswc, topflx, sfcflx (and the optional outputs) for every point at entry
        for name in ("HTSWC", "TOPFLX", "SFCFLX", "HTSWB", "HSW0", "FLXPRF", "FDNCMP"):
            if name in out:
                out[name].fill(0)
        heatfac = _heatfac()
        # Day columns only: one host read of the device day count per SW call (the Fortran NDAY). Chunks
        # are multiples of CHK = 8 columns (taumol19's MINVAL tiles); n = JM*IM is one too for 8-row chunks.
        nday = int(ndayp.get()[0])
        # 32 GB fit: allocate the chunk scratch at the size the chunk uses. It was first allocated for
        # nc = 32768 columns and then again for ncur whenever ncur != nc, and with 8-row rrtm-call chunks (at
        # most 7,632 day columns) that is every call: ~4 GiB of 32768-column blocks (0.9 GB per 112-g-point
        # array) allocated for nothing at the radiation peak. Same values: each chunk still gets a fresh
        # zero-filled scratch of its own size whenever its size differs from nc, as before.
        # One fixed-size block per call (ncap = all columns of this row chunk), and each SW chunk gets a zeroed
        # leading-axis view of it: the same zero-filled [ncur, ...] arrays as before, but no differently sized
        # allocations call to call (they fragmented the CuPy pool until a 32 GB card ran out at the 4th call).
        from namgpu.state import fit_on  # noqa: PLC0415
        _fit = fit_on()
        ncap = sw_ncap(n, nc)
        full = None
        from namgpu.physics.rrtm import rrtm_chunk  # noqa: PLC0415
        ws = rrtm_chunk.workspace(ctx) if _fit else None
        sc = None if _fit else chunk_scratch(cp, nc, nlay, nlp1)
        for cstart in range(0, nday, nc):
            ncur = min(nc, nday - cstart)
            ncur = min(-(-ncur // 8) * 8, n - cstart)
            if _fit and (sc is None or ncur != nc):
                if full is None:
                    # carved from the chunk's shared g-point workspace when rrtm-call made one; every view
                    # is zero-filled just below, as the fresh zeros were
                    full = chunk_scratch(cp, ncap, nlay, nlp1, ws=ws)
                sc = {k: v[:ncur] for k, v in full.items()}
                for v in sc.values():
                    v.fill(0)
            elif not _fit and ncur != nc:
                sc = chunk_scratch(cp, ncur, nlay, nlp1)
            _prep_chunk(inp, sc, ndayp, idxday, cstart, ncur, nlay, nlp1, jm, im, solcon,
                        stream=ctx.stream)
            _seam(ctx, sc, cstart, ncur)
            _post_chunk(sc, out, ndayp, idxday, cstart, ncur, nlay, nlp1, jm, im, heatfac,
                        lhsw0, lhswb, lflxprf, lfdncmp, stream=ctx.stream)
    return out["HTSWC"]


def selftest(seed: int = 1, jc: int = 64, nc: int = NC_DEFAULT):
    """Full-size run of prep, zero_cld and post on random plausible inputs (954 x 835 x 60).

    The domain is run in row chunks of jc rows, each with its own random inputs, so the 42-field
    aerosol array (8.2 GB at full size) never sits on the card. The seam outputs (cldprop ... spcvrtc)
    are random stand-ins with plausible magnitudes; the seam units are not landed.
    Checks: prep scratch finite; delp > 0 and zcf0, zcf1 in [0, 1] on day columns; post outputs
    finite; heating rates are zero off the day list.
    """
    import cupy as cp

    lm, jm, im = 60, 835, 954
    nlay, nlp1 = lm + LTP, lm + LTP + 1
    rng = np.random.default_rng(seed)
    bad = []
    day_total = 0

    def rnd(shape, lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))

    for jstart in range(0, jm, jc):
        rows = min(jc, jm - jstart)
        lev = np.arange(nlp1, dtype=np.float32)
        plvl_np = (1000.0 * (0.955 ** lev))[:, None, None] * np.ones((nlp1, rows, im), np.float32)
        plvl_np = plvl_np * (1.0 + 1.0e-3 * rng.random((nlp1, rows, im), dtype=np.float32))
        plvl_np = np.sort(plvl_np, axis=0)[::-1].copy()
        plvl = cp.asarray(plvl_np)
        plyr = 0.5 * (plvl[:-1] + plvl[1:])
        cosz = rnd((rows, im), -0.2, 1.0)
        cosz = cp.where(cosz < 0.0001, cp.float32(-1.0), cosz)   # night columns
        idx_np = np.flatnonzero(cp.asnumpy(cosz).ravel() > 0.0001).astype(np.int32)
        ndays = int(idx_np.size)
        day_total += ndays
        if ndays == 0:
            continue
        idxday = cp.asarray(idx_np)
        ndayp = cp.asarray(np.array([ndays], dtype=np.int32))
        inp = {
            "plyr": plyr, "plvl": plvl, "tlyr": rnd((nlay, rows, im), 200.0, 300.0),
            "qlyr": rnd((nlay, rows, im), 1.0e-6, 2.0e-2), "olyr": rnd((nlay, rows, im), 1.0e-7, 1.0e-5),
            "gasvmr": rnd((NF_GAS, nlay, rows, im), 1.0e-9, 4.0e-4),
            "clouds": rnd((NF_CLDS, nlay, rows, im), 0.0, 1.0),
            "faersw": rnd((3 * NBDSW, nlay, rows, im), 0.0, 0.5),
            "sfcalb": rnd((4, rows, im), 0.05, 0.6),
            "icsdsw": cp.asarray(rng.integers(0, 1 << 20, size=(rows, im)).astype(np.int32)),
            "cosz": cosz,
        }
        inp["clouds"][2] = cp.asarray(rng.uniform(5.0, 30.0, size=(nlay, rows, im)).astype(np.float32))
        inp["clouds"][4] = cp.asarray(rng.uniform(20.0, 80.0, size=(nlay, rows, im)).astype(np.float32))
        for c in (1, 3, 5):
            inp["clouds"][c] = inp["clouds"][c] * 50.0
        out = {
            "HTSWC": cp.zeros((nlay, rows, im), cp.float32),
            "TOPFLX": cp.zeros((3, rows, im), cp.float32),
            "SFCFLX": cp.zeros((4, rows, im), cp.float32),
            "HTSWB": cp.zeros((NBDSW, nlay, rows, im), cp.float32),
            "HSW0": cp.zeros((nlay, rows, im), cp.float32),
            "FLXPRF": cp.zeros((4, nlp1, rows, im), cp.float32),
            "FDNCMP": cp.zeros((6, rows, im), cp.float32),
            "_dummy": cp.zeros((1,), cp.float32),
        }
        heatfac = _heatfac()
        nrows_cols = rows * im
        for cstart in range(0, ndays, nc):
            ncur = min(nc, ndays - cstart)
            ch = chunk_scratch(cp, ncur, nlay, nlp1)
            _prep_chunk(inp, ch, ndayp, idxday, cstart, ncur, nlay, nlp1, rows, im, SOLCON_DEFAULT)
            cp.cuda.Stream.null.synchronize()
            for name in ("delp", "cfrac", "zcf0", "zcf1", "tauae", "colamt", "coldry", "tavel"):
                if not bool(cp.isfinite(ch[name]).all()):
                    bad.append((jstart, cstart, name))
            if not bool(cp.all(ch["delp"] > 0)):
                bad.append((jstart, cstart, "delp<=0"))
            if not bool(cp.all((ch["zcf0"] >= 0) & (ch["zcf0"] <= 1))):
                bad.append((jstart, cstart, "zcf0 range"))
            # stand-in seam outputs (cldprop .. spcvrtc are not landed)
            for name in ("fxupc", "fxdnc", "fxup0", "fxdn0"):
                ch[name][...] = rnd((ncur, nlp1, NBDSW), 1.0, 400.0)
            for name in ("ftoauc", "ftoau0", "ftoadc", "fsfcuc", "fsfcu0", "fsfcdc", "fsfcd0",
                         "suvbfc", "suvbf0"):
                ch[name][...] = rnd((ncur,), 1.0, 400.0)
            for name in ("sfbmc", "sfdfc", "sfbm0", "sfdf0"):
                ch[name][...] = rnd((ncur, 2), 1.0, 300.0)
            _post_chunk(ch, out, ndayp, idxday, cstart, ncur, nlay, nlp1, rows, im, heatfac,
                        True, True, True, True)
            cp.cuda.Stream.null.synchronize()
        for name in ("HTSWC", "TOPFLX", "SFCFLX", "HTSWB", "HSW0", "FLXPRF", "FDNCMP"):
            if not bool(cp.isfinite(out[name]).all()):
                bad.append((jstart, name))
        # zero off the day list: a night column must hold exactly zero heating
        night = cp.ones(rows * im, dtype=bool)
        night[idxday] = False
        hs = out["HTSWC"].transpose(1, 2, 0).reshape(rows * im, nlay)
        if not bool(cp.all(hs[night] == 0)):
            bad.append((jstart, "night heating nonzero"))
        del out, inp, ch
    return {"day_columns": day_total, "bad": bad}
