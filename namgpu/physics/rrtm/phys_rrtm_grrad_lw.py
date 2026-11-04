"""phys-rrtm-grrad-lw: the LW section of GRRAD_NMMB (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:1963-2075 (LW block under if (lslwr)),
2082-2169 (LW and cloud fluxr, cldcov under if (lssav)) and 2201 (coszen). NOAA US government work;
the NOAA header of grrad_nmmb.f is kept as the notice in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_grrad_lw.cu (grrad_lw_pre, grrad_lw_post_heat, grrad_lw_post_flux;
one launch each, one thread per (i, j) column, no host sync).

What is in this unit: the statements around the two LW calls, not the LW transfer.
  pre        np3d == 3: save clouds(.,.,2) and (.,.,4), scale by CPATHFAC4LW = 1.5 (module_RA_RRTM.F90:212)
  setemis    -> sfcemis      unit phys-rrtm-setemis     (namgpu.physics.rrtm.phys_rrtm_setemis)
  lwrad      -> htlwc, topflw_*, sfcflw_*                unit phys-rrtm-lw-lwrad
             (namgpu.physics.rrtm.phys_rrtm_lw_lwrad)
  post_heat  semis, tsflw, htrlw, restore clouds (np3d == 3)
  post_flux  coszen copy (unconditional), LW fluxr, cloud fluxr (lsswr or lslwr), cldcov

Boundary rows. RRTM is called only for rows JDS+1 .. JDE-1 (module_RADIATION.F90:592), so for the
regional domain grrad_nmmb never runs on j = 1 and j = JM. Those rows get no LW work in any kernel:
their clouds stay unscaled, and htrlw, semis, tsflw, cldcov, fluxr and coszen keep the values they
already hold (zero from the state allocation), as in Fortran. The kernels skip them with a row guard.

Order (grrad_nmmb.f). The SW block (1812-1956) reads clouds before the LW 1.5 scaling (1978). So
phys-rrtm-grrad-sw must run first in the same call; run() requires sw_done=True, which the caller
sets only after the SW unit has run. Within the LW block: setemis (1963), pre (1978), lwrad (1985
and 2001), post_heat, restore (2064), then the post_flux accumulations (2082+) and coszen (2201).

Inputs, bound by CONTRACT name from the producers (a missing producer raises by name):
  RAD_CLOUDS [NF_CLDS, LMK, JM, IM]  clouds, phys-rrtm-grrad-clouds (its copy of progcld2 output)
  sfcemis    [JM, IM]                phys-rrtm-setemis
  TSFA       [JM, IM]                phys-rrtm-grrad-gases (grrad_nmmb.f:1454, ivflip == 0 branch)
  GRRAD_COSZEN_V [JM, IM]            phys-rrtm-coszmn (COSZMN_NMMB daytime mean = Fortran coszen_loc)
  PINT       [LM+1, JM, IM]          state, Pa. Fortran prsi (2157-2158), not the mb plvl of 1253.
  T          [LM, JM, IM]            state, Fortran tgrs (2157-2158 tgrs(i,itop))
  CLDS       [5, JM, IM]             phys-rrtm-progcld2, Fortran cldsa (grrad_nmmb.f:989, 2132-2140)
  MTOP, MBOT [3, JM, IM] int32       phys-rrtm-progcld2, Fortran mtopa, mbota
  htlwc      [LMK, JM, IM], topflw_upfxc, topflw_upfx0, sfcflw_dnfxc, sfcflw_upfxc,
  sfcflw_dnfx0, sfcflw_upfx0 [JM, IM]   phys-rrtm-lw-lwrad (written when lslwr)
  FLUXR_V    [NFLUXR, JM, IM]        ctx.tables["rrtm_driver_call"] (phys-rrtm-driver-call). Zeroed in
                                     that unit's kernel on every RRTM call (module_RA_RRTM.F90:1101-1102),
                                     so the LW and SW accumulations of one call add onto zeros, as in
                                     Fortran. This unit never zeroes it.
Outputs (st, scratch via st.add, read by phys-rrtm-driver-post):
  htrlw [LM, JM, IM], cldcov [LM, JM, IM], cldsa (alias of CLDS, the name driver-post reads),
  semis, tsflw, coszen [JM, IM]

Constants and offsets: kd = 1, kt = 0, kb = 1. The NAM path: lextop = (LTP > 0) = true
(grrad_nmmb.f:211) and ivflip = 0 (toa to sfc; rad_initialize is called with iflip = 0, see
phys-rrtm-rad-initialize). htrlwb: the NAM grrad call does not pass HTRLWB (module_RA_RRTM.F90:1136 is
commented out), so the Fortran present(htrlwb) branch is not taken; the kernels keep it behind has_htlwb.

Scalars (module_RA_RRTM.F90:842-846, 197): dtlw = NRADL*DT, dtsw = NRADS*DT,
lslwr = MOD(NTIMESTEP, NRADL) == 0, lsswr = MOD(NTIMESTEP, NRADS) == 0, lssav = .TRUE.,
raddt = min(dtsw, dtlw) (grrad_nmmb.f:1200). NTIMESTEP = ctx.ntimestep_rad.
np3d = icmphys (grrad_nmmb.f:1157): cfg.np3d, the same value phys-rrtm-driver-call takes as icmphys.

One launch per kernel, no host sync. Scratch is allocated once (ctx.tables, st.add).
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_grrad_lw"
NF_CLDS = 9
NBDLW = 16
NCLDSA = 5
NFLUXR = 39                              # module_RA_RRTM.F90 NFLUXR; phys-rrtm-driver-call
NMTOP = 3
KD, KT, KB = 1, 0, 1                     # grrad_nmmb.f:1169-1193, NAM path (lextop true, ivflip 0)
CPATHFAC4LW = 1.5                        # module_RA_RRTM.F90:212 (PARAMETER)
LSSAV = True                             # module_RA_RRTM.F90:197 (LSSAV=.TRUE.)
HAS_HTLWB = False                        # NAM grrad call does not pass HTRLWB (module_RA_RRTM.F90:1136)
SETEMIS_MODULE = "namgpu.physics.rrtm.phys_rrtm_setemis"
LWRAD_MODULE = "namgpu.physics.rrtm.phys_rrtm_lw_lwrad"
DRIVER_CALL_KEY = "rrtm_driver_call"     # phys-rrtm-driver-call's ctx.tables entry (holds FLUXR_V)
DUMMYS_KEY = "rrtm_lw_dummys"


def _dep(module: str):
    """A dependency unit that is not landed yet fails loudly, it is never skipped silently."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(f"phys-rrtm-grrad-lw needs {module} (unit not landed): {e}") from e


def _scope(stream):
    return stream if stream is not None else contextlib.nullcontext()


def _need(st, name: str, shape, owner: str, dtype=None):
    """A producer's array, checked by name, shape and dtype. Missing is a named error, never a None."""
    arr = st.fields.get(name)
    if arr is None:
        raise KeyError(f"phys-rrtm-grrad-lw: ctx.st['{name}'] is missing (written by {owner})")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"phys-rrtm-grrad-lw: ctx.st['{name}'] shape {tuple(arr.shape)}, expected {tuple(shape)}")
    if dtype is not None and str(arr.dtype) != dtype:
        raise TypeError(f"phys-rrtm-grrad-lw: ctx.st['{name}'] dtype {arr.dtype}, expected {dtype}")
    return arr


def new_dummys(lmk: int, jm: int, im: int):
    """dummys scratch: the two cloud-path slots (clouds Fortran index 2 and 4), allocated once."""
    import cupy as cp

    return cp.zeros((2, lmk, jm, im), dtype=cp.float32)


def pre(clouds, dummys, np3d: int, stream=None):
    """grrad_nmmb.f:1978-1990. clouds [NF_CLDS, LMK, JM, IM], dummys [2, LMK, JM, IM]."""
    _, lmk, jm, im = clouds.shape
    k = kernels.get(UNIT, "grrad_lw_pre")
    with _scope(stream):
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (clouds, dummys, np.int32(lmk), np.int32(jm), np.int32(im), np.int32(np3d),
           np.float32(CPATHFAC4LW)))


def post_heat(a: dict, dummys, lm: int, lmk: int, jm: int, im: int, np3d: int, lslwr: bool,
              stream=None):
    """grrad_nmmb.f:2046-2071. a: sfcemis, tsfa, htlwc, semis, tsflw, htrlw, clouds
    (and htlwb/htrlwb only when HAS_HTLWB)."""
    import cupy as cp

    has_htlwb = HAS_HTLWB
    one = cp.zeros(1, dtype=cp.float32)     # placeholder for a pointer the kernel will not read
    k = kernels.get(UNIT, "grrad_lw_post_heat")
    with _scope(stream):
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (a["htlwc"], a["htlwb"] if has_htlwb else one, a["sfcemis"], a["tsfa"],
           a["semis"], a["tsflw"], a["htrlw"], a["htrlwb"] if has_htlwb else one,
           a["clouds"], dummys,
           np.int32(lm), np.int32(lmk), np.int32(KD), np.int32(jm), np.int32(im),
           np.int32(np3d), np.int32(int(lslwr)), np.int32(int(has_htlwb))))


def post_flux(a: dict, lm: int, lmk: int, jm: int, im: int, dtlw: float, raddt: float,
              lssav: bool, lslwr: bool, lsswr: bool, stream=None):
    """grrad_nmmb.f:2201 and 2082-2169. a: topflw_upfxc, topflw_upfx0, sfcflw_dnfxc, sfcflw_upfxc,
    sfcflw_dnfx0, sfcflw_upfx0, cldsa [5, JM, IM], mtopa, mbota int32 [3, JM, IM], plvl (= PINT, Pa),
    T [LM, JM, IM], clouds, cldcov [LM, JM, IM], fluxr [NFLUXR, JM, IM], coszen_loc, coszen [JM, IM]."""
    k = kernels.get(UNIT, "grrad_lw_post_flux")
    with _scope(stream):
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (a["topflw_upfxc"], a["topflw_upfx0"], a["sfcflw_dnfxc"], a["sfcflw_upfxc"],
           a["sfcflw_dnfx0"], a["sfcflw_upfx0"], a["cldsa"], a["mtopa"], a["mbota"],
           a["plvl"], a["T"], a["clouds"], a["cldcov"], a["fluxr"],
           a["coszen_loc"], a["coszen"],
           np.int32(lm), np.int32(lmk), np.int32(KD), np.int32(KT), np.int32(KB),
           np.int32(jm), np.int32(im),
           np.float32(dtlw), np.float32(raddt),
           np.int32(int(lssav)), np.int32(int(lslwr)), np.int32(int(lsswr))))


def run(ctx, *, sw_done: bool):
    """Driver entry, in grrad_nmmb order: [setemis, pre, lwrad, post_heat] if lslwr, then post_flux.

    sw_done must be True: the caller has run phys-rrtm-grrad-sw in this call, because the SW block
    reads clouds before the LW scaling (grrad_nmmb.f:1812-1956 before 1978). Returns (cldcov, fluxr).
    """
    if not sw_done:
        raise RuntimeError(
            "phys-rrtm-grrad-lw: run phys-rrtm-grrad-sw first in this call (sw_done=True). The SW block "
            "reads clouds before the LW 1.5 scaling (grrad_nmmb.f:1812-1956 before 1978).")
    st = ctx.st
    cfg = ctx.cfg
    lm = int(ctx.grid.lm)
    lmk = lm + 1
    jm = int(ctx.grid.jm)
    im = int(ctx.grid.im)
    nts = int(ctx.ntimestep_rad)
    nradl, nrads = int(cfg.nradl), int(cfg.nrads)
    dtlw = float(nradl * cfg.dt)             # module_RA_RRTM.F90:842 DTLW = NRADL*DT
    dtsw = float(nrads * cfg.dt)             # module_RA_RRTM.F90:843 DTSW = NRADS*DT
    raddt = min(dtsw, dtlw)                  # grrad_nmmb.f:1200
    lslwr = nts % nradl == 0                 # module_RA_RRTM.F90:846
    lsswr = nts % nrads == 0                 # module_RA_RRTM.F90:845
    np3d = int(cfg.np3d)
    stream = ctx.stream

    dummys = ctx.tables.get(DUMMYS_KEY)
    if dummys is None:
        dummys = new_dummys(lmk, jm, im)
        ctx.tables[DUMMYS_KEY] = dummys

    # Inputs that exist before the LW block.
    clouds = _need(st, "RAD_CLOUDS", (NF_CLDS, lmk, jm, im), "phys-rrtm-grrad-clouds")
    tsfa = _need(st, "TSFA", (jm, im), "phys-rrtm-grrad-gases")
    coszen_loc = _need(st, "GRRAD_COSZEN_V", (jm, im), "phys-rrtm-coszmn")
    plvl = _need(st, "PINT", (lm + 1, jm, im), "the state (PINT)")
    tgrs = _need(st, "T", (lm, jm, im), "the state (T)")
    cldsa = _need(st, "CLDS", (NCLDSA, jm, im), "phys-rrtm-progcld2")
    mtopa = _need(st, "MTOP", (NMTOP, jm, im), "phys-rrtm-progcld2", dtype="int32")
    mbota = _need(st, "MBOT", (NMTOP, jm, im), "phys-rrtm-progcld2", dtype="int32")
    st.fields["cldsa"] = cldsa               # the name phys-rrtm-driver-post reads (grrad cldsa)
    tables = ctx.tables.get(DRIVER_CALL_KEY)
    if tables is None or "FLUXR_V" not in tables:
        raise KeyError("phys-rrtm-grrad-lw: ctx.tables['rrtm_driver_call']['FLUXR_V'] missing "
                       "(phys-rrtm-driver-call must run first; it zeroes FLUXR_V each call)")
    fluxr = tables["FLUXR_V"]

    # Outputs (scratch, allocated once).
    cldcov = st.add("cldcov", (lm, jm, im))
    htrlw = st.add("htrlw", (lm, jm, im))
    semis = st.add("semis", (jm, im))
    tsflw = st.add("tsflw", (jm, im))
    coszen = st.add("coszen", (jm, im))

    a = {"clouds": clouds, "tsfa": tsfa, "coszen_loc": coszen_loc, "coszen": coszen,
         "plvl": plvl, "T": tgrs, "cldsa": cldsa, "mtopa": mtopa, "mbota": mbota,
         "cldcov": cldcov, "fluxr": fluxr, "htrlw": htrlw, "semis": semis, "tsflw": tsflw}

    if lslwr:
        _dep(SETEMIS_MODULE).run(ctx)                      # sfcemis (grrad_nmmb.f:1963)
        a["sfcemis"] = _need(st, "sfcemis", (jm, im), "phys-rrtm-setemis")
        pre(clouds, dummys, np3d, stream=stream)          # 1978-1990
        _dep(LWRAD_MODULE).run(ctx)                       # htlwc, topflw_*, sfcflw_* (1985-2001)
        for name, shp in (("htlwc", (lmk, jm, im)),
                          ("topflw_upfxc", (jm, im)), ("topflw_upfx0", (jm, im)),
                          ("sfcflw_dnfxc", (jm, im)), ("sfcflw_upfxc", (jm, im)),
                          ("sfcflw_dnfx0", (jm, im)), ("sfcflw_upfx0", (jm, im))):
            a[name] = _need(st, name, shp, LWRAD_MODULE)
        post_heat(a, dummys, lm, lmk, jm, im, np3d, lslwr, stream=stream)   # 2046-2071
    else:
        for name in ("topflw_upfxc", "topflw_upfx0", "sfcflw_dnfxc", "sfcflw_upfxc",
                     "sfcflw_dnfx0", "sfcflw_upfx0"):
            a[name] = st.add(name, (jm, im))              # not read when lslwr is off

    post_flux(a, lm, lmk, jm, im, dtlw, raddt, LSSAV, lslwr, lsswr, stream=stream)
    return cldcov, fluxr


def selftest(seed: int = 1):
    """Full-size 954 x 835 x 60 run of the three kernels on random inputs (no column test).

    The lwrad and setemis outputs are stand-ins here (random arrays), which is the done-rule form for
    a glue unit. The checks are exact identities of the Fortran statements, not tolerances:
      clouds after pre + post_heat equal the input clouds bitwise on interior rows (np3d = 3 restore),
      and bitwise on boundary rows, which pre and post_heat never touch,
      semis == sfcemis, tsflw == tsfa, htrlw == htlwc shifted by KD, coszen == coszen_loc (interior rows),
      boundary rows of htrlw, cldcov and fluxr stay at their initial zeros.
    """
    import cupy as cp

    lm, jm, im = 60, 835, 954
    lmk = lm + 1
    rng = np.random.default_rng(seed)

    def rnd(shape, lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))

    clouds = rnd((NF_CLDS, lmk, jm, im), 0.0, 1.0e-4)
    clouds_in = clouds.copy()
    dummys = new_dummys(lmk, jm, im)
    a = {
        "htlwc": rnd((lmk, jm, im), 0.0, 5.0e-5),
        "sfcemis": rnd((jm, im), 0.9, 1.0),
        "tsfa": rnd((jm, im), 250.0, 310.0),
        "semis": cp.zeros((jm, im), dtype=cp.float32),
        "tsflw": cp.zeros((jm, im), dtype=cp.float32),
        "htrlw": cp.zeros((lm, jm, im), dtype=cp.float32),
        "coszen_loc": rnd((jm, im), -1.0, 1.0),
        "coszen": cp.zeros((jm, im), dtype=cp.float32),
        "clouds": clouds,
        "topflw_upfxc": rnd((jm, im), 100.0, 300.0),
        "topflw_upfx0": rnd((jm, im), 100.0, 300.0),
        "sfcflw_dnfxc": rnd((jm, im), 200.0, 400.0),
        "sfcflw_upfxc": rnd((jm, im), 300.0, 450.0),
        "sfcflw_dnfx0": rnd((jm, im), 200.0, 400.0),
        "sfcflw_upfx0": rnd((jm, im), 300.0, 450.0),
        "cldsa": rnd((NCLDSA, jm, im), 0.0, 1.0),
        "mtopa": cp.asarray(rng.integers(0, lm + 2, size=(NMTOP, jm, im)).astype(np.int32)),
        "mbota": cp.asarray(rng.integers(0, lm + 2, size=(NMTOP, jm, im)).astype(np.int32)),
        "plvl": rnd((lm + 1, jm, im), 1.0e3, 1.0e5),       # PINT, Pa
        "T": rnd((lm, jm, im), 200.0, 300.0),
        "cldcov": cp.zeros((lm, jm, im), dtype=cp.float32),
        "fluxr": cp.zeros((NFLUXR, jm, im), dtype=cp.float32),
    }
    fluxr_in = a["fluxr"].copy()
    lslwr, lsswr, lssav = True, True, LSSAV
    dtlw, raddt = 1800.0, 150.0
    pre(clouds, dummys, np3d=3)
    post_heat(a, dummys, lm, lmk, jm, im, 3, lslwr)
    post_flux(a, lm, lmk, jm, im, dtlw, raddt, lssav, lslwr, lsswr)
    cp.cuda.Stream.null.synchronize()

    out = {}
    for name in ("htrlw", "semis", "tsflw", "coszen", "cldcov", "fluxr", "clouds"):
        arr = clouds if name == "clouds" else a[name]
        out[name] = bool(cp.isfinite(arr).all())
        assert out[name], f"{name} has NaN/Inf"

    interior = slice(1, jm - 1)
    edges = [0, jm - 1]
    assert bool(cp.array_equal(clouds[:, :, interior, :], clouds_in[:, :, interior, :])), \
        "clouds not restored after np3d=3 pass on interior rows"
    assert bool(cp.array_equal(clouds[:, :, edges, :], clouds_in[:, :, edges, :])), \
        "boundary rows of clouds were touched"
    assert bool(cp.array_equal(a["semis"][interior], a["sfcemis"][interior]))
    assert bool(cp.array_equal(a["tsflw"][interior], a["tsfa"][interior]))
    assert bool(cp.array_equal(a["htrlw"][:, interior, :], a["htlwc"][KD:KD + lm, interior, :]))
    assert bool(cp.array_equal(a["coszen"][interior], a["coszen_loc"][interior])), "coszen copy"
    assert bool(cp.all(a["coszen"][edges] == 0)), "coszen boundary rows written"
    assert bool(cp.all(a["htrlw"][:, edges, :] == 0)), "htrlw boundary rows written"
    assert bool(cp.all(a["cldcov"][:, edges, :] == 0)), "cldcov boundary rows written"
    assert bool(cp.all(a["fluxr"][:, edges, :] == 0)), "fluxr boundary rows written"
    assert bool(cp.allclose(a["fluxr"][0, interior], dtlw * a["topflw_upfxc"][interior], rtol=1e-6))
    assert not bool(cp.array_equal(a["fluxr"], fluxr_in))
    return out
