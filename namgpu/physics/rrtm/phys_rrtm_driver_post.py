"""phys-rrtm-driver-post: the post-grrad_nmmb half of the NMMB RRTM driver.

Ported from phys/module_RA_RRTM.F90:1113-1418 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE RRTM after the
grrad_nmmb call. Kernels: namgpu/kernels/phys_rrtm_driver_post.cu (rrtm_post_outputs, always;
rrtm_post_cloud_old, only when icwp == -1). Fortran-to-CUDA review: the kernel header.

Runs after the phys-rrtm-grrad-* units on each RRTM call (MOD(NTIMESTEP, NRADS) == 0 or MOD(NTIMESTEP, NRADL) == 0,
module_RADIATION.F90 path). The grrad_nmmb call itself is not made here.

Inputs this unit reads. Every grrad input is a view of the field its producer writes (no copy, no zero-fill). A
producer that has not run is a named KeyError, never silent zeros. The mapping is the grrad_nmmb output argument
list (module_RA_RRTM.F90:1131-1132, grrad_nmmb.f:680-681):
  HLW       = ctx.st['htrlw']                            [LM, JM, IM]  phys-rrtm-grrad-lw (post_heat)
  SWH       = ctx.st['HTRSW']                            [LM, JM, IM]  phys-rrtm-grrad-sw
  CLDCOV_V  = ctx.st['cldcov']                           [LM, JM, IM]  phys-rrtm-grrad-lw (post_flux); zeroed in place
  CLDSA_V   = ctx.st['cldsa']                            [5, JM, IM]   phys-rrtm-grrad-lw (post_flux); zeroed in place
  COSZDG_V  = ctx.st['GRRAD_COSZDG_V']                   [JM, IM]      phys-rrtm-coszmn
  COSZEN_V  = ctx.st['GRRAD_COSZEN_V']                   [JM, IM]      phys-rrtm-coszmn
  SFCFLW_DNFXC = ctx.st['sfcflw_dnfxc'], TOPFLW_UPFXC = ctx.st['topflw_upfxc']   [JM, IM]
                                                         phys-rrtm-lw-lwrad (its seam does not write ctx.st yet)
  SFCFSW_UPFXC = ctx.st['SFCFSW'][0], SFCFSW_DNFXC = ctx.st['SFCFSW'][1], SFCFSW_DNFX0 = ctx.st['SFCFSW'][3]
                                                         SFCFSW [4, JM, IM] packed, phys-rrtm-grrad-sw
  TOPFSW_UPFXC = ctx.st['TOPFSW'][0]                     TOPFSW [3, JM, IM] packed, phys-rrtm-grrad-sw
  RRTM_CU_CLOUD = ctx.tables['rrtm_driver_call']['CU_CLOUD']  int32 [JM, IM], phys-rrtm-driver-call
                                                         (Fortran CU_cloud is set at module_RA_RRTM.F90:1074 and 1087)
  CW, T, PD, DSG2, SGML2 (state); PDSG1, PSGML1, PT (grid)
  ctx.tables['rrtm_init']['ptopc'] (float32 [4]) and ['ltop'] (int32 [3]) from phys-rrtm-init
  CSMID, CCMID (float32 [LM, JM, IM]) are written by phys-rrtm-driver-prep (ICWP == -1) and by this unit (ICWP /= -1)

Writes (state, names from the NMMB tables; CSMID and COSZDG are unit-owned scratch added here):
  RLWTT, RSWTT [LM, JM, IM]; RLWIN, RSWIN, RSWINC, RSWOUT, RLWTOA, RSWTOA, CZMEAN, SIGT4, CFRACL, CFRACM, CFRACH,
  CLDFRA [LM, JM, IM]; ACFRST, ACFRCV, NCFRST, NCFRCV [JM, IM] (NCFRST, NCFRCV int32); CSMID [LM, JM, IM];
  COSZDG [JM, IM]; scratch: RRTM_CLDCFR float32 [3, JM, IM], RRTM_MTOP, RRTM_MBOT int32 [3, JM, IM]
  (RRTM locals of the old cloud block).

Launches: one kernel per call, plus rrtm_post_cloud_old when icwp == -1. Thread per column, no host sync, no per-step
allocation (scratch is allocated once through State.add on first use).

Refused (named breakage):
  - lssav false: the Fortran STOPs ('need to set LSSAV=TRUE'), so the port refuses rather than run a different path.
  - lm > 64: the kernels hold the level arrays in registers and local memory sized for 64 levels (NAM LM = 60).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_driver_post"
K_OUTPUTS = "rrtm_post_outputs"
K_OLD_CLOUD = "rrtm_post_cloud_old"
LMAX = 64
NRADL_DEFAULT = 48      # nam_configfile nradl (ctx.cfg.nradl is read when present)
NCLDSA = 5              # CLDSA_V(IX,5), module_RA_RRTM.F90 (cloud fractions L/M/H/TOT/BL)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables (same key as drv-tqadjust, phys-rad-driver)."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _producer(st, name: str, shape, owner: str):
    """A grrad input: the producer's own field, as a view. A missing or misshaped field is a named error."""
    arr = st.fields.get(name)
    if arr is None:
        raise KeyError(f"phys-rrtm-driver-post: ctx.st['{name}'] is missing (written by {owner}, which has not run)")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"phys-rrtm-driver-post: ctx.st['{name}'] has shape {tuple(arr.shape)}, expected {tuple(shape)}")
    return arr


def _grrad_inputs(st, lm: int, jm: int, im: int) -> dict:
    """The grrad_nmmb outputs this unit reads, keyed by the Fortran argument name (mapping in the header)."""
    lw_post = "phys-rrtm-grrad-lw (post_heat / post_flux)"
    lwrad = "phys-rrtm-lw-lwrad (its seam does not write ctx.st yet)"
    sw = "phys-rrtm-grrad-sw"
    cosz = "phys-rrtm-coszmn"
    sfcfsw = _producer(st, "SFCFSW", (4, jm, im), sw)     # upfxc, dnfxc, upfx0, dnfx0
    topfsw = _producer(st, "TOPFSW", (3, jm, im), sw)     # upfxc, dnfxc, upfx0
    return {
        "HLW": _producer(st, "htrlw", (lm, jm, im), lw_post),
        "SWH": _producer(st, "HTRSW", (lm, jm, im), sw),
        "CLDCOV_V": _producer(st, "cldcov", (lm, jm, im), lw_post),
        "CLDSA_V": _producer(st, "cldsa", (NCLDSA, jm, im), lw_post),
        "COSZDG_V": _producer(st, "GRRAD_COSZDG_V", (jm, im), cosz),
        "COSZEN_V": _producer(st, "GRRAD_COSZEN_V", (jm, im), cosz),
        "SFCFLW_DNFXC": _producer(st, "sfcflw_dnfxc", (jm, im), lwrad),
        "TOPFLW_UPFXC": _producer(st, "topflw_upfxc", (jm, im), lwrad),
        "SFCFSW_UPFXC": sfcfsw[0],
        "SFCFSW_DNFXC": sfcfsw[1],
        "SFCFSW_DNFX0": sfcfsw[3],
        "TOPFSW_UPFXC": topfsw[0],
    }


def _cu_cloud(ctx, jm: int, im: int):
    """RRTM_CU_CLOUD: the CU_cloud flag phys-rrtm-driver-call writes into ctx.tables['rrtm_driver_call']."""
    out = ctx.tables.get("rrtm_driver_call")
    if out is None or "CU_CLOUD" not in out:
        raise KeyError("phys-rrtm-driver-post: ctx.tables['rrtm_driver_call']['CU_CLOUD'] is missing "
                       "(written by phys-rrtm-driver-call, which has not run)")
    arr = out["CU_CLOUD"]
    if tuple(arr.shape) != (jm, im) or str(arr.dtype) != "int32":
        raise ValueError(f"phys-rrtm-driver-post: CU_CLOUD has shape {tuple(arr.shape)} dtype {arr.dtype}, "
                         f"expected ({jm}, {im}) int32")
    return arr


def _rrtm_init(ctx, key: str):
    tbl = ctx.tables.get("rrtm_init")
    if tbl is None or key not in tbl:
        raise KeyError(f"phys-rrtm-driver-post: rrtm_init['{key}'] is missing (RRTM_INIT, unit phys-rrtm-init)")
    return tbl[key]


def run(ctx, *, icwp: int = 1, ntimestep: int | None = None, lssav: bool = True, cncld: bool = True):
    """One RRTM post-processing call.

    icwp:  the configured icldflg (RRTM ICWP). NAM 12 km: 1 (ntcw = 3 gives icldflg = 1, see
           phys-rrtm-rad-initialize). -1 selects the old NAM/GFDL cloud block and needs the driver-prep
           CSMID/CCMID fields.
    ntimestep: NTIMESTEP for the LW SIGT4 test (default ctx.ntimestep).
    lssav: the RRTM LSSAV flag; the Fortran value is .TRUE. and False is refused.
    cncld: module CNCLD; the Fortran sets it false for ICWP /= -1 and it is true for the old cloud block.
    """
    import cupy as cp  # noqa: F401  (device arrays come from the State and the driver-call dict)

    if not lssav:
        raise NotImplementedError("phys-rrtm-driver-post: lssav=False makes the Fortran STOP "
                                  "('need to set LSSAV=TRUE'); the port does not run that path")
    g, st = ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > LMAX:
        raise ValueError(f"phys-rrtm-driver-post: lm={lm} exceeds the kernel level capacity {LMAX}")

    nts = int(ctx.ntimestep if ntimestep is None else ntimestep)
    nradl = int(getattr(ctx.cfg, "nradl", NRADL_DEFAULT))
    lw_step = 1 if nts % nradl == 0 else 0

    f = st.fields
    i32, f32 = np.int32, np.float32

    # grrad outputs, bound to their producer fields (views; named error if a producer has not run)
    gin = _grrad_inputs(st, lm, jm, im)
    cu_cloud = _cu_cloud(ctx, jm, im)
    csmid = st.add("CSMID", (lm, jm, im))
    ccmid = st.add("CCMID", (lm, jm, im))

    # state
    cw, t, pd = f["CW"], f["T"], f["PD"]
    dsg2, sgml2 = f["DSG2"], f["SGML2"]
    cldfra = f["CLDFRA"]
    rlwtt, rswtt = f["RLWTT"], f["RSWTT"]
    rlwin, rswin, rswinc, rswout = f["RLWIN"], f["RSWIN"], f["RSWINC"], f["RSWOUT"]
    rlwtoa, rswtoa, czmean, sigt4 = f["RLWTOA"], f["RSWTOA"], f["CZMEAN"], f["SIGT4"]
    cfracl, cfracm, cfrach = f["CFRACL"], f["CFRACM"], f["CFRACH"]
    acfrst, acfrcv, ncfrst, ncfrcv = f["ACFRST"], f["ACFRCV"], f["NCFRST"], f["NCFRCV"]
    cosz_dg = st.add("COSZDG", (jm, im))

    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    psgml1 = _device_vector(ctx, "psgml1", g.psgml1)
    pt = f32(g.pt)

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k1 = kernels.get(UNIT, K_OUTPUTS)
        k1(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm), i32(icwp), i32(1), i32(lw_step),
            gin["HLW"], gin["SWH"],
            gin["COSZDG_V"], gin["COSZEN_V"],
            gin["SFCFLW_DNFXC"], gin["SFCFSW_DNFXC"], gin["SFCFSW_UPFXC"], gin["SFCFSW_DNFX0"],
            gin["TOPFLW_UPFXC"], gin["TOPFSW_UPFXC"],
            gin["CLDCOV_V"], gin["CLDSA_V"],
            cw, t, cu_cloud,
            rlwtt, rswtt, cldfra, csmid,
            rlwin, rswin, rswinc, rswout, rlwtoa, rswtoa,
            czmean, sigt4, cosz_dg,
            cfracl, cfracm, cfrach,
            acfrst, acfrcv, ncfrst, ncfrcv,
        ))

        if icwp == -1:
            ptopc = _rrtm_init(ctx, "ptopc")
            ltop = _rrtm_init(ctx, "ltop")
            cldcfr = st.add("RRTM_CLDCFR", (3, jm, im))
            mtop = st.add("RRTM_MTOP", (3, jm, im), dtype="int32")
            mbot = st.add("RRTM_MBOT", (3, jm, im), dtype="int32")
            k2 = kernels.get(UNIT, K_OLD_CLOUD)
            k2(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
                i32(im), i32(jm), i32(lm), i32(int(cncld)), pt,
                pd, dsg2, sgml2, pdsg1, psgml1,
                csmid, ccmid, ptopc, ltop,
                cldcfr, mtop, mbot,
                cfracl, cfracm, cfrach,
                acfrst, acfrcv, ncfrst, ncfrcv,
            ))
    return None
