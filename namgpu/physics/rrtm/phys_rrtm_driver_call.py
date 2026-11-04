"""phys-rrtm-driver-call: per-column input preparation for grrad (RRTM driver call).

Fortran: phys/module_RA_RRTM.F90:879-1112 (NAM nam.v4.2.11, 83df4e13c1), the daypart body of
subroutine RRTM that fills the grrad_nmmb arguments (prsi, prsl, prslk, gt, gq, gr1, vvel, the
surface scalars, the subgrid and convective bogus cloud) before the call to grrad_nmmb.
Kernel: namgpu/kernels/phys_rrtm_driver_call.cu (phys_rrtm_driver_call, one thread per column,
one launch, no host sync, no device allocation after the first call).

Whole domain, one launch: the DAYPART chunking (dayparts, CHNK_RRTM) is a CPU detail and is dropped
(CONTRACT section 5); the per-daypart dpd flag belongs to the grrad unit.

Inputs (read from ctx.st, then ctx.tables, or from the explicit ``inputs`` dict):
  PINT (Fortran P8W, Pa, [LM+1,JM,IM]) and P_PHY (mid-layer Pa, [LM,JM,IM]) from ctx.st, where
  phys-rrtm-grrad-setup writes P_PHY every call; T, Q, CW, O3, OMGALF, CLDFRA, F_ICE, F_RAIN, F_RIMEF,
  QC, QI, QS, QR, QG, NI, CUPPT, GLON, GLAT, TSKIN, Z0, SICE, SI, SNOWC, MXSNAL, STDH, SM, ALBEDO
  (2-D [JM,IM] for the surface ones); TAUTOTAL [LM,JM,IM], CUTOP and CUBOT [JM,IM] (LM+1-HTOP /
  LM+1-HBOT) from the phys-rrtm-driver-prep output ctx.tables["rrtm_prep:out"]; and fer_gpvs_tbpvs
  [7501] and fer_gpvs_coef [4] (C1XPVS, C2XPVS, ...) from phys-fer-gpvs.
Outputs: ctx.tables["rrtm_driver_call"], a dict of device arrays allocated once and reused.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_driver_call"
KERNEL = "phys_rrtm_driver_call"
KEY = "rrtm_driver_call"
BLOCK = 128
NTRAC = 9
NFLUXR = 39
NSUBG = 5
CUPRATE = 24.0 * 1000.0

# (name, kind): kind "3d" = [LM,JM,IM], "lp1" = [LM+1,JM,IM], "2d" = [JM,IM]
_OUT_3D = ("PRSL", "PRSLK", "GT", "GQ", "VVEL", "F_ICEC", "F_RAINC", "R_RIME", "TAUCLOUDS",
           "CLDF", "CLDCOV_V")


def fortran_nint(x: float) -> int:
    """Fortran NINT: round half away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def convprate(nrads: int, nradl: int, dt: float, icwp: int, cucld: bool) -> float:
    """CONVPRATE exactly as module_RA_RRTM.F90:469-476 (host scalar, computed once per run)."""
    ntsph = fortran_nint(3600.0 / dt)
    nradpp = min(nrads, nradl)
    clstp = 1.0 * nradpp / ntsph
    cp = CUPRATE / clstp
    if icwp > 0 and cucld:
        cp = 1000.0 / clstp
    return cp


# Fortran SAVE initial values of the RRTM module (module_RA_RRTM.F90:74-75). Module variables, not
# namelist keys, so they are constants here; the keyword arguments of run() may override them.
CUCLD_FORTRAN = False
SUBGRID_FORTRAN = True
# Fields that phys-rrtm-driver-prep writes (ctx.tables[prep OUT_KEY]) and that this unit must read
# from there. ctx.st CUTOP/CUBOT, when present, belong to the BMJ convection unit (a different quantity).
PREP_KEY = "rrtm_prep:out"
PREP_NAMES = ("TAUTOTAL", "CUTOP", "CUBOT")


def _take(ctx, inputs, name):
    """Resolve one input: explicit dict, then the prep unit's output (PREP_NAMES), then ctx.st
    (P_PHY and PINT, written by phys-rrtm-grrad-setup and the NMMB state), then bare ctx.tables."""
    if inputs is not None and name in inputs:
        return inputs[name]
    if name in PREP_NAMES:
        prep = ctx.tables.get(PREP_KEY)
        if prep is None:
            raise KeyError(f"phys-rrtm-driver-call needs {name!r} from phys-rrtm-driver-prep "
                           f"(ctx.tables[{PREP_KEY!r}]); run that unit first")
        return prep[name]
    if name in ctx.st:
        return ctx.st[name]
    if name in ctx.tables:
        return ctx.tables[name]
    raise KeyError(f"phys-rrtm-driver-call needs input {name!r} (explicit inputs, ctx.st or ctx.tables)")


def _alloc(cp, shape):
    return cp.empty(shape, dtype=cp.float32)


def run(ctx, inputs=None, *, ntoz: int | None = None, icwp: int | None = None,
        nrads: int | None = None, nradl: int | None = None, dt: float | None = None,
        subgrid: bool | None = None, cucld: bool | None = None, microphysics: str | None = None):
    """Every control value comes from the run configuration unless passed explicitly:
    NTOZ = ioznflg, ICWP = icldflg (-1 when icmphys == 5), NRADS/NRADL, DT = ctx.dt, the microphysics
    string; SUBGRID and CUCLD are the Fortran SAVE values (CUCLD_FORTRAN, SUBGRID_FORTRAN)."""
    import cupy as cp

    cfg = ctx.cfg
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntoz = int(cfg.ioznflg) if ntoz is None else int(ntoz)                 # NTOZ = ioznflg (line 438)
    # icmphys = np3d (NAM sets np3d = icmphys, grrad_nmmb.f:1157); icldflg = 1 when ntcw > 0 else 0
    # (rad_initialize_nmmb.f:151-154). ICWP = icldflg, then -1 when icmphys == 5 (module_RA_RRTM.F90:437-445).
    icmphys = int(cfg.np3d)
    if icwp is None:
        icwp = -1 if icmphys == 5 else (1 if int(cfg.ntcw) > 0 else 0)
    nrads = int(cfg.nrads) if nrads is None else int(nrads)
    nradl = int(cfg.nradl) if nradl is None else int(nradl)
    dt = float(ctx.dt) if dt is None else float(dt)
    subgrid = SUBGRID_FORTRAN if subgrid is None else bool(subgrid)
    cucld = CUCLD_FORTRAN if cucld is None else bool(cucld)
    if microphysics is None:
        microphysics = cfg.microphysics
    f_ni = str(microphysics).strip() == "thompson"       # F_NI
    indx_cw = 4 if f_ni else 3                           # INDX_CW (line 457-459)

    p8w = _take(ctx, inputs, "PINT")
    p_phy = _take(ctx, inputs, "P_PHY")
    s3 = (lm, jm, im)
    s2 = (jm, im)

    tbpvs = ctx.tables["fer_gpvs_tbpvs"]

    out = ctx.tables.get(KEY)
    if out is None or out["PRSL"].shape != s3:
        out = {
            "PRSI": _alloc(cp, (lm + 1, jm, im)),
            "GR1": _alloc(cp, (NTRAC, lm, jm, im)),
            "CLDSA_V": _alloc(cp, (NSUBG, jm, im)),
            "FLUXR_V": _alloc(cp, (NFLUXR, jm, im)),
            "CU_CLOUD": cp.zeros(s2, dtype=cp.int32),
        }
        for name in _OUT_3D:
            out[name] = _alloc(cp, s3)
        for name in ("XLON", "XLAT", "TSEA", "TISFC", "ZORL", "SNWDPH", "SNCOVR", "SNOALB",
                     "HPRIME_V", "SLMSK", "FICE", "FLGMIN_L", "CV", "CVB", "CVT", "SFCALBEDO",
                     "SMX", "SINLAT", "COSLAT"):
            out[name] = _alloc(cp, s2)
        ctx.tables[KEY] = out

    k = kernels.get(UNIT, KERNEL)
    n = im * jm
    grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
    block = (BLOCK, 1, 1)
    i32, f32 = np.int32, np.float32

    args = (
        i32(im), i32(jm), i32(lm),
        i32(ntoz), i32(int(f_ni)), i32(int(subgrid)), i32(int(cucld)), i32(indx_cw),
        f32(convprate(nrads, nradl, dt, icwp, cucld)),
        p8w, p_phy,
        _take(ctx, inputs, "T"), _take(ctx, inputs, "Q"), _take(ctx, inputs, "CW"),
        _take(ctx, inputs, "O3"), _take(ctx, inputs, "OMGALF"), _take(ctx, inputs, "CLDFRA"),
        _take(ctx, inputs, "TAUTOTAL"),
        _take(ctx, inputs, "F_ICE"), _take(ctx, inputs, "F_RAIN"), _take(ctx, inputs, "F_RIMEF"),
        _take(ctx, inputs, "QC"), _take(ctx, inputs, "QI"), _take(ctx, inputs, "QS"),
        _take(ctx, inputs, "QR"), _take(ctx, inputs, "QG"), _take(ctx, inputs, "NI"),
        _take(ctx, inputs, "CUTOP"), _take(ctx, inputs, "CUBOT"), _take(ctx, inputs, "CUPPT"),
        _take(ctx, inputs, "GLON"), _take(ctx, inputs, "GLAT"), _take(ctx, inputs, "TSKIN"),
        _take(ctx, inputs, "Z0"), _take(ctx, inputs, "SICE"), _take(ctx, inputs, "SI"),
        _take(ctx, inputs, "SNOWC"), _take(ctx, inputs, "MXSNAL"), _take(ctx, inputs, "STDH"),
        _take(ctx, inputs, "SM"), _take(ctx, inputs, "ALBEDO"),
        tbpvs, ctx.tables["fer_gpvs_coef"],
        out["PRSI"], out["PRSL"], out["PRSLK"], out["GT"], out["GQ"], out["GR1"], out["VVEL"],
        out["F_ICEC"], out["F_RAINC"], out["R_RIME"], out["TAUCLOUDS"], out["CLDF"],
        out["CLDCOV_V"], out["CLDSA_V"], out["FLUXR_V"], out["CU_CLOUD"],
        out["XLON"], out["XLAT"], out["TSEA"], out["TISFC"], out["ZORL"], out["SNWDPH"],
        out["SNCOVR"], out["SNOALB"], out["HPRIME_V"], out["SLMSK"], out["FICE"],
        out["FLGMIN_L"], out["CV"], out["CVB"], out["CVT"], out["SFCALBEDO"], out["SMX"],
        out["SINLAT"], out["COSLAT"],
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, block, args)
    return out
