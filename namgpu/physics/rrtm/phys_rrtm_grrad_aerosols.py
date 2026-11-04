"""phys-rrtm-grrad-aerosols: the SETAER call inside GRRAD_NMMB (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:1540-1576. NOAA US government work;
the NOAA header of grrad_nmmb.f is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_grrad_aerosols.cu, which has no device code of its own).

The Fortran call has no arithmetic: it hands the layer and level fields to setaer and gets the
per-band aerosol optical properties back. The port is a wrapper that binds the RAD_* arrays that
phys-rrtm-grrad-setup writes, reads the module state that rad_initialize_nmmb and aer_update set
(the Fortran module variables iaerflg, laswflg, lalwflg, lavoflg, ivflip, ivolae, kyrsav, kmonsav),
and launches the setaer stages through phys_rrtm_setaer.run_call.

Bindings (grrad-setup names, all written by phys-rrtm-grrad-setup; shapes [LMK|LMP, JM, IM]):
  plvl    <- RAD_PLVL     [LMP, JM, IM] mb   (Fortran plvl: 10 * prsi, cb to mb, plus extra top)
  plyr    <- RAD_PLYR     [LMK, JM, IM] mb
  prslk1  <- RAD_PRSLK1   [LMK, JM, IM]
  tvly    <- RAD_TVLY     [LMK, JM, IM] K    (Fortran tvly, computed in grrad-setup)
  rhly    <- RAD_RHLY     [LMK, JM, IM]
  tlyr    <- RAD_TLYR     [LMK, JM, IM] K
  qlyr    <- RAD_QLYR     [LMK, JM, IM] kg/kg (Fortran qlyr, computed in grrad-setup)
  tracer1 <- RAD_TRACER1  [NTRAC, LMK, JM, IM]
  slmsk   <- SLMSK        [JM, IM]
  xlon,xlat <- XLON, XLAT [JM, IM] radians
Module state (aer_state, read once at initialisation from ctx.tables['rad_physparam'] and
ctx.tables['aer_ivolae'] / ctx.tables['aer_volc'] when present):
  NAM iaer = 11 gives iaerflg 11, laswflg and lalwflg true, lavoflg false; iflip = 0 gives ivflip 0.
Outputs (device float32, written by setaer): faersw [NF_AESW=3, NBDSW=14, LMK, JM, IM],
  faerlw [NF_AELW=3, NBDLW=16, LMK, JM, IM].
One host read of the 26-int physparam block happens at the first call (initialisation), never per step.
"""
from __future__ import annotations

from namgpu.physics.rrtm.phys_rrtm_rad_initialize import (
    RI_IAERFLG, RI_IVFLIP, RI_LALWFLG, RI_LASWFLG, RI_LAVOFLG,
)

UNIT = "phys_rrtm_grrad_aerosols"
SETAER_MODULE = "namgpu.physics.rrtm.phys_rrtm_setaer"
STATE_KEY = "aer_state"

NF_AESW = 3
NF_AELW = 3
NBDSW = 14
NBDLW = 16


def _setaer():
    """The setaer wrapper (unit phys-rrtm-setaer)."""
    import importlib

    return importlib.import_module(SETAER_MODULE)


def new_outputs(lmk: int, jm: int, im: int):
    """faersw / faerlw scratch, allocated once by the caller and reused every call."""
    import cupy as cp

    faersw = cp.zeros((NF_AESW, NBDSW, lmk, jm, im), dtype=cp.float32)
    faerlw = cp.zeros((NF_AELW, NBDLW, lmk, jm, im), dtype=cp.float32)
    return faersw, faerlw


def aer_state(ctx):
    """Module state setaer reads. Built once; the ivolae array is read from ctx.tables at each call."""
    st = ctx.tables.get(STATE_KEY)
    if st is not None:
        return st
    rip = ctx.tables.get("rad_physparam")
    if rip is None:
        raise RuntimeError(
            "phys-rrtm-grrad-aerosols: ctx.tables['rad_physparam'] is missing; run phys-rrtm-rad-initialize first")
    p = rip.get()  # one-time initialisation read of 26 ints (rad_initialize_nmmb flags)
    volc = ctx.tables.get("aer_volc")
    st = {
        "iaerflg": int(p[RI_IAERFLG]),
        "laswflg": bool(p[RI_LASWFLG]),
        "lalwflg": bool(p[RI_LALWFLG]),
        "lavoflg": bool(p[RI_LAVOFLG]),
        "ivflip": int(p[RI_IVFLIP]),
        "nlwbnd": NBDLW,                       # lalw1bd false: NLWBND = 16
        "kyrsav": int(getattr(volc, "kyrsav", 1)),    # radiation_aerosols.f:591 default 1
        "kmonsav": int(getattr(volc, "kmonsav", 1)),  # radiation_aerosols.f:592 default 1
    }
    ctx.tables[STATE_KEY] = st
    return st


def grrad_aerosols(ctx, faersw, faerlw, *, lsswr: bool, lslwr: bool):
    """Launch the setaer call on the grrad-setup RAD_* arrays. Outputs are written in place.
    lsswr / lslwr are the radiation-call flags of this step (Fortran grrad_nmmb arguments, set by the
    driver schedule); they are not read from cfg, because the config has no such fields."""
    st = ctx.st
    a = aer_state(ctx)
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk, lmp = lm + 1, lm + 2
    cols = im * jm
    ntrac = int(st["RAD_TRACER1"].shape[0])
    rs = lambda x, *shape: x.reshape(*shape)  # contiguous [.., JM, IM] -> [.., JM*IM] views
    return _setaer().run_call(
        prsi=rs(st["RAD_PLVL"], lmp, cols),
        prsl=rs(st["RAD_PLYR"], lmk, cols),
        prslk=rs(st["RAD_PRSLK1"], lmk, cols),
        tvly=rs(st["RAD_TVLY"], lmk, cols),
        rhlay=rs(st["RAD_RHLY"], lmk, cols),
        slmsk=rs(st["SLMSK"], cols),
        tracer=rs(st["RAD_TRACER1"], ntrac, lmk, cols),
        xlon=rs(st["XLON"], cols),
        xlat=rs(st["XLAT"], cols),
        imax=cols, nlay=lmk, nlp1=lmp,
        lsswr=bool(lsswr), lslwr=bool(lslwr),
        tlay=rs(st["RAD_TLYR"], lmk, cols),
        qlay=rs(st["RAD_QLYR"], lmk, cols),
        aerosw=rs(faersw, NF_AESW, NBDSW, lmk, cols),
        aerolw=rs(faerlw, NF_AELW, NBDLW, lmk, cols),
        iaerflg=a["iaerflg"], laswflg=a["laswflg"], lalwflg=a["lalwflg"], lavoflg=a["lavoflg"],
        ivflip=a["ivflip"], nlwbnd=a["nlwbnd"],
        ivolae=ctx.tables.get("aer_ivolae"),
        kyrsav=a["kyrsav"], kmonsav=a["kmonsav"],
        stream=ctx.stream,
        tables=ctx.tables,
    )


def run(ctx, *, lsswr: bool, lslwr: bool):
    """Driver entry: arrays come from the grrad-setup RAD_* scratch (CONTRACT names) and ctx.tables.
    lsswr / lslwr: this step's radiation-call flags (driver schedule)."""
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk = lm + 1
    faersw = ctx.tables.get("rrtm_faersw")
    faerlw = ctx.tables.get("rrtm_faerlw")
    if faersw is None or faerlw is None:
        faersw, faerlw = new_outputs(lmk, jm, im)
        ctx.tables["rrtm_faersw"], ctx.tables["rrtm_faerlw"] = faersw, faerlw
    return grrad_aerosols(ctx, faersw, faerlw, lsswr=lsswr, lslwr=lslwr)
