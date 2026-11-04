"""phys-noah-driver: NOAHLSM driver (NAM nmm/module_LS_NOAHLSM.F90:188-1161, nam.v4.2.11,
83df4e13c1, NMMB physics, NOAA/NCEP/EMC).

Kernel: namgpu/kernels/phys_noah_driver.cu (phys_noah_driver). One launch, one thread per land-sea
point over the B1 limits (its_b1 = 2, ite_b1 = IM-1, jts_b1 = 2, jte_b1 = JM-1, the caller's
ITS_B1..JTE_B1 from module_TURBULENCE.F90 call of NOAHLSM). The SFLX column physics is the
phys-noah-sflx unit, compiled into the same module through phys_noah_sflx.cuh (noah_sflx).

Called from phys-turbl-driver (namgpu/physics/driver/phys_turbl_driver.py) on the physics
cadence (NPHS = 2, DTBL = NPHS * DT). ``run(ctx)`` is the entry point it uses.

Names are the NAM/NMMB caller names (module_TURBULENCE.F90 call of NOAHLSM, CONTRACT section 1);
the scratch names are the ones phys-turbl-driver hands over. Fortran formal -> caller actual:
    DZ8W<-DZ  Q3D<-Q  P8W3D<-PHINT  T3D<-T  TSK<-TSFC  HFX<-TWBS  QFX<-QWBS  LH<-ELFLX
    GRDFLX<-GRNFLX  QGH<-QGH  GSW<-RSW_NET_SFC (unread)  SWDOWN<-RSW_DN_SFC  GLW<-RLW_DN_SFC
    SMSTAV, SMSTOT  SFCRUNOFF<-SSROFF  UDRUNOFF<-BGROFF  IVGTYP, ISLTYP  VEGFRA<-VGFRCK
    ALBEDO  ALBBCK<-ALBASE  ZNT<-Z0 (unread)  Z0<-Z0BASE  TMN<-TG  XLAND  XICE<-SICE  EMISS<-EPSR
    SNOWC  QSFC<-QSH  RAINBL  DZS<-DZSOIL (SLDPTH)  SMOIS<-SMC  TSLB<-STC  SNOW  CANWAT<-CMC
    CHS  CHS2  CQS2  CPM (unread)  ROVCP<-CAPPA (unread)  SR  RIMEF  CHKLOWQ  QZ0 (unread)
    MYJ (unread)  FRPCPN  SH2O  SNOWH (m)  SNOALB<-MXSNAL  ACSNOM  ACSNOW  SNOPCX  POTEVP  RIB
    ITIMESTEP (unread)  UCMCALL (must be 0)  IVEGSRC (1 = IGBP)

Reads (ctx.st, must exist): DZ, T, Q, PHINT, TSFC, RLW_DN_SFC, RSW_DN_SFC, RAINBL, QGH, EPSR,
    XLAND, SICE, IVGTYP, ISLTYP, VGFRCK, ALBEDO, ALBASE, Z0BASE, TG, MXSNAL, SR, RIMEF, CHS,
    CQS2, CHKLOWQ, SMC, STC, SH2O, SNOW, SNOWH, SNOWC, QSH, CMC, RIB, ACSNOM, ACSNOW, SNOPCX,
    POTEVP, SSROFF, BGROFF, GRNFLX, ELFLX, TWBS, QWBS.
Writes (in place, Fortran order): TSFC, TWBS, QWBS, ELFLX, GRNFLX, SMSTAV, SMSTOT, SSROFF, BGROFF,
    ALBEDO, SNOWC, QSH, CHS2, CQS2, CHKLOWQ, SMC, STC, SH2O, SNOW, SNOWH, CMC, ACSNOM, ACSNOW,
    SNOPCX, POTEVP.
Scratch created if absent: TWBS, QWBS, ELFLX, GRNFLX, SMSTAV, SMSTOT, SSROFF, BGROFF, QSH, CHS2,
    CHKLOWQ, SNOWC (zero-filled; the Fortran initialises the accumulators at init, and the
    outputs are written every call), SLDPTH (the NAM soil thicknesses 0.1, 0.3, 0.6, 1.0 m, from
    module_LS_NOAHLSM.F90:31).

Ids passed to SFLX (phys-noah-sflx contract): LUTYPE_ID = 1 for 'IGBP' (IVEGSRC 1) or 0 for
'USGS'; SLTYPE_ID = 1 for 'STAS' (MMINSL, module_LS_NOAHLSM.F90:1307).

Scalars: DT = NPHS * ctx.dt (DTBL = 50 s), NSOIL = 4, IVEGSRC = 1, UCMCALL = 0, FRPCPN = 1.
No host sync and no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_driver"
KERNEL = "phys_noah_driver"
BLOCK = 128

NSOIL = 4
IVEGSRC = 1
UCMCALL = 0
FRPCPN = 1
LUTYPE_ID = 1          # 'IGBP' (IVEGSRC 1)
SLTYPE_ID = 1          # 'STAS' (MMINSL)
SLDPTH_HOST = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)

# Fields the kernel reads that the state tables / turbl hand-over provide (raise if absent).
REQUIRED_3D = ("DZ", "T", "Q", "PHINT")            # PHINT is [LM+1, JM, IM]
REQUIRED_SOIL = ("SMC", "STC", "SH2O")             # [NSOIL, JM, IM]
REQUIRED_2D = (
    "TSFC", "RLW_DN_SFC", "RSW_DN_SFC", "RAINBL", "QGH", "EPSR", "XLAND", "SICE",
    "IVGTYP", "ISLTYP", "VGFRCK", "ALBEDO", "ALBASE", "Z0BASE", "TG", "MXSNAL", "SR", "RIMEF",
    "CHS", "CQS2", "CHKLOWQ", "SNOW", "SNOWH", "SNOWC", "QSH", "CMC", "RIB", "ACSNOM", "ACSNOW",
    "SNOPCX", "POTEVP", "GRNFLX", "ELFLX",
)
def _need(st, names):
    missing = [n for n in names if n not in st.fields]
    if missing:
        raise RuntimeError(f"phys-noah-driver: state fields missing: {', '.join(missing)}")


def _field2(st, name, shape, dtype="float32"):
    """The state array when it exists, otherwise a zero-filled unit scratch array."""
    if name in st.fields:
        return st.fields[name]
    return st.add(name, shape, dtype)


def _soil_thickness(ctx):
    """DZS for NOAHLSM. TURBL is called with the MODULE_LS_NOAHLSM module array SLDPTH
    (module_SOLVER_GRID_COMP.F90:91-92 USE, 5491 actual argument), the constant (/0.1,0.3,0.6,1.0/)
    of module_LS_NOAHLSM.F90:31, not int_state%SLDPTH. Uploaded once to ctx.tables. (The state field
    SLDPTH is a zero-allocated NMMB table entry until phys-noah-init copies the module values in, line
    966; reading it here gave DZS = 0 and NaN soil over every land point.)"""
    key = "noah_module_sldpth"
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(SLDPTH_HOST)
    return ctx.tables[key]


def _tbl_args(ctx):
    """REDPRM table arguments for SFLX (NoahTbl in phys_noah_sflx.cuh): device tables and GENPARM/VEGPARM
    scalars bound once by phys-noah-parm-tables (ctx.tables["noah_parm"]); no fallback."""
    if "noah_tbl_args" not in ctx.tables:
        from namgpu.physics.noah import phys_noah_redprm as rp
        tabs, scal = rp._bound_tables(ctx)
        f32 = np.float32
        ctx.tables["noah_tbl_args"] = (
            tabs["ALBTBL"], tabs["Z0TBL"], tabs["SHDTBL"], tabs["RSTBL"], tabs["RGLTBL"], tabs["HSTBL"],
            tabs["SNUPTBL"], tabs["LAITBL"], tabs["NROTBL"],
            tabs["BB"], tabs["DRYSMC"], tabs["F11"], tabs["MAXSMC"], tabs["REFSMC"], tabs["SATPSI"],
            tabs["SATDK"], tabs["SATDW"], tabs["WLTSMC"], tabs["QTZ"], tabs["SLOPE_DATA"],
            f32(scal["SBETA_DATA"]), f32(scal["FXEXP_DATA"]), f32(scal["CSOIL_DATA"]),
            f32(scal["SALP_DATA"]), f32(scal["REFDK_DATA"]), f32(scal["REFKDT_DATA"]),
            f32(scal["FRZK_DATA"]), f32(scal["ZBOT_DATA"]), f32(scal["CZIL_DATA"]),
            f32(scal["TOPT_DATA"]), f32(scal["CMCMAX_DATA"]), f32(scal["CFACTR_DATA"]),
            f32(scal["RSMAX_DATA"]), np.int32(scal["BARE"]))
    return ctx.tables["noah_tbl_args"]


def run(ctx):
    """Launch the NOAHLSM driver over the B1 land-sea points. Returns the output arrays by name."""
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if int(getattr(ctx.cfg, "ucmcall", 0) if ctx.cfg is not None else 0) != UCMCALL:
        raise RuntimeError("phys-noah-driver: the urban (UCMCALL == 1) branch is not ported; "
                           "ucmcall must be 0")

    _need(st, REQUIRED_3D + REQUIRED_SOIL + REQUIRED_2D)
    if "PHINT" in st.fields and tuple(st["PHINT"].shape) != (lm + 1, jm, im):
        raise RuntimeError("phys-noah-driver: PHINT must be [LM+1, JM, IM]")

    if ctx.cfg is not None:
        nphs = int(getattr(ctx.cfg, "nphs", 2))
    else:
        nphs = 2
    dt = np.float32(nphs * float(ctx.dt))            # DTBL = NPHS * DT

    shape2 = (jm, im)
    twbs = _field2(st, "TWBS", shape2)
    qwbs = _field2(st, "QWBS", shape2)
    smstav = _field2(st, "SMSTAV", shape2)
    smstot = _field2(st, "SMSTOT", shape2)
    ssroff = _field2(st, "SSROFF", shape2)
    bgroff = _field2(st, "BGROFF", shape2)
    qsh = _field2(st, "QSH", shape2)
    chs2 = _field2(st, "CHS2", shape2)
    chklowq = _field2(st, "CHKLOWQ", shape2)
    snowc = _field2(st, "SNOWC", shape2)
    grnflx = _field2(st, "GRNFLX", shape2)
    elflx = _field2(st, "ELFLX", shape2)
    sldpth = _soil_thickness(ctx)

    ni = im - 2 if im > 2 else 0       # B1: i0 in [1, IM-2]
    nj = jm - 2 if jm > 2 else 0
    ncol = ni * nj
    if ncol == 0:
        return {}
    grid = ((ncol + BLOCK - 1) // BLOCK, 1, 1)

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    i32 = np.int32
    f32 = np.float32
    with scope:
        k(grid, (BLOCK, 1, 1), (
            st["DZ"], st["Q"], st["PHINT"], st["T"],
            st["TSFC"], twbs, qwbs, elflx, grnflx,
            st["QGH"], st["RLW_DN_SFC"], st["RSW_DN_SFC"],
            smstav, smstot, ssroff, bgroff,
            st["IVGTYP"], st["ISLTYP"], st["VGFRCK"],
            st["ALBEDO"], st["ALBASE"], st["Z0BASE"], st["TG"], st["XLAND"], st["SICE"], st["EPSR"],
            snowc, qsh, st["RAINBL"], sldpth,
            st["SMC"], st["STC"], st["SNOW"], st["CMC"],
            st["CHS"], chs2, st["CQS2"], st["SR"], st["RIMEF"], chklowq,
            st["SH2O"], st["SNOWH"], st["MXSNAL"],
            st["ACSNOM"], st["ACSNOW"], st["SNOPCX"], st["POTEVP"], st["RIB"],
            f32(dt), i32(NSOIL), i32(IVEGSRC), i32(UCMCALL), i32(FRPCPN),
            i32(LUTYPE_ID), i32(SLTYPE_ID),
            i32(2), i32(im - 1), i32(2), i32(jm - 1),
            i32(im), i32(jm), i32(lm), *_tbl_args(ctx)))

    return {"TSFC": st["TSFC"], "TWBS": twbs, "QWBS": qwbs, "ELFLX": elflx, "GRNFLX": grnflx,
            "SMSTAV": smstav, "SMSTOT": smstot, "SSROFF": ssroff, "BGROFF": bgroff,
            "ALBEDO": st["ALBEDO"], "SNOWC": snowc, "QSH": qsh, "CHS2": chs2, "CQS2": st["CQS2"],
            "CHKLOWQ": chklowq, "SMC": st["SMC"], "STC": st["STC"], "SH2O": st["SH2O"],
            "SNOW": st["SNOW"], "SNOWH": st["SNOWH"], "CMC": st["CMC"], "ACSNOM": st["ACSNOM"],
            "ACSNOW": st["ACSNOW"], "SNOPCX": st["SNOPCX"], "POTEVP": st["POTEVP"]}
