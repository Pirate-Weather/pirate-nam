"""phys-turbl-driver: TURBL outer driver (nmm/module_TURBULENCE.F90:89-1420, nam.v4.2.11, 83df4e13c1).

Ported to nam-gpu. Kernel: namgpu/kernels/phys_turbl_driver.cu (15 entry points, one launch each).
run(ctx) is called on the physics cadence (NPHS = 2 steps, CONTRACT.md section 2 step 16). Inside,
the Fortran's own gates are kept: the surface and PBL blocks run when NTSD == 1 or
mod(NTSD, NPHS) == 0, and the NTSD == 0 block runs only on the first step.

Scheme set (parm/nam_configfile): turbulence myj (PBL 2), sfc_layer myj (JSFC 2), land_surface noah
(LSM 2), gwdflg true, ucmcall 0, pcpflg off, microphysics fer_hires (F_QC, F_QR, F_QS, F_QG true,
F_QI false, graupel out of CWM). Not selected by the 12 km configuration and not called here: GFS
PBL, GFDL surface layer, LISS, GFDL slab, urban canopy.

Launch order (Fortran order; dependency units run between the kernels and read or write the
state by name):
  turbl_prep_column, turbl_uv_phy, turbl_sfc_pre,
  [sfc block: turbl_sfc_save, phys-jsfc-jsfc, phys-noah-driver, turbl_noah_post,
   phys-noah-sfcdiags, turbl_rainbl_reset],
  turbl_akhs_out,
  [pbl block: phys-myj-myjpbl],
  turbl_edge_fill x4 (W, E, S, N),
  [turbulence /= myj: turbl_lpbl],
  turbl_sno_update, turbl_rad_accum,
  [gwdflg: phys-gwd-driver, turbl_dudt_add],
  turbl_tq_update, turbl_flux_accum, turbl_counters.

Scratch arrays are created with st.add on first use, under the Fortran names (the NMMB tables do
not carry them). Z and PHINT are [LM+1,JM,IM] interface arrays. TSFC and THS are required state
fields. They are the hand-off to the dependency units: DZ, PHINT, PHMID, RR, EXNER,
TH, DELP, U_PHY, V_PHY, RTHBLTEN, RQBLTEN, RQCBLTEN, RQIBLTEN, DUDT_GWD, DVDT_GWD (3-D); TSFC,
PSFC_OUT, MAVAIL, CHS, CHS2, CPM, CQS2, QGH, CHKLOWQ, ELFLX, FLHC, FLQC, THLOW, TLOW, QLOW,
CWMLOW, RSW_NET_SFC, RSW_DN_SFC, RLW_DN_SFC, FACTRS, EXNSFC, SFCZ, RAIN, RIMEF, TH2X, Q2X,
SFCEXC, SOILTB, RADOT, RIB, PSIM, PSIH, GZ1OZ0, ONE, CT (2-D); LPBL (int32, 2-D).
Flags handed to NOAHLSM (module_TURBULENCE.F90): the sfclay JSFC branch sets MYJ=.TRUE. (line 786)
and the LSM branch sets FRPCPN=.TRUE. (line 874). phys-noah-driver hard-codes FRPCPN=1; its kernel
does not read MYJ, so the MYJ value has no effect there.
Dependency wrappers: namgpu.physics.jsfc.phys_jsfc_jsfc, namgpu.physics.noah.phys_noah_driver,
namgpu.physics.noah.phys_noah_sfcdiags, namgpu.physics.myj.phys_myj_myjpbl,
namgpu.physics.gwd.phys_gwd_driver. A missing one raises at call time (not landed yet).

Constants: dtphs = NPHS * DT; SNOWH = SI * 0.001 at prep and SI = SNOWH * 1000 after the LPBL block
(SNO_FACTR for noah); one_from_mavail = 1 (LAND_SURFACE noah); NSOIL = 4 (CONTRACT.md section 1).
One stream, no host sync. The host work is the launch sequence; vertical vectors are uploaded
once and kept in ctx.tables.
"""
from __future__ import annotations

import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_turbl_driver"
UNIT_ID = "phys-turbl-driver"

JSFC = "namgpu.physics.jsfc.phys_jsfc_jsfc"
NOAH = "namgpu.physics.noah.phys_noah_driver"
SFCDIAGS = "namgpu.physics.noah.phys_noah_sfcdiags"
MYJ = "namgpu.physics.myj.phys_myj_myjpbl"
GWD = "namgpu.physics.gwd.phys_gwd_driver"

NSOIL = 4
SNO_FACTR_NOAH = np.float32(0.001)     # SNOWH = SI * 0.001 at prep (SURFACE_PHYSICS noah)
SNO_FACTR2_NOAH = np.float32(1000.0)   # SI = SNOWH * 1000 after the LPBL block
EDGE_BLOCK = (128, 1, 1)
ZT = "TURBL_Z"     # TURBL automatic array Z(IMS:IME,JMS:JME,1:LM+1); the state Z belongs to CDZDT/HDIFF

# TSFC (INOUT, Fortran argument) and THS are state fields, not scratch: a missing one is an error
# (see _need), never a silent zero-fill.
REQUIRED_STATE = ("TSFC", "THS")
SCRATCH_2D = (
    "PSFC_OUT", "VGFRCK", "CHS", "CHS2", "CPM", "CQS2", "QGH", "CHKLOWQ", "ELFLX", "FLHC",
    "FLQC", "THLOW", "TLOW", "QLOW", "CWMLOW", "RSW_NET_SFC", "RSW_DN_SFC", "RLW_DN_SFC", "FACTRS",
    "EXNSFC", "SFCZ", "RAIN", "RAINBL", "RIMEF", "TH2X", "Q2X", "SFCEVPX", "SFCEXC", "SOILTB",
    "RADOT", "RIB", "PSIM", "PSIH", "GZ1OZ0", "ONE", "CT", "MAVAIL", "DDATA", "SNOW", "SNOWH",
    "XLAND",
)
SCRATCH_3D = (
    "DZ", "TH", "EXNER", "RR", "PHMID", "DELP", "U_PHY", "V_PHY", "RTHBLTEN", "RQBLTEN",
    "RQCBLTEN", "RQIBLTEN", "DUDT_GWD", "DVDT_GWD", "DUDT", "DVDT",
)


def _dep(modname):
    """A dependency unit's wrapper module; raises if it is not landed yet."""
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"phys-turbl-driver needs {modname} (dependency unit, not landed)") from exc


def _vec(ctx, key, host):
    """A small host vertical vector uploaded once and kept in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _get(st, name, shape, dtype="float32"):
    """Existing array (state table or earlier scratch), else a zeroed scratch array.

    The shape of an existing array is checked: st.add returns whatever was created first, so a
    caller that asks for the wrong shape (Z was allocated [LM,JM,IM] by dyn-cdzdt-nonhydro while
    TURBL needs [LM+1,JM,IM]) would otherwise index out of bounds silently.
    """
    if name in st.fields:
        arr = st.fields[name]
        if tuple(arr.shape) != tuple(shape):
            raise RuntimeError(
                f"phys-turbl-driver: {name} has shape {tuple(arr.shape)}, expected {tuple(shape)}")
        return arr
    return st.add(name, shape, dtype)


# TURBL dummy-argument names that are other int_state names at the call site
# (module_SOLVER_GRID_COMP.F90:5490-5560): TSFC is int_state%TSKIN, PSFC_OUT is int_state%PSFC.
# The dummy name is bound to the same device array (one array under two names), never a copy.
ARG_ALIASES = (("TSFC", "TSKIN"), ("PSFC_OUT", "PSFC"))


def _bind_args(st):
    for dummy, actual in ARG_ALIASES:
        if actual in st.fields and st.fields.get(dummy) is not st.fields[actual]:
            st.fields[dummy] = st.fields[actual]


def _need(st, names):
    for n in names:
        if n not in st.fields:
            raise RuntimeError(f"phys-turbl-driver: state field {n} missing (no init binding)")


def _k(name):
    return kernels.get(UNIT, name)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nphs = int(ctx.cfg.nphs)
    dtphs = np.float32(nphs) * np.float32(ctx.dt)   # Fortran DTPHS = NPHS*DT, single precision
    ntsd = int(ctx.ntimestep)
    sfc_gate = (ntsd == 1) or (ntsd % nphs == 0)   # Fortran: NTSD==1 .OR. MOD(NTSD,NPHS)==0
    ntsd0 = 1 if ntsd == 0 else 0                  # Fortran: IF(NTSD==0)
    pcpflg = 0                                     # parm/nam_configfile pcpflg off
    gwdflg = True                                  # parm/nam_configfile gwdflg true
    one_from_mavail = 1                            # LAND_SURFACE noah (not 'nmm')

    s2, s3, sp = (jm, im), (lm, jm, im), (lm + 1, jm, im)
    _bind_args(st)
    _need(st, REQUIRED_STATE)
    for n in SCRATCH_2D:
        _get(st, n, s2)
    for n in SCRATCH_3D:
        _get(st, n, s3)
    _get(st, "PHINT", sp)
    _get(st, ZT, sp)                               # TURBL-local Z (module_TURBULENCE.F90:339): Z(LM+1)=SFCZ
                                                   # interface heights; NOT int_state%Z (CDZDT), so its own name
    _get(st, "LPBL", s2, dtype="int32")
    _get(st, "RADOT", s2)
    if "MAVAIL_SET" not in ctx.tables:
        st["MAVAIL"].fill(1.0)                     # NMM: MAVAIL = 1 (SOLVER_GRID_COMP:1030)
        ctx.tables["MAVAIL_SET"] = True

    sg2 = _vec(ctx, "turbl_sg2", g.sg2)
    psg1 = _vec(ctx, "turbl_psg1", g.psg1)
    sgml2 = _vec(ctx, "turbl_sgml2", g.sgml2)
    psgml1 = _vec(ctx, "turbl_psgml1", g.psgml1)

    i32, f32 = np.int32, np.float32
    cols = kernels.grid_cols(im, jm)
    blk = kernels.BLOCK_COL
    edge = (((max(im, jm) + EDGE_BLOCK[0] - 1) // EDGE_BLOCK[0]), 1, 1)

    # ---- (1) column prep ----
    _k("turbl_prep_column")(cols, blk, (
        st["SM"], st["SST"], st["THS"], st["TSFC"], st["VGFRCK"], st["VEGFRC"], st["SNO"],
        st["SI"], st["SNOW"], st["SNOWH"], st["XLAND"], st["T2"], st["Z0"], st["Z0BASE"],
        st["SICE"], st["Q2"], st["EXCH_H"], st["DZ"], st[ZT], st["FIS"], st["PD"], st["T"],
        st["Q"], st["PREC"], st["DDATA"], st["CZEN"], st["CZMEAN"], st["SIGT4"], st["RLWIN"],
        st["RSWIN"], st["RSWOUT"], st["F_RIMEF"], st["RAIN"], st["RAINBL"], st["FACTRS"],
        st["RLW_DN_SFC"], st["RSW_NET_SFC"], st["RSW_DN_SFC"], st["SFCZ"], st["EXNSFC"],
        st["TH2X"], st["Q2X"], st["RIMEF"], st["PHINT"], st["PHMID"], st["RR"], st["EXNER"],
        st["TH"], st["DELP"], st["RTHBLTEN"], st["RQBLTEN"], st["RQCBLTEN"], st["RQIBLTEN"],
        sg2, psg1, sgml2, psgml1, st["EPSQ2"], SNO_FACTR_NOAH, i32(pcpflg), i32(ntsd0),
        i32(im), i32(jm), i32(lm)))

    # ---- (2) H->V wind averaging and zeroing of the tendencies ----
    _k("turbl_uv_phy")(cols, blk, (
        st["U"], st["V"], st["U_PHY"], st["V_PHY"], st["DUDT"], st["DVDT"],
        i32(im), i32(jm), i32(lm)))

    # ---- (3) surface pre-state ----
    _k("turbl_sfc_pre")(cols, blk, (
        st["ONE"], st["MAVAIL"], i32(one_from_mavail), st["RMOL"], st["SFCEVPX"], st["TWBS"],
        st["QWBS"], st["THLOW"], st["TLOW"], st["QLOW"], st["CWMLOW"], st["CW"], st["T"],
        st["Q"], st["TH"], st[ZT], st["PBLH"], st["QGH"], st["CHS"], st["CPM"], st["CHS2"],
        st["RAINBL"], st["RAIN"], i32(im), i32(jm), i32(lm)))

    # ---- (4) sfc_and_sfclyr block: JSFC, NOAHLSM, SFCDIAGS ----
    if sfc_gate:
        _k("turbl_sfc_save")(cols, blk, (
            st["PHINT"], st["PSFC_OUT"], i32(im), i32(jm), i32(lm)))
        _dep(JSFC).run(ctx)
        _dep(NOAH).run(ctx)
        _k("turbl_noah_post")(cols, blk, (
            st["QWBS"], st["CHS"], st["STC"], st["SFCEVPX"], st["SFCEXC"], st["SOILTB"],
            f32(dtphs), i32(NSOIL), i32(im), i32(jm)))
        _dep(SFCDIAGS).run(ctx)
        _k("turbl_rainbl_reset")(cols, blk, (st["RAINBL"], i32(im), i32(jm)))

    # ---- (5) AKHS_OUT / AKMS_OUT ----
    _k("turbl_akhs_out")(cols, blk, (
        st["DZ"], st["AKHS"], st["AKMS"], st["AKHS_OUT"], st["AKMS_OUT"],
        i32(im), i32(jm), i32(lm)))

    # ---- (6) PBL block: MYJPBL on the same gate as the surface block ----
    if sfc_gate:
        _dep(MYJ).run(ctx)

    # ---- (7) uncomputed-edge fills, Fortran order W, E, S, N ----
    for side in (0, 1, 2, 3):
        _k("turbl_edge_fill")(edge, EDGE_BLOCK, (
            st["TH10"], st["Q10"], st["U10"], st["V10"], st["TSHLTR"], st["QSHLTR"],
            i32(side), i32(im), i32(jm)))

    # ---- (8) LPBL when turbulence /= myj (not selected by NAM; ported in full, gated) ----
    turbulence = "myj"
    if turbulence != "myj":
        _k("turbl_lpbl")(cols, blk, (
            st[ZT], st["SFCZ"], st["PBLH"], st["LPBL"], i32(im), i32(jm), i32(lm)))

    # ---- (9) snow bookkeeping and radiation accumulation ----
    _k("turbl_sno_update")(cols, blk, (
        st["SNOW"], st["SNOWH"], st["SNO"], st["SI"], f32(SNO_FACTR2_NOAH), i32(im), i32(jm)))
    _k("turbl_rad_accum")(cols, blk, (
        st["EPSR"], st["TSFC"], st["EXNSFC"], st["RSWIN"], st["RSWOUT"], st["RSWTOA"],
        st["RLWTOA"], st["FACTRS"], st["RLW_DN_SFC"], st["RADOT"], st["ASWIN"], st["ASWOUT"],
        st["ASWTOA"], st["ALWIN"], st["ALWOUT"], st["ALWTOA"], st["THS"], st["PREC"],
        i32(im), i32(jm)))

    # ---- (10) gravity wave drag; its tendencies join the PBL tendencies ----
    if gwdflg:
        _dep(GWD).run(ctx)
        _k("turbl_dudt_add")(cols, blk, (
            st["DUDT_GWD"], st["DVDT_GWD"], st["DUDT"], st["DVDT"], i32(im), i32(jm), i32(lm)))

    # ---- (11) T, Q, QC, CWM, F_ICE, F_RAIN (fer_hires: f_qr 1, f_qi 0, f_qs 1, f_qg 1, fer 1) ----
    _k("turbl_tq_update")(cols, blk, (
        st["T"], st["Q"], st["QC"], st["QI"], st["QR"], st["QS"], st["QG"], st["CW"],
        st["F_ICE"], st["F_RAIN"], st["EXNER"], st["RTHBLTEN"], st["RQBLTEN"], st["RQCBLTEN"],
        st["RQIBLTEN"], f32(dtphs), i32(1), i32(0), i32(1), i32(1), i32(1),
        i32(im), i32(jm), i32(lm)))

    # ---- (12) surface flux accumulation and counters ----
    _k("turbl_flux_accum")(cols, blk, (
        st["TWBS"], st["QWBS"], st["CHKLOWQ"], st["SM"], st["GRNFLX"], st["SFCSHX"],
        st["SFCLHX"], st["SFCEVP"], st["POTEVP"], st["POTFLX"], st["SUBSHX"],
        f32(dtphs), i32(im), i32(jm)))
    _k("turbl_counters")(cols, blk, (
        st["APHTIM"], st["ARDSW"], st["ARDLW"], st["ASRFC"], i32(im), i32(jm)))
