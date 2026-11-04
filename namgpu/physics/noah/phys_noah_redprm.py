# Module header copied verbatim from the NAM Fortran (nam.v4.2.11, 83df4e13c1,
# phys/module_LS_NOAHLSM.F90 lines 1-22). NOAA US government work; kept as the notice.
# !-----------------------------------------------------------------------
# !
#       MODULE MODULE_LS_NOAHLSM
# !
# !-----------------------------------------------------------------------
# !***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
# !-----------------------------------------------------------------------
# !
# ! HISTORY LOG:
# !
# !   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
# !               Vasic - SOILW limitted to positive values.
# !
# !-----------------------------------------------------------------------
# !
#       USE ESMF
#       USE MODULE_SF_URBAN
#       USE MODULE_INCLUDE
# !
# !-----------------------------------------------------------------------
# !
#       PRIVATE
# NOTICE, ArWen (Apache-2.0): the start point is ArWen gpuwm/third_party/arwen/core/kernels/noah.cu.
# The ArWen LICENSE (Apache-2.0) is the repo root LICENSE. noah.cu carries no per-file copyright or
# licence header (checked 2026-11-04), so none is reproduced here.
# Copied from ArWen gpuwm/third_party/arwen/core/kernels/noah.cu (Apache-2.0) for nam-gpu, 2026-11-04.
"""phys-noah-redprm: REDPRM, per-point Noah LSM parameters (NAM v4.2.11, phys/module_LS_NOAHLSM.F90:3471-3664).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1). NOAA US government work; the module's NOAA header is kept
as the notice in the kernel file and above.
Kernel: namgpu/kernels/phys_noah_redprm.cu (phys_noah_redprm, one thread per (i, j), one launch).
No host sync in the launch. run() is the init-stage call: it reads the soil thickness and one refusal
counter once, and raises if any point is refused.

Inputs
  VEGTYP, SOILTYP, SLOPETYP  int32 [JM, IM], Fortran 1-based class indices. The static init fills them
                             (phys-noah-init); they must already be in st.fields when run() is called.
                             A class outside 1..NLUS / 1..NSLTYPE / 1..NSLOPE is refused, not clamped.
  Tables                     ctx.tables["noah_parm"] (device, flat keys "veg.X", "soil.X", "gen.X") and
                             ctx.tables["noah_parm_host"] (scalars), both bound by phys-noah-parm-tables.
                             run() never re-parses the table files and has no synthetic fallback: without
                             the bindings it raises.
  SLDPTH  [NSOIL] state field, the run's layer thickness. NOAH_LSM_INIT writes it from DZS (Fortran line
          635). When no one has written it, it is set to the module default 0.1, 0.3, 0.6, 1.0 m
          (module_LS_NOAHLSM.F90:31), as phys-noah-driver does. The number of layers is len(SLDPTH);
          it must equal NSOIL (the state's soil layout), else run() raises.
  ZSOIL   [NSOIL] derived from SLDPTH as SFLX does for the non-ice branch (module_LS_NOAHLSM.F90:1835-1837).
  SHDFAC, Z0BRD, ALBBRD  caller inputs (in/out). The Fortran leaves them as the caller set them; REDPRM
          writes them only under LOCAL (false in NAM) or, for SHDFAC, when VEGTYP is BARE. The caller
          must put them in the state (SHDFAC: green vegetation fraction from the greenness state; Z0BRD:
          background roughness, consumed by phys-noah-snowz0). run() raises if one is missing.

Outputs (CONTRACT/Fortran names, float32 [JM, IM] unless noted): CSOIL BEXP DKSAT DWSAT F1 PSISAT
QUARTZ SMCDRY SMCMAX SMCREF SMCWLT ZBOT SALP SBETA FXEXP REFKDT FRZX KDT SLOPE CZIL PTU TOPT CMCMAX
CFACTR RSMAX SNUP RSMIN RGL HS LAI; NROOT int32 [JM, IM]; RTDIS float32 [NSOIL, JM, IM].
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_parm_tables as parm

UNIT = "phys_noah_redprm"
KERNEL = "phys_noah_redprm"
BLOCK = (32, 8, 1)
NSOIL = parm.NSOIL
SLDPTH_DEFAULT = (0.1, 0.3, 0.6, 1.0)     # module_LS_NOAHLSM.F90:31 (DZSOIL default)
NLUS, NSLTYPE, NSLOPE = 50, 30, 30        # module_LS_NOAHLSM.F90:151,162,171

# Per-point float32 outputs, in the kernel's argument order (29 arrays).
FLOAT_OUT = ("CSOIL", "BEXP", "DKSAT", "DWSAT", "F1", "PSISAT", "QUARTZ", "SMCDRY", "SMCMAX",
             "SMCREF", "SMCWLT", "ZBOT", "SALP", "SBETA", "FXEXP", "REFKDT", "FRZX", "KDT",
             "SLOPE", "CZIL", "PTU", "TOPT", "CMCMAX", "CFACTR", "RSMAX", "SNUP", "RSMIN", "RGL",
             "HS", "LAI")
INOUT = ("SHDFAC", "Z0BRD", "ALBBRD")     # caller-supplied state fields (see run)
TYPES = ("VEGTYP", "SOILTYP", "SLOPETYP")
VEG_TBL = ("ALBTBL", "Z0TBL", "SHDTBL", "RSTBL", "RGLTBL", "HSTBL", "SNUPTBL", "LAITBL")
SOIL_TBL = ("BB", "DRYSMC", "F11", "MAXSMC", "REFSMC", "SATPSI", "SATDK", "SATDW", "WLTSMC", "QTZ")
GEN_SCALARS = ("SBETA_DATA", "FXEXP_DATA", "CSOIL_DATA", "SALP_DATA", "REFDK_DATA", "REFKDT_DATA",
               "FRZK_DATA", "ZBOT_DATA", "CZIL_DATA")
VEG_SCALARS = ("TOPT_DATA", "CMCMAX_DATA", "CFACTR_DATA", "RSMAX_DATA")


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def _bound_tables(ctx):
    """Device arrays (flat keys -> short names) and host scalars from phys-noah-parm-tables.
    parm.run is idempotent: the first call parses and binds, later calls return the binding.
    With no table directory it raises; there is no synthetic fallback."""
    dev = parm.run(ctx)
    host = ctx.tables[parm.KEY_HOST]
    tabs = {k.split(".", 1)[1]: v for k, v in dev.items() if k.startswith(("veg.", "soil.", "gen."))}
    scal = {k: float(host.gen[k]) for k in GEN_SCALARS}
    scal.update({k: float(host.veg[k]) for k in VEG_SCALARS})
    scal["BARE"] = int(host.bare)
    return tabs, scal


def _soil_thickness(st):
    """SLDPTH from the state. NOAH_LSM_INIT writes it from DZS (module_LS_NOAHLSM.F90:635); if nothing
    has, the module default is set, the same as phys-noah-driver._soil_thickness."""
    if "SLDPTH" in st.fields:
        return st.fields["SLDPTH"]
    a = st.add("SLDPTH", (NSOIL,))
    a[...] = st.xp.asarray(np.asarray(SLDPTH_DEFAULT, dtype=np.float32))
    return a


def _sldpth_zsoil(xp, sldpth=SLDPTH_DEFAULT):
    """ZSOIL for the non-ice branch of SFLX (module_LS_NOAHLSM.F90:1835-1837): returns (SLDPTH, ZSOIL)
    as xp arrays. Called by phys-noah-tmpavg with a host tuple."""
    sd = np.asarray(sldpth, dtype=np.float32)
    zs = np.zeros(sd.shape[0], dtype=np.float32)
    zs[0] = -sd[0]
    for k in range(1, sd.shape[0]):
        zs[k] = -sd[k] + zs[k - 1]
    return xp.asarray(sd), xp.asarray(zs)


def redprm(types_, dev, scal, sldpth, zsoil, outs, bad, *, local: int, im: int, jm: int, nsoil: int):
    """One launch. types_: (vegtyp, soiltyp, slopetyp) int32 [JM, IM]; dev: bound device tables;
    sldpth, zsoil: [nsoil] float32 device; bad: int32 [1] refusal counter."""
    k = kernels.get(UNIT, KERNEL)
    vegtyp, soiltyp, slopetyp = types_
    args = [
        vegtyp, soiltyp, slopetyp,
        dev["ALBTBL"], dev["Z0TBL"], dev["SHDTBL"], dev["RSTBL"], dev["RGLTBL"], dev["HSTBL"],
        dev["SNUPTBL"], dev["LAITBL"], dev["NROTBL"],
        dev["BB"], dev["DRYSMC"], dev["F11"], dev["MAXSMC"], dev["REFSMC"], dev["SATPSI"],
        dev["SATDK"], dev["SATDW"], dev["WLTSMC"], dev["QTZ"],
        dev["SLOPE_DATA"], sldpth, zsoil,
        np.float32(scal["SBETA_DATA"]), np.float32(scal["FXEXP_DATA"]), np.float32(scal["CSOIL_DATA"]),
        np.float32(scal["SALP_DATA"]), np.float32(scal["REFDK_DATA"]), np.float32(scal["REFKDT_DATA"]),
        np.float32(scal["FRZK_DATA"]), np.float32(scal["ZBOT_DATA"]), np.float32(scal["CZIL_DATA"]),
        np.float32(scal["TOPT_DATA"]), np.float32(scal["CMCMAX_DATA"]), np.float32(scal["CFACTR_DATA"]),
        np.float32(scal["RSMAX_DATA"]),
        np.int32(scal["BARE"]), np.int32(local), np.int32(im), np.int32(jm), np.int32(nsoil),
    ]
    args += [outs[n] for n in FLOAT_OUT]
    args += [outs["NROOT"], outs["SHDFAC"], outs["Z0BRD"], outs["ALBBRD"], outs["RTDIS"], bad]
    k(_grid(im, jm), BLOCK, tuple(args))
    return outs


def _alloc_outs(st, jm: int, im: int):
    out = {n: st.add(n, (jm, im)) for n in FLOAT_OUT}
    out["NROOT"] = st.add("NROOT", (jm, im), "int32")
    out["RTDIS"] = st.add("RTDIS", (NSOIL, jm, im))
    for n in INOUT:                       # caller-owned: bound, never allocated here
        out[n] = st.fields[n]
    return out


def run(ctx, *, local: int = 0):
    """One launch on the state's full grid. Tables come from phys-noah-parm-tables (ctx.tables);
    VEGTYP/SOILTYP/SLOPETYP and the caller inputs SHDFAC/Z0BRD/ALBBRD must already be in ctx.st.
    Returns the outputs. Raises if any point is refused (class index or NROOT out of range)."""
    st = ctx.st
    missing = [n for n in TYPES if n not in st.fields]
    if missing:
        raise parm.NoahParmTableError(
            f"phys-noah-redprm needs class fields {missing} in the state (filled by phys-noah-init)")
    missing = [n for n in INOUT if n not in st.fields]
    if missing:
        raise parm.NoahParmTableError(
            f"phys-noah-redprm needs caller inputs {missing} in the state: SHDFAC is the green "
            "vegetation fraction from the greenness state, Z0BRD the background roughness (phys-noah-snowz0), "
            "ALBBRD the background albedo; REDPRM writes them only under LOCAL or BARE")
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    for n in INOUT:
        if tuple(st.fields[n].shape) != (jm, im):
            raise parm.NoahParmTableError(f"phys-noah-redprm: {n} has shape {tuple(st.fields[n].shape)}, need {(jm, im)}")
    types_ = tuple(st.fields[n] for n in TYPES)
    tabs, scal = _bound_tables(ctx)
    sd = _soil_thickness(st)
    sd_host = np.asarray(sd.get() if hasattr(sd, "get") else sd, dtype=np.float32)   # init-stage read, once
    nsoil = int(sd_host.shape[0])
    if nsoil != NSOIL:
        raise parm.NoahParmTableError(
            f"phys-noah-redprm: SLDPTH has {nsoil} layers; the state's soil arrays have NSOIL={NSOIL}")
    _, zsoil = _sldpth_zsoil(st.xp, tuple(float(x) for x in sd_host))   # ZSOIL from the run's SLDPTH
    outs = _alloc_outs(st, jm, im)
    bad = st.xp.zeros(1, dtype=np.int32)
    redprm(types_, tabs, scal, sd, zsoil, outs, bad, local=local, im=im, jm=jm, nsoil=nsoil)
    n_bad = int(np.asarray(bad.get() if hasattr(bad, "get") else bad)[0])   # init-stage read, once
    if n_bad:
        raise parm.NoahParmTableError(
            f"phys-noah-redprm: {n_bad} point(s) refused: VEGTYP outside 1..{NLUS}, SOILTYP outside "
            f"1..{NSLTYPE}, SLOPETYP outside 1..{NSLOPE}, or NROOT outside 1..{nsoil}")
    return outs
