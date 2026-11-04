"""phys-noah-tdfcnd: TDFCND, soil thermal diffusivity (Noah LSM, Peters-Lidard approach).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4989-5092. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_tdfcnd.cu (phys_noah_tdfcnd, one thread per (i, j), NSOIL layers
looped in the thread, one launch per call). Layout and launch shape come from the ArWen copy; the
body is the NAM Peters-Lidard routine (ArWen's noah_tdfcnd is WRF's opt_thcnd form and is not used).

DF = AKE*(THKSAT - THKDRY) + THKDRY with the frozen / unfrozen Kersten branch of TDFCND.
No host sync, no per-step allocation, everything on the current stream.

Per-point soil inputs (the review's gaps, fixed here):
  QZ     = QTZ[SOILTYP]      soil-type quartz from the SOILPARM table (SFLX: QUARTZ = QTZ(SOILTYP),
                             module_LS_NOAHLSM.F90:3587 via REDPRM).
  SMCMAX = MAXSMC[SOILTYP]   soil-type porosity (SFLX: SMCMAX = MAXSMC(SOILTYP), :3590), with the SFLX
                             urban override SMCMAX = 0.45 where VEGTYP == IW (:1852-1855).
  SOILTYP driver reset: SOILTYP = 7 where SOILTYP == 14 and XICE == 0 (noahlsm, :787-790), applied
                             before SFLX, so it is applied here too.
The tables come from phys-noah-parm-tables (ctx.tables["noah_parm"]); there is no placeholder and no
fallback. Without a bound table the call raises.

Caller branches (the TDFCND call sites apply these in Fortran; the kernel's MODE argument selects them):
  MODE_SFLX_TOP  SFLX layer 1 (:2003-2035, :3350-3352): ICE == 1 gives DF = 2.2; urban IW gives 3.24.
  MODE_HRT       HRT every layer (:2938-2941, :2970-2973): urban IW gives 3.24.
  MODE_NOPAC_TOP NOPAC layer 1 (:3350-3352): literal VEGTYP == 1 gives 3.24.
run(ctx) launches the HRT form (all layers) and the SFLX top-layer form.
"""
from __future__ import annotations

import os

import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_parm_tables as parm

UNIT = "phys_noah_tdfcnd"
KERNEL = "phys_noah_tdfcnd"
BLOCK = (32, 8, 1)
NSOIL = parm.NSOIL
IVEGSRC = 1                      # IGBP (CONTRACT section 5)
MODE_SFLX_TOP = 0
MODE_HRT = 1
MODE_NOPAC_TOP = 2
URBAN_SMCMAX = 0.45              # SFLX override, module_LS_NOAHLSM.F90:1852-1855
WATER_SOILTYP = 14               # driver reset, module_LS_NOAHLSM.F90:787-790
WATER_RESET_SOILTYP = 7
REQUIRED = ("SOILTYP", "VEGTYP", "ICE", "XICE")


def urban_class(ivegsrc: int = IVEGSRC) -> int:
    """IW (module_LS_NOAHLSM.F90:629-630): 13 for IGBP (IVEGSRC 1), else 1."""
    return 13 if ivegsrc == 1 else 1


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def soil_inputs(soiltyp, vegtyp, xice, qtz, maxsmc, ivegsrc: int = IVEGSRC):
    """QZ and SMCMAX [JM, IM] from the soil-type table, with the driver water reset and the SFLX
    urban SMCMAX rule. qtz, maxsmc are device float32 [NSLTYPE] tables (1-based Fortran index
    SOILTYP is stored at SOILTYP-1). soiltyp, vegtyp, xice are device [JM, IM] arrays.
    Returns (qz, smcmax, soiltyp_effective). Gathers on the device: no host loop, no sync."""
    import cupy as cp

    soil = cp.where((soiltyp == WATER_SOILTYP) & (xice == 0), WATER_RESET_SOILTYP, soiltyp)
    idx = (soil - 1).astype(cp.int32)
    qz = qtz[idx].astype(cp.float32)
    smcmax = cp.where(vegtyp == urban_class(ivegsrc), cp.float32(URBAN_SMCMAX),
                      maxsmc[idx]).astype(cp.float32)
    return qz, smcmax, soil


def tdfcnd(smc, sh2o, qz, smcmax, vegtyp, ice, df, *, mode: int, iw: int):
    """DF[k, j, i] = TDFCND(QZ[j, i], SMC[k, j, i], SMCMAX[j, i], SH2O[k, j, i]) with the caller
    branch selected by mode. smc, sh2o, df are float32 device arrays [NSOIL, JM, IM]; qz, smcmax
    float32 [JM, IM]; vegtyp, ice int32 [JM, IM]."""
    _, jm, im = (int(s) for s in smc.shape)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (smc, sh2o, qz, smcmax, vegtyp, ice, df, np.int32(mode), np.int32(iw),
       np.int32(im), np.int32(jm)))
    return df


def _host_soil_tables(ivegsrc: int = IVEGSRC):
    """QTZ, MAXSMC and SLCATS from NAM_NOAH_FIX_DIR for the ctx-less path (same parser as the bound tables)."""
    import cupy as cp

    tbl_dir = os.environ.get(parm.ENV_DIR)
    if not tbl_dir:
        raise parm.NoahParmTableError(f"no NAM SOILPARM directory: set {parm.ENV_DIR}")
    host = parm.parse_soil_table(tbl_dir, ivegsrc)
    return cp.asarray(host.soil["QTZ"]), cp.asarray(host.soil["MAXSMC"]), host.slcats


def _tables_ctx(ctx):
    """QTZ and MAXSMC device tables bound by phys-noah-parm-tables (soil-only bind) on ctx."""
    dev = parm.run_soil(ctx)
    return dev["soil.QTZ"], dev["soil.MAXSMC"]


def run(ctx=None, *, smc=None, sh2o=None, soiltyp=None, vegtyp=None, ice=None, xice=None,
        df=None, df1=None):
    """Done-rule entry. With ctx: reads SMC, SH2O, SOILTYP, VEGTYP, ICE, XICE from ctx.st (all four
    must be in the state; phys-noah-init and the sea-ice field fill them) and the soil table from
    ctx.tables. Returns {"DF": DF (HRT form, every layer), "DF1": DF1 (SFLX top-layer form)}.
    Without ctx: a full-size 954 x 835 batch with physical ranges and real soil-table classes."""
    import cupy as cp

    if ctx is not None and smc is None:
        st = ctx.st
        missing = [n for n in REQUIRED if n not in st.fields]
        if missing:
            raise parm.NoahParmTableError(
                f"phys-noah-tdfcnd needs state fields {missing} (filled by phys-noah-init / sea ice)")
        smc = st.SMC
        sh2o = st.SH2O
        soiltyp, vegtyp, ice, xice = (st.fields[n] for n in REQUIRED)
        qtz, maxsmc = _tables_ctx(ctx)
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        df = st.add("TDFCND_DF", (NSOIL, jm, im)) if df is None else df
        df1 = st.add("TDFCND_DF1", (NSOIL, jm, im)) if df1 is None else df1
    elif smc is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        qtz, maxsmc, slcats = _host_soil_tables()
        smcmax_h = rng.uniform(0.35, 0.50, size=(jm, im)).astype(np.float32)
        smc_h = (rng.uniform(0.05, 1.0, size=(NSOIL, jm, im)) * smcmax_h[None]).astype(np.float32)
        sh2o_h = (smc_h * rng.uniform(0.0, 1.0, size=(NSOIL, jm, im))).astype(np.float32)
        smc = cp.asarray(smc_h)
        sh2o = cp.asarray(sh2o_h)
        soiltyp = cp.asarray(rng.integers(1, slcats + 1, size=(jm, im)).astype(np.int32))
        vegtyp = cp.asarray(rng.integers(1, 21, size=(jm, im)).astype(np.int32))
        ice = cp.asarray((rng.random((jm, im)) < 0.05).astype(np.int32))
        xice = cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32))
        df = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
        df1 = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    else:
        if any(a is None for a in (soiltyp, vegtyp, ice, xice)):
            raise parm.NoahParmTableError(
                "phys-noah-tdfcnd: with smc given, pass soiltyp, vegtyp, ice and xice too")
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        qtz, maxsmc, _ = _host_soil_tables()
        df = cp.zeros((NSOIL, jm, im), dtype=cp.float32) if df is None else df
        df1 = cp.zeros((NSOIL, jm, im), dtype=cp.float32) if df1 is None else df1
    qz, smcmax, _ = soil_inputs(soiltyp, vegtyp, xice, qtz, maxsmc)
    iw = urban_class()
    tdfcnd(smc, sh2o, qz, smcmax, vegtyp, ice, df, mode=MODE_HRT, iw=iw)
    tdfcnd(smc, sh2o, qz, smcmax, vegtyp, ice, df1, mode=MODE_SFLX_TOP, iw=iw)
    return {"DF": df, "DF1": df1}
