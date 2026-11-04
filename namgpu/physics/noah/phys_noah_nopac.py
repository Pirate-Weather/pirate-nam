"""phys-noah-nopac: NOPAC, the no-snow energy and water balance (Noah LSM), WOOF-NAM.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The module's NOAA header is
kept as the notice in the kernel file.
Fortran: phys/module_LS_NOAHLSM.F90:3204-3387 (SUBROUTINE NOPAC). Kernel:
namgpu/kernels/phys_noah_nopac.cu (phys_noah_nopac, one thread per (i, j), one launch).

NOPAC runs for every column with no snowpack. Its callee closure (EVAPO, DEVAP, TRANSP, SMFLX, SRT,
SSTEP, ROSR12, SHFLX, HRT, HRTICE, HSTEP, TBND, TMPAVG, SNKSRC, FRH2O, TDFCND, WDFCND) is transcribed
from the NAM Fortran inside the kernel file. Inputs and outputs are the NOPAC arguments, bound by
name:

  colf   [NCOLF=36, JM, IM] float32  per-column scalars (CF_* slots in the kernel), inout CMC, T1, BETA
  coli   [2, JM, IM]        int32    NROOT, VEGTYP
  smc, sh2o, stc, zsoil, rtdis, et  [NSOIL=4, JM, IM] float32 (layer 1 at the top); smc/sh2o/stc inout
  out    [NOUT=12, JM, IM] float32  ETA, EDIR, EC, ETT, DEW, DRIP, RUNOFF1-3, SSOIL, FLX1, FLX3

Constants from the configfile and the call site: NSOIL = 4, DT = DTPHS = 50 s (NPHS = 2, DT = 25 s),
ICE = 0 (land), IVEGSRC = 1 (IGBP).

run(ctx=None): the done-rule entry point. With ctx=None it builds a full-size 954 x 835 random column
set (plausible Noah ranges, both the dew and the evaporation branch) and returns the outputs. The
per-column scalars come from the NOAHLSM driver in the full build; they are not state-table fields.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_nopac"
KERNEL = "phys_noah_nopac"
NSOIL = 4
NCOLF = 36
NOUT = 12
ICE = 0
IVEGSRC = 1
DT = 50.0
BLOCK = kernels.BLOCK_COL

CF = {
    "ETP": 0, "PRCP": 1, "SMCMAX": 2, "SMCWLT": 3, "SMCREF": 4, "SMCDRY": 5, "CMC": 6,
    "CMCMAX": 7, "SHDFAC": 8, "SBETA": 9, "Q2": 10, "T1": 11, "SFCTMP": 12, "T24": 13,
    "TH2": 14, "FDOWN": 15, "F1": 16, "EMISSI": 17, "EPSCA": 18, "BEXP": 19, "PC": 20,
    "RCH": 21, "RR": 22, "CFACTR": 23, "SLOPE": 24, "KDT": 25, "FRZFACT": 26, "PSISAT": 27,
    "DKSAT": 28, "DWSAT": 29, "TBOT": 30, "ZBOT": 31, "QUARTZ": 32, "FXEXP": 33, "CSOIL": 34,
    "BETA": 35,
}
OUT = {
    "ETA": 0, "EDIR": 1, "EC": 2, "ETT": 3, "DEW": 4, "DRIP": 5, "RUNOFF1": 6, "RUNOFF2": 7,
    "RUNOFF3": 8, "SSOIL": 9, "FLX1": 10, "FLX3": 11,
}

_scratch: dict = {}


def _grid(n: int):
    return ((n + BLOCK[0] - 1) // BLOCK[0], 1, 1)


def nopac(colf, coli, smc, sh2o, stc, zsoil, rtdis, et, out, *, im: int, jm: int,
          ice: int = ICE, ivegsrc: int = IVEGSRC, dt: float = DT):
    """One launch over every column. colf/smc/sh2o/stc/et/out are updated in place where NOPAC does."""
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im * jm), BLOCK,
      (colf, coli, smc, sh2o, stc, zsoil, rtdis, et, out,
       np.int32(im), np.int32(jm), np.int32(ice), np.int32(ivegsrc), np.float32(dt)))
    return out


def _random_inputs(im: int, jm: int, seed: int = 1):
    import cupy as cp

    rng = np.random.default_rng(seed)
    n = im * jm

    def u(lo, hi, shape):
        return rng.uniform(lo, hi, size=shape).astype(np.float32)

    colf = np.zeros((NCOLF, jm, im), dtype=np.float32)
    smcmax = u(0.35, 0.50, (jm, im))
    smcwlt = u(0.02, 0.10, (jm, im))
    colf[CF["ETP"]] = u(-2.0e-5, 3.0e-4, (jm, im))            # kg m-2 s-1: both branches
    colf[CF["PRCP"]] = u(0.0, 1.0e-4, (jm, im))
    colf[CF["SMCMAX"]] = smcmax
    colf[CF["SMCWLT"]] = smcwlt
    colf[CF["SMCREF"]] = smcwlt + u(0.10, 0.20, (jm, im))
    colf[CF["SMCDRY"]] = smcwlt * u(0.5, 0.9, (jm, im))
    colf[CF["CMC"]] = u(0.0, 2.0e-4, (jm, im))
    colf[CF["CMCMAX"]] = u(2.0e-4, 5.0e-4, (jm, im))
    colf[CF["SHDFAC"]] = u(0.05, 0.95, (jm, im))
    colf[CF["SBETA"]] = -2.0
    colf[CF["Q2"]] = u(0.004, 0.015, (jm, im))
    colf[CF["T1"]] = u(260.0, 300.0, (jm, im))
    colf[CF["SFCTMP"]] = u(260.0, 300.0, (jm, im))
    colf[CF["T24"]] = u(265.0, 295.0, (jm, im))
    colf[CF["TH2"]] = u(260.0, 300.0, (jm, im))
    colf[CF["FDOWN"]] = u(200.0, 400.0, (jm, im))
    colf[CF["F1"]] = u(0.1, 0.3, (jm, im))
    colf[CF["EMISSI"]] = 0.95
    colf[CF["EPSCA"]] = 0.0
    colf[CF["BEXP"]] = u(4.0, 10.0, (jm, im))
    colf[CF["PC"]] = u(0.1, 0.9, (jm, im))
    colf[CF["RCH"]] = u(0.002, 0.006, (jm, im))
    colf[CF["RR"]] = u(1.0, 2.0, (jm, im))
    colf[CF["CFACTR"]] = 0.5
    colf[CF["SLOPE"]] = u(0.1, 1.0, (jm, im))
    colf[CF["KDT"]] = 3.0
    colf[CF["FRZFACT"]] = u(0.1, 0.5, (jm, im))
    colf[CF["PSISAT"]] = u(0.05, 0.7, (jm, im))
    colf[CF["DKSAT"]] = u(1.0e-6, 1.0e-5, (jm, im))
    colf[CF["DWSAT"]] = u(1.0e-6, 1.0e-5, (jm, im))
    colf[CF["TBOT"]] = u(275.0, 290.0, (jm, im))
    colf[CF["ZBOT"]] = -3.0
    colf[CF["QUARTZ"]] = u(0.1, 0.9, (jm, im))
    colf[CF["FXEXP"]] = u(1.0, 3.0, (jm, im))
    colf[CF["CSOIL"]] = 1.26e6
    colf[CF["BETA"]] = 0.0
    coli = np.zeros((2, jm, im), dtype=np.int32)
    coli[0] = rng.integers(1, NSOIL + 1, size=(jm, im))       # NROOT
    coli[1] = rng.integers(1, 20, size=(jm, im))              # VEGTYP (IGBP 1-19)

    zsoil = np.zeros((NSOIL, jm, im), dtype=np.float32)
    for k, z in enumerate((-0.1, -0.4, -1.0, -2.0)):
        zsoil[k] = z
    rtdis = u(0.1, 0.5, (NSOIL, jm, im))
    smc = np.zeros((NSOIL, jm, im), dtype=np.float32)
    sh2o = np.zeros((NSOIL, jm, im), dtype=np.float32)
    for k in range(NSOIL):
        s = u(0.1, 0.45, (jm, im))
        s = np.minimum(s, smcmax)
        smc[k] = s
        sh2o[k] = s * u(0.6, 1.0, (jm, im))
    stc = u(255.0, 300.0, (NSOIL, jm, im))
    return (cp.asarray(colf), cp.asarray(coli), cp.asarray(smc), cp.asarray(sh2o),
            cp.asarray(stc), cp.asarray(zsoil), cp.asarray(rtdis))


def run(ctx=None, *, colf=None, coli=None, smc=None, sh2o=None, stc=None, zsoil=None,
        rtdis=None, im: int = 954, jm: int = 835):
    """Done-rule entry: one launch over a full-size 954 x 835 column set. Returns the arrays it wrote."""
    import cupy as cp

    if colf is None:
        colf, coli, smc, sh2o, stc, zsoil, rtdis = _random_inputs(im, jm)
    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    et = _scratch[key]
    out = cp.zeros((NOUT, jm, im), dtype=cp.float32)
    nopac(colf, coli, smc, sh2o, stc, zsoil, rtdis, et, out, im=im, jm=jm)
    return {"ETA": out[OUT["ETA"]], "EDIR": out[OUT["EDIR"]], "EC": out[OUT["EC"]],
            "ETT": out[OUT["ETT"]], "DEW": out[OUT["DEW"]], "DRIP": out[OUT["DRIP"]],
            "RUNOFF1": out[OUT["RUNOFF1"]], "RUNOFF2": out[OUT["RUNOFF2"]],
            "RUNOFF3": out[OUT["RUNOFF3"]], "SSOIL": out[OUT["SSOIL"]],
            "FLX1": out[OUT["FLX1"]], "FLX3": out[OUT["FLX3"]],
            "ET": et, "SMC": smc, "SH2O": sh2o, "STC": stc, "CMC": colf[CF["CMC"]],
            "T1": colf[CF["T1"]], "BETA": colf[CF["BETA"]]}
