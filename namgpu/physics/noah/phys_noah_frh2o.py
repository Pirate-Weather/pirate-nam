"""phys-noah-frh2o: FRH2O, unfrozen soil water (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2644-2782. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel files.
Kernel: namgpu/kernels/phys_noah_frh2o.cu (phys_noah_frh2o, one launch over (i, j, soil layer));
the device function nam_frh2o lives in namgpu/kernels/phys_noah_frh2o.cuh for other units.
Started from the ArWen noah_frh2o (namgpu/third_party/arwen/core/kernels/noah.cu), Apache-2.0.

FRH2O is a device function in the Fortran: its callers are SNKSRC (per soil layer, TAVG from
TMPAVG) and the SFLX soil paths. The one-launch wrapper is the batch form used by the done rule.

run(ctx): reads the soil temperature TAVG (SFLX soil path, TMPAVG; the interface-weighted layer
average SNKSRC receives, module_LS_NOAHLSM.F90:3003-3004, 4008), SMC and SH2O from ctx.st, and the
soil-type values SMCMAX, BEXP and PSISAT from REDPRM (phys-noah-redprm outputs, the same names as
the Fortran REDPRM locals, module_LS_NOAHLSM.F90:3471-3664). It writes FREE into a cached scratch
array. A missing input is refused by name, never filled with zeros. TAVG is not STC: STC is the
init caller's value (TSLB = STC, line 1263), not SNKSRC's.
Without ctx it builds a full-size random 954 x 835 x NSOIL batch. The explicit form takes
smcmax, bexp and psis as arrays and refuses when any is missing. No host sync, no per-step
allocation, everything on the current stream.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_frh2o"
KERNEL = "phys_noah_frh2o"
BLOCK = (32, 8, 1)
NSOIL = 4
REDPRM_FIELDS = ("SMCMAX", "BEXP", "PSISAT")   # phys-noah-redprm FLOAT_OUT names

_scratch: dict = {}


def frh2o(tkelv, smc, sh2o, smcmax, bexp, psis, freew):
    """FREE[l, j, i] = FRH2O(TKELV, SMC, SH2O, SMCMAX, BEXP, PSIS) for every soil layer.

    tkelv, smc, sh2o, freew: float32 [NSOIL, JM, IM]; smcmax, bexp, psis: float32 [JM, IM].
    """
    nsoil, jm, im = (int(s) for s in smc.shape)
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_2d(im, jm, nsoil), BLOCK,
      (tkelv, smc, sh2o, smcmax, bexp, psis, freew, np.int32(im), np.int32(jm)))
    return freew


def _free_scratch(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = cp.zeros((NSOIL, jm, im), dtype=cp.float32)   # FREE, overwritten every launch
    return _scratch[key]


def run(ctx=None, *, tkelv=None, smc=None, sh2o=None, smcmax=None, bexp=None, psis=None, freew=None):
    """Done-rule entry: one launch over every soil layer of a full-size field set."""
    import cupy as cp

    if ctx is not None and tkelv is None:
        st = ctx.st
        missing = [n for n in ("TAVG",) + REDPRM_FIELDS if n not in st.fields]
        if missing:
            raise KeyError(f"phys-noah-frh2o needs {missing} in the state: TAVG from the SFLX soil path "
                           "(TMPAVG), SMCMAX/BEXP/PSISAT from REDPRM (phys-noah-redprm); not there yet")
        smc = st.SMC
        sh2o = st.SH2O
        tkelv = st.fields["TAVG"]
        smcmax = st.fields["SMCMAX"]
        bexp = st.fields["BEXP"]
        psis = st.fields["PSISAT"]
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        freew = _free_scratch(jm, im)
    elif tkelv is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        tkelv = cp.asarray(rng.uniform(240.0, 300.0, size=(NSOIL, jm, im)).astype(np.float32))
        smc = cp.asarray(rng.uniform(0.05, 0.45, size=(NSOIL, jm, im)).astype(np.float32))
        sh2o = cp.asarray((rng.uniform(0.0, 0.9, size=(NSOIL, jm, im))).astype(np.float32)) * smc
        smcmax = cp.asarray(rng.uniform(0.4, 0.5, size=(jm, im)).astype(np.float32))
        bexp = cp.asarray(rng.uniform(2.0, 12.0, size=(jm, im)).astype(np.float32))
        psis = cp.asarray(rng.uniform(0.05, 0.7, size=(jm, im)).astype(np.float32))
        freew = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    else:
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        if smcmax is None or bexp is None or psis is None:
            raise ValueError("phys-noah-frh2o: smcmax, bexp and psis (REDPRM SMCMAX, BEXP, PSISAT) are "
                             "required; the kernel has no default for them")
        if freew is None:
            freew = _free_scratch(jm, im)
    frh2o(tkelv, smc, sh2o, smcmax, bexp, psis, freew)
    return {"FREE": freew}
