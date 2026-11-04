"""phys-noah-wdfcnd: WDFCND, soil water diffusivity and hydraulic conductivity (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:5312-5369. NOAA US
government work; the module header is kept as the notice in the kernel files. Start point for the
arithmetic: ArWen core/kernels/noah.cu noah_wdfcnd (Apache-2.0, third_party/arwen/LICENSE), which
differs from NAM (NAM FACTR1 = 0.2/SMCMAX with no clamp); the NAM form is what is written.
Kernel: namgpu/kernels/phys_noah_wdfcnd.cu (phys_noah_wdfcnd, one launch over (i, j, soil layer)).
The device function nam_wdfcnd lives in namgpu/kernels/phys_noah_wdfcnd.cuh for other units
(SRT and SFLX call WDFCND for the soil water flux).

Inputs: SMC [NSOIL, JM, IM]; SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX [JM, IM] float32. Outputs: WDF, WCND
[NSOIL, JM, IM]. NSOIL is parm.NSOIL (4); any other layer count is refused by name.

SICEMAX is the Fortran SFLX value: SICEMAX = max(0, max over KS of SICE(KS)), with SICE(KS) = SMC(KS)
- SH2O(KS) (module_LS_NOAHLSM.F90:2881, 3001, and the SICEMAX loop at 4649-4655). It is derived here
from the state, so the frozen-soil branch (SICEMAX > 0) is live on the ctx path.

run(ctx): reads SMC, SH2O from ctx.st and SMCMAX, BEXP, DKSAT, DWSAT from ctx.st.fields (REDPRM,
phys-noah-redprm outputs, the same names as the Fortran REDPRM locals). A missing input is refused by
name, never filled with a placeholder. Writes WDF, WCND into cached scratch arrays.
Without ctx it builds a full-size random 954 x 835 x NSOIL batch with REDPRM-range soil values. No
host sync, no per-step allocation beyond the cached outputs, current stream.

Caller obligation (not this unit): the Fortran WDFCND is also called with scalar MXSMC and SH2OA(NSOIL)
(module_LS_NOAHLSM.F90:4732, 4749, 4787, 4811). The device function nam_wdfcnd serves those per point;
the SRT/SFLX units own that mapping.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_parm_tables as parm

UNIT = "phys_noah_wdfcnd"
KERNEL = "phys_noah_wdfcnd"
BLOCK = (32, 8, 1)
NSOIL = parm.NSOIL
REDPRM_FIELDS = ("SMCMAX", "BEXP", "DKSAT", "DWSAT")   # phys-noah-redprm FLOAT_OUT names

_scratch: dict = {}


def wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd):
    """WDF[l, j, i], WCND[l, j, i] = WDFCND(SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX) per soil layer.

    smc, wdf, wcnd: float32 [NSOIL, JM, IM]; smcmax, bexp, dksat, dwsat, sicemax: float32 [JM, IM].
    """
    nsoil, jm, im = (int(s) for s in smc.shape)
    if nsoil != NSOIL:
        raise ValueError(f"phys-noah-wdfcnd: smc has {nsoil} soil layers, the model has NSOIL={NSOIL}")
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_2d(im, jm, nsoil), BLOCK,
      (smc, smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd, np.int32(im), np.int32(jm), np.int32(nsoil)))
    return wdf, wcnd


def sicemax_from(smc, sh2o, out):
    """SICEMAX[j, i] = max(0, max_l (SMC - SH2O)[l, j, i]), the SFLX column maximum of frozen water."""
    import cupy as cp

    cp.max(cp.maximum(smc - sh2o, 0.0), axis=0, out=out)
    return out


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        _scratch[key] = {
            "SICEMAX": cp.zeros((jm, im), dtype=cp.float32),
            "WDF": cp.zeros((NSOIL, jm, im), dtype=cp.float32),
            "WCND": cp.zeros((NSOIL, jm, im), dtype=cp.float32),
        }
    return _scratch[key]


def run(ctx=None):
    """Done-rule entry: one launch over every soil layer of a full-size 954 x 835 x NSOIL field set."""
    import cupy as cp

    if ctx is not None:
        st = ctx.st
        missing = [n for n in REDPRM_FIELDS if n not in st.fields]
        if missing:
            raise KeyError(f"phys-noah-wdfcnd needs {missing} in the state: SMCMAX/BEXP/DKSAT/DWSAT from "
                           "REDPRM (phys-noah-redprm); not there yet")
        smc = st.SMC
        sh2o = st.SH2O
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        s = _scratch_arrays(jm, im)
        sicemax_from(smc, sh2o, s["SICEMAX"])
        wdfcnd(smc, st.fields["SMCMAX"], st.fields["BEXP"], st.fields["DKSAT"], st.fields["DWSAT"],
               s["SICEMAX"], s["WDF"], s["WCND"])
    else:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        smc = cp.asarray(rng.uniform(0.02, 0.44, size=(NSOIL, jm, im)).astype(np.float32))
        sh2o = smc * cp.asarray(rng.uniform(0.5, 1.0, size=(NSOIL, jm, im)).astype(np.float32))
        smcmax = cp.asarray(rng.uniform(0.40, 0.50, size=(jm, im)).astype(np.float32))
        bexp = cp.asarray(rng.uniform(2.0, 12.0, size=(jm, im)).astype(np.float32))
        dksat = cp.asarray(rng.uniform(1.0e-6, 1.0e-5, size=(jm, im)).astype(np.float32))
        dwsat = cp.asarray(rng.uniform(1.0e-6, 1.0e-5, size=(jm, im)).astype(np.float32))
        s = _scratch_arrays(jm, im)
        sicemax_from(smc, sh2o, s["SICEMAX"])
        wdfcnd(smc, smcmax, bexp, dksat, dwsat, s["SICEMAX"], s["WDF"], s["WCND"])
    return {"WDF": s["WDF"], "WCND": s["WCND"]}
