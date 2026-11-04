// phys_noah_wdfcnd.cuh -- WDFCND device function: soil water diffusivity and hydraulic
// conductivity (Noah LSM). NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work.
// Author headers of the source module are kept as notices: MODULE MODULE_LS_NOAHLSM
// (module_LS_NOAHLSM.F90). Ported from phys/module_LS_NOAHLSM.F90:5312-5369.
// Start point for the arithmetic: namgpu/third_party/arwen/core/kernels/noah.cu noah_wdfcnd (Apache-2.0,
// ArWen LICENSE). The ArWen copy is NOT the NAM routine: it uses WRF v4.6.1 FACTR1 = 0.05/SMCMAX
// clamped by min(FACTR1, FACTR2). NAM uses FACTR1 = 0.2/SMCMAX with no clamp, so the NAM form is
// written here.
//
// Executed statements, every one kept (Fortran order):
//   FACTR1 = 0.2 / SMCMAX
//   FACTR2 = SMC / SMCMAX
//   EXPON  = BEXP + 2.0
//   WDF    = DWSAT * FACTR2 ** EXPON
//   IF (SICEMAX .gt. 0.0) THEN
//      VKWGT = 1. / (1. + (500.* SICEMAX)**3.)
//      WDF   = VKWGT * WDF + (1. - VKWGT) * DWSAT * FACTR1 ** EXPON
//   END IF
//   EXPON  = (2.0 * BEXP) + 3.0
//   WCND   = DKSAT * FACTR2 ** EXPON
// No MPI, halo, OpenMP or print statements in this range. Float32 throughout (kfpt = single).
// Real exponents use powf, as the Fortran REAL ** REAL does.

#pragma once

__device__ static inline void nam_wdfcnd(float& wdf, float& wcnd, float smc,
                                         float smcmax, float bexp, float dksat,
                                         float dwsat, float sicemax)
{
    const float factr1 = 0.2f / smcmax;
    const float factr2 = smc / smcmax;
    float expon = bexp + 2.0f;
    wdf = dwsat * powf(factr2, expon);
    if (sicemax > 0.0f) {
        const float vkwgt = 1.0f / (1.0f + powf(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat * powf(factr1, expon);
    }
    expon = (2.0f * bexp) + 3.0f;
    wcnd = dksat * powf(factr2, expon);
}
