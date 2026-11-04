#pragma once
// phys_fer_fpvs.cu -- nam-gpu unit phys-fer-fpvs: FPVS, FPVS0, FPVSX, FPVSX0 (device functions).
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). Module header block kept as in phys_fer_growth_rates.cu (NOAA "Modified" fer_hires,
// 11 July 2016 and later revisions). Function headers below are reproduced from the source.
//
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90 (nam.v4.2.11, 83df4e13c1):
//   FPVS(T)    2840-2891  table interpolation in TBPVS on [XMIN,XMAX]; Magnus-Tetens outside
//                         (water above XMAX, ice below XMIN).                 -> nam_fpvs
//   FPVS0(T)   2894-2911  same interpolation on TBPVS0; Magnus water everywhere outside range. -> nam_fpvs0
//   FPVSX(T)   2915-2971  exact Clausius-Clapeyron integral, liquid for T>=TTP, ice below.   -> nam_fpvsx
//   FPVSX0(T)  2974-2989  exact liquid-only form.                                 -> nam_fpvsx0
// FPVSX and FPVSX0 are defined once in fer_fpvs_common.cuh (shared with phys_fer_gpvs.cu, which builds
// the table from them). This file includes it and keeps the two table lookups.
//
// Tables and their coefficients are built by GPVS_hr (phys-fer-gpvs) and passed in by pointer:
//   tbpvs / tbpvs0 : float[NX=7501], Fortran TBPVS(1:NX) / TBPVS0(1:NX), 1-based in Fortran, 0-based here.
//   c1/c2          : C1XPVS, C2XPVS (and C1XPVS0, C2XPVS0), SAVEd module scalars set by GPVS_hr.
// Fortran "JX=MIN(XJ,NX-1.)" assigns a real to an integer, which truncates toward zero; XJ >= 1 so this is floor.
//
// Constants: NX, XMIN, XMAX from module_MP_FER_HIRES.F90 lines 148-149; R_V (RV) from nmmb_common.cuh.
#include "nmmb_common.cuh"
#include "fer_fpvs_common.cuh"

#define FPVS_NX     7501
#define FPVS_XMIN   180.0f
#define FPVS_XMAX   330.0f

// FPVS(T): table lookup with Magnus-Tetens fallbacks. Fortran lines 2840-2891.
__device__ inline float nam_fpvs(float T, const float* __restrict__ tbpvs,
                                 float c1xpvs, float c2xpvs)
{
    if (T >= FPVS_XMIN && T <= FPVS_XMAX) {
        float xj = fminf(fmaxf(c1xpvs + c2xpvs * T, 1.0f), (float)FPVS_NX);
        int jx = (int)fminf(xj, (float)(FPVS_NX - 1));       // Fortran JX = MIN(XJ, NX-1.)
        // Fortran TBPVS(JX) + (XJ-JX)*(TBPVS(JX+1)-TBPVS(JX)), 1-based -> 0-based index jx-1, jx
        return tbpvs[jx - 1] + (xj - (float)jx) * (tbpvs[jx] - tbpvs[jx - 1]);
    } else if (T > FPVS_XMAX) {
        // Magnus-Tetens water saturation (Murray 1967), kPa
        return 0.61078f * expf(17.2694f * (T - 273.16f) / (T - 35.86f));
    } else {
        // Magnus-Tetens ice saturation (Murray 1967), kPa
        return 0.61078f * expf(21.8746f * (T - 273.16f) / (T - 7.66f));
    }
}

// FPVS0(T): table lookup with water Magnus fallback everywhere outside the range. Fortran lines 2894-2911.
__device__ inline float nam_fpvs0(float T, const float* __restrict__ tbpvs0,
                                  float c1xpvs0, float c2xpvs0)
{
    if (T >= FPVS_XMIN && T <= FPVS_XMAX) {
        float xj1 = fminf(fmaxf(c1xpvs0 + c2xpvs0 * T, 1.0f), (float)FPVS_NX);
        int jx1 = (int)fminf(xj1, (float)(FPVS_NX - 1));     // Fortran JX1 = MIN(XJ1, NX-1.)
        return tbpvs0[jx1 - 1] + (xj1 - (float)jx1) * (tbpvs0[jx1] - tbpvs0[jx1 - 1]);
    } else {
        return 0.61078f * expf(17.2694f * (T - 273.16f) / (T - 35.86f));
    }
}

// Evaluation kernel: one thread per element of T (flattened [LM*JM*IM]). Writes the four functions.
// coef is the device array fer_gpvs_coef[4] written by GPVS_hr (phys-fer-gpvs).
// Used by the units wrapper for the done-rule run and by the dependent units tests; the production
// callers (phys-fer-condense, phys-fer-deposit, phys-fer-gpvs, ...) call the __device__ functions above.
extern "C" __global__ void fer_fpvs_eval(const float* __restrict__ T,
                                         const float* __restrict__ tbpvs,
                                         const float* __restrict__ tbpvs0,
                                         const float* __restrict__ coef,
                                         float* __restrict__ out_fpvs,
                                         float* __restrict__ out_fpvs0,
                                         float* __restrict__ out_fpvsx,
                                         float* __restrict__ out_fpvsx0,
                                         int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float t = T[k];
    // coef = fer_gpvs_coef from GPVS_hr: {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0}, read on device (no host sync).
    const float c1xpvs = coef[0], c2xpvs = coef[1], c1xpvs0 = coef[2], c2xpvs0 = coef[3];
    out_fpvs[k]   = nam_fpvs(t, tbpvs, c1xpvs, c2xpvs);
    out_fpvs0[k]  = nam_fpvs0(t, tbpvs0, c1xpvs0, c2xpvs0);
    out_fpvsx[k]  = nam_fpvsx(t);
    out_fpvsx0[k] = nam_fpvsx0(t);
}
