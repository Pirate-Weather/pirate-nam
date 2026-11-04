// nam-gpu unit drv-exptbl: EXPTBL exponential look-up table (and the scalars zjexp reads).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_CONTROL.F90:1780-1820 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine exptbl; kexm = 10001 from module_CONTROL.F90:33, kfpt = single):
//   xmax = 30.;  xmin = -30.
//   kexm2 = kexm - 2
//   xrng  = xmax - xmin
//   dex   = xrng / (kexm - 1)
//   rdex  = 1. / dex
//   x = xmin - dex
//   do k = 1, kexm:  x = x + dex;  expf(k) = exp(x)
//
// Single thread by design: the Fortran builds x by repeated float32 addition, so expf(k) carries the
// accumulated rounding of k additions. A parallel evaluation of exp(xmin + k*dex) would not reproduce
// that table. The loop is 10001 iterations, run once at init (not on the step path).
// Outputs: expf[KEXM] (1-based Fortran expf(k) is expf_tab[k-1]); scal[0..3] = xmin, xmax, dex, rdex.
// kexm2 (= KEXM-2) is an integer constant, set on the host (ctx.tables['EXPTBL_KEXM2']).
// The zjexp function (Fortran 1822+, outside this unit's line range) is provided below as a device
// helper for the units that call it; it reads expf_tab with the same 0-based offsets.
// Not ported (no effect on the table): the in-routine declarations and the end-of-routine labels.

#include "nmmb_common.cuh"

#define EXPTBL_KEXM 10001

extern "C" __global__ void exptbl_build(
    float* __restrict__ expf_tab,       // EXPF [KEXM]   out, Fortran expf(1:kexm)
    float* __restrict__ scal)           // [4]           out: xmin, xmax, dex, rdex
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;   // single thread, see header

    const float xmax = 30.0f;
    const float xmin = -30.0f;
    const float xrng = xmax - xmin;
    const float dex = xrng / (float)(EXPTBL_KEXM - 1);
    const float rdex = 1.0f / dex;

    float x = xmin - dex;
    for (int k = 1; k <= EXPTBL_KEXM; ++k) {
        x = x + dex;
        expf_tab[k - 1] = expf(x);
    }

    scal[0] = xmin;
    scal[1] = xmax;
    scal[2] = dex;
    scal[3] = rdex;
}

// zjexp (Fortran module_CONTROL.F90, function zjexp). Helper for the callers; 0-based table.
//   ak = (x - xmin)*rdex;  k = int(ak);  k = max(k,0);  k = min(k, kexm2)
//   zjexp = (expf(k+2) - expf(k+1))*(ak - real(k)) + expf(k+1)      (Fortran 1-based)
__device__ __forceinline__ float zjexp_dev(const float* __restrict__ expf_tab, float xmin, float rdex, float x)
{
    const float ak = (x - xmin) * rdex;
    int k = (int)ak;
    k = max(k, 0);
    k = min(k, EXPTBL_KEXM - 2);            // kexm2 = kexm-2 (Fortran k in 0..kexm2)
    const float e1 = expf_tab[k];           // Fortran expf(k+1)
    const float e2 = expf_tab[k + 1];       // Fortran expf(k+2)
    return (e2 - e1) * (ak - (float)k) + e1;
}
