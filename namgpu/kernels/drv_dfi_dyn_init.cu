// drv_dfi_dyn_init.cu -- DIGITAL_FILTER_DYN_INIT_NMM: set up the TDFI dynamics filter sums.
//
// NOTICE: NOAA/NCEP NMMB source, nmm/module_DIGITAL_FILTER_NMM.F90 (US Government work, no licence
// file in the NAM clone; see NOTICE at the repo root). Module header kept as the notice:
//   MODULE module_DIGITAL_FILTER_NMM -- "a generic digital filter for any model under ESMF"
//   March 2007 Hann-Ming Henry Juang; Feb 2008-Sep 2011 Weiyu Yang; July 2012 T Black
// Ported to CUDA for nam-gpu from nmm/module_DIGITAL_FILTER_NMM.F90:46-203 (nam.v4.2.11, 83df4e13c1).
//
// The routine has two parts, split into two launches (CONTRACT.md section 3):
//
//   (a) dfi_dolph_window: the call DOLPH(dt, taus, nstep, dolph_wgts) at line 108 (the DOLPH window
//       subroutine itself is nmm/module_DIGITAL_FILTER_NMM.F90:929-1010, part of unit drv-dfi-dolph).
//       Dolph-Chebyshev window with M = nstep, N = 2M+1, THETAS = 2 pi |dt/taus|, X0 = 1/cos(THETAS/2),
//       R = 1/(0.5 (TERM1 + TERM2)), W(NT) = (1 + sum_{I=1..M} R 2 T_{N-1}(X0 cos(I pi/N)) cos(2 NT pi I/N)) / N
//       for NT = 0..M, and the symmetric fill W2(M+NT) = W2(M-NT) = W(NT). The Fortran passes
//       dolph_wgts(-nstep:nstep) to window(0:2M) by sequence association, so Fortran dolph_wgts(k)
//       is device element k + nstep. One thread per NT in 0..M writes both mirror elements.
//       Chebyshev T_{N-1} is evaluated by the three-term recurrence of CHEBY (module_DIGITAL_FILTER_NMM
//       .F90:1012-1039) with a rolling pair instead of the work array, same arithmetic.
//       The SUMW accumulation (Fortran 1002-1008 region) is a print-only quantity and has no effect.
//
//   (b) dfi_dyn_init_zero: the zeroing at lines 188-200 (array_save_2d = 0, array_save_3d = 0,
//       totalsum = 0). The array_save_4d zeroing has no 4-D fields to zero in the TDFI dynamics list
//       (see the wrapper; the Fortran SPEC_MAX at line 179 is never assigned, so no 4-D size exists).
//       Grid-stride loop over both sum arrays, one launch.
//
// The rank counting loop (lines 134-158, ESMF FieldBundle lookups and the tot_rank_2d/3d/4d
// counters) and the name bookkeeping (name_save_2d/3d/4d) are host-side bookkeeping on the field
// list, done in the wrapper (namgpu/steps/drv_dfi_dyn_init.py), not in this kernel.
// The ESMF allocation-failure aborts (lines 97-100, 163-167, 172-176, 181-185) are replaced by the
// CuPy allocation exception.
//
// Mapping: array_save_2d(i,j,n) -> sum2d[n][j][i] (n is the 2-D field index, zero-based);
//          array_save_3d(i,j,l,n) -> sum3d[n][l][j][i], l = 1..LM+1 as in NMMB (l = 1 top).
// Precision: REAL = kfpt = single, so every value here is float (CONTRACT.md section 3). The PI of
// the Fortran is 4*ATAN(1.D0) assigned to REAL, i.e. float PI.
// No MPI/halo or OpenMP lines in this routine apart from the ESMF calls (dropped, single card).

#include "nmmb_common.cuh"

extern "C" __global__ void dfi_dolph_window(float* __restrict__ window, const int m,
                                            const float deltat, const float taus)
{
    const int nt = blockIdx.x * blockDim.x + threadIdx.x;   // NT = 0..M
    if (nt > m) return;

    const float PI = 3.14159265358979f;                     // 4*ATAN(1.D0) as REAL
    const int N = 2 * m + 1;
    const int nm1 = N - 1;

    const float thetas = 2.0f * PI * fabsf(deltat / taus);
    const float x0 = 1.0f / cosf(thetas / 2.0f);
    const float term1 = powf(x0 + sqrtf(x0 * x0 - 1.0f), (float)(N - 1));
    const float term2 = powf(x0 - sqrtf(x0 * x0 - 1.0f), (float)(N - 1));
    const float rr = 0.5f * (term1 + term2);
    const float r = 1.0f / rr;

    float sum = 1.0f;
    for (int i = 1; i <= m; ++i) {
        const float arg = x0 * cosf((float)i * PI / (float)N);
        // CHEBY(T, NM1, ARG): T(0)=1, T(1)=ARG, T(n)=2 ARG T(n-1) - T(n-2); we need T(NM1).
        float tkm2 = 1.0f;
        float tkm1 = arg;
        float tk = (nm1 == 0) ? 1.0f : arg;
        for (int nn = 2; nn <= nm1; ++nn) {
            tk = 2.0f * arg * tkm1 - tkm2;
            tkm2 = tkm1;
            tkm1 = tk;
        }
        const float tnm1 = tk;
        const float cosine = cosf(2.0f * (float)nt * PI * (float)i / (float)N);
        sum = sum + r * 2.0f * tnm1 * cosine;
    }
    const float w = sum / (float)N;
    window[m + nt] = w;      // W2(M+NT)
    window[m - nt] = w;      // W2(M-NT)
}

extern "C" __global__ void dfi_dyn_init_zero(float* __restrict__ sum2d, const size_t n2,
                                             float* __restrict__ sum3d, const size_t n3,
                                             float* __restrict__ totalsum)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n = n2 > n3 ? n2 : n3;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        if (k < n2) sum2d[k] = 0.0f;
        if (k < n3) sum3d[k] = 0.0f;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) totalsum[0] = 0.0f;   // totalsum = 0. (line 200)
}
