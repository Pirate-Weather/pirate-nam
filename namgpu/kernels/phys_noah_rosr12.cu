// phys_noah_rosr12.cu -- ROSR12: tridiagonal solver (Noah LSM, soil column).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:3666-3725 (nam.v4.2.11, 83df4e13c1).
//
// Starting point: ArWen (gpuwm, Apache-2.0) namgpu/third_party/arwen/core/kernels/noah.cu, function
// noah_rosr12 (the same tridiagonal algorithm), copied with its Apache-2.0 notice (the ArWen header
// names gpuwm/core/kernels/noah.cu). Adapted to the NAM Fortran:
//   - NAM ROSR12 has INTENT(INOUT) C, P and DELTA. The Fortran first sets C(NSOIL) = 0.0 in the
//     caller's array, so that write is done back to global memory here (c[NSOIL-1] = 0).
//   - Fortran is 1-based; the back-substitution loop "DO K = 2,NSOIL; KK = NSOIL-K+1" is the
//     0-based loop kk = NSOIL-2 .. 0 below (same operations, same order).
//   - Soil arrays use the CONTRACT layout [NSOIL, JM, IM] (layer 1 at index 0), one thread per
//     column (grid_cols), so the solve for each (i, j) column is independent.
//
// Fortran (SUBROUTINE ROSR12(P,A,B,C,D,DELTA,NSOIL)), every executed statement:
//   C (NSOIL) = 0.0                                          (line 3697)
//   P (1) = - C (1) / B (1)                                  (line 3698)
//   DELTA (1) = D (1) / B (1)                                (line 3706)
//   DO K = 2,NSOIL                                           (line 3707)
//      P (K) = - C (K) * ( 1.0 / (B (K) + A (K) * P (K -1)) )        (line 3708)
//      DELTA (K) = (D (K) - A (K)* DELTA (K -1))* (1.0/ (B (K) + A (K) * P (K -1)))  (line 3709)
//   END DO                                                   (line 3711)
//   P (NSOIL) = DELTA (NSOIL)                                (line 3715)
//   DO K = 2,NSOIL                                           (line 3720)
//      KK = NSOIL - K + 1                                    (line 3721)
//      P (KK) = P (KK) * P (KK +1) + DELTA (KK)              (line 3722)
//   END DO                                                   (line 3723)
// Every argument of the Fortran signature is used (A, B, C, D, P, DELTA); NSOIL = 4 here.
//
// Fortran REAL = float (kfpt=single), fast numerics (FMA contraction on, IEEE division).

#include "nmmb_common.cuh"

#define NSOIL_ROSR 4

// One column: solve the NSOIL x NSOIL tridiagonal system. Arrays are indexed by soil layer k
// (0-based) with stride `ls` between layers; the caller passes the column's base pointers.
__device__ __forceinline__ void phys_noah_rosr12_col(
    const float* a, const float* b, float* c, const float* d,
    float* p, float* delta, size_t ls)
{
    // C (NSOIL) = 0.0   (INTENT(INOUT) write-back to the caller's array)
    c[(NSOIL_ROSR - 1) * ls] = 0.0f;

    // P (1) = - C (1) / B (1)
    p[0] = -c[0] / b[0];
    // DELTA (1) = D (1) / B (1)
    delta[0] = d[0] / b[0];

    // DO K = 2,NSOIL
    for (int k = 1; k < NSOIL_ROSR; ++k) {
        const float ak = a[k * ls];
        const float bk = b[k * ls];
        p[k * ls] = -c[k * ls] * (1.0f / (bk + ak * p[(k - 1) * ls]));
        delta[k * ls] = (d[k * ls] - ak * delta[(k - 1) * ls])
                        * (1.0f / (bk + ak * p[(k - 1) * ls]));
    }

    // P (NSOIL) = DELTA (NSOIL)
    p[(NSOIL_ROSR - 1) * ls] = delta[(NSOIL_ROSR - 1) * ls];

    // DO K = 2,NSOIL ; KK = NSOIL-K+1 (1-based)  ==  kk = NSOIL-2 .. 0 (0-based)
    for (int kk = NSOIL_ROSR - 2; kk >= 0; --kk) {
        p[kk * ls] = p[kk * ls] * p[(kk + 1) * ls] + delta[kk * ls];
    }
}

// One thread per (i, j) column. Soil fields are [NSOIL, JM, IM] (IDX3 with l0 = soil layer).
// P and DELTA are outputs (device arrays, same shape); C is INOUT (C[NSOIL-1] set to 0).
extern "C" __global__ void phys_noah_rosr12(
    const float* __restrict__ a,       // A      sub-diagonal                 [NSOIL, JM, IM]
    const float* __restrict__ b,       // B      diagonal                     [NSOIL, JM, IM]
    float* __restrict__ c,             // C      super-diagonal (INOUT)       [NSOIL, JM, IM]
    const float* __restrict__ d,       // D      right-hand side              [NSOIL, JM, IM]
    float* __restrict__ p,             // P      solution (output)            [NSOIL, JM, IM]
    float* __restrict__ delta,         // DELTA  forward sweep (INOUT)        [NSOIL, JM, IM]
    int im, int jm)
{
    COLUMN_IJ(im, jm)
    const size_t ls = (size_t)jm * (size_t)im;          // stride between soil layers
    const size_t n0 = IDX3(i0, j0, 0, im, jm);          // layer 1 of this column
    phys_noah_rosr12_col(a + n0, b + n0, c + n0, d + n0, p + n0, delta + n0, ls);
}
