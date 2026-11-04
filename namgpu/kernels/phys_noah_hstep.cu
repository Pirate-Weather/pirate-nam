// phys_noah_hstep.cu -- HSTEP: soil temperature step (Noah LSM), with its tridiagonal solve ROSR12.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The NOAA module header of the
// source file is kept as the notice (phys/module_LS_NOAHLSM.F90, MODULE MODULE_LS_NOAHLSM):
//
//   !-----------------------------------------------------------------------
//   !
//         MODULE MODULE_LS_NOAHLSM
//   !
//   !-----------------------------------------------------------------------
//   !***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
//   !-----------------------------------------------------------------------
//   ! HISTORY LOG:
//   !
//   !   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
//   !               Vasic - SOILW limitted to positive values.
//
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:3153-3201 (HSTEP) and
// phys/module_LS_NOAHLSM.F90:3666-3725 (ROSR12, the solve HSTEP calls) (nam.v4.2.11, 83df4e13c1).
//
// Adapted from ArWen gpuwm/core/kernels/noah.cu (noah_hstep, noah_rosr12), namgpu/third_party/arwen/
// core/kernels/noah.cu, Copyright 2026 ArWen contributors, Licensed under the Apache License,
// Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The ArWen version drops the INOUT
// write-backs; this kernel keeps them, because the Fortran leaves the scaled AI, BI, the solution
// in CI and DELTA in RHSTS in the caller's arrays.
//
// Fortran HSTEP (STCOUT, STCIN, RHSTS, DT, NSOIL, AI, BI, CI):
//   RHSTS(K) = RHSTS(K)*DT ; AI(K) = AI(K)*DT ; BI(K) = 1. + BI(K)*DT ; CI(K) = CI(K)*DT
//   CALL ROSR12(CI, AI, BI, CIin, RHSTSin, RHSTS, NSOIL)   (CIin, RHSTSin: copies of CI, RHSTS)
//   STCOUT(K) = STCIN(K) + CI(K)
// Fortran ROSR12(P=CI, A, B, C=CIin, D=RHSTSin, DELTA=RHSTS, NSOIL): C(NSOIL) = 0 on the copy;
// forward elimination for K = 1..NSOIL; P(NSOIL) = DELTA(NSOIL); back substitution for K = 2..NSOIL
// with KK = NSOIL-K+1 (1-based), P(KK) = P(KK)*P(KK+1) + DELTA(KK).
//
// Layout: point kernel, one thread per (i, j) column; the NSOIL soil layers are looped inside the
// thread (CONTRACT section 3). Soil arrays are float32 [NSOIL, JM, IM] (C order, layer 1 at the top),
// indexed IDX3(i0, j0, k0, IM, JM). Fortran REAL = float (kfpt=single). fmad is left on.

#include "nmmb_common.cuh"

#define NSOIL 4

extern "C" __global__ void phys_noah_hstep(
    const float* __restrict__ stcin,   // STCIN   soil temperature in           [NSOIL, JM, IM]
    float* __restrict__ stcf,          // STCOUT  soil temperature out          [NSOIL, JM, IM]
    float* __restrict__ rhsts,         // RHSTS   INOUT: rhs in, DELTA out      [NSOIL, JM, IM]
    float* __restrict__ ai,            // AI      INOUT: scaled by DT           [NSOIL, JM, IM]
    float* __restrict__ bi,            // BI      INOUT: 1 + BI*DT              [NSOIL, JM, IM]
    float* __restrict__ ci,            // CI      INOUT: scaled, then solution  [NSOIL, JM, IM]
    float dt,                          // DT      time step (s)
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;

    float rs[NSOIL], a[NSOIL], b[NSOIL], c[NSOIL], cin[NSOIL], p[NSOIL], delta[NSOIL];

    // HSTEP: scale the finite-difference values by DT (the DO K loops, module_LS_NOAHLSM.F90:3153-3201).
    for (int k = 0; k < NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        rs[k] = rhsts[n] * dt;             // RHSTS(K) = RHSTS(K)*DT
        a[k] = ai[n] * dt;                 // AI(K)    = AI(K)*DT
        b[k] = 1.0f + bi[n] * dt;          // BI(K)    = 1. + BI(K)*DT
        c[k] = ci[n] * dt;                 // CI(K)    = CI(K)*DT
    }

    // ROSR12 (module_LS_NOAHLSM.F90:3666-3725). C is the copy CIin with C(NSOIL) = 0.
    for (int k = 0; k < NSOIL; ++k) cin[k] = c[k];
    cin[NSOIL - 1] = 0.0f;

    // Initialise the first layer: P(1) = -C(1)/B(1), DELTA(1) = D(1)/B(1).
    p[0] = -cin[0] / b[0];
    delta[0] = rs[0] / b[0];

    // Forward elimination for layers 2 .. NSOIL.
    for (int k = 1; k < NSOIL; ++k) {
        const float inv = 1.0f / (b[k] + a[k] * p[k - 1]);
        p[k] = -cin[k] * inv;
        delta[k] = (rs[k] - a[k] * delta[k - 1]) * inv;
    }

    // Lowest layer takes DELTA; back substitution for KK = NSOIL-1 .. 1 (1-based).
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        const int kk = NSOIL - k - 1;      // 0-based form of KK = NSOIL-K+1, K = 2..NSOIL
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }

    // Write back: STCOUT = STCIN + CI (CI now the solution P), and the INOUT arrays as the Fortran leaves them.
    for (int k = 0; k < NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        stcf[n] = stcin[n] + p[k];         // STCOUT(K) = STCIN(K) + CI(K)
        rhsts[n] = delta[k];               // RHSTS = DELTA (ROSR12 output)
        ai[n] = a[k];                      // AI scaled by DT
        bi[n] = b[k];                      // BI = 1 + BI*DT
        ci[n] = p[k];                      // CI = solution P
    }
}
