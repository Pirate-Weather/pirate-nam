// phys_noah_sstep.cu -- SSTEP: soil moisture step (Noah LSM), with its tridiagonal solve ROSR12.
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
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:4855-4942 (SSTEP) and
// phys/module_LS_NOAHLSM.F90:3666-3725 (ROSR12, the solve SSTEP calls) (nam.v4.2.11, 83df4e13c1).
//
// Start point and notice: ArWen (Apache-2.0, namgpu/third_party/arwen/LICENSE, Copyright 2026 ArWen
// contributors), core/kernels/noah.cu, functions noah_rosr12 and noah_sstep. Changes from that copy:
//   * The INOUT write-backs are kept, as the Fortran has them: AI and BI are left scaled by DT, CI is
//     left holding the ROSR12 solution P, RHSTT holds DELTA, and CMC is always updated. ArWen drops
//     the AI/BI/CI/RHSTT write-backs.
//   * The SMCWLT floor is applied (SMC(K) = MAX(SMC(K), SMCWLT), module_LS_NOAHLSM.F90:4918), which the
//     ArWen copy omits.
//   * The ArWen update_cmc flag is not here. The Fortran always updates CMC; the SFLX caller's first
//     SSTEP call passes DUMMY as CMC, so the phys-noah-sflx unit hands this kernel a scratch CMC there.
//   * Soil arrays are [NSOIL=4, JM, IM], layer 1 at the top (CONTRACT section 1). SMCMAX, SMCWLT and
//     CMCMAX are per column (SFLX passes them per point). ZSOIL is per column, [NSOIL, JM, IM]: SFLX
//     builds ZSOIL per point (module_LS_NOAHLSM.F90:1823-1837), so sea-ice points (ICE==1,
//     ZSOIL(KZ) = -3*KZ/NSOIL) and land points (cumulative SLDPTH) differ. The wrapper builds it.
//
// Fortran (SUBROUTINE SSTEP, 4855-4942), every executed statement in order:
//   RHSTT(K) = RHSTT(K)*DT; AI(K) = AI(K)*DT; BI(K) = 1.+BI(K)*DT; CI(K) = CI(K)*DT   (K=1..NSOIL)
//   RHSTTin = RHSTT; CIin = CI
//   CALL ROSR12(CI, AI, BI, CIin, RHSTTin, RHSTT, NSOIL)     -> CI = P, RHSTT = DELTA
//   WPLUS = 0; RUNOFF3 = 0; DDZ = -ZSOIL(1)
//   DO K: IF (K/=1) DDZ = ZSOIL(K-1)-ZSOIL(K)
//         SH2OOUT(K) = SH2OIN(K) + CI(K) + WPLUS/DDZ
//         STOT = SH2OOUT(K) + SICE(K)
//         IF (STOT > SMCMAX): DDZ = -ZSOIL(1) if K==1, else -ZSOIL(K)+ZSOIL(K-1); WPLUS=(STOT-SMCMAX)*DDZ
//         ELSE WPLUS = 0
//         SMC(K) = MAX(MIN(STOT,SMCMAX),0.02); SMC(K) = MAX(SMC(K),SMCWLT)
//         SH2OOUT(K) = MAX(SMC(K)-SICE(K), 0.0)
//   RUNOFF3 = WPLUS; CMC = CMC + DT*RHSCT; IF (CMC<1.E-20) CMC=0; CMC = MIN(CMC,CMCMAX)
// ROSR12 (3666-3725): C(NSOIL)=0; P(1)=-C(1)/B(1); DELTA(1)=D(1)/B(1);
//   DO K=2,NSOIL: P(K) = -C(K)*(1/(B(K)+A(K)*P(K-1))); DELTA(K) = (D(K)-A(K)*DELTA(K-1))*(1/(B(K)+A(K)*P(K-1)))
//   P(NSOIL) = DELTA(NSOIL); DO K=2,NSOIL: KK=NSOIL-K+1; P(KK) = P(KK)*P(KK+1)+DELTA(KK)
// No MPI, halo, OpenMP or print statements in this range.
//
// Layout: one thread per (i, j) column, all NSOIL layers looped inside the thread. Arrays are float32.
// SH2OIN and SH2OOUT may alias (SFLX calls SSTEP(SH2O,SH2O,...)); each layer reads SH2OIN(K) before it
// writes SH2OOUT(K) and no layer reads another layer's SH2OIN, so aliasing is safe. No __restrict__.

#include "nmmb_common.cuh"

#define SS_NSOIL 4

// ---------------------------------------------------------------- ROSR12 (3666-3725)
__device__ static void noah_sstep_rosr12(float* p, const float* a, const float* b,
                                         const float* c_in, const float* d, float* delta)
{
    float c[SS_NSOIL];
    for (int k = 0; k < SS_NSOIL; ++k) c[k] = c_in[k];
    c[SS_NSOIL - 1] = 0.0f;
    p[0] = -c[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < SS_NSOIL; ++k) {
        float den = 1.0f / (b[k] + a[k] * p[k - 1]);
        p[k] = -c[k] * den;
        delta[k] = (d[k] - a[k] * delta[k - 1]) * den;
    }
    p[SS_NSOIL - 1] = delta[SS_NSOIL - 1];
    for (int k = 1; k < SS_NSOIL; ++k) {
        int kk = SS_NSOIL - k - 1;   // Fortran KK = NSOIL-K+1 (1-based), zero-based here
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

// ---------------------------------------------------------------- SSTEP (4855-4942)
extern "C" __global__ void phys_noah_sstep(
    const float* sh2oin,      // [NSOIL, JM, IM] SH2OIN (may alias sh2oout)
    float* sh2oout,           // [NSOIL, JM, IM] SH2OOUT
    float* cmc,               // [JM, IM]        CMC, INOUT
    float* rhstt,             // [NSOIL, JM, IM] RHSTT, INOUT (scaled, then DELTA)
    const float* rhsct,       // [JM, IM]        RHSCT
    float dt,                 // DT (s)
    const float* smcmax,      // [JM, IM]        SMCMAX
    const float* smcwlt,      // [JM, IM]        SMCWLT
    const float* cmcmax,      // [JM, IM]        CMCMAX
    const float* zsoil,       // [NSOIL, JM, IM] ZSOIL per column (negative depths, layer 1 at top)
    float* smc,               // [NSOIL, JM, IM] SMC, written
    const float* sice,        // [NSOIL, JM, IM] SICE
    float* ai,                // [NSOIL, JM, IM] AI, INOUT (scaled by DT)
    float* bi,                // [NSOIL, JM, IM] BI, INOUT (1 + BI*DT)
    float* ci,                // [NSOIL, JM, IM] CI, INOUT (scaled, then ROSR12 P)
    float* runoff3,           // [JM, IM]        RUNOFF3, out
    int im, int jm)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t c2 = IDX2(i0, j0, im);

    float ai_l[SS_NSOIL], bi_l[SS_NSOIL], ci_l[SS_NSOIL], rhstt_l[SS_NSOIL];
    float p[SS_NSOIL], delta[SS_NSOIL];

    // Scale to 'amount' values (first loop of SSTEP).
    for (int k = 0; k < SS_NSOIL; ++k) {
        const size_t idx = IDX3(i0, j0, k, im, jm);
        rhstt_l[k] = rhstt[idx] * dt;
        ai_l[k] = ai[idx] * dt;
        bi_l[k] = 1.0f + bi[idx] * dt;
        ci_l[k] = ci[idx] * dt;
    }

    // ROSR12(CI, AI, BI, CIin, RHSTTin, RHSTT): P -> CI, DELTA -> RHSTT.
    noah_sstep_rosr12(p, ai_l, bi_l, ci_l, rhstt_l, delta);
    for (int k = 0; k < SS_NSOIL; ++k) {
        const size_t idx = IDX3(i0, j0, k, im, jm);
        ai[idx] = ai_l[k];
        bi[idx] = bi_l[k];
        ci[idx] = p[k];
        rhstt[idx] = delta[k];
    }

    const float smcmax_p = smcmax[c2];
    const float smcwlt_p = smcwlt[c2];

    float zs[SS_NSOIL];
    for (int k = 0; k < SS_NSOIL; ++k) zs[k] = zsoil[IDX3(i0, j0, k, im, jm)];

    float wplus = 0.0f;
    float ddz = -zs[0];
    for (int k = 0; k < SS_NSOIL; ++k) {
        const size_t idx = IDX3(i0, j0, k, im, jm);
        if (k != 0) ddz = zs[k - 1] - zs[k];
        sh2oout[idx] = sh2oin[idx] + p[k] + wplus / ddz;
        const float stot = sh2oout[idx] + sice[idx];
        if (stot > smcmax_p) {
            if (k == 0) ddz = -zs[0];
            else        ddz = -zs[k] + zs[k - 1];
            wplus = (stot - smcmax_p) * ddz;
        } else {
            wplus = 0.0f;
        }
        float s = fminf(stot, smcmax_p);
        s = fmaxf(s, 0.02f);
        s = fmaxf(s, smcwlt_p);
        smc[idx] = s;
        sh2oout[idx] = fmaxf(s - sice[idx], 0.0f);
    }
    runoff3[c2] = wplus;

    // Canopy water: CMC = CMC + DT*RHSCT, floor at 1e-20 then cap at CMCMAX.
    float cm = cmc[c2] + dt * rhsct[c2];
    if (cm < 1.0e-20f) cm = 0.0f;
    cm = fminf(cm, cmcmax[c2]);
    cmc[c2] = cm;
}
