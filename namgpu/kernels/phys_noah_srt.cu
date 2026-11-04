// phys_noah_srt.cu -- SRT: soil water diffusion right-hand side and tridiagonal coefficients (Noah LSM).
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
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:4603-4852 (SRT) and
// phys/module_LS_NOAHLSM.F90:5312-5369 (WDFCND, which SRT calls) (nam.v4.2.11, 83df4e13c1). Adapted from ArWen gpuwm/core/kernels/noah.cu (noah_srt, noah_wdfcnd),
// namgpu/third_party/arwen/core/kernels/noah.cu, Copyright 2026 ArWen contributors, Licensed under the
// Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
//
// Deviation from the ArWen copy, followed from the NAM Fortran: the infiltration cap is
// INFMAX = MIN(INFMAX, PX) with PX = PCPDRP*DT (module_LS_NOAHLSM.F90, the line after the FCR block).
// The ArWen (WRF-lineage) copy divides PX by DT there. The NAM Fortran is the reference here.
//
// Fortran SRT(RHSTT, EDIR, ET, SH2O, SH2OA, NSOIL, PCPDRP, ZSOIL, DWSAT, DKSAT, SMCMAX, BEXP,
//             RUNOFF1, RUNOFF2, DT, SMCWLT, SLOPE, KDT, FRZX, SICE, AI, BI, CI)
//
// Layout: point kernel, one thread per (i, j) column; the NSOIL soil layers are looped inside the
// thread (CONTRACT section 3). Soil arrays are float32 [NSOIL, JM, IM] (layer 1 at the top), indexed
// IDX3(i0, j0, k0, IM, JM). Per-column scalars are float32 [JM, IM], indexed IDX2(i0, j0, IM).
// ZSOIL is a float32 [NSOIL] profile shared by all columns. Fortran REAL = float (kfpt=single).
// fmad is left on (fast numerics). 1-based Fortran indices become 0-based: K -> k = K-1.

#include "nmmb_common.cuh"

#define NSOIL 4
#define CVFRZ 3

// ----------------------------------------------------------------- WDFCND (module_LS_NOAHLSM.F90)
__device__ static void noah_wdfcnd(float& wdf, float& wcnd, float smc,
                                   float smcmax, float bexp, float dksat,
                                   float dwsat, float sicemax)
{
    // Fortran WDFCND: FACTR1 = 0.2 / SMCMAX (no MIN against FACTR2; that term is not in NAM).
    float factr1 = 0.2f / smcmax;
    float factr2 = smc / smcmax;
    float expon = bexp + 2.0f;
    wdf = dwsat * powf(factr2, expon);
    if (sicemax > 0.0f) {
        float vkwgt = 1.0f / (1.0f + powf(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat * powf(factr1, expon);
    }
    expon = 2.0f * bexp + 3.0f;
    wcnd = dksat * powf(factr2, expon);
}

// ------------------------------------------------------------------ SRT
__device__ static void noah_srt(float* rhstt, float edir, const float* et,
                                const float* sh2o, const float* sh2oa,
                                float pcpdrp, const float* zsoil,
                                float dwsat, float dksat, float smcmax,
                                float bexp, float dt, float smcwlt,
                                float slope, float kdt, float frzx,
                                const float* sice, float* ai, float* bi,
                                float* ci, float& runoff1, float& runoff2)
{
    // Fortran SSTT (SSTT = WDF*DSMDZ + WCND + EDIR + ET(1)) is computed in NAM SRT and never read;
    // it is dead code and is not written here.
    // Fortran IOHINF (= 1) and the commented-out MXSMC = MAX(...) lines are unused in NAM; omitted.
    float dmax[NSOIL];
    float sicemax = 0.0f;
    for (int ks = 0; ks < NSOIL; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    float pddum = pcpdrp;
    runoff1 = 0.0f;
    runoff2 = 0.0f;

    if (pcpdrp != 0.0f) {
        float dt1 = dt / 86400.0f;
        float smcav = smcmax - smcwlt;
        // FROZEN GROUND VERSION of the saturation deficit, layer 1 then layers 2..NSOIL.
        dmax[0] = -zsoil[0] * smcav;
        float dice = -zsoil[0] * sice[0];
        dmax[0] = dmax[0] * (1.0f - (sh2oa[0] + sice[0] - smcwlt) / smcav);
        float dd = dmax[0];
        for (int ks = 1; ks < NSOIL; ++ks) {
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks];
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav;
            dmax[ks] = dmax[ks] * (1.0f - (sh2oa[ks] + sice[ks] - smcwlt) / smcav);
            dd = dd + dmax[ks];
        }
        float val = 1.0f - expf(-kdt * dt1);
        float ddt = dd * val;
        // PX is a depth (PCPDRP*DT), as in NAM. The ArWen copy divides it by DT; NAM does not.
        float px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;

        // FROZEN GROUND VERSION: reduction of infiltration by frozen ground.
        float infmax = (px * (ddt / (px + ddt))) / dt;
        float fcr = 1.0f;
        if (dice > 1.0e-2f) {
            float acrt = (float)CVFRZ * frzx / dice;
            float ssum = 1.0f;
            int ialp1 = CVFRZ - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int kk = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) kk = kk * jj;
                // Fortran uses integer powers (CVFRZ - J); powf is used here, so results may differ in the
            // last ulp. Not claimed bitwise.
            ssum = ssum + powf(acrt, (float)(CVFRZ - j)) / (float)kk;
            }
            fcr = 1.0f - expf(-acrt) * ssum;
        }

        // Correction of the infiltration limitation (Schaake and Koren, modified by Q. Duan).
        infmax = infmax * fcr;
        float wdf0, wcnd0;
        noah_wdfcnd(wdf0, wcnd0, sh2oa[0], smcmax, bexp, dksat, dwsat, sicemax);
        infmax = fmaxf(infmax, wcnd0);
        // Fortran: INFMAX = MIN(INFMAX, PX). INFMAX is a rate and PX a depth, so this cap is a units
        // mismatch carried over from NAM on purpose (faithful port); it is effectively inert because
        // the rate is almost always below the depth. Decision recorded here, not a transcription error.
        infmax = fminf(infmax, px);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }

    // Top-layer conductivities and matrix coefficients.
    float wdf, wcnd;
    noah_wdfcnd(wdf, wcnd, sh2oa[0], smcmax, bexp, dksat, dwsat, sicemax);
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    float dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];

    // Remaining layers, Fortran K = 2..NSOIL (k = 1..NSOIL-1).
    float ddz2 = 0.0f;
    for (int k = 1; k < NSOIL; ++k) {
        float denom2 = zsoil[k - 1] - zsoil[k];
        float wdf2, wcnd2, dsmdz2, slopx;
        if (k != NSOIL - 1) {
            slopx = 1.0f;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[k], smcmax, bexp, dksat, dwsat, sicemax);
            float denom = zsoil[k - 1] - zsoil[k + 1];
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k] = -wdf2 * ddz2 / denom2;
        } else {
            // Bottom layer: slope of the bottom layer is introduced.
            slopx = slope;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[NSOIL - 1], smcmax, bexp, dksat, dwsat, sicemax);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        float numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf * dsmdz - wcnd + et[k];
        rhstt[k] = numer / (-denom2);
        ai[k] = -wdf * ddz / denom2;
        bi[k] = -(ai[k] + ci[k]);
        if (k == NSOIL - 1) runoff2 = slopx * wcnd2;
        if (k != NSOIL - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// One thread per column; one launch. Outputs RHSTT, AI, BI, CI, RUNOFF1, RUNOFF2 as the Fortran SRT.
extern "C" __global__ void phys_noah_srt(
    float* __restrict__ rhstt,         // RHSTT   out                       [NSOIL, JM, IM]
    const float* __restrict__ et,      // ET      transpiration per layer   [NSOIL, JM, IM]
    const float* __restrict__ sh2o,    // SH2O    liquid soil water (new)   [NSOIL, JM, IM]
    const float* __restrict__ sh2oa,   // SH2OA   liquid soil water (old)   [NSOIL, JM, IM]
    const float* __restrict__ sice,    // SICE    soil ice                  [NSOIL, JM, IM]
    const float* __restrict__ zsoil,   // ZSOIL   layer depths (negative)   [NSOIL]
    const float* __restrict__ edir,    // EDIR    direct soil evaporation   [JM, IM]
    const float* __restrict__ pcpdrp,  // PCPDRP  rain reaching the ground  [JM, IM]
    const float* __restrict__ dwsat,   // DWSAT   saturated diffusivity     [JM, IM]
    const float* __restrict__ dksat,   // DKSAT   saturated conductivity    [JM, IM]
    const float* __restrict__ smcmax,  // SMCMAX  porosity                  [JM, IM]
    const float* __restrict__ bexp,    // BEXP    B exponent                [JM, IM]
    const float* __restrict__ smcwlt,  // SMCWLT  wilting point             [JM, IM]
    const float* __restrict__ slope,   // SLOPE   bottom-layer slope        [JM, IM]
    const float* __restrict__ kdt,     // KDT     infiltration factor       [JM, IM]
    const float* __restrict__ frzx,    // FRZX    frozen-ground factor      [JM, IM]
    float* __restrict__ ai,            // AI      out                       [NSOIL, JM, IM]
    float* __restrict__ bi,            // BI      out                       [NSOIL, JM, IM]
    float* __restrict__ ci,            // CI      out                       [NSOIL, JM, IM]
    float* __restrict__ runoff1,       // RUNOFF1 surface runoff            [JM, IM]
    float* __restrict__ runoff2,       // RUNOFF2 baseflow runoff           [JM, IM]
    float dt,                          // DT      time step (s)
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;

    const size_t n2 = IDX2(i0, j0, im);
    float rh[NSOIL], e[NSOIL], s[NSOIL], sa[NSOIL], si[NSOIL], a[NSOIL], b[NSOIL], c[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        e[k] = et[n];
        s[k] = sh2o[n];
        sa[k] = sh2oa[n];
        si[k] = sice[n];
    }
    float r1 = 0.0f, r2 = 0.0f;
    noah_srt(rh, edir[n2], e, s, sa, pcpdrp[n2], zsoil, dwsat[n2], dksat[n2],
             smcmax[n2], bexp[n2], dt, smcwlt[n2], slope[n2], kdt[n2], frzx[n2],
             si, a, b, c, r1, r2);
    for (int k = 0; k < NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        rhstt[n] = rh[k];
        ai[n] = a[k];
        bi[n] = b[k];
        ci[n] = c[k];
    }
    runoff1[n2] = r1;
    runoff2[n2] = r2;
}
