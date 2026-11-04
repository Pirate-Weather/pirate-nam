// phys_noah_smflx.cu -- SMFLX: soil moisture flux (Noah LSM), with its callees SRT, SSTEP,
// ROSR12 and WDFCND.
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
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:3799-3916 (SMFLX),
// 4603-4852 (SRT), 4855-4942 (SSTEP), 3666-3725 (ROSR12) and 5312-5369 (WDFCND)
// (nam.v4.2.11, 83df4e13c1).
//
// Starting point: ArWen (gpuwm, Apache-2.0) namgpu/third_party/arwen/core/kernels/noah.cu, functions
// noah_srt, noah_sstep, noah_rosr12, noah_wdfcnd, noah_smflx, copied with the ArWen notice:
//   Copyright 2026 ArWen contributors, Licensed under the Apache License, Version 2.0
//   (https://www.apache.org/licenses/LICENSE-2.0).
// The ArWen copy is WRF v4.6.1 phys/module_sf_noahlsm.F and differs from the NAM Fortran in
// places that are NOT used here. Adapted to the NAM Fortran, which is the law for this kernel:
//   - SMFLX gate: NAM tests only (PCPDRP*DT) > 0.001*1000*(-ZSOIL(1))*SMCMAX (line 3900). The ArWen
//     version uses 0.0001 and an extra FAC2 > FLIMIT test. Neither is in NAM, so neither is here.
//   - WDFCND: NAM FACTR1 = 0.2/SMCMAX (ArWen 0.05/SMCMAX).
//   - SRT infiltration cap: NAM INFMAX = MIN(INFMAX, PX) with PX = PCPDRP*DT (ArWen caps at PX/DT).
//   - SSTEP: NAM clamps SMC again at SMCWLT after the 0.02 floor (ArWen does not).
//   - SSTEP CMC update is always executed in the NAM call; the first (dummy) SSTEP call of the
//     two-pass branch passes DUMMY as CMC, so its update is applied to a dummy and skipped here.
//
// Fortran SMFLX (SMC, NSOIL, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE, KDT, FRZFACT, SMCMAX, BEXP, SMCWLT,
// DKSAT, DWSAT, SHDFAC, CMCMAX, RUNOFF1, RUNOFF2, RUNOFF3, EDIR, EC, ET, DRIP), every executed statement:
//   DUMMY = 0.
//   RHSCT = SHDFAC*PRCP1 - EC ; DRIP = 0 ; TRHSCT = DT*RHSCT ; EXCESS = CMC + TRHSCT
//   IF (EXCESS > CMCMAX) DRIP = EXCESS - CMCMAX
//   PCPDRP = (1 - SHDFAC)*PRCP1 + DRIP/DT
//   SICE(I) = SMC(I) - SH2O(I)                                   (all layers, before the solves)
//   IF (PCPDRP*DT > 0.001*1000*(-ZSOIL(1))*SMCMAX) THEN          two-pass ("F") branch:
//     SRT(RHSTT,EDIR,ET,SH2O,SH2O,PCPDRP,...)                    -> RHSTT, AI, BI, CI, RUNOFF1, RUNOFF2
//     SSTEP(SH2OFG,SH2O,DUMMY,RHSTT,RHSCT,...)                   -> SH2OFG, SMC, RUNOFF3
//     SH2OA(K) = (SH2O(K) + SH2OFG(K))*0.5
//     SRT(RHSTT,EDIR,ET,SH2O,SH2OA,PCPDRP,...)
//     SSTEP(SH2O,SH2O,CMC,RHSTT,RHSCT,...)                       -> SH2O (in place), CMC, SMC, RUNOFF3
//   ELSE                                                          one-pass ("D") branch:
//     SRT(RHSTT,EDIR,ET,SH2O,SH2O,PCPDRP,...)
//     SSTEP(SH2O,SH2O,CMC,RHSTT,RHSCT,...)
//   END IF
//
// SRT (RHSTT, EDIR, ET, SH2O, SH2OA, NSOIL, PCPDRP, ZSOIL, DWSAT, DKSAT, SMCMAX, BEXP, RUNOFF1,
// RUNOFF2, DT, SMCWLT, SLOPE, KDT, FRZX, SICE, AI, BI, CI), every executed statement: the frozen-
// ground infiltration block (PCPDRP /= 0), the top-layer and interior/bottom matrix coefficients,
// RHSTT for each layer, and RUNOFF2 = SLOPX*WCND2 on the bottom layer. CVFRZ = 3 (PARAMETER).
// The Fortran (ACRT ** (CVFRZ - J)) with real base and integer exponent is powf here; ACRT > 0 on
// that path.
//
// SSTEP (SH2OOUT, SH2OIN, CMC, RHSTT, RHSCT, DT, NSOIL, SMCMAX, CMCMAX, SMCWLT, RUNOFF3, ZSOIL, SMC,
// SICE, AI, BI, CI): RHSTT, AI, BI, CI scaled by DT in place; ROSR12(CI, AI, BI, CIin, RHSTTin,
// RHSTT); the sequential soil water sum with WPLUS; SMC clamped to [max(0.02, SMCWLT), SMCMAX];
// SH2OOUT = max(SMC - SICE, 0); CMC = min(CMC + DT*RHSCT, CMCMAX) with CMC < 1e-20 set to 0.
//
// ROSR12(P=CI, A=AI, B=BI, C=CIin, D=RHSTTin, DELTA=RHSTT, NSOIL): see phys_noah_hstep.cu.
// WDFCND(WDF, WCND, SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX): FACTR1 = 0.2/SMCMAX;
// FACTR2 = SMC/SMCMAX; EXPON = BEXP + 2; WDF = DWSAT*FACTR2**EXPON; if SICEMAX > 0:
// VKWGT = 1/(1 + (500*SICEMAX)**3), WDF = VKWGT*WDF + (1 - VKWGT)*DWSAT*FACTR1**EXPON;
// EXPON = 2*BEXP + 3; WCND = DKSAT*FACTR2**EXPON.
//
// Layout: point kernel over columns (CONTRACT section 3, grid_cols), one thread per (i, j); the
// NSOIL = 4 layers and the local tridiagonal work arrays live in the thread. Soil arrays are float32
// [NSOIL, JM, IM] (layer 1 at the top, IDX3); column arrays are [JM, IM] (IDX2). Fortran REAL = float
// (kfpt=single). fmad is left on.

#include "nmmb_common.cuh"

#define NSOIL 4
#define CVFRZ 3

// -------------------------------------------------------------------- WDFCND
__device__ static void nam_wdfcnd(float smc, float smcmax, float bexp,
                                  float dksat, float dwsat, float sicemax,
                                  float& wdf, float& wcnd)
{
    float factr1 = 0.2f / smcmax;
    float factr2 = smc / smcmax;
    float expon = bexp + 2.0f;
    wdf = dwsat * powf(factr2, expon);
    if (sicemax > 0.0f) {
        float vkwgt = 1.0f / (1.0f + powf(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat * powf(factr1, expon);
    }
    expon = (2.0f * bexp) + 3.0f;
    wcnd = dksat * powf(factr2, expon);
}

// -------------------------------------------------------------------- ROSR12
// Fortran ROSR12(P, A, B, C, D, DELTA, NSOIL). C is overwritten (C(NSOIL) = 0), as in the Fortran.
__device__ static void nam_rosr12(float* p, const float* a, const float* b,
                                  float* c, const float* d, float* delta)
{
    c[NSOIL - 1] = 0.0f;
    p[0] = -c[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < NSOIL; ++k) {
        p[k] = -c[k] * (1.0f / (b[k] + a[k] * p[k - 1]));
        delta[k] = (d[k] - a[k] * delta[k - 1]) * (1.0f / (b[k] + a[k] * p[k - 1]));
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        int kk = NSOIL - k - 1;                 // Fortran KK = NSOIL - K + 1 (1-based), K = 2..NSOIL
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

// ----------------------------------------------------------------------- SRT
__device__ static void nam_srt(float* rhstt, float edir, const float* et,
                               const float* sh2o, const float* sh2oa,
                               float pcpdrp, const float* zsoil,
                               float dwsat, float dksat, float smcmax,
                               float bexp, float dt, float smcwlt,
                               float slope, float kdt, float frzx,
                               const float* sice, float* ai, float* bi,
                               float* ci, float& runoff1, float& runoff2)
{
    float dmax[NSOIL];
    float sicemax = 0.0f;
    for (int ks = 0; ks < NSOIL; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    float pddum = pcpdrp;
    float wdf, wcnd;
    runoff1 = 0.0f;
    runoff2 = 0.0f;

    if (pcpdrp != 0.0f) {
        float dt1 = dt / 86400.0f;
        float smcav = smcmax - smcwlt;
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
        float px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;
        float infmax = (px * (ddt / (px + ddt))) / dt;
        float fcr = 1.0f;
        if (dice > 1.0e-2f) {
            float acrt = (float)CVFRZ * frzx / dice;
            float sum = 1.0f;
            int ialp1 = CVFRZ - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int k = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) k = k * jj;
                sum = sum + powf(acrt, (float)(CVFRZ - j)) / (float)k;
            }
            fcr = 1.0f - expf(-acrt) * sum;
        }
        infmax = infmax * fcr;
        nam_wdfcnd(sh2oa[0], smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd);
        infmax = fmaxf(infmax, wcnd);
        infmax = fminf(infmax, px);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }

    nam_wdfcnd(sh2oa[0], smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd);
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    float dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];

    float ddz2 = 0.0f;
    for (int k = 1; k < NSOIL; ++k) {
        float denom2 = zsoil[k - 1] - zsoil[k];
        float wdf2, wcnd2, dsmdz2, slopx, denom;
        if (k != NSOIL - 1) {
            slopx = 1.0f;
            nam_wdfcnd(sh2oa[k], smcmax, bexp, dksat, dwsat, sicemax, wdf2, wcnd2);
            denom = zsoil[k - 1] - zsoil[k + 1];
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k] = -wdf2 * ddz2 / denom2;
        } else {
            slopx = slope;
            nam_wdfcnd(sh2oa[NSOIL - 1], smcmax, bexp, dksat, dwsat, sicemax, wdf2, wcnd2);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        float numer = (wdf2 * dsmdz2) + slopx * wcnd2 - (wdf * dsmdz) - wcnd + et[k];
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

// ---------------------------------------------------------------------- SSTEP
// SH2OIN and SH2OOUT may be the same array (the Fortran passes SH2O, SH2O): each layer reads its own
// SH2OIN before writing its SH2OOUT, so in-place use is exact. update_cmc is false for the dummy
// first call of the two-pass branch.
__device__ static void nam_sstep(float* sh2oout, const float* sh2oin,
                                 float& cmc, float* rhstt, float rhsct,
                                 float dt, float smcmax, float cmcmax,
                                 float smcwlt, float& runoff3,
                                 const float* zsoil, float* smc,
                                 const float* sice, float* ai, float* bi,
                                 float* ci, bool update_cmc)
{
    float rhsttin[NSOIL], ciin[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhstt[k] = rhstt[k] * dt;
        ai[k] = ai[k] * dt;
        bi[k] = 1.0f + bi[k] * dt;
        ci[k] = ci[k] * dt;
    }
    for (int k = 0; k < NSOIL; ++k) rhsttin[k] = rhstt[k];
    for (int k = 0; k < NSOIL; ++k) ciin[k] = ci[k];
    nam_rosr12(ci, ai, bi, ciin, rhsttin, rhstt);      // ci <- solution P, rhstt <- DELTA

    float wplus = 0.0f;
    float ddz = -zsoil[0];
    runoff3 = 0.0f;
    for (int k = 0; k < NSOIL; ++k) {
        if (k != 0) ddz = zsoil[k - 1] - zsoil[k];
        sh2oout[k] = sh2oin[k] + ci[k] + wplus / ddz;
        float stot = sh2oout[k] + sice[k];
        if (stot > smcmax) {
            if (k == 0) ddz = -zsoil[0];
            else ddz = -zsoil[k] + zsoil[k - 1];
            wplus = (stot - smcmax) * ddz;
        } else {
            wplus = 0.0f;
        }
        smc[k] = fmaxf(fminf(stot, smcmax), 0.02f);
        smc[k] = fmaxf(smc[k], smcwlt);
        sh2oout[k] = fmaxf(smc[k] - sice[k], 0.0f);
    }
    runoff3 = wplus;
    if (update_cmc) {
        cmc = cmc + dt * rhsct;
        if (cmc < 1.0e-20f) cmc = 0.0f;
        cmc = fminf(cmc, cmcmax);
    }
}

// ------------------------------------------------------------------- SMFLX
__device__ static void nam_smflx(float* smc, float& cmc, float dt,
                                 float prcp1, const float* zsoil,
                                 float* sh2o, float slope, float kdt,
                                 float frzfact, float smcmax, float bexp,
                                 float smcwlt, float dksat, float dwsat,
                                 float shdfac, float cmcmax, float edir,
                                 float ec, const float* et,
                                 float& runoff1, float& runoff2,
                                 float& runoff3, float& drip)
{
    float rhsct = shdfac * prcp1 - ec;
    drip = 0.0f;
    float trhsct = dt * rhsct;
    float excess = cmc + trhsct;
    if (excess > cmcmax) drip = excess - cmcmax;
    float pcpdrp = (1.0f - shdfac) * prcp1 + drip / dt;

    float sice[NSOIL];
    for (int i = 0; i < NSOIL; ++i) sice[i] = smc[i] - sh2o[i];

    float rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
    float dummy = 0.0f;
    if ((pcpdrp * dt) > (0.001f * 1000.0f * (-zsoil[0]) * smcmax)) {
        float sh2ofg[NSOIL], sh2oa[NSOIL];
        nam_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat, dksat,
                smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice,
                ai, bi, ci, runoff1, runoff2);
        nam_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, smcmax, cmcmax,
                  smcwlt, runoff3, zsoil, smc, sice, ai, bi, ci, false);
        for (int k = 0; k < NSOIL; ++k)
            sh2oa[k] = (sh2o[k] + sh2ofg[k]) * 0.5f;
        nam_srt(rhstt, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat, dksat,
                smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice,
                ai, bi, ci, runoff1, runoff2);
        nam_sstep(sh2o, sh2o, cmc, rhstt, rhsct, dt, smcmax, cmcmax,
                  smcwlt, runoff3, zsoil, smc, sice, ai, bi, ci, true);
    } else {
        nam_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat, dksat,
                smcmax, bexp, dt, smcwlt, slope, kdt, frzfact, sice,
                ai, bi, ci, runoff1, runoff2);
        nam_sstep(sh2o, sh2o, cmc, rhstt, rhsct, dt, smcmax, cmcmax,
                  smcwlt, runoff3, zsoil, smc, sice, ai, bi, ci, true);
    }
}

// -------------------------------------------------------------- kernel entry
// One thread per column (i, j). ET is [NSOIL, JM, IM]; SMC and SH2O are [NSOIL, JM, IM] and are
// updated in place; CMC is [JM, IM] and is updated in place. Column parameters are [JM, IM] per
// point (the soil-type and vegetation tables are looked up by the caller). ZSOIL [NSOIL] is the
// column-independent negative layer depth, layer 1 at the top.
extern "C" __global__ void phys_noah_smflx(
    float* smc,                 // SMC     [NSOIL, JM, IM] INOUT
    float* sh2o,                // SH2O    [NSOIL, JM, IM] INOUT
    float* cmc,                 // CMC     [JM, IM]        INOUT
    const float* zsoil,         // ZSOIL   [NSOIL]
    const float* et,            // ET      [NSOIL, JM, IM]
    const float* prcp1,         // PRCP1   [JM, IM]
    const float* edir,          // EDIR    [JM, IM]
    const float* ec,            // EC      [JM, IM]
    const float* shdfac,        // SHDFAC  [JM, IM]
    const float* slope,         // SLOPE   [JM, IM]
    const float* kdt,           // KDT     [JM, IM]
    const float* frzfact,       // FRZFACT [JM, IM]
    const float* smcmax,        // SMCMAX  [JM, IM]
    const float* bexp,          // BEXP    [JM, IM]
    const float* smcwlt,        // SMCWLT  [JM, IM]
    const float* dksat,         // DKSAT   [JM, IM]
    const float* dwsat,         // DWSAT   [JM, IM]
    const float* cmcmax,        // CMCMAX  [JM, IM]
    float* drip,                // DRIP    [JM, IM] output
    float* runoff1,             // RUNOFF1 [JM, IM] output
    float* runoff2,             // RUNOFF2 [JM, IM] output
    float* runoff3,             // RUNOFF3 [JM, IM] output
    float dt,                   // DT      s (NAM calls SMFLX every NPHS steps: DTPHS = 50 s)
    int im, int jm)
{
    COLUMN_IJ(im, jm)
    const size_t n2 = IDX2(i0, j0, im);

    float sm[NSOIL], sh[NSOIL], et_[NSOIL], zs[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        const size_t nk = IDX3(i0, j0, k, im, jm);
        sm[k] = smc[nk];
        sh[k] = sh2o[nk];
        et_[k] = et[nk];
        zs[k] = zsoil[k];
    }
    float cmc_ = cmc[n2];
    float r1, r2, r3, dr;
    nam_smflx(sm, cmc_, dt, prcp1[n2], zs, sh, slope[n2], kdt[n2],
              frzfact[n2], smcmax[n2], bexp[n2], smcwlt[n2], dksat[n2],
              dwsat[n2], shdfac[n2], cmcmax[n2], edir[n2], ec[n2], et_,
              r1, r2, r3, dr);
    for (int k = 0; k < NSOIL; ++k) {
        const size_t nk = IDX3(i0, j0, k, im, jm);
        smc[nk] = sm[k];
        sh2o[nk] = sh[k];
    }
    cmc[n2] = cmc_;
    drip[n2] = dr;
    runoff1[n2] = r1;
    runoff2[n2] = r2;
    runoff3[n2] = r3;
}
