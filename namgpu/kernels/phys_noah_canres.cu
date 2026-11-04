// phys_noah_canres.cu -- CANRES: canopy resistance and plant coefficient (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2331-2469 (nam.v4.2.11, 83df4e13c1).
//
// Start point and notice: ArWen (Apache-2.0, third_party/arwen/LICENSE, Copyright 2026 ArWen
// contributors), core/kernels/noah.cu, function noah_canres. Changes from that copy:
//   * CP is the NAM module constant (module_LS_NOAHLSM.F90:44, CP = 1004.6), not the 1004.5 of the
//     ArWen launch-time define.
//   * RD and SIGMA are the NAM module values (RD = 287.04, SIGMA = 5.67E-8), same as ArWen.
//   * The Fortran OUT terms RCS, RCT, RCQ, RCSOIL are written to arrays as well as RC and PC.
//   * Fortran REAL = float (kfpt=single). SFCTMP**4 and (TOPT-SFCTMP)**2 are written as explicit
//     products (integer exponents).
//
// Fortran (SUBROUTINE CANRES, lines 2331-2469), every executed statement in order:
//   RCS = 0; RCT = 0; RCQ = 0; RCSOIL = 0; RC = 0
//   FF = 0.55*2.0*SOLAR/(RGL*XLAI); RCS = (FF + RSMIN/RSMAX)/(1+FF)
//   RCS = MAX(RCS, 0.0001); RCT = 1.0 - 0.0016*((TOPT-SFCTMP)**2.0)
//   RCT = MAX(RCT, 0.0001); RCQ = 1.0/(1.0 + HS*(Q2SAT-Q2))
//   RCQ = MAX(RCQ, 0.01)
//   GX = (SH2O(1)-SMCWLT)/(SMCREF-SMCWLT), clamped to [0,1]
//   PART(1) = (ZSOIL(1)/ZSOIL(NROOT))*GX
//   DO K=2,NROOT: GX = (SH2O(K)-SMCWLT)/(SMCREF-SMCWLT) clamped to [0,1];
//                 PART(K) = ((ZSOIL(K)-ZSOIL(K-1))/ZSOIL(NROOT))*GX
//   RCSOIL = SUM over K=1..NROOT of PART(K); RCSOIL = MAX(RCSOIL, 0.0001)
//   RC = RSMIN/(XLAI*RCS*RCT*RCQ*RCSOIL)
//   RR = (4*EMISSI*SIGMA*RD/CP)*(SFCTMP**4)/(SFCPRS*CH) + 1.0
//   DELTA = (SLV/CP)*DQSDT2, SLV = 2.501E6
//   PC = (RR + DELTA)/(RR*(1 + RC*CH) + DELTA)
// Every term above has a counterpart below. The Fortran range has no MPI, halo, OpenMP or print
// statements. The local P declared in the Fortran is never assigned in this range and has no
// counterpart.
//
// Caller guard (Fortran 2154-2158, IF (SHDFAC > 0 .AND. XLAI > 0) CALL CANRES) is reproduced in
// this kernel from the SHDFAC input: columns outside it are left untouched, as in Fortran.
// NROOT outside 1..NSOIL poisons that column's outputs with NaN instead of indexing out of range.
// Clamps are NaN-propagating (Fortran IF tests), not fminf/fmaxf.
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3).
// 2-D fields are float32 [JM, IM]; SH2O is [NSOIL, JM, IM]; ZSOIL is [NSOIL] (column independent);
// NROOT is int32 [JM, IM].

#include "nmmb_common.cuh"

#define NSOIL_C 4

#define CANRES_RD      287.04f      // module_LS_NOAHLSM RD
#define CANRES_SIGMA   5.67E-8f     // module_LS_NOAHLSM SIGMA
#define CANRES_CP      1004.6f      // module_LS_NOAHLSM CP (module parameter, line 44)
#define CANRES_SLV     2.501000E6f  // CANRES local SLV

extern "C" __global__ void phys_noah_canres(
    const float* __restrict__ solar,   // SOLAR   incoming shortwave                 [JM, IM]
    const float* __restrict__ ch,      // CH      surface exchange coeff heat/moist  [JM, IM]
    const float* __restrict__ sfctmp,  // SFCTMP  air temperature, 1st level         [JM, IM]
    const float* __restrict__ q2,      // Q2      air humidity, 1st level            [JM, IM]
    const float* __restrict__ sfcprs,  // SFCPRS  surface pressure                   [JM, IM]
    const float* __restrict__ sh2o,    // SH2O    unfrozen soil moisture             [NSOIL, JM, IM]
    const float* __restrict__ zsoil,   // ZSOIL   soil layer depths (negative)       [NSOIL]
    const int*   __restrict__ nroot,   // NROOT   root-zone soil layers, 1..NSOIL    [JM, IM]
    const float* __restrict__ smcwlt,  // SMCWLT  wilting point                      [JM, IM]
    const float* __restrict__ smcref,  // SMCREF  reference soil moisture            [JM, IM]
    const float* __restrict__ rsmin,   // RSMIN   minimum stomatal resistance        [JM, IM]
    const float* __restrict__ rsmax,   // RSMAX   maximum stomatal resistance        [JM, IM]
    const float* __restrict__ rgl,     // RGL     solar radiation stress parameter   [JM, IM]
    const float* __restrict__ hs,      // HS      vapour pressure deficit parameter  [JM, IM]
    const float* __restrict__ xlai,    // XLAI    leaf area index                    [JM, IM]
    const float* __restrict__ topt,    // TOPT    optimum transpiration temperature  [JM, IM]
    const float* __restrict__ q2sat,   // Q2SAT   saturation humidity, 1st level     [JM, IM]
    const float* __restrict__ dqsdt2,  // DQSDT2  d(q_sat)/dT at 1st level           [JM, IM]
    const float* __restrict__ emissi,  // EMISSI  surface emissivity                 [JM, IM]
    const float* __restrict__ shdfac,  // SHDFAC  green vegetation fraction          [JM, IM]
    float* __restrict__ rc,          // RC      canopy resistance (output)         [JM, IM]
    float* __restrict__ pc,            // PC      plant coefficient (output)         [JM, IM]
    float* __restrict__ rcs,           // RCS     solar term (output)                [JM, IM]
    float* __restrict__ rct,           // RCT     temperature term (output)          [JM, IM]
    float* __restrict__ rcq,           // RCQ     vapour deficit term (output)       [JM, IM]
    float* __restrict__ rcsoil,        // RCSOIL  soil moisture term (output)        [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    const size_t nsl = (size_t)jm * (size_t)im;   // stride between soil layers of SH2O

    // Caller guard, module_LS_NOAHLSM.F90:2154-2158: CALL CANRES only IF (SHDFAC > 0 .AND.
    // XLAI > 0). Columns outside it are not touched: RC, PC, RCS, RCT, RCQ, RCSOIL keep their
    // previous values, exactly as the Fortran leaves them. This kernel is the only place that
    // guard is reproduced, so the unit needs no active-mask input from its caller.
    const float shdfac_c = shdfac[n];
    const float xlai_g = xlai[n];
    if (!(shdfac_c > 0.0f && xlai_g > 0.0f)) return;

    const int nroot_c = nroot[n];
    // NROOT is 1..NSOIL by contract. part[] holds NSOIL_C entries and zsoil[nroot-1] must be a
    // real layer, so an out-of-range value is refused loudly: the outputs of that column are
    // poisoned with NaN (they are checked for NaN by the gates and by the caller) instead of
    // reading or writing outside the arrays. Fortran has the same undefined behaviour.
    if (nroot_c < 1 || nroot_c > NSOIL_C) {
        const float nan_v = __int_as_float(0x7fc00000);
        rc[n] = nan_v; pc[n] = nan_v; rcs[n] = nan_v; rct[n] = nan_v;
        rcq[n] = nan_v; rcsoil[n] = nan_v;
        return;
    }

    const float solar_c = solar[n];
    const float ch_c = ch[n];
    const float sfctmp_c = sfctmp[n];
    const float q2_c = q2[n];
    const float sfcprs_c = sfcprs[n];
    const float smcwlt_c = smcwlt[n];
    const float smcref_c = smcref[n];
    const float rsmin_c = rsmin[n];
    const float rsmax_c = rsmax[n];
    const float rgl_c = rgl[n];
    const float hs_c = hs[n];
    const float xlai_c = xlai[n];
    const float topt_c = topt[n];
    const float q2sat_c = q2sat[n];
    const float dqsdt2_c = dqsdt2[n];
    const float emissi_c = emissi[n];

    // Clamps are written as the Fortran IF tests, not fminf/fmaxf: a NaN compares false and
    // passes through unchanged, so a bad SH2O, SMCREF or RCS shows up in the output instead of
    // being silently turned into 0 or 1. (Fortran IF (GX > 1.) / IF (GX < 0.) and MAX(RCS, ...)
    // propagate NaN in the same way.)

    // ---- initialise the canopy resistance multiplier terms --------------------------------
    float rcs_c = 0.0f;
    float rct_c = 0.0f;
    float rcq_c = 0.0f;
    float rcsoil_c = 0.0f;
    float rc_c = 0.0f;

    // ---- contribution due to incoming solar radiation -------------------------------------
    rc_c = 0.0f;
    float ff = 0.55f * 2.0f * solar_c / (rgl_c * xlai_c);
    rcs_c = (ff + rsmin_c / rsmax_c) / (1.0f + ff);

    // ---- contribution due to air temperature at first model level (Noilhan and Planton) ----
    if (rcs_c < 0.0001f) rcs_c = 0.0001f;
    float dt_opt = topt_c - sfctmp_c;
    rct_c = 1.0f - 0.0016f * (dt_opt * dt_opt);
    // ---- contribution due to vapour pressure deficit at first level (SSiB) -----------------
    if (rct_c < 0.0001f) rct_c = 0.0001f;
    rcq_c = 1.0f / (1.0f + hs_c * (q2sat_c - q2_c));
    if (rcq_c < 0.01f) rcq_c = 0.01f;

    // ---- contribution due to soil moisture availability (per soil layer, summed) -----------
    float gx = (sh2o[n] - smcwlt_c) / (smcref_c - smcwlt_c);   // SH2O(1), layer 1 at top
    if (gx > 1.0f) gx = 1.0f;
    if (gx < 0.0f) gx = 0.0f;
    float part[NSOIL_C];
    const float zsoil_nroot = zsoil[nroot_c - 1];
    part[0] = (zsoil[0] / zsoil_nroot) * gx;                    // PART(1)
    for (int k = 2; k <= nroot_c; ++k) {                         // DO K = 2, NROOT
        gx = (sh2o[(size_t)(k - 1) * nsl + n] - smcwlt_c) / (smcref_c - smcwlt_c);
        if (gx > 1.0f) gx = 1.0f;
        if (gx < 0.0f) gx = 0.0f;
        part[k - 1] = ((zsoil[k - 1] - zsoil[k - 2]) / zsoil_nroot) * gx;   // PART(K)
    }
    for (int k = 1; k <= nroot_c; ++k) {                         // DO K = 1, NROOT
        rcsoil_c = rcsoil_c + part[k - 1];
    }
    if (rcsoil_c < 0.0001f) rcsoil_c = 0.0001f;

    // ---- canopy resistance and plant coefficient -------------------------------------------
    rc_c = rsmin_c / (xlai_c * rcs_c * rct_c * rcq_c * rcsoil_c);
    float rr = (4.0f * emissi_c * CANRES_SIGMA * CANRES_RD / CANRES_CP)
               * (sfctmp_c * sfctmp_c * sfctmp_c * sfctmp_c) / (sfcprs_c * ch_c) + 1.0f;
    float delta = (CANRES_SLV / CANRES_CP) * dqsdt2_c;
    pc[n] = (rr + delta) / (rr * (1.0f + rc_c * ch_c) + delta);
    rc[n] = rc_c;

    rcs[n] = rcs_c;
    rct[n] = rct_c;
    rcq[n] = rcq_c;
    rcsoil[n] = rcsoil_c;
}
