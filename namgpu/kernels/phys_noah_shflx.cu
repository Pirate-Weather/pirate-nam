// phys_noah_shflx.cu -- SHFLX: soil heat flux and soil temperature update (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:3729-3796 (nam.v4.2.11, 83df4e13c1),
// together with the routines SHFLX calls, each ported from its own Fortran lines:
//   HRT      2785-3035   (land-mass soil heat RHS and tri-diagonal coefficients)
//   HRTICE   3038-3150   (sea-ice case, ICE == 1)
//   HSTEP    3153-3201   (soil temperature step through ROSR12)
//   ROSR12   3666-3725   (tri-diagonal solver)
//   SNKSRC   3960-4076   (freeze/thaw heat sink and source)
//   FRH2O    2644-2782   (unfrozen supercooled water, Koren et al. 1999 eqn 17)
//   TBND     4945-4985   (temperature at a layer interface)
//   TDFCND   4989-5092   (soil thermal diffusivity, Peters-Lidard et al. 2000)
//   TMPAVG   5095-5197   (layer temperature average over the frozen/thawed interface)
// The helper routines are device functions in this file, not other units kernels, because each
// unit compiles as its own RawModule (CONTRACT section 3). Their Fortran is the same text as the
// units phys-noah-hrt, phys-noah-hrtice, phys-noah-hstep, phys-noah-rosr12, phys-noah-snksrc,
// phys-noah-frh2o, phys-noah-tbnd, phys-noah-tdfcnd and phys-noah-tmpavg, so the two copies must
// be kept in step if either is changed.
//
// SHFLX executed statements, every one ported:
//   IF (ICE == 1): HRTICE(RHSTS,STC,...) then HSTEP(STCF,STC,RHSTS,...)
//   ELSE:          HRT(RHSTS,STC,SMC,...,SH2O,...,AI,BI,CI,VEGTYP,IVEGSRC) then HSTEP(...)
//   STC(I) = STCF(I), I = 1..NSOIL
//   T1   = (YY + (ZZ1-1.0)*STC(1)) / ZZ1          (updated STC(1); T1 is INOUT, input unused)
//   SSOIL = DF1*(STC(1)-T1) / (0.5*ZSOIL(1))
// SH2O is updated in place by SNKSRC inside HRT (land case), so it is INOUT and written back.
// F1 is passed to HRT by the Fortran but never read there; it is kept in the signature.
//
// Constants (single precision, Fortran kfpt=single): HRT CAIR=1004.0, CICE=2.106E6, CH2O=4.2E6,
// T0=273.15; HRTICE HCPCT=1.72396E+6, TBOT=271.16 (DATA); SNKSRC DH2O=1.0E3, HLICE=3.3350E5,
// T0=2.7315E2; FRH2O CK=8.0, BLIM=5.5, ERROR=0.005, HLICE=3.335E5, GS=9.81, T0=273.15;
// TMPAVG T0=2.7315E2; TDFCND THKICE=2.2, THKW=0.57, THKO=2.0, THKQTZ=7.7, GAMMD = 2700 form.
// IW = 13 when IVEGSRC == 1 (IGBP), urban vegetation (VEGTYP == IW) uses CSOIL=3.0E6 and DF1N=3.24.
//
// Layout: one thread per (i, j) column, grid.z = 1 (CONTRACT section 3). Soil fields are
// [NSOIL, JM, IM] float32 with layer 1 (top) at index 0; zsoil is the [NSOIL] device vector
// (negative, cumulative: ZSOIL(1) = -SLDPTH(1)); the per-column scalars SMCMAX, PSISAT, BEXP, F1,
// DF1, QUARTZ, CSOIL, TBOT, ZBOT, YY, ZZ1, T1 are [JM, IM]; ICE and VEGTYP are int32 [JM, IM];
// DT and IVEGSRC are scalars. Outputs: STC, SH2O (INOUT), T1 (INOUT), SSOIL (OUT).
// Fortran REAL = float (kfpt=single). No host sync, no device allocation inside the kernel.

#include "nmmb_common.cuh"

#define NSOIL 4

// ---------------------------------------------------------------- FRH2O (module 2644-2782)
// Unfrozen supercooled liquid water. Fortran is 1-based; no index arithmetic here.
__device__ static float nam_frh2o(float tkelv, float smc, float sh2o, float smcmax,
                                  float bexp, float psis)
{
    const float ck = 8.0f, blim = 5.5f, error = 0.005f;
    const float hlice = 3.335e5f, gs = 9.81f, t0 = 273.15f;
    float bx = bexp;
    if (bexp > blim) bx = blim;
    int nlog = 0;
    int kcount = 0;
    float free_w;
    if (tkelv > (t0 - 1.0e-3f)) {
        free_w = smc;
    } else {
        if (ck != 0.0f) {
            float swl = smc - sh2o;
            if (swl > (smc - 0.02f)) swl = smc - 0.02f;
            if (swl < 0.0f) swl = 0.0f;
            while ((nlog < 10) && (kcount == 0)) {          // label 1001 .. GOTO 1002
                nlog = nlog + 1;
                const float ratio = smcmax / (smc - swl);
                const float onecks = 1.0f + ck * swl;
                const float df = logf((psis * gs / hlice) * (onecks * onecks) * powf(ratio, bx))
                               - logf(-(tkelv - t0) / tkelv);
                const float denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
                float swlk = swl - df / denom;
                if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
                if (swlk < 0.0f) swlk = 0.0f;
                const float dswl = fabsf(swlk - swl);
                swl = swlk;
                if (dswl <= error) kcount = kcount + 1;
            }
            free_w = smc - swl;
        } else {
            free_w = 0.0f;   // unreachable: CK = 8.0 (the Fortran branch is CK /= 0)
        }
        if (kcount == 0) {
            float fk = powf((hlice / (gs * (-psis))) * ((tkelv - t0) / tkelv), -1.0f / bx) * smcmax;
            if (fk < 0.02f) fk = 0.02f;
            free_w = fminf(fk, smc);
        }
    }
    return free_w;
}

// ---------------------------------------------------------------- TBND (module 4945-4985)
// K is the 1-based Fortran layer index; zsoil is 0-based storage of ZSOIL(1..NSOIL).
__device__ static float nam_tbnd(float tu, float tb, const float* zsoil, float zbot, int k)
{
    float zup, zb;
    if (k == 1) zup = 0.0f;
    else        zup = zsoil[k - 2];                         // ZSOIL(K-1)
    if (k == NSOIL) zb = 2.0f * zbot - zsoil[k - 1];        // ZB = 2*ZBOT - ZSOIL(K)
    else            zb = zsoil[k];                          // ZSOIL(K+1)
    return tu + (tb - tu) * (zup - zsoil[k - 1]) / (zup - zb);
}

// ------------------------------------------------------- TDFCND (module 4989-5092)
__device__ static float nam_tdfcnd(float smc, float qz, float smcmax, float sh2o)
{
    const float thkice = 2.2f, thkw = 0.57f, thko = 2.0f, thkqtz = 7.7f;
    const float satratio = smc / smcmax;
    const float thks = powf(thkqtz, qz) * powf(thko, 1.0f - qz);
    const float xunfroz = sh2o / smc;
    const float xu = xunfroz * smcmax;
    const float thksat = powf(thks, 1.0f - smcmax) * powf(thkice, smcmax - xu) * powf(thkw, xu);
    const float gammd = (1.0f - smcmax) * 2700.0f;
    const float thkdry = (0.135f * gammd + 64.7f) / (2700.0f - 0.947f * gammd);
    float ake;
    if ((sh2o + 0.0005f) < smc) {
        ake = satratio;
    } else {
        if (satratio > 0.1f) ake = log10f(satratio) + 1.0f;
        else                 ake = 0.0f;
    }
    return ake * (thksat - thkdry) + thkdry;
}

// ------------------------------------------------------- TMPAVG (module 5095-5197)
__device__ static float nam_tmpavg(float tup, float tm, float tdn, const float* zsoil, int k)
{
    const float t0 = 2.7315e2f;
    float dz;
    if (k == 1) dz = -zsoil[0];
    else        dz = zsoil[k - 2] - zsoil[k - 1];
    const float dzh = dz * 0.5f;
    float tavg, x0, xup, xdn;
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                tavg = (tup + 2.0f * tm + tdn) / 4.0f;
            } else {
                x0 = (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (tup * dzh + tm * (dzh + x0) + t0 * (2.0f * dzh - x0)) / dz;
            }
        } else {
            if (tdn < t0) {
                xup = (t0 - tup) * dzh / (tm - tup);
                xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (tup * xup + t0 * (2.0f * dz - xup - xdn) + tdn * xdn) / dz;
            } else {
                xup = (t0 - tup) * dzh / (tm - tup);
                tavg = 0.5f * (tup * xup + t0 * (2.0f * dz - xup)) / dz;
            }
        }
    } else {
        if (tm < t0) {
            if (tdn < t0) {
                xup = dzh - (t0 - tup) * dzh / (tm - tup);
                tavg = 0.5f * (t0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh) / dz;
            } else {
                xup = dzh - (t0 - tup) * dzh / (tm - tup);
                xdn = (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (t0 * (2.0f * dz - xup - xdn) + tm * (xup + xdn)) / dz;
            }
        } else {
            if (tdn < t0) {
                xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
                tavg = (t0 * (dz - xdn) + 0.5f * (t0 + tdn) * xdn) / dz;
            } else {
                tavg = (tup + 2.0f * tm + tdn) / 4.0f;
            }
        }
    }
    return tavg;
}

// -------------------------------------------------------- SNKSRC (module 3960-4076)
// SH2O is INTENT(INOUT): updated to the new liquid water. Returns TSNSR.
__device__ static float nam_snksrc(float tavg, float smc, float& sh2o, const float* zsoil,
                                   float smcmax, float psisat, float bexp, float dt,
                                   int k, float qtot)
{
    const float dh2o = 1.0e3f, hlice = 3.3350e5f;
    float dz;
    if (k == 1) dz = -zsoil[0];
    else        dz = zsoil[k - 2] - zsoil[k - 1];
    const float free_w = nam_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    float xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
    if (xh2o < sh2o && xh2o < free_w) {
        if (free_w > sh2o) xh2o = sh2o;
        else               xh2o = free_w;
    }
    if (xh2o > sh2o && xh2o > free_w) {
        if (free_w < sh2o) xh2o = sh2o;
        else               xh2o = free_w;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc)  xh2o = smc;
    const float tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

// ---------------------------------------------------------- ROSR12 (module 3666-3725)
// P, C, DELTA are INOUT; A, B, D are IN. Tri-diagonal solve, NSOIL = 4.
__device__ static void nam_rosr12(float* p, const float* a, const float* b, float* c,
                                  const float* d, float* delta)
{
    c[NSOIL - 1] = 0.0f;
    p[0] = -c[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 2; k <= NSOIL; ++k) {
        const int k0 = k - 1;
        p[k0] = -c[k0] * (1.0f / (b[k0] + a[k0] * p[k0 - 1]));
        delta[k0] = (d[k0] - a[k0] * delta[k0 - 1]) * (1.0f / (b[k0] + a[k0] * p[k0 - 1]));
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 2; k <= NSOIL; ++k) {
        const int kk = NSOIL - k + 1;                       // 1-based KK
        const int kk0 = kk - 1;
        p[kk0] = p[kk0] * p[kk0 + 1] + delta[kk0];
    }
}

// ----------------------------------------------------------- HSTEP (module 3153-3201)
__device__ static void nam_hstep(float* stcout, const float* stcin, float* rhsts, float dt,
                                 float* ai, float* bi, float* ci)
{
    for (int k = 0; k < NSOIL; ++k) {
        rhsts[k] = rhsts[k] * dt;
        ai[k] = ai[k] * dt;
        bi[k] = 1.0f + bi[k] * dt;
        ci[k] = ci[k] * dt;
    }
    float rhstsin[NSOIL], cin[NSOIL];
    for (int k = 0; k < NSOIL; ++k) rhstsin[k] = rhsts[k];
    for (int k = 0; k < NSOIL; ++k) cin[k] = ci[k];
    nam_rosr12(ci, ai, bi, cin, rhstsin, rhsts);            // CALL ROSR12(CI,AI,BI,CIin,RHSTSin,RHSTS)
    for (int k = 0; k < NSOIL; ++k) stcout[k] = stcin[k] + ci[k];
}

// ----------------------------------------------------------- HRTICE (module 3038-3150)
__device__ static void nam_hrtice(float* rhsts, const float* stc, const float* zsoil,
                                  float yy, float zz1, float df1,
                                  float* ai, float* bi, float* ci)
{
    const float hcpct = 1.72396e6f;
    const float tbot = 271.16f;
    const float zbot = zsoil[NSOIL - 1];
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    float dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    const float ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    rhsts[0] = (df1 * dtsdz - ssoil) / (zsoil[0] * hcpct);
    float ddz2 = 0.0f;
    for (int k = 2; k <= NSOIL; ++k) {
        const int k0 = k - 1;
        float dtsdz2, denom;
        if (k != NSOIL) {
            denom = 0.5f * (zsoil[k0 - 1] - zsoil[k0 + 1]);
            dtsdz2 = (stc[k0] - stc[k0 + 1]) / denom;
            ddz2 = 2.0f / (zsoil[k0 - 1] - zsoil[k0 + 1]);
            ci[k0] = -df1 * ddz2 / ((zsoil[k0 - 1] - zsoil[k0]) * hcpct);
        } else {
            dtsdz2 = (stc[k0] - tbot) / (0.5f * (zsoil[k0 - 1] + zsoil[k0]) - zbot);
            ci[k0] = 0.0f;
        }
        denom = (zsoil[k0] - zsoil[k0 - 1]) * hcpct;
        rhsts[k0] = (df1 * dtsdz2 - df1 * dtsdz) / denom;
        ai[k0] = -df1 * ddz / ((zsoil[k0 - 1] - zsoil[k0]) * hcpct);
        bi[k0] = -(ai[k0] + ci[k0]);
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ------------------------------------------------------------ HRT (module 2785-3035)
// Land-mass soil heat RHS and tri-diagonal coefficients. ITAVG is .TRUE. in the Fortran (constant),
// so the temperature-averaged branches are the executed ones; the ITAVG=.FALSE. branches are kept.
__device__ static void nam_hrt(float* rhsts, const float* stc, const float* smc, float smcmax,
                               const float* zsoil, float yy, float zz1, float tbot, float zbot,
                               float psisat, float* sh2o, float dt, float bexp, float f1,
                               float df1, float quartz, float csoil, float* ai, float* bi,
                               float* ci, int vegtyp, int ivegsrc)
{
    (void)f1;                                               // passed, never read in HRT
    const float cair = 1004.0f, cice = 2.106e6f, ch2o = 4.2e6f, t0 = 273.15f;
    const bool itavg = true;
    const int iw = (ivegsrc == 1) ? 13 : 1;
    const float csoil_loc = (vegtyp == iw) ? 3.0e6f : csoil;

    float hcpct, ddz, ddz2, denom, df1n, df1k, dtsdz, dtsdz2, qtot, sice, tavg, tbk, tbk1, tsnsr, tsurf;
    float ssoil;

    // ---- top soil layer
    hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc + (smcmax - smc[0]) * cair
          + (smc[0] - sh2o[0]) * cice;
    ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    qtot = -1.0f * rhsts[0] * denom;
    tbk = 0.0f;
    tbk1 = 0.0f;
    sice = smc[0] - sh2o[0];
    if (itavg) {
        tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
        tbk = nam_tbnd(stc[0], stc[1], zsoil, zbot, 1);
        if ((sice > 0.0f) || (stc[0] < t0) || (tsurf < t0) || (tbk < t0)) {
            tavg = nam_tmpavg(tsurf, stc[0], tbk, zsoil, 1);
            tsnsr = nam_snksrc(tavg, smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 1, qtot);
            rhsts[0] = rhsts[0] - tsnsr / denom;
        }
    } else {
        if ((sice > 0.0f) || (stc[0] < t0)) {
            tsnsr = nam_snksrc(stc[0], smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 1, qtot);
            rhsts[0] = rhsts[0] - tsnsr / denom;
        }
    }

    // ---- remaining layers
    ddz2 = 0.0f;
    df1k = df1;
    for (int k = 2; k <= NSOIL; ++k) {
        const int k0 = k - 1;
        hcpct = sh2o[k0] * ch2o + (1.0f - smcmax) * csoil_loc + (smcmax - smc[k0]) * cair
              + (smc[k0] - sh2o[k0]) * cice;
        if (k != NSOIL) {
            df1n = nam_tdfcnd(smc[k0], quartz, smcmax, sh2o[k0]);
            if (vegtyp == iw) df1n = 3.24f;
            denom = 0.5f * (zsoil[k0 - 1] - zsoil[k0 + 1]);
            dtsdz2 = (stc[k0] - stc[k0 + 1]) / denom;
            ddz2 = 2.0f / (zsoil[k0 - 1] - zsoil[k0 + 1]);
            ci[k0] = -df1n * ddz2 / ((zsoil[k0 - 1] - zsoil[k0]) * hcpct);
            if (itavg) tbk1 = nam_tbnd(stc[k0], stc[k0 + 1], zsoil, zbot, k);
        } else {
            df1n = nam_tdfcnd(smc[k0], quartz, smcmax, sh2o[k0]);
            if (vegtyp == iw) df1n = 3.24f;
            denom = 0.5f * (zsoil[k0 - 1] + zsoil[k0]) - zbot;
            dtsdz2 = (stc[k0] - tbot) / denom;
            ci[k0] = 0.0f;
            if (itavg) tbk1 = nam_tbnd(stc[k0], tbot, zsoil, zbot, k);
        }
        denom = (zsoil[k0] - zsoil[k0 - 1]) * hcpct;
        rhsts[k0] = (df1n * dtsdz2 - df1k * dtsdz) / denom;
        qtot = -1.0f * denom * rhsts[k0];
        sice = smc[k0] - sh2o[k0];
        if (itavg) {
            tavg = nam_tmpavg(tbk, stc[k0], tbk1, zsoil, k);
            if ((sice > 0.0f) || (stc[k0] < t0) || (tbk < t0) || (tbk1 < t0)) {
                tsnsr = nam_snksrc(tavg, smc[k0], sh2o[k0], zsoil, smcmax, psisat, bexp, dt, k, qtot);
                rhsts[k0] = rhsts[k0] - tsnsr / denom;
            }
        } else {
            if ((sice > 0.0f) || (stc[k0] < t0)) {
                tsnsr = nam_snksrc(stc[k0], smc[k0], sh2o[k0], zsoil, smcmax, psisat, bexp, dt, k, qtot);
                rhsts[k0] = rhsts[k0] - tsnsr / denom;
            }
        }
        ai[k0] = -df1k * ddz / ((zsoil[k0 - 1] - zsoil[k0]) * hcpct);
        bi[k0] = -(ai[k0] + ci[k0]);
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ----------------------------------------------------------- SHFLX (module 3729-3796)
// One column: the HRT/HRTICE + HSTEP soil temperature update, then T1 and SSOIL.
__device__ static void nam_shflx(float* stc, const float* smc, float* sh2o, float smcmax,
                                 float& t1, float& ssoil, const float* zsoil, float yy,
                                 float zz1, float tbot, float zbot, float psisat, float bexp,
                                 float f1, float df1, float quartz, float csoil, float dt,
                                 int ice, int vegtyp, int ivegsrc)
{
    float rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], stcf[NSOIL];
    if (ice == 1) {
        nam_hrtice(rhsts, stc, zsoil, yy, zz1, df1, ai, bi, ci);
    } else {
        nam_hrt(rhsts, stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt, bexp,
                f1, df1, quartz, csoil, ai, bi, ci, vegtyp, ivegsrc);
    }
    nam_hstep(stcf, stc, rhsts, dt, ai, bi, ci);
    for (int i = 0; i < NSOIL; ++i) stc[i] = stcf[i];
    t1 = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    ssoil = df1 * (stc[0] - t1) / (0.5f * zsoil[0]);
}

// ---------------------------------------------------------------- kernel
extern "C" __global__ void phys_noah_shflx(
    float* __restrict__ stc,            // STC    soil temperature        [NSOIL, JM, IM] inout
    const float* __restrict__ smc,      // SMC    total soil moisture     [NSOIL, JM, IM]
    float* __restrict__ sh2o,           // SH2O   liquid soil moisture    [NSOIL, JM, IM] inout
    const float* __restrict__ zsoil,    // ZSOIL  layer depths, negative  [NSOIL]
    const float* __restrict__ smcmax,   // SMCMAX                          [JM, IM]
    float* __restrict__ t1,             // T1     skin/ground temp        [JM, IM] inout
    float* __restrict__ ssoil,          // SSOIL  soil heat flux          [JM, IM] out
    const float* __restrict__ yy,       // YY                              [JM, IM]
    const float* __restrict__ zz1,      // ZZ1                             [JM, IM]
    const float* __restrict__ tbot,     // TBOT                            [JM, IM]
    const float* __restrict__ zbot,     // ZBOT                            [JM, IM]
    const float* __restrict__ psisat,   // PSISAT                          [JM, IM]
    const float* __restrict__ bexp,     // BEXP                            [JM, IM]
    const float* __restrict__ f1,       // F1                              [JM, IM]
    const float* __restrict__ df1,      // DF1    thermal diffusivity      [JM, IM]
    const float* __restrict__ quartz,   // QUARTZ                          [JM, IM]
    const float* __restrict__ csoil,    // CSOIL                           [JM, IM]
    const int* __restrict__ ice,        // ICE    sea-ice flag             [JM, IM]
    const int* __restrict__ vegtyp,     // VEGTYP                          [JM, IM]
    float dt,                           // DT     time step (s)
    int ivegsrc,                        // IVEGSRC (1 = IGBP)
    int im, int jm,
    int i_lo, int i_hi, int j_lo, int j_hi)   // interior, 0-based inclusive (wrapper _interior)
{
    POINT_IJL
    (void)l0;
    // NOAHLSM is called with ITS_B1..ITE_B1, JTS_B1..JTE_B1 (module_TURBULENCE.F90:899-900), so
    // boundary rows and columns are never updated.
    if (i0 < i_lo || i0 > i_hi || j0 < j_lo || j0 > j_hi) return;
    const size_t n = IDX2(i0, j0, im);
    const size_t plane = (size_t)jm * (size_t)im;

    float stc_c[NSOIL], smc_c[NSOIL], sh2o_c[NSOIL], zs_c[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        stc_c[k] = stc[(size_t)k * plane + n];
        smc_c[k] = smc[(size_t)k * plane + n];
        sh2o_c[k] = sh2o[(size_t)k * plane + n];
        zs_c[k] = zsoil[k];
    }

    float t1_c = t1[n];
    float ssoil_c = 0.0f;
    nam_shflx(stc_c, smc_c, sh2o_c, smcmax[n], t1_c, ssoil_c, zs_c, yy[n], zz1[n], tbot[n],
              zbot[n], psisat[n], bexp[n], f1[n], df1[n], quartz[n], csoil[n], dt, ice[n],
              vegtyp[n], ivegsrc);

    for (int k = 0; k < NSOIL; ++k) {
        stc[(size_t)k * plane + n] = stc_c[k];
        sh2o[(size_t)k * plane + n] = sh2o_c[k];
    }
    t1[n] = t1_c;
    ssoil[n] = ssoil_c;
}
