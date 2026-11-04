// phys_noah_nopac.cu -- NOPAC: no-snow energy and water balance (Noah LSM), WOOF-NAM.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:3204-3387 (nam.v4.2.11, 83df4e13c1).
//
// One thread per column (i, j), one launch, no host sync. The body is the Fortran NOPAC in order.
// Its callee closure is transcribed from the NAM Fortran, not from ArWen's WRF-lineage noah.cu
// (ArWen's TDFCND, DEVAP and FRH2O differ from NAM's):
//   EVAPO 2557, DEVAP 2514, TRANSP 5200, SMFLX 3799, SRT 4603, SSTEP 4855, ROSR12 3666,
//   SHFLX 3729, HRT 2785, HRTICE 3038, HSTEP 3153, TBND 4945, TMPAVG 5095, SNKSRC 3960,
//   FRH2O 2644, TDFCND 4989, WDFCND 5312.
// Those callees are separate units with their own files; the closure is duplicated here so this
// unit is self-contained until a shared device-function header exists. Helpers keep the Fortran
// K (1-based) in their arguments where the Fortran tests K, and index arrays as x[K-1].
//
// Layout: colf [NCOLF, JM, IM] per-column scalars (inout: CMC, T1, BETA), coli [2, JM, IM] int32
// (NROOT, VEGTYP), smc / sh2o / stc / zsoil / rtdis / et [NSOIL, JM, IM] (layer 1 at the top),
// out [NOUT, JM, IM]. Fortran REAL = float (kfpt=single). SIGMA = 5.67E-8 (noahlsm module
// PARAMETER, module_LS_NOAHLSM.F90:34). NSOIL = 4.

#include "nmmb_common.cuh"

#define NS 4
#define NOPAC_SIGMA 5.67E-8f
#define NOPAC_T0 273.15f

// colf per-column scalar slots (NOPAC arguments; inout ones written back)
#define CF_ETP      0
#define CF_PRCP     1
#define CF_SMCMAX   2
#define CF_SMCWLT   3
#define CF_SMCREF   4
#define CF_SMCDRY   5
#define CF_CMC      6   // inout
#define CF_CMCMAX   7
#define CF_SHDFAC   8
#define CF_SBETA    9
#define CF_Q2       10
#define CF_T1       11  // inout
#define CF_SFCTMP   12
#define CF_T24      13
#define CF_TH2      14
#define CF_FDOWN    15
#define CF_F1       16
#define CF_EMISSI   17
#define CF_EPSCA    18
#define CF_BEXP     19
#define CF_PC       20
#define CF_RCH      21
#define CF_RR       22
#define CF_CFACTR   23
#define CF_SLOPE    24
#define CF_KDT      25
#define CF_FRZFACT  26
#define CF_PSISAT   27
#define CF_DKSAT    28
#define CF_DWSAT    29
#define CF_TBOT     30
#define CF_ZBOT     31
#define CF_QUARTZ   32
#define CF_FXEXP    33
#define CF_CSOIL    34
#define CF_BETA     35  // inout
#define NCOLF       36

// out slots (INTENT(OUT) of NOPAC)
#define OUT_ETA     0
#define OUT_EDIR    1
#define OUT_EC      2
#define OUT_ETT     3
#define OUT_DEW     4
#define OUT_DRIP    5
#define OUT_RUNOFF1 6
#define OUT_RUNOFF2 7
#define OUT_RUNOFF3 8
#define OUT_SSOIL   9
#define OUT_FLX1    10
#define OUT_FLX3    11
#define NOUT        12

// ---------------------------------------------------------------- WDFCND (5312-5369)
__device__ static void nam_wdfcnd(float smc, float smcmax, float bexp, float dksat,
                                  float dwsat, float sicemax, float& wdf, float& wcnd)
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

// ---------------------------------------------------------------- TDFCND (4989-5092)
__device__ static float nam_tdfcnd(float smc, float qz, float smcmax, float sh2o)
{
    const float thkice = 2.2f, thkw = 0.57f, thko = 2.0f, thkqtz = 7.7f;
    float satratio = smc / smcmax;
    float thks = powf(thkqtz, qz) * powf(thko, 1.0f - qz);
    float xunfroz = sh2o / smc;
    float xu = xunfroz * smcmax;
    float thksat = powf(thks, 1.0f - smcmax) * powf(thkice, smcmax - xu) * powf(thkw, xu);
    float gammd = (1.0f - smcmax) * 2700.0f;
    float thkdry = (0.135f * gammd + 64.7f) / (2700.0f - 0.947f * gammd);
    float ake;
    if ((sh2o + 0.0005f) < smc) {
        ake = satratio;
    } else {
        if (satratio > 0.1f) ake = log10f(satratio) + 1.0f;
        else ake = 0.0f;
    }
    return ake * (thksat - thkdry) + thkdry;
}

// ---------------------------------------------------------------- FRH2O (2644-2782)
__device__ static float nam_frh2o(float tkelv, float smc, float sh2o, float smcmax,
                                  float bexp, float psis)
{
    const float ck = 8.0f, blim = 5.5f, error = 0.005f, hlice = 3.335E5f, gs = 9.81f;
    float bx = bexp;
    if (bexp > blim) bx = blim;
    int nlog = 0, kcount = 0;
    if (tkelv > (NOPAC_T0 - 1.0E-3f)) return smc;
    float swl = smc - sh2o;
    if (swl > (smc - 0.02f)) swl = smc - 0.02f;
    if (swl < 0.0f) swl = 0.0f;
    while (nlog < 10 && kcount == 0) {
        nlog = nlog + 1;
        float df = logf((psis * gs / hlice) * powf(1.0f + ck * swl, 2.0f)
                        * powf(smcmax / (smc - swl), bx))
                   - logf(-(tkelv - NOPAC_T0) / tkelv);
        float denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
        float swlk = swl - df / denom;
        if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
        if (swlk < 0.0f) swlk = 0.0f;
        float dswl = fabsf(swlk - swl);
        swl = swlk;
        if (dswl <= error) kcount = kcount + 1;
    }
    float freev = smc - swl;
    if (kcount == 0) {
        float fk = powf((hlice / (gs * (-psis))) * ((tkelv - NOPAC_T0) / tkelv), -1.0f / bx) * smcmax;
        if (fk < 0.02f) fk = 0.02f;
        freev = fminf(fk, smc);
    }
    return freev;
}

// ---------------------------------------------------------------- TBND (4945-4985)
__device__ static float nam_tbnd(float tu, float tb, const float* zsoil, float zbot, int k, int nsoil)
{
    float zup = (k == 1) ? 0.0f : zsoil[k - 2];
    float zb = (k == nsoil) ? (2.0f * zbot - zsoil[k - 1]) : zsoil[k];
    return tu + (tb - tu) * (zup - zsoil[k - 1]) / (zup - zb);
}

// ---------------------------------------------------------------- TMPAVG (5095-5197)
__device__ static float nam_tmpavg(float tup, float tm, float tdn, const float* zsoil, int k)
{
    const float t0 = 2.7315E2f;
    float dz = (k == 1) ? -zsoil[0] : zsoil[k - 2] - zsoil[k - 1];
    float dzh = dz * 0.5f;
    float tavg;
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                tavg = (tup + 2.0f * tm + tdn) / 4.0f;
            } else {
                float x0 = (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (tup * dzh + tm * (dzh + x0) + t0 * (2.0f * dzh - x0)) / dz;
            }
        } else {
            if (tdn < t0) {
                float xup = (t0 - tup) * dzh / (tm - tup);
                float xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (tup * xup + t0 * (2.0f * dz - xup - xdn) + tdn * xdn) / dz;
            } else {
                float xup = (t0 - tup) * dzh / (tm - tup);
                tavg = 0.5f * (tup * xup + t0 * (2.0f * dz - xup)) / dz;
            }
        }
    } else {
        if (tm < t0) {
            if (tdn < t0) {
                float xup = dzh - (t0 - tup) * dzh / (tm - tup);
                tavg = 0.5f * (t0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh) / dz;
            } else {
                float xup = dzh - (t0 - tup) * dzh / (tm - tup);
                float xdn = (t0 - tm) * dzh / (tdn - tm);
                tavg = 0.5f * (t0 * (2.0f * dz - xup - xdn) + tm * (xup + xdn)) / dz;
            }
        } else {
            if (tdn < t0) {
                float xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
                tavg = (t0 * (dz - xdn) + 0.5f * (t0 + tdn) * xdn) / dz;
            } else {
                tavg = (tup + 2.0f * tm + tdn) / 4.0f;
            }
        }
    }
    return tavg;
}

// ---------------------------------------------------------------- SNKSRC (3960-4047)
__device__ static void nam_snksrc(float& tsnsr, float tavg, float smc, float& sh2o,
                                  const float* zsoil, float smcmax, float psisat,
                                  float bexp, float dt, int k, float qtot)
{
    const float dh2o = 1.0000E3f, hlice = 3.3350E5f;
    float dz = (k == 1) ? -zsoil[0] : zsoil[k - 2] - zsoil[k - 1];
    float freev = nam_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    float xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
    if (xh2o < sh2o && xh2o < freev) {
        if (freev > sh2o) xh2o = sh2o;
        else xh2o = freev;
    }
    if (xh2o > sh2o && xh2o > freev) {
        if (freev < sh2o) xh2o = sh2o;
        else xh2o = freev;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc) xh2o = smc;
    tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
}

// ---------------------------------------------------------------- ROSR12 (3666-3725)
// P, C, DELTA as in the Fortran (P and DELTA outputs, C modified: C(NSOIL) = 0).
__device__ static void nam_rosr12(float* p, const float* a, const float* b, float* c,
                                  const float* d, float* delta, int n)
{
    c[n - 1] = 0.0f;
    p[0] = -c[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < n; ++k) {
        float rden = 1.0f / (b[k] + a[k] * p[k - 1]);
        p[k] = -c[k] * rden;
        delta[k] = (d[k] - a[k] * delta[k - 1]) * rden;
    }
    p[n - 1] = delta[n - 1];
    for (int kk = n - 2; kk >= 0; --kk) {
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

// ---------------------------------------------------------------- SRT (4603-4852)
__device__ static void nam_srt(float* rhstt, float edir, const float* et, const float* sh2o,
                               const float* sh2oa, int nsoil, float pcpdrp, const float* zsoil,
                               float dwsat, float dksat, float smcmax, float bexp,
                               float& runoff1, float& runoff2, float dt, float smcwlt,
                               float slope, float kdt, float frzx, const float* sice,
                               float* ai, float* bi, float* ci)
{
    const int cvfrz = 3;
    float sicemax = 0.0f;
    for (int ks = 0; ks < nsoil; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    float pddum = pcpdrp;
    float wdf, wcnd;
    float dmax[NS];
    runoff1 = 0.0f;
    if (pcpdrp != 0.0f) {
        float dt1 = dt / 86400.0f;
        float smcav = smcmax - smcwlt;
        dmax[0] = -zsoil[0] * smcav;
        float dice = -zsoil[0] * sice[0];
        dmax[0] = dmax[0] * (1.0f - (sh2oa[0] + sice[0] - smcwlt) / smcav);
        float dd = dmax[0];
        for (int ks = 1; ks < nsoil; ++ks) {
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks];
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav;
            dmax[ks] = dmax[ks] * (1.0f - (sh2oa[ks] + sice[ks] - smcwlt) / smcav);
            dd = dd + dmax[ks];
        }
        float val = (1.0f - expf(-kdt * dt1));
        float ddt = dd * val;
        float px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;
        float infmax = (px * (ddt / (px + ddt))) / dt;
        float fcr = 1.0f;
        if (dice > 1.0E-2f) {
            float acrt = cvfrz * frzx / dice;
            // SUM = 1 + ACRT**2/2 + ACRT**1/1 (the Fortran J loop with IALP1 = CVFRZ-1 = 2)
            float sum = 1.0f;
            sum = sum + (acrt * acrt) / 2.0f;
            sum = sum + acrt / 1.0f;
            fcr = 1.0f - expf(-acrt) * sum;
        }
        infmax = infmax * fcr;
        float mxsmc = sh2oa[0];
        nam_wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd);
        infmax = fmaxf(infmax, wcnd);
        infmax = fminf(infmax, px);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }
    float mxsmc = sh2oa[0];
    nam_wdfcnd(mxsmc, smcmax, bexp, dksat, dwsat, sicemax, wdf, wcnd);
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    float dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];
    float ddz2 = 0.0f;
    for (int k = 2; k <= nsoil; ++k) {
        float denom2 = (zsoil[k - 2] - zsoil[k - 1]);
        float slopx, wdf2, wcnd2, dsmdz2, denom = 0.0f;
        if (k != nsoil) {
            slopx = 1.0f;
            float mxsmc2 = sh2oa[k - 1];
            nam_wdfcnd(mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, wdf2, wcnd2);
            denom = (zsoil[k - 2] - zsoil[k]);
            dsmdz2 = (sh2o[k - 1] - sh2o[k]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k - 1] = -wdf2 * ddz2 / denom2;
        } else {
            slopx = slope;
            nam_wdfcnd(sh2oa[nsoil - 1], smcmax, bexp, dksat, dwsat, sicemax, wdf2, wcnd2);
            dsmdz2 = 0.0f;
            ci[k - 1] = 0.0f;
        }
        float numer = (wdf2 * dsmdz2) + slopx * wcnd2 - (wdf * dsmdz) - wcnd + et[k - 1];
        rhstt[k - 1] = numer / (-denom2);
        ai[k - 1] = -wdf * ddz / denom2;
        bi[k - 1] = -(ai[k - 1] + ci[k - 1]);
        if (k == nsoil) runoff2 = slopx * wcnd2;
        if (k != nsoil) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// ---------------------------------------------------------------- SSTEP (4855-4942)
// SH2OOUT may alias SH2OIN (the Fortran final call passes SH2O twice); each K reads SH2OIN(K)
// before it writes SH2OOUT(K) and no other K is read, so the alias is safe.
__device__ static void nam_sstep(float* sh2oout, const float* sh2oin, float& cmc, float* rhstt,
                                 float rhsct, float dt, int nsoil, float smcmax, float cmcmax,
                                 float smcwlt, float& runoff3, const float* zsoil, float* smc,
                                 const float* sice, float* ai, float* bi, float* ci)
{
    float rhsttin[NS], ciin[NS];
    for (int k = 0; k < nsoil; ++k) {
        rhstt[k] = rhstt[k] * dt;
        ai[k] = ai[k] * dt;
        bi[k] = 1.0f + bi[k] * dt;
        ci[k] = ci[k] * dt;
    }
    for (int k = 0; k < nsoil; ++k) rhsttin[k] = rhstt[k];
    for (int k = 0; k < nsoil; ++k) ciin[k] = ci[k];
    nam_rosr12(ci, ai, bi, ciin, rhsttin, rhstt, nsoil);
    float wplus = 0.0f;
    runoff3 = 0.0f;
    float ddz = -zsoil[0];
    for (int k = 0; k < nsoil; ++k) {
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
        sh2oout[k] = fmaxf((smc[k] - sice[k]), 0.0f);
    }
    runoff3 = wplus;
    cmc = cmc + dt * rhsct;
    if (cmc < 1.0E-20f) cmc = 0.0f;
    cmc = fminf(cmc, cmcmax);
}

// ---------------------------------------------------------------- SMFLX (3799-3916)
__device__ static void nam_smflx(float* smc, int nsoil, float& cmc, float dt, float prcp1,
                                 const float* zsoil, float* sh2o, float slope, float kdt,
                                 float frzfactor, float smcmax, float bexp, float smcwlt,
                                 float dksat, float dwsat, float shdfac, float cmcmax,
                                 float& runoff1, float& runoff2, float& runoff3,
                                 float edir, float ec, const float* et, float& drip)
{
    float dummy = 0.0f;
    float rhsct = shdfac * prcp1 - ec;
    drip = 0.0f;
    float trhsct = dt * rhsct;
    float excess = cmc + trhsct;
    if (excess > cmcmax) drip = excess - cmcmax;
    float pcpdrp = (1.0f - shdfac) * prcp1 + drip / dt;
    float sice[NS], sh2oa[NS], sh2ofg[NS], rhstt[NS], ai[NS], bi[NS], ci[NS];
    for (int k = 0; k < nsoil; ++k) sice[k] = smc[k] - sh2o[k];
    if ((pcpdrp * dt) > (0.001f * 1000.0f * (-zsoil[0]) * smcmax)) {
        nam_srt(rhstt, edir, et, sh2o, sh2o, nsoil, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp,
                runoff1, runoff2, dt, smcwlt, slope, kdt, frzfactor, sice, ai, bi, ci);
        nam_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, nsoil, smcmax, cmcmax, smcwlt,
                  runoff3, zsoil, smc, sice, ai, bi, ci);
        for (int k = 0; k < nsoil; ++k) sh2oa[k] = (sh2o[k] + sh2ofg[k]) * 0.5f;
        nam_srt(rhstt, edir, et, sh2o, sh2oa, nsoil, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp,
                runoff1, runoff2, dt, smcwlt, slope, kdt, frzfactor, sice, ai, bi, ci);
        nam_sstep(sh2o, sh2o, cmc, rhstt, rhsct, dt, nsoil, smcmax, cmcmax, smcwlt,
                  runoff3, zsoil, smc, sice, ai, bi, ci);
    } else {
        nam_srt(rhstt, edir, et, sh2o, sh2o, nsoil, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp,
                runoff1, runoff2, dt, smcwlt, slope, kdt, frzfactor, sice, ai, bi, ci);
        nam_sstep(sh2o, sh2o, cmc, rhstt, rhsct, dt, nsoil, smcmax, cmcmax, smcwlt,
                  runoff3, zsoil, smc, sice, ai, bi, ci);
    }
}

// ---------------------------------------------------------------- DEVAP (2514-2554)
// NAM DEVAP takes SH2O(1) for the soil-moisture ratio (ArWen's noah.cu uses SMC).
__device__ static float nam_devap(float etp1, float sh2o1, float shdfac, float smcmax,
                                  float smcdry, float fxexp)
{
    float sratio = (sh2o1 - smcdry) / (smcmax - smcdry);
    float fx;
    if (sratio > 0.0f) {
        fx = powf(sratio, fxexp);
        fx = fmaxf(fminf(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return fx * (1.0f - shdfac) * etp1;
}

// ---------------------------------------------------------------- TRANSP (5200-5309)
// Fortran TRANSP(ET,NSOIL,ETP1,SMC,CMC,...): EVAPO passes SH2O as its SMC.
__device__ static void nam_transp(float* et, int nsoil, float etp1, const float* smc, float cmc,
                                  float shdfac, float smcwlt, float cmcmax, float pc,
                                  float cfactr, float smcref, int nroot, const float* rtdis)
{
    for (int k = 0; k < nsoil; ++k) et[k] = 0.0f;
    float etp1a;
    if (cmc != 0.0f) etp1a = shdfac * pc * etp1 * (1.0f - powf(cmc / cmcmax, cfactr));
    else etp1a = shdfac * pc * etp1;
    float gx[NS];
    float sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (smc[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = fmaxf(fminf(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (float)nroot;
    float denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        float rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * fmaxf(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

// ---------------------------------------------------------------- EVAPO (2557-2641)
__device__ static void nam_evapo(float& eta1, const float* sh2o, float cmc, float etp1, float dt,
                                 float smcmax, float bexp, float pc, float smcwlt, float smcref,
                                 float shdfac, float cmcmax, float smcdry, float cfactr,
                                 float fxexp, int nroot, const float* rtdis, float& edir,
                                 float& ec, float* et, float& ett, int nsoil)
{
    edir = 0.0f;
    ec = 0.0f;
    for (int k = 0; k < nsoil; ++k) et[k] = 0.0f;
    ett = 0.0f;
    if (etp1 > 0.0f) {
        if (shdfac < 1.0f)
            edir = nam_devap(etp1, sh2o[0], shdfac, smcmax, smcdry, fxexp);
        if (shdfac > 0.0f) {
            nam_transp(et, nsoil, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax, pc, cfactr, smcref,
                       nroot, rtdis);
            for (int k = 0; k < nsoil; ++k) ett = ett + et[k];
            if (cmc > 0.0f) ec = shdfac * powf(cmc / cmcmax, cfactr) * etp1;
            else ec = 0.0f;
            float cmc2ms = cmc / dt;
            ec = fminf(cmc2ms, ec);
        }
    }
    eta1 = edir + ett + ec;
}

// ---------------------------------------------------------------- HSTEP (3153-3201)
__device__ static void nam_hstep(float* stcout, const float* stcin, float* rhsts, float dt,
                                 int nsoil, float* ai, float* bi, float* ci)
{
    for (int k = 0; k < nsoil; ++k) {
        rhsts[k] = rhsts[k] * dt;
        ai[k] = ai[k] * dt;
        bi[k] = 1.0f + bi[k] * dt;
        ci[k] = ci[k] * dt;
    }
    float rhstsin[NS], ciin[NS];
    for (int k = 0; k < nsoil; ++k) rhstsin[k] = rhsts[k];
    for (int k = 0; k < nsoil; ++k) ciin[k] = ci[k];
    nam_rosr12(ci, ai, bi, ciin, rhstsin, rhsts, nsoil);
    for (int k = 0; k < nsoil; ++k) stcout[k] = stcin[k] + ci[k];
}

// ---------------------------------------------------------------- HRTICE (3038-3150)
__device__ static void nam_hrtice(float* rhsts, const float* stc, int nsoil, const float* zsoil,
                                  float yy, float zz1, float df1, float* ai, float* bi, float* ci)
{
    const float hcpct = 1.72396E+6f;
    const float tbot = 271.16f;
    float zbot = zsoil[nsoil - 1];
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    float dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    float ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    rhsts[0] = (df1 * dtsdz - ssoil) / (zsoil[0] * hcpct);
    float ddz2 = 0.0f;
    for (int k = 2; k <= nsoil; ++k) {
        float dtsdz2;
        if (k != nsoil) {
            float denom = 0.5f * (zsoil[k - 2] - zsoil[k]);
            dtsdz2 = (stc[k - 1] - stc[k]) / denom;
            ddz2 = 2.0f / (zsoil[k - 2] - zsoil[k]);
            ci[k - 1] = -df1 * ddz2 / ((zsoil[k - 2] - zsoil[k - 1]) * hcpct);
        } else {
            dtsdz2 = (stc[k - 1] - tbot) / (0.5f * (zsoil[k - 2] + zsoil[k - 1]) - zbot);
            ci[k - 1] = 0.0f;
        }
        float denom = (zsoil[k - 1] - zsoil[k - 2]) * hcpct;
        rhsts[k - 1] = (df1 * dtsdz2 - df1 * dtsdz) / denom;
        ai[k - 1] = -df1 * ddz / ((zsoil[k - 2] - zsoil[k - 1]) * hcpct);
        bi[k - 1] = -(ai[k - 1] + ci[k - 1]);
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ---------------------------------------------------------------- HRT (2785-3035)
// ITAVG = .TRUE. in NAM: the temperature-averaged branch is the only one run.
__device__ static void nam_hrt(float* rhsts, const float* stc, const float* smc, float smcmax,
                               int nsoil, const float* zsoil, float yy, float zz1, float tbot,
                               float zbot, float psisat, float* sh2o, float dt, float bexp,
                               float df1, float quartz, float csoil, float* ai, float* bi,
                               float* ci, int vegtyp, int ivegsrc)
{
    const float cair = 1004.0f, cice = 2.106E6f, ch2o = 4.2E6f;
    int iw = 1;
    if (ivegsrc == 1) iw = 13;
    float csoil_loc = (vegtyp == iw) ? 3.0E6f : csoil;
    float hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc + (smcmax - smc[0]) * cair
                  + (smc[0] - sh2o[0]) * cice;
    float ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    float dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    float ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    float denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    float qtot = -1.0f * rhsts[0] * denom;
    float sice = smc[0] - sh2o[0];
    float tsnsr, tavg;
    float tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    float tbk = nam_tbnd(stc[0], stc[1], zsoil, zbot, 1, nsoil);
    float tbk1 = 0.0f;
    if ((sice > 0.0f) || (stc[0] < NOPAC_T0) || (tsurf < NOPAC_T0) || (tbk < NOPAC_T0)) {
        tavg = nam_tmpavg(tsurf, stc[0], tbk, zsoil, 1);
        nam_snksrc(tsnsr, tavg, smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 1, qtot);
        rhsts[0] = rhsts[0] - tsnsr / denom;
    }
    float ddz2 = 0.0f;
    float df1k = df1;
    for (int k = 2; k <= nsoil; ++k) {
        float hc = sh2o[k - 1] * ch2o + (1.0f - smcmax) * csoil_loc + (smcmax - smc[k - 1]) * cair
                   + (smc[k - 1] - sh2o[k - 1]) * cice;
        float df1n, dtsdz2, dn;
        if (k != nsoil) {
            df1n = nam_tdfcnd(smc[k - 1], quartz, smcmax, sh2o[k - 1]);
            if (vegtyp == iw) df1n = 3.24f;
            dn = 0.5f * (zsoil[k - 2] - zsoil[k]);
            dtsdz2 = (stc[k - 1] - stc[k]) / dn;
            ddz2 = 2.0f / (zsoil[k - 2] - zsoil[k]);
            ci[k - 1] = -df1n * ddz2 / ((zsoil[k - 2] - zsoil[k - 1]) * hc);
            tbk1 = nam_tbnd(stc[k - 1], stc[k], zsoil, zbot, k, nsoil);
        } else {
            df1n = nam_tdfcnd(smc[k - 1], quartz, smcmax, sh2o[k - 1]);
            if (vegtyp == iw) df1n = 3.24f;
            dn = 0.5f * (zsoil[k - 2] + zsoil[k - 1]) - zbot;
            dtsdz2 = (stc[k - 1] - tbot) / dn;
            ci[k - 1] = 0.0f;
            tbk1 = nam_tbnd(stc[k - 1], tbot, zsoil, zbot, k, nsoil);
        }
        denom = (zsoil[k - 1] - zsoil[k - 2]) * hc;
        rhsts[k - 1] = (df1n * dtsdz2 - df1k * dtsdz) / denom;
        qtot = -1.0f * denom * rhsts[k - 1];
        sice = smc[k - 1] - sh2o[k - 1];
        tavg = nam_tmpavg(tbk, stc[k - 1], tbk1, zsoil, k);
        if ((sice > 0.0f) || (stc[k - 1] < NOPAC_T0) || (tbk < NOPAC_T0) || (tbk1 < NOPAC_T0)) {
            nam_snksrc(tsnsr, tavg, smc[k - 1], sh2o[k - 1], zsoil, smcmax, psisat, bexp, dt, k,
                       qtot);
            rhsts[k - 1] = rhsts[k - 1] - tsnsr / denom;
        }
        ai[k - 1] = -df1k * ddz / ((zsoil[k - 2] - zsoil[k - 1]) * hc);
        bi[k - 1] = -(ai[k - 1] + ci[k - 1]);
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ---------------------------------------------------------------- SHFLX (3729-3796)
__device__ static void nam_shflx(float& ssoil, float* stc, const float* smc, float smcmax,
                                 int nsoil, float& t1, float dt, float yy, float zz1,
                                 const float* zsoil, float tbot, float zbot, float psisat,
                                 float* sh2o, float bexp, float df1, int ice, float quartz,
                                 float csoil, int vegtyp, int ivegsrc)
{
    float ai[NS], bi[NS], ci[NS], rhsts[NS], stcf[NS];
    if (ice == 1) {
        nam_hrtice(rhsts, stc, nsoil, zsoil, yy, zz1, df1, ai, bi, ci);
        nam_hstep(stcf, stc, rhsts, dt, nsoil, ai, bi, ci);
    } else {
        nam_hrt(rhsts, stc, smc, smcmax, nsoil, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt,
                bexp, df1, quartz, csoil, ai, bi, ci, vegtyp, ivegsrc);
        nam_hstep(stcf, stc, rhsts, dt, nsoil, ai, bi, ci);
    }
    for (int k = 0; k < nsoil; ++k) stc[k] = stcf[k];
    t1 = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    ssoil = df1 * (stc[0] - t1) / (0.5f * zsoil[0]);
}

// ---------------------------------------------------------------- NOPAC (3204-3387)
extern "C" __global__ void phys_noah_nopac(
    float* __restrict__ colf,          // [NCOLF, JM, IM] inout (CMC, T1, BETA written back)
    const int* __restrict__ coli,      // [2, JM, IM]: NROOT, VEGTYP
    float* __restrict__ smc,           // [NSOIL, JM, IM] inout
    float* __restrict__ sh2o,          // [NSOIL, JM, IM] inout
    float* __restrict__ stc,           // [NSOIL, JM, IM] inout
    const float* __restrict__ zsoil,   // [NSOIL, JM, IM]
    const float* __restrict__ rtdis,   // [NSOIL, JM, IM]
    float* __restrict__ et,            // [NSOIL, JM, IM] out
    float* __restrict__ out,           // [NOUT, JM, IM] out
    const int im, const int jm, const int ice, const int ivegsrc, const float dt)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jm) return;
    const int i0 = col % im;
    const int j0 = col / im;
    const size_t ns = (size_t)im * (size_t)jm;
    const size_t base = (size_t)j0 * (size_t)im + (size_t)i0;

    float c[NCOLF];
    for (int f = 0; f < NCOLF; ++f) c[f] = colf[(size_t)f * ns + base];
    const int nroot = coli[base];
    const int vegtyp = coli[ns + base];

    float smc_l[NS], sh2o_l[NS], stc_l[NS], zs[NS], rt[NS], et_l[NS], et1[NS];
    for (int k = 0; k < NS; ++k) {
        size_t o = (size_t)k * ns + base;
        smc_l[k] = smc[o];
        sh2o_l[k] = sh2o[o];
        stc_l[k] = stc[o];
        zs[k] = zsoil[o];
        rt[k] = rtdis[o];
        et_l[k] = 0.0f;
        et1[k] = 0.0f;
    }

    float cmc = c[CF_CMC];
    float beta = c[CF_BETA];
    float t1 = c[CF_T1];
    float eta = 0.0f, dew = 0.0f, edir = 0.0f, ec = 0.0f, ett = 0.0f, drip = 0.0f;
    float runoff1 = 0.0f, runoff2 = 0.0f, runoff3 = 0.0f, ssoil = 0.0f;
    const float flx1 = 0.0f, flx3 = 0.0f;
    float prcp1 = c[CF_PRCP] * 0.001f;
    float etp = c[CF_ETP];
    float etp1 = etp * 0.001f;
    float edir1 = 0.0f, ec1 = 0.0f, ett1 = 0.0f, eta1 = 0.0f;

    // ETP and PRCP to m/s; DEW; evaporation terms; SMFLX
    if (etp > 0.0f) {
        nam_evapo(eta1, sh2o_l, cmc, etp1, dt, c[CF_SMCMAX], c[CF_BEXP], c[CF_PC], c[CF_SMCWLT],
                  c[CF_SMCREF], c[CF_SHDFAC], c[CF_CMCMAX], c[CF_SMCDRY], c[CF_CFACTR],
                  c[CF_FXEXP], nroot, rt, edir1, ec1, et1, ett1, NS);
        nam_smflx(smc_l, NS, cmc, dt, prcp1, zs, sh2o_l, c[CF_SLOPE], c[CF_KDT], c[CF_FRZFACT],
                  c[CF_SMCMAX], c[CF_BEXP], c[CF_SMCWLT], c[CF_DKSAT], c[CF_DWSAT], c[CF_SHDFAC],
                  c[CF_CMCMAX], runoff1, runoff2, runoff3, edir1, ec1, et1, drip);
        eta = eta1 * 1000.0f;
    } else {
        dew = -etp1;
        prcp1 = prcp1 + dew;
        nam_smflx(smc_l, NS, cmc, dt, prcp1, zs, sh2o_l, c[CF_SLOPE], c[CF_KDT], c[CF_FRZFACT],
                  c[CF_SMCMAX], c[CF_BEXP], c[CF_SMCWLT], c[CF_DKSAT], c[CF_DWSAT], c[CF_SHDFAC],
                  c[CF_CMCMAX], runoff1, runoff2, runoff3, edir1, ec1, et1, drip);
    }

    // BETA from ETP and ETA
    if (etp < 0.000001f) {
        beta = 1.0f;
        eta = etp;
    } else if (etp == 0.000001f) {
        beta = 0.0f;
    } else {
        beta = eta / etp;
    }

    // evaporation components to kg m-2 s-1
    edir = edir1 * 1000.0f;
    ec = ec1 * 1000.0f;
    for (int k = 0; k < NS; ++k) et_l[k] = et1[k] * 1000.0f;
    ett = ett1 * 1000.0f;

    // top-layer thermal diffusivity, adjusted top-layer temperature, SHFLX
    float df1 = nam_tdfcnd(smc_l[0], c[CF_QUARTZ], c[CF_SMCMAX], sh2o_l[0]);
    if (vegtyp == 1) df1 = 3.24f;
    df1 = df1 * expf(c[CF_SBETA] * c[CF_SHDFAC]);
    float yynum = c[CF_FDOWN] - c[CF_EMISSI] * NOPAC_SIGMA * c[CF_T24];
    float yy = c[CF_SFCTMP] + (yynum / c[CF_RCH] + c[CF_TH2] - c[CF_SFCTMP] - beta * c[CF_EPSCA]) / c[CF_RR];
    float zz1 = df1 / (-0.5f * zs[0] * c[CF_RCH] * c[CF_RR]) + 1.0f;
    nam_shflx(ssoil, stc_l, smc_l, c[CF_SMCMAX], NS, t1, dt, yy, zz1, zs, c[CF_TBOT], c[CF_ZBOT],
              c[CF_PSISAT], sh2o_l, c[CF_BEXP], df1, ice, c[CF_QUARTZ], c[CF_CSOIL], vegtyp,
              ivegsrc);

    // write back
    out[(size_t)OUT_ETA * ns + base] = eta;
    out[(size_t)OUT_EDIR * ns + base] = edir;
    out[(size_t)OUT_EC * ns + base] = ec;
    out[(size_t)OUT_ETT * ns + base] = ett;
    out[(size_t)OUT_DEW * ns + base] = dew;
    out[(size_t)OUT_DRIP * ns + base] = drip;
    out[(size_t)OUT_RUNOFF1 * ns + base] = runoff1;
    out[(size_t)OUT_RUNOFF2 * ns + base] = runoff2;
    out[(size_t)OUT_RUNOFF3 * ns + base] = runoff3;
    out[(size_t)OUT_SSOIL * ns + base] = ssoil;
    out[(size_t)OUT_FLX1 * ns + base] = flx1;
    out[(size_t)OUT_FLX3 * ns + base] = flx3;
    colf[(size_t)CF_CMC * ns + base] = cmc;
    colf[(size_t)CF_T1 * ns + base] = t1;
    colf[(size_t)CF_BETA * ns + base] = beta;
    for (int k = 0; k < NS; ++k) {
        size_t o = (size_t)k * ns + base;
        et[o] = et_l[k];
        smc[o] = smc_l[k];
        sh2o[o] = sh2o_l[k];
        stc[o] = stc_l[k];
    }
}
