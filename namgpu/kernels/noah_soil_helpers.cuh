// noah_soil_helpers.cuh -- shared soil device helpers for the NAM Noah LSM port (nam-gpu).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module (MODULE MODULE_LS_NOAHLSM, module_LS_NOAHLSM.F90) are kept as notices.
// Helpers and Fortran line ranges:
//   nam_frh2o   FRH2O   2644-2782   (unfrozen soil water, Flerchinger iteration)
//   nam_tbnd    TBND    4945-4988   (layer bottom interface temperature)
//   nam_tdfcnd  TDFCND  4989-5017   (NAM form: AKE = SATRATIO when liquid < total moisture)
//   nam_tmpavg  TMPAVG  5095-5199   (layer-mean temperature)
//   nam_snksrc  SNKSRC  3960-4047   (phase-change sink/source, SH2O updated in place)
// This file is the single source for these helpers. Included by phys_noah_hrt.cu,
// phys_noah_tbnd.cu and phys_noah_tmpavg.cu. Do not copy them into a kernel again.
//
// Start point for the HRT-family routines: ArWen core/kernels/noah.cu (Apache-2.0, see
// namgpu/third_party/arwen/LICENSE), which transcribes WRF's Noah LSM. The ArWen Apache-2.0 notice applies
// to the transcribed text.
//
// Float32 throughout (Fortran REAL = float, kfpt = single). Fast numerics (fmad on).
// Layer index k is 0-based (Fortran K - 1). zsoil[k] is ZSOIL(K+1), the layer bottom (m, negative).

#ifndef NAM_NOAH_SOIL_HELPERS_CUH
#define NAM_NOAH_SOIL_HELPERS_CUH

#define NAM_SOIL_NSOIL  4
#define NAM_SOIL_T0      273.15f     // module T0
#define NAM_SOIL_T0_TM   2.7315e2f   // TMPAVG T0 (PARAMETER)
#define NAM_SOIL_DH2O    1.0000e3f   // SNKSRC DH2O
#define NAM_SOIL_HLICE   3.3350e5f   // FRH2O / SNKSRC HLICE
#define NAM_SOIL_GS      9.81f       // FRH2O GS
#define NAM_SOIL_CK      8.0f        // FRH2O CK
#define NAM_SOIL_BLIM    5.5f        // FRH2O BLIM
#define NAM_SOIL_ERROR   0.005f      // FRH2O ERROR

// FRH2O (2644-2782). Returns FREE, the equilibrium unfrozen water content.
// Note on the FK fallback (kcount == 0): the base HLICE/(GS*(-PSIS)) * (TKELV-T0)/TKELV is
// positive for TKELV < T0 (both factors negative), so the power is real for every table PSIS.
__device__ static float nam_frh2o(float tkelv, float smc, float sh2o,
                                  float smcmax, float bexp, float psis)
{
    float bx = bexp;
    if (bexp > NAM_SOIL_BLIM) bx = NAM_SOIL_BLIM;
    int nlog = 0;
    int kcount = 0;
    float freev;
    if (tkelv > (NAM_SOIL_T0 - 1.0e-3f)) {
        freev = smc;
    } else {
        // CK /= 0.0 is always true (CK = 8.0), so the iteration branch is always executed.
        float swl = smc - sh2o;
        if (swl > (smc - 0.02f)) swl = smc - 0.02f;
        if (swl < 0.0f) swl = 0.0f;
        while (nlog < 10 && kcount == 0) {                     // label 1001 / 1002
            nlog = nlog + 1;
            float a = (psis * NAM_SOIL_GS / NAM_SOIL_HLICE)
                      * ((1.0f + NAM_SOIL_CK * swl) * (1.0f + NAM_SOIL_CK * swl))   // **2. -> x*x
                      * powf(smcmax / (smc - swl), bx);
            float df = logf(a) - logf(-(tkelv - NAM_SOIL_T0) / tkelv);
            float denom = 2.0f * NAM_SOIL_CK / (1.0f + NAM_SOIL_CK * swl) + bx / (smc - swl);
            float swlk = swl - df / denom;
            if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
            if (swlk < 0.0f) swlk = 0.0f;
            float dswl = fabsf(swlk - swl);
            swl = swlk;
            if (dswl <= NAM_SOIL_ERROR) kcount = kcount + 1;
        }
        freev = smc - swl;
        if (kcount == 0) {                                     // Flerchinger explicit fallback
            float fk = powf((NAM_SOIL_HLICE / (NAM_SOIL_GS * (-psis)))
                            * ((tkelv - NAM_SOIL_T0) / tkelv), -1.0f / bx) * smcmax;
            if (fk < 0.02f) fk = 0.02f;
            freev = fminf(fk, smc);
        }
    }
    return freev;
}

// TBND (4945-4988). k is 0-based. Returns the temperature at the bottom interface of layer k.
__device__ static float nam_tbnd(float tu, float tb, const float* zsoil, float zbot, int k)
{
    float zup, zb;
    if (k == 0) zup = 0.0f;
    else zup = zsoil[k - 1];
    if (k == NAM_SOIL_NSOIL - 1) zb = 2.0f * zbot - zsoil[k];
    else zb = zsoil[k + 1];
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb);
}

// TDFCND (4989-5017). NAM form (not the WRF opt_thcnd form).
__device__ static float nam_tdfcnd(float smc, float qz, float smcmax, float sh2o)
{
    float satratio = smc / smcmax;
    const float thkice = 2.2f, thkw = 0.57f, thko = 2.0f, thkqtz = 7.7f;
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

// TMPAVG (5095-5199). k is 0-based. tup, tm, tdn are the top, mid and bottom temperatures of layer k.
__device__ static float nam_tmpavg(float tup, float tm, float tdn,
                                   const float* zsoil, int k)
{
    const float t0 = NAM_SOIL_T0_TM;
    float dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    float dzh = dz * 0.5f;
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

// SNKSRC (3960-4047). Updates SH2O (INTENT INOUT) in place and returns TSNSR. k is 0-based.
__device__ static float nam_snksrc(float qtot, float tavg, float smc, float& sh2o,
                                   const float* zsoil, float smcmax, float psisat,
                                   float bexp, float dt, int k)
{
    float dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    float freev = nam_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    float xh2o = sh2o + qtot * dt / (NAM_SOIL_DH2O * NAM_SOIL_HLICE * dz);
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
    float tsnsr = -NAM_SOIL_DH2O * NAM_SOIL_HLICE * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

#endif  // NAM_NOAH_SOIL_HELPERS_CUH
