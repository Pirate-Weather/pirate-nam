// phys_noah_sflx.cuh -- SFLX for one land column (Noah LSM, NCEP/NAM version), called by NOAHLSM
// (phys-noah-driver).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the source
// module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:1550-2278 (SFLX, nam.v4.2.11,
// 83df4e13c1). Integration (Opus integrator, 2026-11-04): this is the NAM SFLX body, written from
// the Fortran, calling the NAM callee ports that the Noah units hold:
//   NOPAC callee closure (EVAPO, DEVAP, TRANSP, SMFLX, SRT, SSTEP, ROSR12, SHFLX, HRT, HRTICE,
//   HSTEP, TBND, TMPAVG, SNKSRC, FRH2O, TDFCND, WDFCND)    phys_noah_nopac.cu (unit phys-noah-nopac)
//   SNOPAC                                                 phys_noah_snopac.cuh (phys-noah-snopac)
//   PENMAN, SNOW_NEW, CSNOW, SNOWZ0, SNFRAC, ALCALC, CANRES, SNOWPACK, REDPRM: the bodies of the unit
//   kernels phys_noah_{penman,snow_new,csnow,snowz0,snfrac,alcalc,canres,snowpack,redprm}.cu, here
//   as device functions with the Fortran argument lists.
// The earlier phys_noah_sflx.cu body (ArWen WRF v4.6.1 lineage, never reconciled to NAM) is retired.
//
// Tables: REDPRM reads the NAM VEGPARM/SOILPARM/GENPARM tables bound by phys-noah-parm-tables
// (ctx.tables["noah_parm"]); the driver hands them in as NoahTbl (device pointers and scalars,
// built in the driver kernel from its arguments).
//
// Definitions where the Fortran leaves a value undefined:
//   - ICE == 1: SFLX skips REDPRM, so the soil/vegetation locals hold whatever the previous call
//     left. Here REDPRM runs for every point (that point's own class values), so they are defined.
//   - SNOWZ0 in this NAM version sets Z0 = Z0BRD (unit phys-noah-snowz0).
#pragma once
#include "nmmb_common.cuh"

// ---- the NOPAC callee closure (NAM). Its EVAPO/SMFLX/SHFLX take other argument lists than the
// SNOPAC prototypes, so they are renamed on include and SNOPAC gets thin adapters below.
#define nam_evapo nopac_evapo
#define nam_smflx nopac_smflx
#define nam_shflx nopac_shflx
#include "phys_noah_nopac.cu"
#undef nam_evapo
#undef nam_smflx
#undef nam_shflx

#include "phys_noah_snopac.cuh"

// SNOPAC prototypes (phys_noah_snopac.cuh) -> the NAM closure. Fortran EVAPO/SMFLX/SHFLX argument
// lists; arguments the Fortran bodies do not read (EVAPO: SMC, ZSOIL, DKSAT, DWSAT, SFCTMP, Q2,
// VEGTYP, STC; SHFLX: SMCWLT, F1) are accepted and unused, as in the Fortran.
__device__ void nam_evapo(float& eta1, float* smc, int nsoil, float& cmc, float etp1, float dt,
                          const float* zsoil, const float* sh2o, float smcmax, float bexp,
                          float pc, float smcwlt, float dksat, float dwsat, float smcref,
                          float shdfac, float cmcmax, float smcdry, float cfactr,
                          float& edir, float& ec, float* et, float& ett, float sfctmp,
                          float q2, int nroot, const float* rtdis, float fxexp, int vegtyp,
                          const float* stc)
{
    (void)smc; (void)zsoil; (void)dksat; (void)dwsat; (void)sfctmp; (void)q2; (void)vegtyp; (void)stc;
    nopac_evapo(eta1, sh2o, cmc, etp1, dt, smcmax, bexp, pc, smcwlt, smcref, shdfac, cmcmax, smcdry,
                cfactr, fxexp, nroot, rtdis, edir, ec, et, ett, nsoil);
}
__device__ void nam_smflx(float* smc, int nsoil, float& cmc, float dt, float prcp1,
                          const float* zsoil, float* sh2o, float slope, float kdt, float frzfact,
                          float smcmax, float bexp, float smcwlt, float dksat, float dwsat,
                          float shdfac, float cmcmax, float& runoff1, float& runoff2,
                          float& runoff3, float& edir, float& ec, float* et, float& drip)
{
    nopac_smflx(smc, nsoil, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzfact, smcmax, bexp, smcwlt,
                dksat, dwsat, shdfac, cmcmax, runoff1, runoff2, runoff3, edir, ec, et, drip);
}
__device__ void nam_shflx(float& ssoil, float* stc, float* smc, float smcmax, int nsoil,
                          float& t1, float dt, float yy, float zz1, const float* zsoil,
                          float tbot, float zbot, float smcwlt, float psisat, float* sh2o,
                          float bexp, float f1, float df1, int ice, float quartz, float csoil,
                          int vegtyp, int ivegsrc)
{
    (void)smcwlt; (void)f1;
    nopac_shflx(ssoil, stc, smc, smcmax, nsoil, t1, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o,
                bexp, df1, ice, quartz, csoil, vegtyp, ivegsrc);
}
// SNOWPACK (4378-4505), body of unit phys-noah-snowpack (its kernel also holds the SNOPAC call-site
// gate ESD > 0, which nam_snopac applies itself). NAM 2013 changes: density cap 0.9.
__device__ void nam_snowpack(float& esd, float dtsec, float& snowh, float& sndens,
                             float& tsnow, float tsoil)
{
    const float esdc = esd * 100.0f;
    const float dthr = dtsec / 3600.0f;
    const float tsnowc = tsnow - 273.15f;
    const float tsoilc = tsoil - 273.15f;
    const float tavgc = 0.5f * (tsnowc + tsoilc);
    const float esdcx = (esdc > 1.0e-2f) ? esdc : 1.0e-2f;
    const float sd = sndens;
    const float bfac = dthr * 0.01f * expf(0.08f * tavgc - 21.0f * sd);
    float pexp = 0.0f;
    for (int j = 4; j >= 1; --j) pexp = (1.0f + pexp) * bfac * esdcx / (float)(j + 1);
    pexp = pexp + 1.0f;
    float dsx = sd * pexp;
    if (dsx > 0.9f) dsx = 0.9f;
    if (dsx < 0.05f) dsx = 0.05f;
    float sdn = dsx;
    if (tsnowc >= 0.0f) {
        const float dw = 0.13f * dthr / 24.0f;
        sdn = sdn * (1.0f - dw) + dw;
        if (sdn >= 0.9f) sdn = 0.9f;
    }
    sndens = sdn;
    snowh = (esdc / sdn) * 0.01f;
}

// ---- PENMAN (3390-3468), body of unit phys-noah-penman.
__device__ __forceinline__ void sflx_penman(
    float sfctmp, float sfcprs, float ch, float t2v, float th2, float prcp,
    float fdown, float ssoil, float q2, float q2sat, float dqsdt2,
    int snowng, int frzgra, float emissi_in, float sneqv,
    float& t24, float& etp, float& rch, float& rr, float& epsca, float& flx2)
{
    float emissi, elcp1, lvs;
    if (sneqv == 0.0f) { emissi = emissi_in; elcp1 = 2.4888e+3f; lvs = 2.501000e+6f; }
    else { emissi = 0.95f; elcp1 = 2.4888e+3f * 2.83e+6f / 2.501000e+6f; lvs = 2.83e+6f; }
    flx2 = 0.0f;
    const float delta = elcp1 * dqsdt2;
    t24 = sfctmp * sfctmp * sfctmp * sfctmp;
    rr = emissi * t24 * 6.48e-8f / (sfcprs * ch) + 1.0f;
    const float rho = sfcprs / (287.04f * t2v);
    rch = rho * 1004.6f * ch;
    if (snowng == 0) { if (prcp > 0.0f) rr = rr + 4.218e+3f * prcp / rch; }
    else rr = rr + 2.106e+3f * prcp / rch;
    float fnet = fdown - emissi * 5.67e-8f * t24 - ssoil;
    if (frzgra != 0) { flx2 = -3.335e+5f * prcp; fnet = fnet - flx2; }
    const float rad = fnet / rch + th2 - sfctmp;
    const float a = elcp1 * (q2sat - q2);
    epsca = (a * rr + rad * delta) / (delta + rr);
    etp = epsca * rch / lvs;
}

// ---- CANRES (2331-2469), body of unit phys-noah-canres (the caller guard SHDFAC > 0 and
// XLAI > 0 is in SFLX). NROOT is 1..NSOIL (REDPRM refuses other values at init).
__device__ void sflx_canres(float solar, float ch, float sfctmp, float q2, float sfcprs,
                            const float* sh2o, const float* zsoil, int nroot, float smcwlt,
                            float smcref, float rsmin, float& rc, float& pc, float q2sat,
                            float dqsdt2, float topt, float rsmax, float rgl, float hs, float xlai,
                            float& rcs, float& rct, float& rcq, float& rcsoil, float emissi)
{
    rcs = 0.0f; rct = 0.0f; rcq = 0.0f; rcsoil = 0.0f; rc = 0.0f;
    const float ff = 0.55f * 2.0f * solar / (rgl * xlai);
    rcs = (ff + rsmin / rsmax) / (1.0f + ff);
    rcs = fmaxf(rcs, 0.0001f);
    const float dt_opt = topt - sfctmp;
    rct = 1.0f - 0.0016f * (dt_opt * dt_opt);
    rct = fmaxf(rct, 0.0001f);
    rcq = 1.0f / (1.0f + hs * (q2sat - q2));
    rcq = fmaxf(rcq, 0.01f);
    float part[4];
    const int nr = (nroot < 1) ? 1 : (nroot > 4 ? 4 : nroot);
    float gx = (sh2o[0] - smcwlt) / (smcref - smcwlt);
    gx = fmaxf(fminf(gx, 1.0f), 0.0f);
    part[0] = (zsoil[0] / zsoil[nr - 1]) * gx;
    for (int k = 2; k <= nr; ++k) {
        gx = (sh2o[k - 1] - smcwlt) / (smcref - smcwlt);
        gx = fmaxf(fminf(gx, 1.0f), 0.0f);
        part[k - 1] = ((zsoil[k - 1] - zsoil[k - 2]) / zsoil[nr - 1]) * gx;
    }
    for (int k = 1; k <= nr; ++k) rcsoil = rcsoil + part[k - 1];
    rcsoil = fmaxf(rcsoil, 0.0001f);
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil);
    const float rr = (4.0f * emissi * 5.67E-8f * 287.04f / 1004.6f)
                     * (sfctmp * sfctmp * sfctmp * sfctmp) / (sfcprs * ch) + 1.0f;
    const float delta = (2.501000E6f / 1004.6f) * dqsdt2;
    pc = (rr + delta) / (rr * (1.0f + rc * ch) + delta);
}

// ---- REDPRM tables (phys-noah-parm-tables), 0-based class index.
struct NoahTbl {
    const float *albtbl, *z0tbl, *shdtbl, *rstbl, *rgltbl, *hstbl, *snuptbl, *laitbl;
    const int* nrotbl;
    const float *bb, *drysmc, *f11, *maxsmc, *refsmc, *satpsi, *satdk, *satdw, *wltsmc, *qtz;
    const float* slope_data;
    float sbeta_data, fxexp_data, csoil_data, salp_data, refdk_data, refkdt_data, frzk_data,
          zbot_data, czil_data, topt_data, cmcmax_data, cfactr_data, rsmax_data;
    int bare;
};

// ---- REDPRM (3471-3664), body of unit phys-noah-redprm for one point. Class indices are checked
// once at init by phys-noah-redprm; the clamp here only keeps reads in bounds.
__device__ void sflx_redprm(const NoahTbl& tb, int vegtyp, int soiltyp, int slopetyp,
    float& cfactr, float& cmcmax, float& rsmax, float& topt, float& refkdt, float& kdt,
    float& sbeta, float& shdfac, float& rsmin, float& rgl, float& hs, float& zbot, float& frzx,
    float& psisat, float& slope, float& snup, float& salp, float& bexp, float& dksat, float& dwsat,
    float& smcmax, float& smcwlt, float& smcref, float& smcdry, float& f1, float& quartz,
    float& fxexp, float* rtdis, const float* sldpth, const float* zsoil, int& nroot, int nsoil,
    float& z0brd, float& czil, float& xlai, float& csoil, float& alb, float& ptu, int local)
{
    const int vt = min(max(vegtyp, 1), 50) - 1;
    const int st = min(max(soiltyp, 1), 30) - 1;
    const int sp = min(max(slopetyp, 1), 30) - 1;
    csoil = tb.csoil_data;
    bexp = tb.bb[st]; dksat = tb.satdk[st]; dwsat = tb.satdw[st]; f1 = tb.f11[st];
    psisat = tb.satpsi[st]; quartz = tb.qtz[st]; smcdry = tb.drysmc[st]; smcmax = tb.maxsmc[st];
    smcref = tb.refsmc[st]; smcwlt = tb.wltsmc[st];
    zbot = tb.zbot_data; salp = tb.salp_data; sbeta = tb.sbeta_data;
    const float refdk = tb.refdk_data;
    const float frzk = tb.frzk_data;
    fxexp = tb.fxexp_data; refkdt = tb.refkdt_data; ptu = 0.0f;
    kdt = refkdt * dksat / refdk;
    czil = tb.czil_data;
    slope = tb.slope_data[sp];
    const float frzfact = (smcmax / smcref) * (0.412f / 0.468f);
    frzx = frzk * frzfact;
    topt = tb.topt_data; cmcmax = tb.cmcmax_data; cfactr = tb.cfactr_data; rsmax = tb.rsmax_data;
    nroot = tb.nrotbl[vt];
    snup = tb.snuptbl[vt]; rsmin = tb.rstbl[vt]; rgl = tb.rgltbl[vt]; hs = tb.hstbl[vt];
    xlai = tb.laitbl[vt];
    if (local) { alb = tb.albtbl[vt]; z0brd = tb.z0tbl[vt]; shdfac = tb.shdtbl[vt]; }
    if (vt + 1 == tb.bare) shdfac = 0.0f;
    const int nr = min(max(nroot, 1), nsoil);
    for (int k = 0; k < nsoil; ++k) rtdis[k] = (k < nr) ? -sldpth[k] / zsoil[nr - 1] : 0.0f;
}

// ---- SFLX (1550-2278). Argument order = the Fortran CALL SFLX list (phys_noah_driver.cu), plus the
// table struct. LLANDUSE/LSOIL arrive as the table ids (tables are bound at init).
__device__ void noah_sflx(
    float FFROZP, int ICE, float DT, float ZLVL, int NSOIL, const float* SLDPTH,
    int LOCAL, int IVEGSRC, int LUTYPE_ID, int SLTYPE_ID,
    float LWDN, float SOLDN, float SOLNET, float SFCPRS, float PRCP, float SFCTMP, float Q2,
    float SFCSPD, float* COSZ, float PRCPRAIN, float* SOLARDIRECT,
    float TH2, float Q2SAT, float DQSDT2,
    int VEGTYP, int SOILTYP, int SLOPETYP, float SHDFAC_in,
    float ALB_in, float SNOALB, float TBOT, float Z0BRD_in, float* Z0, float EMISSI,
    float* CMC, float* T1, float* STC, float* SMC, float* SH2O, float* SNOWH, float* SNEQV,
    float* ALBEDO, float* CH, float* CM,
    float* ETA, float* SHEAT, float* ETA_KINEMATIC, float* FDOWN,
    float* EC, float* EDIR, float* ET, float* ETT, float* ESNOW, float* DRIP, float* DEW,
    float* BETA, float* ETP, float* SSOIL,
    float* FLX1, float* FLX2, float* FLX3,
    float* SNOMLT, float* SNCOVR, float* RIMEF1,
    float* RUNOFF1, float* RUNOFF2, float* RUNOFF3,
    float* RC, float* PC, float* RSMIN, float* XLAI, float* RCS, float* RCT, float* RCQ,
    float* RCSOIL, float* SOILW, float* SOILM, float* Q1,
    float* SMCWLT, float* SMCDRY, float* SMCREF, float* SMCMAX, int* NROOT, float* RIBB,
    const NoahTbl& tb)
{
    (void)LUTYPE_ID; (void)SLTYPE_ID; (void)SFCSPD; (void)COSZ; (void)PRCPRAIN; (void)SOLARDIRECT;
    (void)CM; (void)ZLVL;
    const float TFREEZ = 273.15f, LVH2O = 2.501E+6f, LSUBS = 2.83E+6f, R = 287.04f;
    const float CP = 1004.6f;     // module_LS_NOAHLSM CP (Q1 and SHEAT)
    float ZSOIL[4], RTDIS[4];
    float SHDFAC = SHDFAC_in, ALB = ALB_in, Z0BRD = Z0BRD_in;
    float CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, RGL, HS, ZBOT, FRZX, PSISAT, SLOPE,
          SNUP, SALP, BEXP, DKSAT, DWSAT, F1, QUARTZ, FXEXP, CZIL, CSOIL, PTU, rsmin, xlai,
          smcmax, smcwlt, smcref, smcdry;
    int nroot;

    *RUNOFF1 = 0.0f; *RUNOFF2 = 0.0f; *RUNOFF3 = 0.0f; *SNOMLT = 0.0f;
    const int IW = (IVEGSRC == 1) ? 13 : 1;

    if (ICE == 1) {
        for (int kz = 1; kz <= NSOIL; ++kz) ZSOIL[kz - 1] = -3.0f * (float)kz / (float)NSOIL;
    } else {
        ZSOIL[0] = -SLDPTH[0];
        for (int kz = 2; kz <= NSOIL; ++kz) ZSOIL[kz - 1] = -SLDPTH[kz - 1] + ZSOIL[kz - 2];
    }
    // REDPRM (see the header: run for every point so ICE points have defined parameters).
    sflx_redprm(tb, VEGTYP, SOILTYP, SLOPETYP, CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA,
                SHDFAC, rsmin, RGL, HS, ZBOT, FRZX, PSISAT, SLOPE, SNUP, SALP, BEXP, DKSAT, DWSAT,
                smcmax, smcwlt, smcref, smcdry, F1, QUARTZ, FXEXP, RTDIS, SLDPTH, ZSOIL, nroot,
                NSOIL, Z0BRD, CZIL, xlai, CSOIL, ALB, PTU, LOCAL);
    (void)REFKDT; (void)CZIL; (void)PTU;

    if (VEGTYP == IW) {                     // wetland
        SHDFAC = 0.05f; rsmin = 400.0f; smcmax = 0.45f; smcref = 0.42f; smcwlt = 0.40f;
        smcdry = 0.40f; BEXP = 17.0f;
        const float SMHIGH = 3.02f, SMLOW = 0.4978f;
        DKSAT = 5.79E-9f / powf((smcref * SMHIGH - smcmax) / (smcmax * (SMHIGH - 1.0f)),
                                 2.0f * BEXP + 3.0f);
        PSISAT = 200.0f / powf(smcwlt / (smcmax * (1.0f - SMLOW)), -BEXP);
        DWSAT = BEXP * DKSAT * (PSISAT / smcmax);
        F1 = log10f(PSISAT) + BEXP * log10f(smcmax) + 2.0f;
    }
    smcref = smcref + (SMC[0] - SH2O[0]);   // 2015

    bool SNOWNG = false, FRZGRA = false;
    if (ICE == 1) { *SNEQV = 0.01f; *SNOWH = 0.05f; }
    float SNDENS, SNCOND;
    if (*SNEQV == 0.0f) { SNDENS = 0.0f; *SNOWH = 0.0f; SNCOND = 1.0f; }
    else { SNDENS = *SNEQV / *SNOWH; SNCOND = 2.0f * 0.11631f * (0.328f * powf(10.0f, 2.25f * SNDENS)); }
    if (PRCP > 0.0f) {
        if (FFROZP > 0.0f) SNOWNG = true;
        if ((*T1 <= TFREEZ) && (FFROZP < 1.0f)) FRZGRA = true;
    }
    float PRCPF;
    if (SNOWNG || FRZGRA) {
        float SN_NEW;
        if (FRZGRA) { SN_NEW = PRCP * DT * 0.001f; *SNEQV = *SNEQV + SN_NEW; PRCPF = 0.0f; }
        else { SN_NEW = PRCP * FFROZP * DT * 0.001f; *SNEQV = *SNEQV + SN_NEW; PRCPF = PRCP * (1.0f - FFROZP); }
        // SNOW_NEW (TEMP = SFCTMP), body of unit phys-noah-snow-new
        {
            float snowhc = *SNOWH * 100.0f;
            const float newsnc = SN_NEW * 100.0f;
            const float tempc = SFCTMP - 273.15f;
            float dsnew = (tempc <= -15.0f) ? 0.05f : 0.05f + 0.0017f * powf(tempc + 15.0f, 1.5f);
            const float rf = *RIMEF1;
            if (rf < 2.0f) { /* DSNEW = DSNEW */ }
            else if (rf < 5.0f) dsnew = fminf(0.9f, 2.0f * dsnew);
            else if (rf < 20.0f) dsnew = fminf(0.9f, 4.0f * dsnew);
            else dsnew = fminf(0.9f, 10.0f * dsnew);
            const float hnewc = newsnc / dsnew;
            if (snowhc + hnewc < 1.0e-3f) SNDENS = fmaxf(dsnew, SNDENS);
            else SNDENS = (snowhc * SNDENS + hnewc * dsnew) / (snowhc + hnewc);
            snowhc = snowhc + hnewc;
            *SNOWH = snowhc * 0.01f;
        }
        SNCOND = 2.0f * 0.11631f * (0.328f * powf(10.0f, 2.25f * SNDENS));
    } else {
        PRCPF = PRCP;
    }

    if (ICE == 0) {
        if (*SNEQV == 0.0f) {
            *SNCOVR = 0.0f;
            *ALBEDO = ALB;
        } else {
            SNUP = 4.0f * SNUP;
            if (*SNEQV < SNUP) {                         // SNFRAC
                const float rsnow = *SNEQV / SNUP;
                *SNCOVR = 1.0f - (expf(-SALP * rsnow) - rsnow * expf(-SALP));
            } else {
                *SNCOVR = 1.0f;
            }
            *SNCOVR = fminf(*SNCOVR, 0.98f);
            *ALBEDO = ALB + *SNCOVR * (SNOALB - ALB);    // ALCALC
            if (*ALBEDO > SNOALB) *ALBEDO = SNOALB;
        }
    } else {
        *SNCOVR = 1.0f;
        *ALBEDO = 0.65f;
    }

    float DF1;
    if (ICE == 1) {
        DF1 = 2.2f;
    } else {
        DF1 = nam_tdfcnd(SMC[0], QUARTZ, smcmax, SH2O[0]);
        if (VEGTYP == IW) DF1 = 3.24f;
        DF1 = DF1 * expf(SBETA * SHDFAC);
    }
    const float DSOIL = -(0.5f * ZSOIL[0]);
    float ssoil;
    if (*SNEQV == 0.0f) {
        ssoil = DF1 * (*T1 - STC[0]) / DSOIL;
    } else {
        const float DTOT = *SNOWH + DSOIL;
        const float FRCSNO = *SNOWH / DTOT;
        const float FRCSOI = DSOIL / DTOT;
        const float DF1A = FRCSNO * SNCOND + FRCSOI * DF1;   // DF1H is computed and never used
        DF1 = DF1A * *SNCOVR + DF1 * (1.0f - *SNCOVR);
        ssoil = DF1 * (*T1 - STC[0]) / DTOT;
    }
    *Z0 = Z0BRD;                            // SNOWZ0 (NAM: Z0 = Z0BRD) or Z0 = Z0BRD

    *FDOWN = SOLNET + LWDN;
    const float T2V = SFCTMP * (1.0f + 0.61f * Q2);
    float T24, etp, RCH, RR, EPSCA, flx2;
    sflx_penman(SFCTMP, SFCPRS, *CH, T2V, TH2, PRCP, *FDOWN, ssoil, Q2, Q2SAT, DQSDT2,
                SNOWNG ? 1 : 0, FRZGRA ? 1 : 0, EMISSI, *SNEQV, T24, etp, RCH, RR, EPSCA, flx2);

    float rc = 0.0f, pc = 0.0f, rcs = 0.0f, rct = 0.0f, rcq = 0.0f, rcsoil = 0.0f;
    if (SHDFAC > 0.0f && xlai > 0.0f) {
        sflx_canres(SOLDN, *CH, SFCTMP, Q2, SFCPRS, SH2O, ZSOIL, nroot, smcwlt, smcref, rsmin, rc, pc,
                    Q2SAT, DQSDT2, TOPT, RSMAX, RGL, HS, xlai, rcs, rct, rcq, rcsoil, EMISSI);
    }

    float esnow = 0.0f, eta = 0.0f, eta_kin = 0.0f, edir = 0.0f, ec = 0.0f, ett = 0.0f;
    float beta = *BETA, drip = 0.0f, dew = 0.0f, flx1 = 0.0f, flx3 = 0.0f, etns = 0.0f;
    float et[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (etp <= 0.0f && *RIBB > 0.90f) etp = fminf(etp * (1.0f - *RIBB), 0.0f);

    if (*SNEQV == 0.0f) {
        // NOPAC (3204-3387), body of unit phys-noah-nopac
        float prcp1 = PRCP * 0.001f;
        const float etp1 = etp * 0.001f;
        float edir1 = 0.0f, ec1 = 0.0f, ett1 = 0.0f, eta1 = 0.0f;
        float et1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float cmc = *CMC;
        float runoff1 = 0.0f, runoff2 = 0.0f, runoff3 = 0.0f;
        if (etp > 0.0f) {
            nopac_evapo(eta1, SH2O, cmc, etp1, DT, smcmax, BEXP, pc, smcwlt, smcref, SHDFAC, CMCMAX,
                        smcdry, CFACTR, FXEXP, nroot, RTDIS, edir1, ec1, et1, ett1, NSOIL);
            nopac_smflx(SMC, NSOIL, cmc, DT, prcp1, ZSOIL, SH2O, SLOPE, KDT, FRZX, smcmax, BEXP,
                        smcwlt, DKSAT, DWSAT, SHDFAC, CMCMAX, runoff1, runoff2, runoff3, edir1, ec1,
                        et1, drip);
            eta = eta1 * 1000.0f;
        } else {
            dew = -etp1;
            prcp1 = prcp1 + dew;
            nopac_smflx(SMC, NSOIL, cmc, DT, prcp1, ZSOIL, SH2O, SLOPE, KDT, FRZX, smcmax, BEXP,
                        smcwlt, DKSAT, DWSAT, SHDFAC, CMCMAX, runoff1, runoff2, runoff3, edir1, ec1,
                        et1, drip);
        }
        if (etp < 0.000001f) { beta = 1.0f; eta = etp; }
        else if (etp == 0.000001f) beta = 0.0f;
        else beta = eta / etp;
        edir = edir1 * 1000.0f;
        ec = ec1 * 1000.0f;
        for (int k = 0; k < 4; ++k) et[k] = et1[k] * 1000.0f;
        ett = ett1 * 1000.0f;
        float df1n = nam_tdfcnd(SMC[0], QUARTZ, smcmax, SH2O[0]);
        if (VEGTYP == 1) df1n = 3.24f;                 // NOPAC literal VEGTYP==1 (NOPAC body line 150)
        df1n = df1n * expf(SBETA * SHDFAC);
        const float yynum = *FDOWN - EMISSI * NOPAC_SIGMA * T24;
        const float yy = SFCTMP + (yynum / RCH + TH2 - SFCTMP - beta * EPSCA) / RR;
        const float zz1 = df1n / (-0.5f * ZSOIL[0] * RCH * RR) + 1.0f;
        float t1 = *T1;
        nopac_shflx(ssoil, STC, SMC, smcmax, NSOIL, t1, DT, yy, zz1, ZSOIL, TBOT, ZBOT, PSISAT, SH2O,
                    BEXP, df1n, ICE, QUARTZ, CSOIL, VEGTYP, IVEGSRC);
        *T1 = t1;
        *CMC = cmc;
        *RUNOFF1 = runoff1; *RUNOFF2 = runoff2; *RUNOFF3 = runoff3;
        flx1 = 0.0f; flx3 = 0.0f;
        eta_kin = eta;
    } else {
        NamSnopacCol s;
        s.etp = etp; s.prcp = PRCP; s.prcpf = PRCPF; s.snowng = SNOWNG ? 1 : 0;
        s.smcmax = smcmax; s.smcwlt = smcwlt; s.smcref = smcref; s.smcdry = smcdry; s.cmcmax = CMCMAX;
        s.dt = DT; s.sbeta = SBETA; s.df1 = DF1; s.q2 = Q2; s.sfctmp = SFCTMP; s.t24 = T24;
        s.th2 = TH2; s.fdown = *FDOWN; s.f1 = F1; s.bexp = BEXP; s.pc = pc; s.rch = RCH; s.rr = RR;
        s.cfactr = CFACTR; s.slope = SLOPE; s.kdt = KDT; s.frzfact = FRZX; s.psisat = PSISAT;
        s.dwsat = DWSAT; s.dksat = DKSAT; s.tbot = TBOT; s.zbot = ZBOT; s.shdfac = SHDFAC;
        s.nroot = nroot; s.vegtyp = VEGTYP; s.ice = ICE; s.ivegsrc = IVEGSRC;
        s.quartz = QUARTZ; s.fxexp = FXEXP; s.csoil = CSOIL; s.emissi = EMISSI;
        for (int k = 0; k < 4; ++k) {
            s.smc[k] = SMC[k]; s.sh2o[k] = SH2O[k]; s.stc[k] = STC[k]; s.zsoil[k] = ZSOIL[k];
            s.rtdis[k] = RTDIS[k]; s.et[k] = 0.0f;
        }
        s.cmc = *CMC; s.t1 = *T1; s.sncovr = *SNCOVR; s.esd = *SNEQV; s.sndens = SNDENS;
        s.snowh = *SNOWH; s.beta = beta; s.flx2 = flx2; s.ribb = *RIBB;
        s.ssoil = ssoil;
        nam_snopac(s);
        for (int k = 0; k < 4; ++k) { SMC[k] = s.smc[k]; SH2O[k] = s.sh2o[k]; STC[k] = s.stc[k]; et[k] = s.et[k]; }
        *CMC = s.cmc; *T1 = s.t1; *SNCOVR = s.sncovr; *SNEQV = s.esd; SNDENS = s.sndens;
        *SNOWH = s.snowh; beta = s.beta; flx2 = s.flx2;
        dew = s.dew; edir = s.edir; ec = s.ec; ett = s.ett; etns = s.etns; esnow = s.esnow;
        flx1 = s.flx1; flx3 = s.flx3; ssoil = s.ssoil; *SNOMLT = s.snomlt; drip = s.drip;
        *RUNOFF1 = s.runoff1; *RUNOFF2 = s.runoff2; *RUNOFF3 = s.runoff3;
        eta_kin = esnow + etns;
    }

    *Q1 = Q2 + eta_kin * CP / RCH;
    *SHEAT = -(*CH * CP * SFCPRS) / (R * T2V) * (TH2 - *T1);

    edir = edir * LVH2O;
    ec = ec * LVH2O;
    for (int k = 0; k < 4; ++k) et[k] = et[k] * LVH2O;
    ett = ett * LVH2O;
    esnow = esnow * LSUBS;
    etp = etp * ((1.0f - *SNCOVR) * LVH2O + *SNCOVR * LSUBS);
    if (etp > 0.0f) eta = edir + ec + ett + esnow;
    else eta = etp;
    if (etp == 0.0f) beta = 0.0f;
    else beta = eta / etp;

    ssoil = -1.0f * ssoil;
    *RUNOFF3 = *RUNOFF3 / DT;
    *RUNOFF2 = *RUNOFF2 + *RUNOFF3;

    float soilm, soilw;
    if (ICE == 0) {
        soilm = -1.0f * SMC[0] * ZSOIL[0];
        for (int k = 2; k <= NSOIL; ++k) soilm = soilm + SMC[k - 1] * (ZSOIL[k - 2] - ZSOIL[k - 1]);
        float soilwm = -1.0f * (smcmax - smcwlt) * ZSOIL[0];
        float soilww = -1.0f * (SMC[0] - smcwlt) * ZSOIL[0];
        if (nroot >= 2) {
            for (int k = 2; k <= nroot; ++k) {
                soilwm = soilwm + (smcmax - smcwlt) * (ZSOIL[k - 2] - ZSOIL[k - 1]);
                soilww = soilww + (SMC[k - 1] - smcwlt) * (ZSOIL[k - 2] - ZSOIL[k - 1]);
            }
        }
        if (soilwm < 1.E-6f) { soilw = 0.0f; soilm = 0.0f; }
        else soilw = fmaxf(soilww / soilwm, 0.0f);
    } else {
        soilw = 0.0f; soilm = 0.0f;
    }

    *ETA = eta; *ETA_KINEMATIC = eta_kin; *EC = ec; *EDIR = edir;
    for (int k = 0; k < 4; ++k) ET[k] = et[k];
    *ETT = ett; *ESNOW = esnow; *DRIP = drip; *DEW = dew; *BETA = beta; *ETP = etp; *SSOIL = ssoil;
    *FLX1 = flx1; *FLX2 = flx2; *FLX3 = flx3;
    *RC = rc; *PC = pc; *RSMIN = rsmin; *XLAI = xlai; *RCS = rcs; *RCT = rct; *RCQ = rcq;
    *RCSOIL = rcsoil; *SOILW = soilw; *SOILM = soilm;
    *SMCWLT = smcwlt; *SMCDRY = smcdry; *SMCREF = smcref; *SMCMAX = smcmax; *NROOT = nroot;
}
