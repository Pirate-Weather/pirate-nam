// phys_rrtm_driver_call.cu -- nam-gpu unit phys-rrtm-driver-call.
//
// NOTICE: NOAA/NCEP NAM source phys/module_RA_RRTM.F90 (US government work, no licence file in the
// NAM clone). The source header block is reproduced below as the notice.
//
// !
//       MODULE MODULE_RA_RRTM
// !
// !-----------------------------------------------------------------------
// !
// !***  THE RADIATION DRIVERS AND PACKAGES
// !
// !-----------------------------------------------------------------------
// !
//       USE MODULE_INCLUDE
// !
//       use physparam,     only : icldflg, ioznflg, kind_phys, icmphys
// 
//       USE MODULE_CONSTANTS, ONLY : R,CP,PI,EPSQ,STBOLT,EP_2
//       USE MODULE_MP_FER_HIRES, ONLY : FPVS
// 
//       use module_radiation_driver_nmmb,  only : grrad_nmmb,dayparts
// 
//       use module_radsw_parameters,  only : topfsw_type, sfcfsw_type
//       use module_radlw_parameters,  only : topflw_type, sfcflw_type
// 
//       use module_radsw_main_nmmb, only : buggal, buggaloff, buggalon
// 
// !-----------------------------------------------------------------------
// !
//       IMPLICIT NONE
// !
// !-----------------------------------------------------------------------
// !
//       PRIVATE
// !
//       PUBLIC :: RRTM, RRTM_INIT
// !
// !-----------------------------------------------------------------------
// !
// !--- Used for Gaussian look up tables
// !
// 
// !Moved to here from out of the RRTM routine to fix races found by Intel Inspector, jm 20131222,20131226
//       LOGICAL       :: CNCLD=.TRUE.
//
// Ported to CUDA for nam-gpu from phys/module_RA_RRTM.F90:879-1112 (nam.v4.2.11, 83df4e13c1).
//
// What it computes (per column, whole domain, one thread per (i,j); the Fortran DAYPART chunking
// is a CPU detail and is dropped, CONTRACT section 5):
//   - surface and column inputs for grrad: slmsk, fice, tsea, zorl, snwdph, sncovr, snoalb, hprime,
//     sinlat/coslat, flgmin, the zeroed CV/CVB/CVT, sfcalbedo, smx;
//   - interface and mid-layer pressure in kPa (prsi, prsl, prslk), virtual-temperature factor and
//     vertical velocity (vvel), gt/gq, the GR1 species array (ozone, total condensate, Thompson or
//     Ferrier hydrometeors), the Ferrier-path cloud arrays (f_icec, f_rainc, r_rime);
//   - subgrid-scale cloud for RH>95% when no condensate (SUBGRID, .TRUE. by default);
//   - the convective bogus cloud (CUCLD, .FALSE. by default: dead on the default path, ported in full);
//   - zeroed cldsa_v (5) and fluxr_v (39) accumulators, and the cu_cloud flag.
// The grrad call that follows (phys-rrtm-grrad) is a separate unit.
//
// Layout: 3-D fields [LM, JM, IM] (CONTRACT section 1), GR1 [NTRAC, LM, JM, IM], interfaces
// [LM+1, JM, IM], 2-D [JM, IM]. Fortran A(i,j,l) is at zero-based (i0, j0, l0). Fortran L is 1-based
// with l=1 the model top; here l0 = L-1, so PRSI(:,L+1) is l0 = L.
//
// Constants: module_CONSTANTS (R, CP, EP_2 = R/R_V, EPSQ) from nmmb_common.cuh; RRTM module PARAMETERs
// below from module_RA_RRTM.F90 lines 57, 74-86, 287.
#include "nmmb_common.cuh"
#include "phys_fer_fpvs.cu"

#define RRTM_NTRAC      9
#define RRTM_NFLUXR     39
#define RRTM_NSUBG      5
#define RRTM_R_OVER_CP  (NMMB_R_D / NMMB_CP)
#define RRTM_EP_2       (NMMB_R_D / NMMB_R_V)      /* module_CONSTANTS: EP_2 = r/r_v */
#define RRTM_TRAD_ICE   (273.15f - 30.0f)          /* TRAD_ice = 273.15-30. (line 57) */
#define RRTM_EPSO3      1.0e-10f                   /* EPSO3 (line 287 block) */
#define RRTM_QW_CU      (0.003e-3f)                /* QW_Cu (line 85) */
#define RRTM_QWMAX      1.0e-7f                    /* QWmax (line 85) */
#define RRTM_CUPPT_MIN  1.0e-5f                    /* CUPPT_min (line 85) */
#define RRTM_CU_DEEP_MIN 50.0e2f                   /* CU_DEEP_MIN (Pa, line 86) */
#define RRTM_CU_DEEP_MAX 200.0e2f                  /* CU_DEEP_MAX (Pa, line 86) */

extern "C" __global__ void phys_rrtm_driver_call(
    const int IM, const int JM, const int LM,
    const int ntoz,          /* NTOZ = ioznflg; ozone column read when > 0 */
    const int f_ni,          /* F_NI: Thompson microphysics species path */
    const int subgrid,       /* SUBGRID (SAVE .TRUE.) */
    const int cucld,         /* CUCLD (SAVE .FALSE.) */
    const int indx_cw,       /* INDX_CW: 4 for Thompson, 3 otherwise (1-based GR1 index) */
    const float convprate,   /* CONVPRATE (host scalar from convprate(), see wrapper) */
    /* inputs */
    const float* __restrict__ P8W,      /* [LM+1,JM,IM] interface pressure, Pa (state PINT) */
    const float* __restrict__ P_PHY,    /* [LM,JM,IM] mid-layer pressure, Pa */
    const float* __restrict__ T,
    const float* __restrict__ Q,
    const float* __restrict__ CW,
    const float* __restrict__ O3,
    const float* __restrict__ OMGALF,
    const float* __restrict__ CLDFRA,
    const float* __restrict__ TAUTOTAL, /* RRTM local, zeroed upstream (line 512) */
    const float* __restrict__ F_ICE,
    const float* __restrict__ F_RAIN,
    const float* __restrict__ F_RIMEF,
    const float* __restrict__ QC,
    const float* __restrict__ QI,
    const float* __restrict__ QS,
    const float* __restrict__ QR,
    const float* __restrict__ QG,
    const float* __restrict__ NI,
    const float* __restrict__ CUTOP,    /* [JM,IM] LM+1-HTOP, flipped index (1 top, LM surface) */
    const float* __restrict__ CUBOT,    /* [JM,IM] LM+1-HBOT */
    const float* __restrict__ CUPPT,
    const float* __restrict__ GLON,
    const float* __restrict__ GLAT,
    const float* __restrict__ TSKIN,
    const float* __restrict__ Z0,
    const float* __restrict__ SICE,
    const float* __restrict__ SI,
    const float* __restrict__ SNOWC,
    const float* __restrict__ MXSNAL,
    const float* __restrict__ STDH,
    const float* __restrict__ SM,
    const float* __restrict__ ALBEDO,
    const float* __restrict__ tbpvs,    /* fer_gpvs_tbpvs [7501] from phys-fer-gpvs */
    const float* __restrict__ coef,     /* fer_gpvs_coef [4]: C1XPVS, C2XPVS, C1XPVS0, C2XPVS0 */
    /* outputs */
    float* __restrict__ PRSI,           /* [LM+1,JM,IM] kPa */
    float* __restrict__ PRSL,           /* [LM,JM,IM] kPa */
    float* __restrict__ PRSLK,
    float* __restrict__ GT,
    float* __restrict__ GQ,
    float* __restrict__ GR1,            /* [NTRAC,LM,JM,IM] */
    float* __restrict__ VVEL,
    float* __restrict__ F_ICEC,
    float* __restrict__ F_RAINC,
    float* __restrict__ R_RIME,
    float* __restrict__ TAUCLOUDS,
    float* __restrict__ CLDF,
    float* __restrict__ CLDCOV_V,
    float* __restrict__ CLDSA_V,        /* [5,JM,IM] */
    float* __restrict__ FLUXR_V,        /* [39,JM,IM] */
    int*   __restrict__ CU_CLOUD,       /* [JM,IM] LOGICAL */
    float* __restrict__ XLON,
    float* __restrict__ XLAT,
    float* __restrict__ TSEA,
    float* __restrict__ TISFC,
    float* __restrict__ ZORL,
    float* __restrict__ SNWDPH,
    float* __restrict__ SNCOVR,
    float* __restrict__ SNOALB,
    float* __restrict__ HPRIME_V,
    float* __restrict__ SLMSK,
    float* __restrict__ FICE,
    float* __restrict__ FLGMIN_L,
    float* __restrict__ CV,
    float* __restrict__ CVB,
    float* __restrict__ CVT,
    float* __restrict__ SFCALBEDO,
    float* __restrict__ SMX,
    float* __restrict__ SINLAT,
    float* __restrict__ COSLAT)
{
    const int col_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_ >= IM * JM) return;
    const int i0 = col_ % IM;
    const int j0 = col_ / IM;
    const size_t c2 = IDX2(i0, j0, IM);

    /* ---- surface and column scalars (Fortran 879-1112, daypart body) ---- */
    const float glon = GLON[c2];
    const float glat = GLAT[c2];
    XLON[c2] = glon;
    XLAT[c2] = glat;
    SINLAT[c2] = sinf(glat);
    COSLAT[c2] = cosf(glat);
    TSEA[c2] = TSKIN[c2];
    TISFC[c2] = TSKIN[c2];                       /* change later if necessary (source) */
    /* Fortran Z0*100.d0 is exact in double (100 has 5 significant bits), so one rounding to single
     * gives the same value as this single-precision product. */
    ZORL[c2] = Z0[c2] * 100.0f;
    SNWDPH[c2] = SI[c2];                         /* snwdph [mm] */
    SNCOVR[c2] = SNOWC[c2];
    SNOALB[c2] = MXSNAL[c2];
    HPRIME_V[c2] = STDH[c2];
    if (SICE[c2] > 0.5f) {                       /* WHERE (SICE > 0.5): sea ice, slmsk = 2 */
        SLMSK[c2] = 2.0f;
        FICE[c2] = SICE[c2];
    } else {                                     /* ELSEWHERE: slmsk = 1 - SM */
        SLMSK[c2] = 1.0f - SM[c2];
        FICE[c2] = 0.0f;
    }
    FLGMIN_L[c2] = 0.20f;                        /* for ferrier */
    CV[c2] = 0.0f;                               /* not in use */
    CVB[c2] = 0.0f;
    CVT[c2] = 0.0f;
    SFCALBEDO[c2] = ALBEDO[c2];
    SMX[c2] = SM[c2];

    /* ---- interface pressure and the level loop ---- */
    PRSI[IDX3(i0, j0, 0, IM, JM)] = P8W[IDX3(i0, j0, 0, IM, JM)] / 1000.0f;
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t cl = IDX3(i0, j0, l0, IM, JM);
        PRSI[IDX3(i0, j0, l0 + 1, IM, JM)] = P8W[IDX3(i0, j0, l0 + 1, IM, JM)] / 1000.0f;
        const float prsl = P_PHY[cl] / 1000.0f;
        PRSL[cl] = prsl;
        /* Fortran PRSLK = (PRSL*0.01d0)**(R/CP): double base, R/CP formed in single then promoted. */
        PRSLK[cl] = (float)pow((double)prsl * 0.01, (double)RRTM_R_OVER_CP);
        const float t = T[cl];
        const float q = Q[cl];
        const float cw = CW[cl];
        /* RTvR = 1/(R*(Q*0.608+1-CW)*T) in single (Fortran REAL); VVEL = OMGALF*1000.d0*PRSL*RTvR
         * evaluated left to right in double, stored single (not used by grrad, kept). */
        const float rtvr = 1.0f / (NMMB_R_D * (q * 0.608f + 1.0f - cw) * t);
        VVEL[cl] = (float)((double)OMGALF[cl] * 1000.0 * (double)prsl * (double)rtvr);
        GT[cl] = t;
        GQ[cl] = q;
        /* GR1: 1 ozone, 2 reserved (aerosol), 3 total condensate, 4-9 Thompson hydrometeors */
        for (int n0 = 0; n0 < RRTM_NTRAC; ++n0)
            GR1[IDX4(i0, j0, l0, n0, IM, JM, LM)] = 0.0f;
        if (ntoz > 0)
            GR1[IDX4(i0, j0, l0, 0, IM, JM, LM)] = fmaxf(O3[cl], RRTM_EPSO3);
        CLDCOV_V[cl] = 0.0f;                     /* used for prognostic cloud (zeroed) */
        TAUCLOUDS[cl] = TAUTOTAL[cl];
        CLDF[cl] = CLDFRA[cl];
        GR1[IDX4(i0, j0, l0, 2, IM, JM, LM)] = cw;   /* total condensate */
        if (f_ni) {
            GR1[IDX4(i0, j0, l0, 3, IM, JM, LM)] = QC[cl];
            GR1[IDX4(i0, j0, l0, 4, IM, JM, LM)] = QI[cl];
            GR1[IDX4(i0, j0, l0, 5, IM, JM, LM)] = QS[cl];
            GR1[IDX4(i0, j0, l0, 6, IM, JM, LM)] = QR[cl];
            GR1[IDX4(i0, j0, l0, 7, IM, JM, LM)] = QG[cl];
            GR1[IDX4(i0, j0, l0, 8, IM, JM, LM)] = NI[cl];
        } else {
            F_ICEC[cl] = F_ICE[cl];
            F_RAINC[cl] = F_RAIN[cl];
            R_RIME[cl] = F_RIMEF[cl];
        }
        /* Thompson (F_NI): Fortran writes F_ICEC, F_RAINC, R_RIME only on the non-F_NI branch and leaves
         * caller-owned values in place. Here they are written as zero, so no uninitialised device memory
         * reaches grrad. No reader outside module_RA_RRTM.F90 was found in the NAM tree (grrad path
         * unverified). The F_NI subgrid block below also leaves them untouched. */
        if (f_ni) {
            F_ICEC[cl] = 0.0f;
            F_RAINC[cl] = 0.0f;
            R_RIME[cl] = 0.0f;
        }
    }

    /* ---- subgrid-scale cloud: RH > 95% with no condensate (SUBGRID) ---- */
    if (subgrid) {
        for (int l0 = 0; l0 < LM; ++l0) {
            const size_t cl = IDX3(i0, j0, l0, IM, JM);
            const float gq = GQ[cl];
            const float wv = gq / (1.0f - gq);                 /* water vapour mixing ratio */
            const float tcld = GT[cl];
            float esat = nam_fpvs(tcld, tbpvs, coef[0], coef[1]); /* kPa */
            const float p1 = PRSL[cl];                         /* kPa */
            esat = fminf(esat, 0.99f * p1);
            const float qsat = RRTM_EP_2 * esat / (p1 - esat);
            const float rhum = wv / qsat;
            const size_t c_cw = IDX4(i0, j0, l0, 2, IM, JM, LM);
            if (GR1[c_cw] < NMMB_EPSQ && rhum > 0.95f) {
                const float arg = fminf(0.01f, rhum - 0.95f) * qsat;
                GR1[c_cw] = fminf(0.01e-3f, arg);
                if (f_ni) {
                    if (tcld > RRTM_TRAD_ICE)
                        GR1[IDX4(i0, j0, l0, 3, IM, JM, LM)] = GR1[c_cw];
                    else
                        GR1[IDX4(i0, j0, l0, 4, IM, JM, LM)] = GR1[c_cw];
                } else {
                    F_ICEC[cl] = (tcld > RRTM_TRAD_ICE) ? 0.0f : 1.0f;
                    F_RAINC[cl] = 0.0f;
                    R_RIME[cl] = 1.0f;
                }
            }
        }
    }

    /* ---- convective bogus cloud (CUCLD; .FALSE. by default) ---- */
    CU_CLOUD[c2] = 0;
    if (cucld) {
        int lcnvt = (int)roundf(CUTOP[c2]);
        int lcnvb = (int)roundf(CUBOT[c2]);
        if (lcnvt > LM) lcnvt = LM;                    /* MIN(LM, NINT(..)) */
        if (lcnvb > LM) lcnvb = LM;
        float cu_depth = 0.0f;
        /* DECLARED DIVERGENCE (default-off branch): lcnvt < 1 makes the Fortran read PRSL(ii,0) and
         * GR1(ii,0,3), out of bounds (undefined native memory, no value to match). Such a column is
         * skipped here. Not exercised on the default path; not proven against native. */
        if (lcnvb - lcnvt > 1 && lcnvt >= 1) {
            cu_depth = 1000.0f * (PRSL[IDX3(i0, j0, lcnvb - 1, IM, JM)] -
                                  PRSL[IDX3(i0, j0, lcnvt - 1, IM, JM)]);   /* Pa */
            if (cu_depth >= RRTM_CU_DEEP_MIN && cu_depth <= RRTM_CU_DEEP_MAX) {
                float qcld = -3.402823466e+38f;                        /* MAXVAL(GR1(:,:,3)) */
                for (int l0 = 0; l0 < LM; ++l0) {
                    const float v = GR1[IDX4(i0, j0, l0, 2, IM, JM, LM)];
                    if (v > qcld) qcld = v;
                }
                const float pmod = CUPPT[c2] * convprate;
                if (qcld < RRTM_QWMAX && pmod <= RRTM_CUPPT_MIN) {
                    CU_CLOUD[c2] = 1;
                    for (int l = lcnvt; l <= lcnvb; ++l) {
                        const int l0 = l - 1;
                        GR1[IDX4(i0, j0, l0, 2, IM, JM, LM)] += RRTM_QW_CU;       /* cloud water */
                        if (f_ni)
                            GR1[IDX4(i0, j0, l0, indx_cw - 1, IM, JM, LM)] += RRTM_QW_CU;
                    }
                }
            }
        }
    }

    /* ---- accumulators zeroed before grrad ---- */
    for (int nc0 = 0; nc0 < RRTM_NSUBG; ++nc0)
        CLDSA_V[IDX3(i0, j0, nc0, IM, JM)] = 0.0f;
    for (int nc0 = 0; nc0 < RRTM_NFLUXR; ++nc0)
        FLUXR_V[IDX3(i0, j0, nc0, IM, JM)] = 0.0f;
}
