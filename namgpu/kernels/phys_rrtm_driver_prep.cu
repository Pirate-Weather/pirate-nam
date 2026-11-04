// phys_rrtm_driver_prep.cu -- nam-gpu unit phys-rrtm-driver-prep.
//
// NOAA notice: module MODULE_RA_RRTM (phys/module_RA_RRTM.F90, NCEP/EMC NAM nam.v4.2.11,
// 83df4e13c1), "THE RADIATION DRIVERS AND PACKAGES". NOAA/NCEP US government work; the NAM clone
// carries no licence file. Kept as the notice of the source; the module text has no per-author
// header block beyond that.
//
// Ported to CUDA for nam-gpu from phys/module_RA_RRTM.F90:98-878 (nam.v4.2.11, 83df4e13c1).
// Subroutine RRTM, the cloud-input and cloud-layer part that runs before the DO J column loop
// (the column loop itself, 879-1112, is unit phys-rrtm-driver-call).
//
// Two kernels, one launch each (CONTRACT section 3):
//   rrtm_prep_p8w      one thread per I: P8W and P_PHY from the last J row (see note 1).
//   rrtm_prep_columns  one thread per (I,J) column; every term of 98-878 for that column.
//
// Notes on the Fortran that the port reproduces on purpose:
//   1. P8W(I,K) and P_PHY(I,K) are declared (ITS:ITE,1:LM+1) and (ITS:ITE,1:LM) with no J
//      dimension, and the first nest loops DO J / DO I. They therefore hold the values of the
//      LAST row J = JTE, and every later read (the cloud-property block, the non-OPER branch)
//      uses that row for all J. rrtm_prep_p8w writes exactly those values.
//   2. QCLD = QICE + MAX(QS,0) with QICE = MAX(QS,0) (module_RA_RRTM.F90 ICWP_Test block): both
//      terms are QS, not QC/QI. Kept literally.
//   3. CNCLD (module variable, initially .TRUE.) becomes .FALSE. on the first call with ICWP/=-1
//      and never returns, so for a consistent configuration CNCLD == (ICWP == -1). Host side.
//   4. The NAM configuration (ntcw=3 -> icldflg=1; np3d=3 -> icmphys=3; cld_fraction=0 for the
//      default cloud fraction) takes NEITHER the ICWP_Test block nor the GFDL_Conv block. Both
//      stay ported and gated by icwp_m1 / cnc. With those off, CSMID and CCMID stay zero and the
//      layering block finds no cloud. That is the Fortran behaviour, not a port choice.
//   5. GFDL_Conv loop start: the Fortran DO L=LCNVT,LCNVB has no clamp. When LCNVT < 1 (CUTOP < 1,
//      i.e. HTOP > LM) it writes CCMID(I,J,0), out of bounds in the Fortran (undefined). The port
//      starts the loop at L = max(LCNVT,1). Identical wherever the Fortran is defined (CUTOP >= 1);
//      this is the one deliberate deviation and it only changes undefined input.
//   5b. Omitted on purpose (recorded): Fortran 430-435 zeroes ICSDSW/ICSDLW, which the Fortran
//      itself documents as unused in this version. The port neither allocates nor writes them.
//   6. TauC is accumulated in the cloud-property loop and not used afterwards in this range; it
//      is kept so the loop reads as the Fortran does.
//   7. NXSD = XSDmax/DXSD = 310 (the AXSD table length; the Fortran builds AXSD in RRTM_INIT,
//      module_RA_RRTM.F90:1480-1486, unit phys-rrtm-init).
//
// Index convention (CONTRACT section 1): Fortran A(I,J,L) -> a[IDX3(I-1,J-1,L-1,IM,JM)];
// 2-D A(I,J) -> a[IDX2(I-1,J-1,IM)]. The kernel body uses the Fortran 1-based indices through the
// macros P2/P3/P8/PP below, so every line reads against the source.
#include "nmmb_common.cuh"

// --- module_RA_RRTM.F90 PARAMETERs used in 98-878 (single precision) ---
#define RRTM_RHGRD        1.00f
#define RRTM_TRAD_ICE     (273.15f - 30.0f)
#define RRTM_ABSCOEF_W    800.0f
#define RRTM_ABSCOEF_I    500.0f
#define RRTM_QCONV        0.1e-3f
#define RRTM_CTAUCW       (RRTM_ABSCOEF_W * RRTM_QCONV)
#define RRTM_CTAUCI       (RRTM_ABSCOEF_I * RRTM_QCONV)
#define RRTM_EPSQ         1.0e-12f      /* module_CONSTANTS epsq (grrad.f parameter EPSQ) */
#define RRTM_EP_2         (NMMB_R_D / NMMB_R_V)   /* module_CONSTANTS ep_2 = r/r_v */
#define RRTM_H0           0.0f
#define RRTM_H1           1.0f
#define RRTM_HALF         0.5f
#define RRTM_CLFRMIN      0.01f
#define RRTM_NXSD         310
#define RRTM_XSDMAX       3.1f
#define RRTM_DXSD         0.01f
#define RRTM_XSDMIN       (-3.1f)
#define RRTM_DXSD1        (-0.01f)
#define RRTM_STSDM        0.01f
#define RRTM_CVSDM        0.04f
#define RRTM_DXSD2        (0.5f * 0.01f)
#define RRTM_DXSD2N       (-(0.5f * 0.01f))

// 1-based Fortran accessors (see header).
#define P2(a, J, I)        a[IDX2((I) - 1, (J) - 1, IM)]
#define P3(a, L, J, I)     a[IDX3((I) - 1, (J) - 1, (L) - 1, IM, JM)]
#define P8(K, I)           P8W[((K) - 1) * IM + ((I) - 1)]
#define PP(K, I)           P_PHY[((K) - 1) * IM + ((I) - 1)]

// ---------------------------------------------------------------------------
// Kernel 1: P8W(I,1:LM+1) and P_PHY(I,1:LM) from the last row J = JM (Fortran JTE).
// Launch: grid ceil(IM/128), block 128, one thread per I.
// Layouts: P8W[(K-1)*IM + (I-1)] with K = 1..LM+1; P_PHY[(K-1)*IM + (I-1)] with K = 1..LM.
// Fortran source: the first DO J / DO I nest of 98-878 (P8W(I,1)=PT; P8W(I,K+1) = P8W(I,K)
// + PDSG1(K) + DSG2(K)*PD(I,J); P_PHY(I,K) = SGML2(K)*PD(I,J) + PSGML1(K)).
// ---------------------------------------------------------------------------
extern "C" __global__ void rrtm_prep_p8w(
    const int IM, const int JM, const int LM,
    const float PT,
    const float* __restrict__ PD,        /* [JM, IM] */
    const float* __restrict__ PDSG1,     /* [LM] */
    const float* __restrict__ DSG2,      /* [LM] */
    const float* __restrict__ SGML2,     /* [LM] */
    const float* __restrict__ PSGML1,    /* [LM] */
    float* __restrict__ P8W,             /* [LM+1, IM] */
    float* __restrict__ P_PHY)           /* [LM, IM] */
{
    const int I = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (I > IM) return;
    const int J = JM;                                 /* JTE: last row written by the Fortran nest */
    const float PDIJ = P2(PD, J, I);

    P8(1, I) = PT;                                    /* P8W(I,1) = PT */
    for (int K = 1; K <= LM; ++K) {
        P8(K + 1, I) = P8(K, I) + PDSG1[K - 1] + DSG2[K - 1] * PDIJ;   /* P8W(I,K+1) */
        PP(K, I) = SGML2[K - 1] * PDIJ + PSGML1[K - 1];               /* P_PHY(I,K) */
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: every column (I,J). One thread per column (grid_cols over IM*JM), an L loop inside.
// Order inside a thread is the Fortran order of 98-878:
//   zeroing (CFRAC*, CZMEAN, SIGT4, CLDCFR, MTOP, MBOT, CUTOP, CUBOT, TAUTOTAL, CCMID, CSMID,
//   CLDFRA if CLD_FRACTION == 0); ICWP_Test block (icwp_m1); GFDL_Conv block (cnc); the
//   KTOP/KBTM/CAMT/KCLD initialisation; the layering loop; NCLDS; the cloud-property block.
// Arguments that are 2-D are [JM, IM]; 3-D [LM, JM, IM]; KTOP/KBTM/CAMT are [LM+1, JM, IM];
// CLDCFR/MTOP/MBOT are [3, JM, IM]. FPVS is read only when oper == 0 (filled by the wrapper with
// phys-fer-fpvs); AXSD is the 310-entry Gaussian table of RRTM_INIT. P8W/P_PHY come from kernel 1.
// ---------------------------------------------------------------------------
extern "C" __global__ void rrtm_prep_columns(
    const int IM, const int JM, const int LM,
    const int icwp_m1,          /* 1 when ICWP == -1 (icmphys == 5): old NAM/GFDL cloud inputs */
    const int cnc,              /* 1 when CNCLD is .TRUE. (ICWP == -1): GFDL convective clouds */
    const int oper,             /* module flag OPER (configfile oper: true -> 1) */
    const int cld_fraction_zero,/* 1 when CLD_FRACTION == 0 (default cloud fraction) */
    const float convprate,      /* CONVPRATE (host-computed from DT, NRADS, NRADL; see wrapper) */
    float* __restrict__ CLDFRA,            /* [LM, JM, IM] INOUT */
    const float* __restrict__ T,           /* [LM, JM, IM] */
    const float* __restrict__ Q,           /* [LM, JM, IM] */
    const float* __restrict__ QC,          /* [LM, JM, IM] */
    const float* __restrict__ QS,          /* [LM, JM, IM] */
    const float* __restrict__ CUPPT,       /* [JM, IM] */
    const float* __restrict__ HTOP,        /* [JM, IM] */
    const float* __restrict__ HBOT,        /* [JM, IM] */
    const float* __restrict__ P8W,         /* [LM+1, IM] from kernel 1 */
    const float* __restrict__ P_PHY,       /* [LM, IM]   from kernel 1 */
    const float* __restrict__ FPVS,        /* [LM, JM, IM] (oper == 0 only) */
    const float* __restrict__ AXSD,        /* [310] */
    float* __restrict__ CFRACH,            /* [JM, IM] */
    float* __restrict__ CFRACL,            /* [JM, IM] */
    float* __restrict__ CFRACM,            /* [JM, IM] */
    float* __restrict__ CZMEAN,            /* [JM, IM] */
    float* __restrict__ SIGT4,             /* [JM, IM] */
    float* __restrict__ CSMID,             /* [LM, JM, IM] scratch */
    float* __restrict__ CCMID,             /* [LM, JM, IM] scratch */
    float* __restrict__ TAUTOTAL,          /* [LM, JM, IM] scratch */
    float* __restrict__ CUTOP,             /* [JM, IM] scratch */
    float* __restrict__ CUBOT,             /* [JM, IM] scratch */
    int* __restrict__ KTOP,                /* [LM+1, JM, IM] scratch */
    int* __restrict__ KBTM,                /* [LM+1, JM, IM] scratch */
    float* __restrict__ CAMT,              /* [LM+1, JM, IM] scratch */
    int* __restrict__ NCLDS,               /* [JM, IM] scratch */
    int* __restrict__ KCLD,                /* [JM, IM] scratch */
    float* __restrict__ CLDCFR,            /* [3, JM, IM] scratch */
    int* __restrict__ MTOP,                /* [3, JM, IM] scratch */
    int* __restrict__ MBOT)                /* [3, JM, IM] scratch */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int I = col % IM + 1;
    const int J = col / IM + 1;
    const float LMP1 = (float)(LM + 1);

    // ---- 2-D zeroing and CUTOP/CUBOT (Fortran DO J / DO I loops after P8W) ----
    P2(CFRACH, J, I) = 0.0f;
    P2(CFRACL, J, I) = 0.0f;
    P2(CFRACM, J, I) = 0.0f;
    P2(CZMEAN, J, I) = 0.0f;
    P2(SIGT4, J, I) = 0.0f;
    for (int K = 1; K <= 3; ++K) {
        P3(CLDCFR, K, J, I) = 0.0f;
        P3(MTOP, K, J, I) = 0;
        P3(MBOT, K, J, I) = 0;
    }
    P2(CUTOP, J, I) = LMP1 - P2(HTOP, J, I);          /* CUTOP(I,J) = LM+1-HTOP(I,J) */
    P2(CUBOT, J, I) = LMP1 - P2(HBOT, J, I);          /* CUBOT(I,J) = LM+1-HBOT(I,J) */

    // ---- the first nest zeroing, and the TAUTOTAL/CLDFRA zeroing ----
    for (int L = 1; L <= LM; ++L) {
        P3(TAUTOTAL, L, J, I) = 0.0f;
        P3(CCMID, L, J, I) = 0.0f;
        P3(CSMID, L, J, I) = 0.0f;
    }
    if (cld_fraction_zero) {
        for (int L = 1; L <= LM; ++L) P3(CLDFRA, L, J, I) = 0.0f;
    }

    // ---- ICWP_Test: old NAM/GFDL cloud inputs (ICWP == -1 only) ----
    if (icwp_m1) {
        for (int L = 1; L <= LM; ++L) {
            const float qs_pos = fmaxf(P3(QS, L, J, I), 0.0f);
            const float qice = qs_pos;                       /* QICE = MAX(QS,0.) */
            const float qcld = qice + qs_pos;                /* QCLD = QICE + MAX(QS,0.) */
            if (oper) {
                /* OPER_flag: IF (OPER) branch */
                float clfr = fminf(RRTM_H1, fmaxf(RRTM_H0, 1.e5f * qcld));
                clfr = sqrtf(clfr);
                if (clfr >= RRTM_CLFRMIN) P3(CSMID, L, J, I) = clfr;
            } else {
                /* else OPER_flag: non-OPER branch */
                if (qcld <= RRTM_EPSQ) continue;             /* GO TO 255 */
                const float qv = fmaxf(RRTM_EPSQ, P3(Q, L, J, I));
                const float wv = qv / (1.0f - qv);           /* WV */
                const float esat0 = 1000.0f * P3(FPVS, L, J, I);   /* 1000*FPVS(T) */
                const float esat = fminf(esat0, 0.99f * PP(L, I));
                const float qsat = RRTM_EP_2 * esat / (PP(L, I) - esat);
                const float rhtot = (wv + qcld) / qsat;      /* RHtot */
                int lcnvt = (int)lroundf(P2(CUTOP, J, I));
                lcnvt = min(LM, lcnvt);
                int lcnvb = (int)lroundf(P2(CUBOT, J, I));
                lcnvb = min(LM, lcnvb);
                const float sdm = (L >= lcnvt && L <= lcnvb) ? RRTM_CVSDM : RRTM_STSDM;
                const float arg = (rhtot - RRTM_RHGRD) / sdm;
                float clfr;
                if (arg <= RRTM_DXSD2 && arg >= RRTM_DXSD2N) {
                    clfr = RRTM_HALF;
                } else if (arg > RRTM_DXSD2) {
                    if (arg >= RRTM_XSDMAX) {
                        clfr = RRTM_H1;
                    } else {
                        int ixsd = (int)(arg / RRTM_DXSD + RRTM_HALF);
                        ixsd = min(RRTM_NXSD, max(ixsd, 1));
                        clfr = RRTM_HALF + AXSD[ixsd - 1];
                    }
                } else {
                    if (arg <= RRTM_XSDMIN) {
                        clfr = RRTM_H0;
                    } else {
                        int ixsd = (int)(arg / RRTM_DXSD1 + RRTM_HALF);
                        ixsd = min(RRTM_NXSD, max(ixsd, 1));
                        clfr = RRTM_HALF - AXSD[ixsd - 1];
                        if (clfr < RRTM_CLFRMIN) clfr = RRTM_H0;
                    }
                }
                P3(CSMID, L, J, I) = clfr;
            }
        }
    }

    // ---- GFDL_Conv: convective cloud fractions (CNCLD only) ----
    if (cnc) {
        const float CC[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        const float PPT[10] = {0.f, .14f, .31f, .70f, 1.6f, 3.4f, 7.7f, 17.f, 38.f, 85.f};
        if (P2(CUBOT, J, I) - P2(CUTOP, J, I) > 1.0f) {
            float clfr = CC[0];
            const float pmod = P2(CUPPT, J, I) * convprate;
            if (pmod > PPT[0]) {
                int nmod = 0;
                for (int nc = 1; nc <= 10; ++nc) {
                    if (pmod > PPT[nc - 1]) nmod = nc;
                }
                if (nmod >= 10) {
                    clfr = CC[9];
                } else {
                    const float cc1 = CC[nmod - 1];
                    const float cc2 = CC[nmod];
                    const float p1 = PPT[nmod - 1];
                    const float p2 = PPT[nmod];
                    clfr = cc1 + (cc2 - cc1) * (pmod - p1) / (p2 - p1);
                }
                clfr = fminf(RRTM_H1, clfr);
            }
            int lcnvt = (int)lroundf(P2(CUTOP, J, I));
            lcnvt = min(LM, lcnvt);
            int lcnvb = (int)lroundf(P2(CUBOT, J, I));
            lcnvb = min(LM, lcnvb);
            for (int L = max(lcnvt, 1); L <= lcnvb; ++L) {
                const float arg = fmaxf(RRTM_H0, RRTM_H1 - P3(CSMID, L, J, I));
                P3(CCMID, L, J, I) = fminf(arg, clfr);
            }
        }
    }

    // ---- initialise the layer arrays (KTOP/KBTM/CAMT/KCLD) ----
    P3(KTOP, 1, J, I) = LM + 1;
    P3(KBTM, 1, J, I) = LM + 1;
    P3(CAMT, 1, J, I) = 1.0f;
    P2(KCLD, J, I) = 2;
    for (int L = 2; L <= LM + 1; ++L) {
        P3(CAMT, L, J, I) = 0.0f;
        P3(KTOP, L, J, I) = 1;
        P3(KBTM, L, J, I) = 1;
    }

    // ---- layering: amount, top, bottom and type of each cloud layer ----
    int new_cloud = 1;
    for (int L = 2; L <= LM; ++L) {
        const int LL = LM - L + 1;
        const float clfr = fmaxf(P3(CCMID, LL, J, I), P3(CSMID, LL, J, I));
        const float clfr1 = fmaxf(P3(CCMID, LL + 1, J, I), P3(CSMID, LL + 1, J, I));
        if (clfr >= RRTM_CLFRMIN) {
            if (new_cloud) {
                if (L == 2 && clfr1 >= RRTM_CLFRMIN) {
                    P3(KBTM, P2(KCLD, J, I), J, I) = LL + 1;
                    P3(CAMT, P2(KCLD, J, I), J, I) = clfr1;
                } else {
                    P3(KBTM, P2(KCLD, J, I), J, I) = LL;
                    P3(CAMT, P2(KCLD, J, I), J, I) = clfr;
                }
                new_cloud = 0;
            } else {
                P3(CAMT, P2(KCLD, J, I), J, I) = fmaxf(P3(CAMT, P2(KCLD, J, I), J, I), clfr);
            }
        } else if (clfr1 >= RRTM_CLFRMIN) {
            if (L == 2) {
                P3(KBTM, P2(KCLD, J, I), J, I) = LL + 1;
                P3(CAMT, P2(KCLD, J, I), J, I) = clfr1;
            }
            P3(KTOP, P2(KCLD, J, I), J, I) = LL + 1;
            new_cloud = 1;
            P2(KCLD, J, I) = P2(KCLD, J, I) + 1;
            P3(CAMT, P2(KCLD, J, I), J, I) = 0.0f;
        }
    }

    // ---- number of cloud layers ----
    P2(NCLDS, J, I) = P2(KCLD, J, I) - 2;
    const int ncld = P2(NCLDS, J, I);

    // ---- cloud radiative properties per layer ----
    if (ncld >= 1) {
        for (int NC = 2; NC <= ncld + 1; ++NC) {
            float tauc = 0.0f;                             /* TauC: accumulated, unused (note 6) */
            const int bitx = (P3(CAMT, NC, J, I) >= RRTM_CLFRMIN);
            int nktp = LM + 1;
            int nbtm = 0;
            nktp = min(nktp, P3(KTOP, NC, J, I));
            nbtm = max(nbtm, P3(KBTM, NC, J, I));
            for (int LLq = nktp; LLq <= nbtm; ++LLq) {
                const int L = nbtm - LLq + nktp;
                if (LLq >= P3(KTOP, NC, J, I) && LLq <= P3(KBTM, NC, J, I) && bitx) {
                    const float prs1 = P8(L, I) * 0.01f;
                    const float prs2 = P8(L + 1, I) * 0.01f;
                    const float delp = prs2 - prs1;
                    float ctau = 0.0f;
                    if (P3(CCMID, L, J, I) >= RRTM_CLFRMIN) {
                        if (P3(T, L, J, I) >= RRTM_TRAD_ICE) ctau = RRTM_CTAUCW;   /* convective water */
                        else ctau = RRTM_CTAUCI;                                   /* convective ice */
                    }
                    ctau = ctau + RRTM_ABSCOEF_W * P3(QC, L, J, I) + RRTM_ABSCOEF_I * P3(QS, L, J, I);
                    P3(TAUTOTAL, L, J, I) = ctau * delp;
                    /* CLDFRA(I,J,L) = MAX(CCMID(I,J,LL), CSMID(I,J,LL)) (module_RA_RRTM.F90:824):
                       read at the Fortran LL (loop variable), write at the mapped L. */
                    P3(CLDFRA, L, J, I) = fmaxf(P3(CCMID, LLq, J, I), P3(CSMID, LLq, J, I));
                    tauc = tauc + delp * ctau;
                }
            }
            (void)tauc;
        }
    }
}
