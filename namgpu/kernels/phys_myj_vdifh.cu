// WOOF-NAM unit phys-myj-vdifh: VDIFH, vertical diffusion of the mass variables (heat, water
// vapour, cloud condensate) in the NMMB MYJ PBL scheme. The momentum sibling is VDIFV (phys-myj-vdifv).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Module
// header kept as the notice:
//   MODULE MODULE_BL_MYJPBL (NMMB nam.v4.2.11, 83df4e13c1), "THE MYJ PBL SCHEME".
// Ported to CUDA for WOOF-NAM from phys/module_BL_MYJPBL.F90:552-625 (caller column prep and
// tendencies, MYJPBL main_integration) and 1425-1522 (SUBROUTINE VDIFH), nam.v4.2.11, 83df4e13c1.
//
// Three kernels, one launch sequence (wrapper namgpu/physics/myj/phys_myj_vdifh.py):
//   1. phys_myj_vdifh_prep_column  caller column prep, Fortran 552-605 (copy-in, ZHK, RHOK, AKHK,
//                                  THSK and THZ0 blend, AKHS_DENS, QSFC sea/land branch, QZ0).
//   2. phys_myj_vdifh_column       VDIFH, Fortran 1425-1522.
//   3. phys_myj_vdifh_tend_column  primary-variable tendencies, Fortran 619-622.
//
// Layout (CONTRACT.md sections 1 and 3). The Fortran dummy arrays are one column of the caller's
// arrays, so every array here is 3-D [level, j, i] with level 1 = model top (Z(K)-Z(K+1) > 0):
//   th, q, cwm (Fortran TH, Q, CWM = caller TH, Q, QC)      [LM,   JM, IM]
//   t, pmid (Fortran T, PMID)                                [LM,   JM, IM]
//   pinh (Fortran PINH, LM+1 levels), zint (Fortran ZINT)    [LM+1, JM, IM]
//   akh (Fortran AKH, from phys-myj-difcof)                  [LM,   JM, IM]
//   thk, qk, cwmk (Fortran THK, QK, CWMK, scratch)           [LM,   JM, IM]
//   zhk (Fortran ZHK, scratch)                               [LM+1, JM, IM]
//   rhok (Fortran RHOK, scratch), akhk (Fortran AKHK, scratch) [LM, JM, IM]
//   2-D [JM, IM]: tsk (TSK = TSFC), xland, sice, snow, chklowq, elflx, akhs, akhs_dens (scratch),
//   thz0, qz0, qsh (Fortran THZ0, QZ0, QSFC; inout, as in the Fortran).
//   Level LM of AKH and AKHK (Fortran RKH uses LM-1 levels) is not read.
//   dtdif (= DTTURBL), lmh (= LM), im, jm, lm: scalars.
//
// Fortran-to-CUDA mapping of the caller prep (552-605):
//   DO K=LM,1,-1: THK, QK, CWMK, ZHK, RHOK          -> prep, first loop
//   DO K=1,LM-1: AKHK                               -> prep
//   ZHK(LM+1)=ZINT(LM+1)                            -> prep
//   SEAMASK, THSK=TSK*(1E5/PSFC)**CAPPA, THZ0 blend -> prep (THSK as main_integration line 536)
//   AKHS_DENS=AKHS*RHOK(LM)                         -> prep
//   QFC1 / QSFC land branch, PSFC/QSFC sea branch   -> prep (ELWV, RLIVWV, PQ0SEA, A2, A3, A4)
//   QZ0=(1-SEAMASK)*QSFC+SEAMASK*QZ0                -> prep
//   LMH=LM                                          -> the wrapper passes lmh = LM
// Fortran-to-CUDA mapping of VDIFH (1425-1522):
//   CTHF = 0.5*CT                                  -> cthf
//   DO K=1,LMH-1: DTOZ, CR, RKCT                   -> DTOZ_(k), CR_(k), RKCT_(k) (macros below)
//   CM(1), RST(1), RSQ(1), RSC(1)                  -> lines marked FORWARD 1
//   DO K=2,LMH-1: CF, CM, RST, RSQ, RSC            -> forward loop
//   DTOZS, RKHH, CF, RKQS, CMB, CMTB, CMQB, CMCB,
//   RSTB, RSQB, RSCB                               -> bottom boundary block
//   TH(LMH), Q(LMH), CWM(LMH)                      -> bottom boundary block
//   DO K=LMH-1,1,-1: RCML, TH, Q, CWM              -> back substitution
// DTOZ(K), CR(K) and RKCT(K) are not stored: each is recomputed at its use from RKH and Z with
// the same float operations in the same order as the Fortran assignment, so the values are the
// same and the local-memory footprint stays at four arrays of LM entries. The RKCT term is
// the RKCT(K) of the Fortran (CT enters only through CTHF).
// Tendencies (619-622): RTHBLTEN, RQBLTEN, RQCBLTEN = (THK-TH, QK-Q, CWMK-CWM)*RDTTURBL, with
// the TH, Q, CWM of the caller before VDIFH (the VDIFH kernel does not touch them).
// Nothing here is MPI, halo, OpenMP or print. The MYJPBL PRINT_DIAG blocks are not numerics.
//
// Precision: float32 (kfpt single). Fast numerics: fmad contraction on, IEEE division as NVRTC
// default. One thread per (i, j) column, the l-loops run inside the thread (CONTRACT section 3).
// Refusals: the wrapper refuses lm > VDIFH_LMAX and lm < 2 before launch. The kernels keep a
// guard that returns without writing, which the wrapper makes unreachable.

#include "nmmb_common.cuh"

#define VDIFH_LMAX 64
#define VDIFH_SEAFC   0.98f                                    // module_BL_MYJPBL SEAFC
#define VDIFH_PQ0SEA  (NMMB_PQ0 * VDIFH_SEAFC)                 // PQ0SEA=PQ0*SEAFC
#define VDIFH_RLIVWV  (NMMB_ELIV / NMMB_ELWV)                  // RLIVWV=XLS/XLV

// Fortran X(K) for the current column, K 1-based (level 1 = model top).
#define KAT(a, k) a[IDX3(i0, j0, (k) - 1, im, jm)]

// Recomputed Fortran temporaries (DTOZ, CR, RKCT) for level k, with the Fortran operation order.
#define DTOZ_(k)  (dtdif / (KAT(z, k) - KAT(z, (k) + 1)))
#define CR_(k)    (-DTOZ_(k) * KAT(rkh, k))
#define RKCT_(k)  (KAT(rkh, k) * (KAT(z, k) - KAT(z, (k) + 2)) * cthf)

// ---------------------------------------------------------------------------------------------
// 1. Caller column prep, Fortran module_BL_MYJPBL.F90:552-605.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_myj_vdifh_prep_column(
    const float* __restrict__ th_g,       // TH   (caller)
    const float* __restrict__ q_g,        // Q    (caller)
    const float* __restrict__ cwm_g,      // CWM  (caller QC)
    const float* __restrict__ t_g,        // T    (caller)
    const float* __restrict__ pmid_g,     // PMID (caller PHMID)
    const float* __restrict__ pinh_g,     // PINH (caller PHINT), LM+1 levels
    const float* __restrict__ zint_g,     // ZINT (caller MYJ_ZINT), LM+1 levels
    const float* __restrict__ akh_g,      // AKH  (phys-myj-difcof MYJ_AKH), levels 1..LM-1 used
    const float* __restrict__ tsk_g,      // TSK  (caller TSFC)
    const float* __restrict__ xland_g,    // XLAND
    const float* __restrict__ sice_g,     // SICE
    const float* __restrict__ snow_g,     // SNOW
    const float* __restrict__ chklowq_g,  // CHKLOWQ
    const float* __restrict__ elflx_g,    // ELFLX
    const float* __restrict__ akhs_g,     // AKHS
    float* __restrict__ thk,              // THK   (scratch, out)
    float* __restrict__ qk,               // QK    (scratch, out)
    float* __restrict__ cwmk,             // CWMK  (scratch, out)
    float* __restrict__ zhk,              // ZHK   (scratch, out), LM+1 levels
    float* __restrict__ rhok,             // RHOK  (scratch, out)
    float* __restrict__ akhk,             // AKHK  (scratch, out), levels 1..LM-1
    float* __restrict__ akhs_dens_g,      // AKHS_DENS (scratch, out, 2-D)
    float* __restrict__ thz0_g,           // THZ0  (inout, 2-D)
    float* __restrict__ qz0_g,            // QZ0   (inout, 2-D)
    float* __restrict__ qsh_g,            // QSFC  (inout, 2-D)
    int im, int jm, int lm)
{
    COLUMN_IJ(im, jm)
    if (lm > VDIFH_LMAX || lm < 2) return;

    const size_t c2 = IDX2(i0, j0, im);
#define P3(a, k) a[IDX3(i0, j0, (k) - 1, im, jm)]

    // DO K=LM,1,-1: THK, QK, CWMK, ZHK, RHOK
    for (int k = lm; k >= 1; --k) {
        P3(thk, k)  = P3(th_g, k);
        P3(qk, k)   = P3(q_g, k);
        P3(cwmk, k) = P3(cwm_g, k);
        P3(zhk, k)  = P3(zint_g, k);
        P3(rhok, k) = P3(pmid_g, k) / (NMMB_R_D * P3(t_g, k) * (1.0f + NMMB_P608 * P3(qk, k) - P3(cwmk, k)));
    }
    // AKHK(K)=AKH(K)*0.5*(RHOK(K)+RHOK(K+1)), K=1..LM-1
    for (int k = 1; k <= lm - 1; ++k) {
        P3(akhk, k) = P3(akh_g, k) * 0.5f * (P3(rhok, k) + P3(rhok, k + 1));
    }
    P3(zhk, lm + 1) = P3(zint_g, lm + 1);                  // ZHK(LM+1)=ZINT(LM+1)

    // SEAMASK and THSK (main_integration 536: THSK=TSK*(1E5/PSFC)**CAPPA, PSFC=PINH(LM+1))
    const float seamask = xland_g[c2] - 1.0f;
    const float psfc_s = P3(pinh_g, lm + 1);
    const float thsk = tsk_g[c2] * powf(1.0e5f / psfc_s, NMMB_CAPPA);
    thz0_g[c2] = (1.0f - seamask) * thsk + seamask * thz0_g[c2];   // THZ0 blend (578)

    // AKHS_DENS=AKHS*RHOK(LM)
    const float akhs_dens = akhs_g[c2] * P3(rhok, lm);
    akhs_dens_g[c2] = akhs_dens;

    // QSFC: land branch (583-595) or sea branch (596-600)
    if (seamask < 0.5f) {
        float qfc1 = NMMB_ELWV * chklowq_g[c2] * akhs_dens;     // XLV*CHKLOWQ*AKHS_DENS
        if (snow_g[c2] > 0.0f || sice_g[c2] > 0.5f) {
            qfc1 = qfc1 * VDIFH_RLIVWV;
        }
        if (qfc1 > 0.0f) {
            const float qlow = P3(qk, lm);                      // QLOW=QK(LLOW), LLOW=LM
            qsh_g[c2] = qlow + elflx_g[c2] / qfc1;
        }
    } else {
        const float rxnsfc = powf(1.0e5f / psfc_s, NMMB_CAPPA);
        qsh_g[c2] = VDIFH_PQ0SEA / psfc_s
                    * expf(NMMB_A2 * (thsk - NMMB_A3 * rxnsfc) / (thsk - NMMB_A4 * rxnsfc));
    }

    // QZ0=(1-SEAMASK)*QSFC+SEAMASK*QZ0 (603)
    qz0_g[c2] = (1.0f - seamask) * qsh_g[c2] + seamask * qz0_g[c2];
#undef P3
}

// ---------------------------------------------------------------------------------------------
// 2. VDIFH, Fortran 1425-1522.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_myj_vdifh_column(
    float* __restrict__ th,              // inout TH   (THK)
    float* __restrict__ q,               // inout Q    (QK)
    float* __restrict__ cwm,             // inout CWM  (CWMK)
    const float* __restrict__ rkh,       // RKH        (AKHK), levels 1..LM-1
    const float* __restrict__ rho,       // RHO        (RHOK)
    const float* __restrict__ z,         // Z          (ZHK), LM+1 interfaces
    const float* __restrict__ thz0,      // THZ0(I,J)
    const float* __restrict__ qz0,       // QZ0(I,J)
    const float* __restrict__ rkhs,      // RKHS       (AKHS_DENS(I,J))
    const float* __restrict__ chklowq,   // CHKLOWQ(I,J)
    const float* __restrict__ ct,        // CT(I,J)
    float dtdif, int lmh, int im, int jm, int lm)
{
    COLUMN_IJ(im, jm)
    if (lm > VDIFH_LMAX || lm < 2) return;

    const size_t c2 = IDX2(i0, j0, im);
    const float cthf = 0.5f * ct[c2];                       // CTHF=0.5*CT

    float cm[VDIFH_LMAX + 1];
    float rst[VDIFH_LMAX + 1];
    float rsq[VDIFH_LMAX + 1];
    float rsc[VDIFH_LMAX + 1];

    // FORWARD 1: CM(1), RST(1), RSQ(1), RSC(1)
    cm[1]  = DTOZ_(1) * KAT(rkh, 1) + KAT(rho, 1);
    rst[1] = KAT(th, 1) * KAT(rho, 1) - RKCT_(1) * DTOZ_(1);
    rsq[1] = KAT(q, 1) * KAT(rho, 1);
    rsc[1] = KAT(cwm, 1) * KAT(rho, 1);

    // FORWARD 2: DO K=2,LMH-1
    for (int k = 2; k <= lmh - 1; ++k) {
        float dtozl = DTOZ_(k);
        float cf = -dtozl * KAT(rkh, k - 1) / cm[k - 1];
        cm[k]  = -CR_(k - 1) * cf + (KAT(rkh, k - 1) + KAT(rkh, k)) * dtozl + KAT(rho, k);
        rst[k] = -rst[k - 1] * cf + (RKCT_(k - 1) - RKCT_(k)) * dtozl + KAT(th, k) * KAT(rho, k);
        rsq[k] = -rsq[k - 1] * cf + KAT(q, k) * KAT(rho, k);
        rsc[k] = -rsc[k - 1] * cf + KAT(cwm, k) * KAT(rho, k);
    }

    // BOTTOM BOUNDARY (LMH)
    const float dtozs = dtdif / (KAT(z, lmh) - KAT(z, lmh + 1));
    const float rkhh  = KAT(rkh, lmh - 1);
    const float cf    = -dtozs * rkhh / cm[lmh - 1];
    const float rkhs_c = rkhs[c2];
    const float rkqs  = rkhs_c * chklowq[c2];
    const float cmb   = CR_(lmh - 1) * cf;
    const float cmtb  = -cmb + (rkhh + rkhs_c) * dtozs + KAT(rho, lmh);
    const float cmqb  = -cmb + (rkhh + rkqs) * dtozs + KAT(rho, lmh);
    const float cmcb  = -cmb + (rkhh) * dtozs + KAT(rho, lmh);
    const float rstb  = -rst[lmh - 1] * cf + RKCT_(lmh - 1) * dtozs + KAT(th, lmh) * KAT(rho, lmh);
    const float rsqb  = -rsq[lmh - 1] * cf + KAT(q, lmh) * KAT(rho, lmh);
    const float rscb  = -rsc[lmh - 1] * cf + KAT(cwm, lmh) * KAT(rho, lmh);
    KAT(th, lmh)  = (dtozs * rkhs_c * thz0[c2] + rstb) / cmtb;
    KAT(q, lmh)   = (dtozs * rkqs * qz0[c2] + rsqb) / cmqb;
    KAT(cwm, lmh) = (rscb) / cmcb;

    // BACK SUBSTITUTION: DO K=LMH-1,1,-1 (RCML = 1/CM(K), then multiply, as the Fortran)
    for (int k = lmh - 1; k >= 1; --k) {
        const float rcml = 1.0f / cm[k];
        KAT(th, k)  = (-CR_(k) * KAT(th, k + 1) + rst[k]) * rcml;
        KAT(q, k)   = (-CR_(k) * KAT(q, k + 1) + rsq[k]) * rcml;
        KAT(cwm, k) = (-CR_(k) * KAT(cwm, k + 1) + rsc[k]) * rcml;
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Primary-variable tendencies, Fortran 619-622. TH, Q, CWM are the caller values before VDIFH.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_myj_vdifh_tend_column(
    const float* __restrict__ th_g,      // TH   (caller, before VDIFH)
    const float* __restrict__ q_g,       // Q    (caller)
    const float* __restrict__ cwm_g,     // CWM  (caller QC)
    const float* __restrict__ thk,       // THK  after VDIFH
    const float* __restrict__ qk,        // QK   after VDIFH
    const float* __restrict__ cwmk,      // CWMK after VDIFH
    float* __restrict__ rthblten_g,      // RTHBLTEN
    float* __restrict__ rqblten_g,       // RQBLTEN
    float* __restrict__ rqcblten_g,      // RQCBLTEN
    float dttb, int im, int jm, int lm)
{
    COLUMN_IJ(im, jm)
    if (lm > VDIFH_LMAX || lm < 2) return;
#define P3(a, k) a[IDX3(i0, j0, (k) - 1, im, jm)]
    const float rdtturbl = 1.0f / dttb;                    // RDTTURBL=1./DTTURBL
    for (int k = 1; k <= lm; ++k) {
        P3(rthblten_g, k) = (P3(thk, k) - P3(th_g, k)) * rdtturbl;
        P3(rqblten_g, k)  = (P3(qk, k) - P3(q_g, k)) * rdtturbl;
        P3(rqcblten_g, k) = (P3(cwmk, k) - P3(cwm_g, k)) * rdtturbl;
    }
#undef P3
}

#undef KAT
#undef DTOZ_
#undef CR_
#undef RKCT_
#undef VDIFH_LMAX
#undef VDIFH_SEAFC
#undef VDIFH_PQ0SEA
#undef VDIFH_RLIVWV
