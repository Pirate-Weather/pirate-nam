// nam-gpu unit phys-myj-vdifq: VDIFQ, vertical diffusion of Q2 (2 x TKE) in the MYJ PBL,
// one column per thread, one launch over the whole [IM, JM] grid.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   module module_BL_MYJPBL (NMMB, nam.v4.2.11), subroutine VDIFQ (phys/module_BL_MYJPBL.F90:1342-1420).
//   PARAMETERs ESQ = 5.0 and ELZ0 = 0. are module_BL_MYJPBL.F90 line 58 (inside the PARAMETER block).
//   The EPSQ2 floor is the caller's Q2(I,J,K)=MAX(Q2K(K),EPSQ2(K)) for K=1..LM (module_BL_MYJPBL.F90:515-517).
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:1342-1420 (nam.v4.2.11, 83df4e13c1).
//
// Apache-2.0 origin of the copied arithmetic: the expression shapes are adapted from
// third_party/arwen/core/kernels/myjpbl.cu (ArWen, Apache-2.0), its VDIFQ block. The ArWen kernel keeps
// the MYJ index downward from the top (zh[m] = Fortran ZHK(m+1)), which is the NAM convention too, so
// the block carries over with the names and the per-column loads changed. ArWen's Apache-2.0 header and
// LICENSE stay in the ArWen tree; this file keeps the attribution.
//
// Fortran -> CUDA line map (K is Fortran 1-based, m = K-1 zero-based; level 1 = model top):
//   1388      ESQHF = 0.5*ESQ                                   -> esqhf
//   1390-1394 DO K=1,LMH-2: DTOZ(K), AKQ(K), CR(K)              -> s1[m], s2[m], s3[m] (m = 0..LMH-3)
//   1396      CM(1) = DTOZ(1)*AKQ(1)+1., RSQ2(1) = Q2(1)        -> s4[0], rhok[0]
//   1398-1403 DO K=2,LMH-2: CF, CM(K), RSQ2(K)                  -> m = 1..LMH-3
//   1406-1407 DTOZS, AKQS (uses EL(LMH-1)+ELZ0)                  -> dtozs, akqs
//   1409-1410 Q2(LMH-1) (solve for the bottom layer)            -> q2c[lmh-2]
//   1412-1414 DO K=LMH-2,1,-1: Q2(K) back substitution          -> m = lmh-3 .. 0
//   515       Q2(I,J,K)=MAX(Q2K(K),EPSQ2(K)), K=1..LM            -> EPSQ2 floor, all LM levels (below)
// Q2(LMH) is read but not written by VDIFQ. The caller's floor still applies to it (K = LM when LMH = LM).
//
// Inputs and outputs (NAM names, CuPy float32, C order [l, j, i], l = 0 is the model top):
//   q2    [LM, JM, IM]    Q2 (inout). Levels 1..LM all written: solved levels 1..LMH-1, floored by EPSQ2.
//   el    [LM, JM, IM]    EL after PRODQ2 (EL_PBL). Fortran EL(1:LM-1) is at [K-1]; the top LM-1 levels are used.
//   zint  [LM+1, JM, IM]  ZINT = ZHK, interface heights, top-down. Fortran Z(K) is at [K-1]. ZINT(LM+1) = HT.
//   epsq2 [LM]            EPSQ2, the 2*TKE floor per level (rank 1, shared by every column).
// Precision: kfpt = single, so float. Fast numerics (fmad on, IEEE sqrt and divide).
// No host sync, no device malloc, no printf. The column is held in local arrays of VDIFQ_MAXLEV.

#include "nmmb_common.cuh"

#define VDIFQ_MAXLEV 72
#define VDIFQ_ESQ    5.0f      // module_BL_MYJPBL PARAMETER ESQ (line 58)
#define VDIFQ_ELZ0   0.0f      // module_BL_MYJPBL PARAMETER ELZ0 (line 58)

extern "C" __global__ void phys_myj_vdifq(
    const int im, const int jm, const int lm, const int lmh, const float dtdif,
    float* __restrict__ q2,
    const float* __restrict__ el,
    const float* __restrict__ zint,
    const float* __restrict__ epsq2)
{
    COLUMN_IJ(im, jm)
    // Fortran VDIFQ needs LMH >= 3 (CM(1), the K=2 loop and Q2(LMH-1) index below the column).
    // The NAM calls pass LMH = LM = 60 (module_BL_MYJPBL.F90:364 and :605).
    if (lmh < 3 || lmh > VDIFQ_MAXLEV || lmh > lm) return;

    const size_t sl = (size_t)im * (size_t)jm;      // level stride
    const size_t base = IDX3(i0, j0, 0, im, jm);    // level-0 base, same for every array here

    float q2c[VDIFQ_MAXLEV];
    float ec[VDIFQ_MAXLEV];
    float zh[VDIFQ_MAXLEV + 1];
    float s1[VDIFQ_MAXLEV];   // DTOZ
    float s2[VDIFQ_MAXLEV];   // AKQ
    float s3[VDIFQ_MAXLEV];   // CR
    float s4[VDIFQ_MAXLEV];   // CM
    float rhok[VDIFQ_MAXLEV]; // RSQ2

    for (int m = 0; m < lmh; ++m) {
        q2c[m] = q2[base + (size_t)m * sl];
        zh[m] = zint[base + (size_t)m * sl];
    }
    zh[lmh] = zint[base + (size_t)lmh * sl];        // ZINT(LMH+1), the ground interface (HT)
    for (int m = 0; m < lmh - 1; ++m) {
        ec[m] = el[base + (size_t)m * sl];          // EL(1:LMH-1)
    }

    // ---- 1388: ESQHF ----
    const float esqhf = 0.5f * VDIFQ_ESQ;

    // ---- 1390-1394: first pass, K = 1..LMH-2 ----
    const int nq = lmh - 2;
    for (int m = 0; m < nq; ++m) {
        s1[m] = (dtdif + dtdif) / (zh[m] - zh[m + 2]);
        s2[m] = sqrtf((q2c[m] + q2c[m + 1]) * 0.5f) * (ec[m] + ec[m + 1]) * esqhf
                / (zh[m + 1] - zh[m + 2]);
        s3[m] = -s1[m] * s2[m];
    }

    // ---- 1396: CM(1), RSQ2(1) ----
    s4[0] = s1[0] * s2[0] + 1.0f;
    rhok[0] = q2c[0];

    // ---- 1398-1403: K = 2..LMH-2 ----
    for (int m = 1; m < nq; ++m) {
        const float cf = -s1[m] * s2[m - 1] / s4[m - 1];
        s4[m] = -s3[m - 1] * cf + (s2[m - 1] + s2[m]) * s1[m] + 1.0f;
        rhok[m] = -rhok[m - 1] * cf + q2c[m];
    }

    // ---- 1406-1410: bottom layer, Q2(LMH-1) ----
    const float dtozs = (dtdif + dtdif) / (zh[lmh - 2] - zh[lmh]);
    const float akqs = sqrtf((q2c[lmh - 2] + q2c[lmh - 1]) * 0.5f)
                       * (ec[lmh - 2] + VDIFQ_ELZ0) * esqhf
                       / (zh[lmh - 1] - zh[lmh]);
    const float cf_b = -dtozs * s2[nq - 1] / s4[nq - 1];
    q2c[lmh - 2] = (dtozs * akqs * q2c[lmh - 1] - rhok[nq - 1] * cf_b + q2c[lmh - 2])
                   / ((s2[nq - 1] + akqs) * dtozs - s3[nq - 1] * cf_b + 1.0f);

    // ---- 1412-1414: back substitution, K = LMH-2 .. 1 ----
    for (int m = nq - 1; m >= 0; --m) {
        q2c[m] = (-s3[m] * q2c[m + 1] + rhok[m]) / s4[m];
    }

    // ---- 515: EPSQ2 floor, Q2(K)=MAX(Q2K(K),EPSQ2(K)) for every level K=1..LM ----
    // Levels 1..LMH-1 (m = 0..lmh-2) take the solved values. Levels LMH..LM (m = lmh-1..lm-1) are not
    // changed by VDIFQ, so the floor reads their global value. Together the two loops floor all LM levels.
    for (int m = 0; m < lmh - 1; ++m) {
        q2[base + (size_t)m * sl] = fmaxf(q2c[m], epsq2[m]);
    }
    for (int m = lmh - 1; m < lm; ++m) {
        q2[base + (size_t)m * sl] = fmaxf(q2[base + (size_t)m * sl], epsq2[m]);
    }
}
