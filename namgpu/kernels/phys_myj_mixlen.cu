// nam-gpu unit phys-myj-mixlen: MIXLEN, the level 2.5 master mixing length of the Mellor-Yamada-
// Janjic (MYJ) PBL scheme (NMMB module_BL_MYJPBL, nam.v4.2.11).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). The
// module header of module_BL_MYJPBL is kept as the notice:
//   MODULE MODULE_BL_MYJPBL, "THE MYJ PBL SCHEME" (NMMB, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:758-965 (nam.v4.2.11, 83df4e13c1).
// Constants: module_BL_MYJPBL.F90:36-118 (PARAMETER block), evaluated in double and rounded to
// float here (Fortran single-precision evaluation may differ by one ulp; not claimed bitwise).
//
// Structure: one thread per column (grid_cols). The column is NAM top-down: Fortran level K = 1
// is the model top and K = LMH the lowest mass level, so Z(K) decreases with K. Every Fortran
// 1-based index K is read as [K-1] (F1/F3 macros below). Per-column scratch Q1, ELM and REL are
// per-thread arrays of MIXLEN_KMAX entries (Fortran dimensions 1:LM and 1:LM-1).
//
// Inputs (CuPy float32, CONTRACT section 1):
//   U, V      [LM, JM, IM]    wind (Fortran U(K), V(K))
//   THV, THE  [LM, JM, IM]    virtual and equivalent potential temperature
//   Q2        [LM, JM, IM]    2*tke
//   EPSQ2     [LM]            floor of 2*tke, per level (Fortran EPSQ2(1:LM))
//   EPSL      [LM]            floor mixing length (Fortran EPSL(1:LM-1); entries 1..LM-1 read)
//   ZINT      [LM+1, JM, IM]  interface heights Z(1:LM+1); Z(LM+1) is the ground
//   P         [LM, JM, IM]    mid-level pressure (Fortran P(K))
//   PSK       [LM, JM, IM]    per-level pressure argument PS(K) of MIXLEN (MYJPBL PSK)
//   LMH       [JM, IM] int32  lowest mass level above ground
//   RSQDT     scalar          SQRT(1 / (DT*NPHS)), Fortran RSQDT (MYJPBL:307-309)
// Outputs:
//   GM, GH, EL  [LM-1, JM, IM]  Fortran GM(K), GH(K), EL(K). Entries outside the range the
//                               Fortran writes are zeroed first (Fortran leaves them undefined).
//   PBLH, MIXHT [JM, IM]        PBL height and mixed-layer height
//   LPBL, LMXL  [JM, IM] int32  PBL top level and mixing-length top level
//   CT          [JM, IM]        INOUT, set to 0 (Fortran CT=0.)
//
// RXNER (a Fortran argument) is not read by MIXLEN's body and is not passed.
// No host sync, no printf, no device allocation.

#include "nmmb_common.cuh"

#define MIXLEN_KMAX 128

// module_BL_MYJPBL.F90:36-118, single-precision values of the derived PARAMETER constants.
#define MYJ_VKARMAN 0.4f
#define MYJ_EPSRU   1.0e-7f
#define MYJ_FH      1.10f
#define MYJ_ALPH    0.30f
#define MYJ_EL0MAX  1000.0f
#define MYJ_EL0MIN  1.0f
#define MYJ_ELFC    1.0f
#define MYJ_EPSGH   1.0e-9f
#define MYJ_EPSGM   7.118342026934954e-11f   // REQU*EPSGH (:100)
#define MYJ_REQU    0.07118342071771622f     // -AEQH/AEQM (:99)
#define MYJ_ADNH    0.09802711009979248f     // :67
#define MYJ_ADNM    0.9726041555404663f      // :68
#define MYJ_BDNH    0.8392161130905151f      // :71
#define MYJ_BDNM    2.6127171516418457f      // :72
#define MYJ_AUBH    0.02958165667951107f     // :113
#define MYJ_AUBM    -0.41556957364082336f    // :114
#define MYJ_BUBH    0.2907695472240448f     // :115
#define MYJ_BUBM    -1.1187937259674072f     // :116
#define MYJ_CUBR    0.5692963600158691f      // :117 (1 - 3*UBRY)
#define MYJ_RCUBR   1.7565544843673706f      // :118 (1/CUBR)
#define MYJ_CUBRY   0.21535183489322662f     // CUBRY=UBRY*1.5 (:820), UBRY=0.14356788992881775

// 1-based Fortran indexing for this thread's column (i0, j0 set by COLUMN_IJ).
#define F3(arr, K)    ((arr)[IDX3(i0, j0, (K) - 1, IM, JM)])
#define F1(arr, K)    ((arr)[(K) - 1])
#define LOCAL(arr, K) ((arr)[(K) - 1])

extern "C" __global__ void phys_myj_mixlen(
    const float* __restrict__ U, const float* __restrict__ V,
    const float* __restrict__ THV, const float* __restrict__ THE,
    const float* __restrict__ Q2,
    const float* __restrict__ EPSQ2, const float* __restrict__ EPSL,
    const float* __restrict__ ZINT, const float* __restrict__ P, const float* __restrict__ PSK,
    const int* __restrict__ LMH,
    float rsqdt,
    float* __restrict__ GM, float* __restrict__ GH, float* __restrict__ EL,
    float* __restrict__ PBLH, int* __restrict__ LPBL, int* __restrict__ LMXL,
    float* __restrict__ CT, float* __restrict__ MIXHT,
    int IM, int JM, int LM)
{
    COLUMN_IJ(IM, JM)

    const int lmh = LMH[IDX2(i0, j0, IM)];
    float q1[MIXLEN_KMAX];
    float elm[MIXLEN_KMAX];
    float rel[MIXLEN_KMAX];

    // Zero the outputs the Fortran leaves undefined (GM, GH, EL beyond the written ranges).
    for (int k = 1; k <= LM - 1; ++k) {
        F3(GM, k) = 0.0f;
        F3(GH, k) = 0.0f;
        F3(EL, k) = 0.0f;
    }

    // Guard: LMH < 2 has no mixing layer (Fortran would read ELM(0)); LMH > LM would index past the
    // column. Write zero heights and return instead of reading out of bounds.
    if (lmh < 2 || lmh > LM) {
        LPBL[IDX2(i0, j0, IM)] = 1;
        LMXL[IDX2(i0, j0, IM)] = 1;
        PBLH[IDX2(i0, j0, IM)] = 0.0f;
        MIXHT[IDX2(i0, j0, IM)] = 0.0f;
        CT[IDX2(i0, j0, IM)] = 0.0f;
        return;
    }

    // ---------------- FIND THE HEIGHT OF THE PBL (:796-812) ----------------
    int lpbl = 1;                                   // LPBL=1 unless the loop exits early
    for (int k = lmh - 1; k >= 1; --k) {
        if (F3(Q2, k) - F1(EPSQ2, k) + F1(EPSQ2, LM) <= F1(EPSQ2, LM) * MYJ_FH) {
            lpbl = k;
            break;
        }
    }
    LPBL[IDX2(i0, j0, IM)] = lpbl;

    // THE HEIGHT OF THE PBL (:814): PBLH = Z(LPBL+1) - Z(LMH+1)
    PBLH[IDX2(i0, j0, IM)] = F3(ZINT, lpbl + 1) - F3(ZINT, lmh + 1);

    // Q1 = 0 for K = 1..LMH (:816-818)
    for (int k = 1; k <= lmh; ++k) LOCAL(q1, k) = 0.0f;

    // ---------------- GM and GH (:822-870) ----------------
    for (int k = 1; k <= lmh - 1; ++k) {
        float dz = (F3(ZINT, k) - F3(ZINT, k + 2)) * 0.5f;   // DZ=(Z(K)-Z(K+2))*0.5
        float rdz = 1.0f / dz;
        float du = F3(U, k) - F3(U, k + 1);
        float dv = F3(V, k) - F3(V, k + 1);
        float gml = (du * du + dv * dv) * rdz * rdz;
        F3(GM, k) = fmaxf(gml, MYJ_EPSGM);

        float dthv = F3(THV, k) - F3(THV, k + 1);
        if (dthv > 0.0f) {
            if (F3(THE, k + 1) > F3(THE, k)) {
                if (F3(PSK, k + 1) > F3(P, k)) {                 // !>12KM
                    float wcon = (F3(P, k + 1) - F3(PSK, k + 1)) / (F3(P, k + 1) - F3(P, k));
                    float t = dz * wcon * rsqdt;
                    if ((F3(Q2, k) > F1(EPSQ2, k))
                        && (F3(Q2, k) * MYJ_CUBRY > t * t)) {
                        dthv = (F3(THE, k) - F3(THE, k + 1)) + dthv;
                    }
                }
            }
        }
        float ghl = dthv * rdz;
        if (fabsf(ghl) <= MYJ_EPSGH) ghl = MYJ_EPSGH;
        F3(GH, k) = ghl;
    }

    // CT = 0. (:872)
    CT[IDX2(i0, j0, IM)] = 0.0f;

    // ---------------- MAXIMUM MIXING LENGTHS AND PBL TOP (:874-906) ----------------
    int lmxl = lmh;
    for (int k = 1; k <= lmh - 1; ++k) {
        float gml = F3(GM, k);
        float ghl = F3(GH, k);
        if (ghl >= MYJ_EPSGH) {
            if (gml / ghl <= MYJ_REQU) {
                LOCAL(elm, k) = F1(EPSL, k);
                lmxl = k + 1;
            } else {
                float aubr = (MYJ_AUBM * gml + MYJ_AUBH * ghl) * ghl;
                float bubr = MYJ_BUBM * gml + MYJ_BUBH * ghl;
                float qol2st = (-0.5f * bubr
                                + sqrtf(bubr * bubr * 0.25f - aubr * MYJ_CUBR)) * MYJ_RCUBR;
                float eloq2x = 1.0f / fmaxf(MYJ_EPSGH, qol2st);
                LOCAL(elm, k) = fmaxf(sqrtf(eloq2x * F3(Q2, k)), F1(EPSL, k));
            }
        } else {
            float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
            float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
            float qol2un = -0.5f * bden + sqrtf(bden * bden * 0.25f - aden);
            float eloq2x = 1.0f / (qol2un + MYJ_EPSRU);
            LOCAL(elm, k) = fmaxf(sqrtf(eloq2x * F3(Q2, k)), F1(EPSL, k));
        }
    }
    if (LOCAL(elm, lmh - 1) == F1(EPSL, lmh - 1)) lmxl = lmh;        // :904
    LMXL[IDX2(i0, j0, IM)] = lmxl;

    // THE HEIGHT OF THE MIXED LAYER (:908-911): BLMX = Z(LMXL) - Z(LMH+1)
    MIXHT[IDX2(i0, j0, IM)] = F3(ZINT, lmxl) - F3(ZINT, lmh + 1);

    // Q1 = SQRT(Q2) for K = LPBL..LMH (:913-915)
    for (int k = lpbl; k <= lmh; ++k) LOCAL(q1, k) = sqrtf(F3(Q2, k));

    // ---------------- ASYMPTOTIC L IN BLACKADAR FORMULA (:917-934) ----------------
    float szq = 0.0f;
    float sq = 0.0f;
    for (int k = 1; k <= lmh - 1; ++k) {
        float qdzl = (LOCAL(q1, k) + LOCAL(q1, k + 1)) * (F3(ZINT, k + 1) - F3(ZINT, k + 2));
        szq = (F3(ZINT, k + 1) + F3(ZINT, k + 2) - F3(ZINT, lmh + 1) - F3(ZINT, lmh + 1)) * qdzl + szq;
        sq = qdzl + sq;
    }
    float el0 = fminf(MYJ_ALPH * szq * 0.5f / sq, MYJ_EL0MAX);
    el0 = fmaxf(el0, MYJ_EL0MIN);

    // ---------------- ABOVE THE PBL TOP (:938-944) ----------------
    int lpblm = lpbl - 1;
    if (lpblm < 1) lpblm = 1;                                        // MAX(LPBL-1, 1)
    for (int k = 1; k <= lpblm; ++k) {
        float elk = fminf((F3(ZINT, k) - F3(ZINT, k + 2)) * MYJ_ELFC, LOCAL(elm, k));
        F3(EL, k) = elk;
        LOCAL(rel, k) = elk / LOCAL(elm, k);
    }

    // ---------------- INSIDE THE PBL (:948-955) ----------------
    if (lpbl < lmh) {
        for (int k = lpbl; k <= lmh - 1; ++k) {
            float vkrmz = (F3(ZINT, k + 1) - F3(ZINT, lmh + 1)) * MYJ_VKARMAN;
            float elk = fminf(vkrmz / (vkrmz / el0 + 1.0f), LOCAL(elm, k));
            F3(EL, k) = elk;
            LOCAL(rel, k) = elk / LOCAL(elm, k);
        }
    }

    // ---------------- SMOOTH REL, K = LPBL+1 .. LMH-2 (:957-961) ----------------
    for (int k = lpbl + 1; k <= lmh - 2; ++k) {
        float srel = fminf(((LOCAL(rel, k - 1) + LOCAL(rel, k + 1)) * 0.5f + LOCAL(rel, k)) * 0.5f,
                           LOCAL(rel, k));
        F3(EL, k) = fmaxf(srel * LOCAL(elm, k), F1(EPSL, k));
    }
}
