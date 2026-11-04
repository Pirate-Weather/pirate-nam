// phys_myj_myjpbl.cu -- NAM MYJ PBL column driver (MYJPBL) and its column subroutines, one thread per column.
//
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:180-753 (MYJPBL) and the subroutines it calls
// per column: MIXLEN 758-965, PRODQ2 969-1173, DIFCOF 1178-1337, VDIFQ 1342-1420, VDIFH 1425-1522,
// VDIFV 1527-1607 (nam.v4.2.11, MatthewPyle-NOAA/NAM @ 83df4e13c115d197db67b972cb3befec8511291a).
// The NAM source file carries no author header block; the origin is the file named above.
// Units phys-myj-mixlen/prodq2/difcof/vdifq/vdifh/vdifv own their own kernel files; the device
// copies below are transcribed from the same lines so this unit's kernel is self-contained.
//
// Source is NAM (top-down): level K=1 is the model top, K=LM the lowest layer (CONTRACT section 1).
// Fortran arrays are 1-based; the device arrays below keep that by using index 0 unused, so each
// statement reads like its Fortran line. Global 3-D fields are CuPy [LM, JM, IM] float32 and a
// Fortran X(I,J,K) is element ((K-1)*JM*IM + (J-1)*IM + (I-1)), i.e. (K-1)*NCOL + col.
//
// Not in this kernel, by reason:
//  * The I/J loops are the one-thread-per-column launch (grid_cols).
//  * The MPI/OpenMP directives (ZJ OMP lines) are dropped (single card, CONTRACT section 1).
//  * The debugging WRITE blocks (PRINT_DIAG) are not part of the numerics and are not ported.
//  * DIFCOF LMXL argument and its commented-out K-inversion block (Fortran 1260-1321) are inactive
//    in the source; LMXL is passed and not read, as in the Fortran.
//  * RSTDH and ZSF (Fortran ~720-745) feed only DCOL, which the source sets to 0 (line ~720, ZJ).
//    With DCOL zero the term D(K)*RKMS in VDIFV is zero, so RSTDH/ZSF are not computed.
//  * Z0 is in the MYJPBL argument list but has no reference in its body; it is not an argument here.
//
// Constants: module_CONSTANTS values come from nmmb_common.cuh (NMMB_*). MYJPBL's own PARAMETER
// block is evaluated in double and rounded once to float (CONTRACT: fast numerics, FMA on).

#include "nmmb_common.cuh"

#define MYJ_MAXL   128        // largest LM the local arrays hold; the wrapper refuses larger
#define MYJ_ITBL   401
#define MYJ_JTBL   1201
#define MYJ_THL    210.0f      // lower bound of potential temperature range (THL)

// ---- MYJPBL PARAMETER block (module_BL_MYJPBL.F90:44-110), double then float --------------------
#define MYJ_BETA_D  (1.0/273.0)
#define MYJ_A1_D    0.659888514560862645
#define MYJ_A2X_D   0.6574209922667784586
#define MYJ_B1_D    11.87799326209552761
#define MYJ_B2_D    7.226971804046074028
#define MYJ_C1_D    0.000830955950095854396
#define MYJ_BTG_D   (MYJ_BETA_D*9.8060226)
#define MYJ_ADNH_D  (9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*(12.*MYJ_A1_D+3.*MYJ_B2_D)*MYJ_BTG_D*MYJ_BTG_D)
#define MYJ_ADNM_D  (18.*MYJ_A1_D*MYJ_A1_D*MYJ_A2X_D*(MYJ_B2_D-3.*MYJ_A2X_D)*MYJ_BTG_D)
#define MYJ_ANMH_D  (-9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*MYJ_BTG_D*MYJ_BTG_D)
#define MYJ_ANMM_D  (-3.*MYJ_A1_D*MYJ_A2X_D*(3.*MYJ_A2X_D+3.*MYJ_B2_D*MYJ_C1_D+18.*MYJ_A1_D*MYJ_C1_D-MYJ_B2_D)*MYJ_BTG_D)
#define MYJ_BDNH_D  (3.*MYJ_A2X_D*(7.*MYJ_A1_D+MYJ_B2_D)*MYJ_BTG_D)
#define MYJ_BDNM_D  (6.*MYJ_A1_D*MYJ_A1_D)
#define MYJ_BEQH_D  (MYJ_A2X_D*MYJ_B1_D*MYJ_BTG_D+3.*MYJ_A2X_D*(7.*MYJ_A1_D+MYJ_B2_D)*MYJ_BTG_D)
#define MYJ_BEQM_D  (-MYJ_A1_D*MYJ_B1_D*(1.-3.*MYJ_C1_D)+6.*MYJ_A1_D*MYJ_A1_D)
#define MYJ_BNMH_D  (-MYJ_A2X_D*MYJ_BTG_D)
#define MYJ_BNMM_D  (MYJ_A1_D*(1.-3.*MYJ_C1_D))
#define MYJ_BSHH_D  (9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*MYJ_BTG_D)
#define MYJ_BSHM_D  (18.*MYJ_A1_D*MYJ_A1_D*MYJ_A2X_D*MYJ_C1_D)
#define MYJ_BSMH_D  (-3.*MYJ_A1_D*MYJ_A2X_D*(3.*MYJ_A2X_D+3.*MYJ_B2_D*MYJ_C1_D+12.*MYJ_A1_D*MYJ_C1_D-MYJ_B2_D)*MYJ_BTG_D)
#define MYJ_CESH_D  (MYJ_A2X_D)
#define MYJ_CESM_D  (MYJ_A1_D*(1.-3.*MYJ_C1_D))
#define MYJ_AEQH_D  (9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*MYJ_B1_D*MYJ_BTG_D*MYJ_BTG_D+9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*(12.*MYJ_A1_D+3.*MYJ_B2_D)*MYJ_BTG_D*MYJ_BTG_D)
#define MYJ_AEQM_D  (3.*MYJ_A1_D*MYJ_A2X_D*MYJ_B1_D*(3.*MYJ_A2X_D+3.*MYJ_B2_D*MYJ_C1_D+18.*MYJ_A1_D*MYJ_C1_D-MYJ_B2_D)*MYJ_BTG_D+18.*MYJ_A1_D*MYJ_A1_D*MYJ_A2X_D*(MYJ_B2_D-3.*MYJ_A2X_D)*MYJ_BTG_D)
#define MYJ_REQU_D  (-MYJ_AEQH_D/MYJ_AEQM_D)
#define MYJ_UBRYL_D ((18.*MYJ_REQU_D*MYJ_A1_D*MYJ_A1_D*MYJ_A2X_D*MYJ_B2_D*MYJ_C1_D*MYJ_BTG_D+9.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*MYJ_B2_D*MYJ_BTG_D*MYJ_BTG_D)/(MYJ_REQU_D*MYJ_ADNM_D+MYJ_ADNH_D))
#define MYJ_UBRY_D  ((1.+1.0e-7)*MYJ_UBRYL_D)
#define MYJ_UBRY3_D (3.*MYJ_UBRY_D)
#define MYJ_AUBH_D  (27.*MYJ_A1_D*MYJ_A2X_D*MYJ_A2X_D*MYJ_B2_D*MYJ_BTG_D*MYJ_BTG_D-MYJ_ADNH_D*MYJ_UBRY3_D)
#define MYJ_AUBM_D  (54.*MYJ_A1_D*MYJ_A1_D*MYJ_A2X_D*MYJ_B2_D*MYJ_C1_D*MYJ_BTG_D-MYJ_ADNM_D*MYJ_UBRY3_D)
#define MYJ_BUBH_D  ((9.*MYJ_A1_D*MYJ_A2X_D+3.*MYJ_A2X_D*MYJ_B2_D)*MYJ_BTG_D-MYJ_BDNH_D*MYJ_UBRY3_D)
#define MYJ_BUBM_D  (18.*MYJ_A1_D*MYJ_A1_D*MYJ_C1_D-MYJ_BDNM_D*MYJ_UBRY3_D)
#define MYJ_CUBR_D  (1.-MYJ_UBRY3_D)

#define MYJ_ADNH    ((float)(MYJ_ADNH_D))
#define MYJ_ADNM    ((float)(MYJ_ADNM_D))
#define MYJ_ANMH    ((float)(MYJ_ANMH_D))
#define MYJ_ANMM    ((float)(MYJ_ANMM_D))
#define MYJ_BDNH    ((float)(MYJ_BDNH_D))
#define MYJ_BDNM    ((float)(MYJ_BDNM_D))
#define MYJ_BEQH    ((float)(MYJ_BEQH_D))
#define MYJ_BEQM    ((float)(MYJ_BEQM_D))
#define MYJ_BNMH    ((float)(MYJ_BNMH_D))
#define MYJ_BNMM    ((float)(MYJ_BNMM_D))
#define MYJ_BSHH    ((float)(MYJ_BSHH_D))
#define MYJ_BSHM    ((float)(MYJ_BSHM_D))
#define MYJ_BSMH    ((float)(MYJ_BSMH_D))
#define MYJ_CESH    ((float)(MYJ_CESH_D))
#define MYJ_CESM    ((float)(MYJ_CESM_D))
#define MYJ_AEQH    ((float)(MYJ_AEQH_D))
#define MYJ_AEQM    ((float)(MYJ_AEQM_D))
#define MYJ_REQU    ((float)(MYJ_REQU_D))
#define MYJ_UBRY    ((float)(MYJ_UBRY_D))
#define MYJ_UBRY3   ((float)(MYJ_UBRY3_D))
#define MYJ_AUBH    ((float)(MYJ_AUBH_D))
#define MYJ_AUBM    ((float)(MYJ_AUBM_D))
#define MYJ_BUBH    ((float)(MYJ_BUBH_D))
#define MYJ_BUBM    ((float)(MYJ_BUBM_D))
#define MYJ_CUBR    ((float)(MYJ_CUBR_D))
#define MYJ_RCUBR   ((float)(1.0/MYJ_CUBR_D))
#define MYJ_B1      ((float)(MYJ_B1_D))
#define MYJ_RB1     ((float)(1.0/MYJ_B1_D))

#define MYJ_EPSGH   1.0e-9f
#define MYJ_EPSGM   (MYJ_REQU*MYJ_EPSGH)          // EPSGM=REQU*EPSGH
#define MYJ_EPSRU   1.0e-7f
#define MYJ_EPSTRB  1.0e-24f
#define MYJ_EPS1    1.0e-12f
#define MYJ_EPS2    0.0f
#define MYJ_VKARMAN 0.4f
#define MYJ_ALPH    0.30f
#define MYJ_EL0MAX  1000.0f
#define MYJ_EL0MIN  1.0f
#define MYJ_ELFC    1.0f
#define MYJ_FH      1.10f
#define MYJ_ELEVFC  0.6f
#define MYJ_ESQ     5.0f
#define MYJ_ESQHF   (0.5f*MYJ_ESQ)                // ESQHF=0.5*ESQ (module-level, MYJPBL:88)
#define MYJ_ELZ0    0.0f
#define MYJ_SEAFC   0.98f
#define MYJ_PQ0SEA  (NMMB_PQ0*MYJ_SEAFC)         // PQ0SEA=PQ0*SEAFC
#define MYJ_RLIVWV  (NMMB_ELIV/NMMB_ELWV)         // RLIVWV=XLS/XLV, XLS=ELIV, XLV=ELWV
#define MYJ_ELOCP   ((float)(2.72e6/1004.6))     // ELOCP=2.72E6/CP

// 1-based device array helpers ------------------------------------------------------------
// Each per-column local array is a[MYJ_MAXL+2]; a[0] is unused, a[1..n] are Fortran K=1..n.

// MIXLEN (Fortran 758-965). Z is ZHK (1..LM+1), P is PK, PS is PSK.
__device__ void myj_mixlen(const int lmh, const float rsqdt,
                           const float* U, const float* V, const float* THV, const float* THE,
                           const float* Q2, const float* EPSL, const float* EPSQ2, const float* Z,
                           const float* P, const float* PS,
                           float* GM, float* GH, float* EL, float* PBLH, int* LPBL, int* LMXL,
                           float* CT, float* MIXHT, const int lm)
{
    float Q1[MYJ_MAXL + 2], ELM[MYJ_MAXL + 2], REL[MYJ_MAXL + 2];
    const float cubry = MYJ_UBRY * 1.5f;             // CUBRY=UBRY*1.5

    // FIND THE HEIGHT OF THE PBL
    int lpbl = lmh;
    int found = 0;
    for (int k = lmh - 1; k >= 1; --k) {
        if (Q2[k] - EPSQ2[k] + EPSQ2[lm] <= EPSQ2[lm] * MYJ_FH) {
            lpbl = k;
            found = 1;
            break;
        }
    }
    if (!found) lpbl = 1;
    *LPBL = lpbl;
    // 110 PBLH=Z(LPBL+1)-Z(LMH+1)
    *PBLH = Z[lpbl + 1] - Z[lmh + 1];

    for (int k = 1; k <= lmh; k++) Q1[k] = 0.0f;

    // GM, GH
    for (int k = 1; k <= lmh - 1; k++) {
        float dz = (Z[k] - Z[k + 2]) * 0.5f;
        float rdz = 1.0f / dz;
        float gml = ((U[k] - U[k + 1]) * (U[k] - U[k + 1]) + (V[k] - V[k + 1]) * (V[k] - V[k + 1])) * rdz * rdz;
        GM[k] = fmaxf(gml, MYJ_EPSGM);
        float dthv = THV[k] - THV[k + 1];
        if (dthv > 0.0f) {
            if (THE[k + 1] > THE[k]) {
                if (PS[k + 1] > P[k]) {                          // >12KM
                    float wcon = (P[k + 1] - PS[k + 1]) / (P[k + 1] - P[k]);
                    float x = dz * wcon * rsqdt;
                    if ((Q2[k] > EPSQ2[k]) && (Q2[k] * cubry > x * x)) {
                        dthv = (THE[k] - THE[k + 1]) + dthv;
                    }
                }
            }
        }
        float ghl = dthv * rdz;
        if (fabsf(ghl) <= MYJ_EPSGH) ghl = MYJ_EPSGH;
        GH[k] = ghl;
    }
    *CT = 0.0f;                                                  // CT=0.

    // MAXIMUM MIXING LENGTHS
    int lmxl = lmh;
    for (int k = 1; k <= lmh - 1; k++) {
        float gml = GM[k];
        float ghl = GH[k];
        if (ghl >= MYJ_EPSGH) {
            if (gml / ghl <= MYJ_REQU) {
                ELM[k] = EPSL[k];
                lmxl = k + 1;
            } else {
                float aubr = (MYJ_AUBM * gml + MYJ_AUBH * ghl) * ghl;
                float bubr = MYJ_BUBM * gml + MYJ_BUBH * ghl;
                float qol2st = (-0.5f * bubr + sqrtf(bubr * bubr * 0.25f - aubr * MYJ_CUBR)) * MYJ_RCUBR;
                float eloq2x = 1.0f / fmaxf(MYJ_EPSGH, qol2st);
                ELM[k] = fmaxf(sqrtf(eloq2x * Q2[k]), EPSL[k]);
            }
        } else {
            float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
            float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
            float qol2un = -0.5f * bden + sqrtf(bden * bden * 0.25f - aden);
            float eloq2x = 1.0f / (qol2un + MYJ_EPSRU);
            ELM[k] = fmaxf(sqrtf(eloq2x * Q2[k]), EPSL[k]);
        }
    }
    if (ELM[lmh - 1] == EPSL[lmh - 1]) lmxl = lmh;
    *LMXL = lmxl;

    // MIXED LAYER HEIGHT
    float blmx = Z[lmxl] - Z[lmh + 1];
    *MIXHT = blmx;

    for (int k = lpbl; k <= lmh; k++) Q1[k] = sqrtf(Q2[k]);

    // BLACKADAR ASYMPTOTIC L
    float szq = 0.0f, sq = 0.0f;
    for (int k = 1; k <= lmh - 1; k++) {
        float qdzl = (Q1[k] + Q1[k + 1]) * (Z[k + 1] - Z[k + 2]);
        szq = (Z[k + 1] + Z[k + 2] - Z[lmh + 1] - Z[lmh + 1]) * qdzl + szq;
        sq = qdzl + sq;
    }
    float el0 = fminf(MYJ_ALPH * szq * 0.5f / sq, MYJ_EL0MAX);
    el0 = fmaxf(el0, MYJ_EL0MIN);

    // ABOVE THE PBL TOP
    int lpblm = lpbl - 1;
    if (lpblm < 1) lpblm = 1;
    for (int k = 1; k <= lpblm; k++) {
        EL[k] = fminf((Z[k] - Z[k + 2]) * MYJ_ELFC, ELM[k]);
        REL[k] = EL[k] / ELM[k];
    }
    // INSIDE THE PBL
    if (lpbl < lmh) {
        for (int k = lpbl; k <= lmh - 1; k++) {
            float vkrmz = (Z[k + 1] - Z[lmh + 1]) * MYJ_VKARMAN;
            EL[k] = fminf(vkrmz / (vkrmz / el0 + 1.0f), ELM[k]);
            REL[k] = EL[k] / ELM[k];
        }
    }
    for (int k = lpbl + 1; k <= lmh - 2; k++) {
        float srel = fminf(((REL[k - 1] + REL[k + 1]) * 0.5f + REL[k]) * 0.5f, REL[k]);
        EL[k] = fmaxf(srel * ELM[k], EPSL[k]);
    }
}

// PRODQ2 (Fortran 969-1173). Updates EL and Q2 in place; Q2(LMH) lower boundary from USTAR.
__device__ void myj_prodq2(const int lmh, const float dttb, const float ustar,
                           const float* GM, const float* GH, float* EL, float* Q2,
                           const float* EPSL, const float* EPSQ2)
{
    for (int k = 1; k <= lmh - 1; k++) {
        float gml = GM[k];
        float ghl = GH[k];
        float aequ = (MYJ_AEQM * gml + MYJ_AEQH * ghl) * ghl;
        float bequ = MYJ_BEQM * gml + MYJ_BEQH * ghl;
        float eqol2 = -0.5f * bequ + sqrtf(bequ * bequ * 0.25f - aequ);
        if ((gml + ghl * ghl <= MYJ_EPSTRB)
            || (ghl >= MYJ_EPSGH && gml / ghl <= MYJ_REQU)
            || (eqol2 <= MYJ_EPS2)) {
            Q2[k] = EPSQ2[k];
            EL[k] = EPSL[k];
        } else {
            float anum = (MYJ_ANMM * gml + MYJ_ANMH * ghl) * ghl;
            float bnum = MYJ_BNMM * gml + MYJ_BNMH * ghl;
            float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
            float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
            float cden = 1.0f;
            float arhs = -(anum * bden - bnum * aden) * 2.0f;
            float brhs = -anum * 4.0f;
            float crhs = -bnum * 2.0f;
            float dloq1 = EL[k] / sqrtf(Q2[k]);
            float eloq21 = 1.0f / eqol2;
            float eloq11 = sqrtf(eloq21);
            float eloq31 = eloq21 * eloq11;
            float eloq41 = eloq21 * eloq21;
            float eloq51 = eloq21 * eloq31;
            float rden1 = 1.0f / (aden * eloq41 + bden * eloq21 + cden);
            float rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1;
            float eloq12 = eloq11 + (dloq1 - eloq11) * expf(rhsp1 * dttb);
            eloq12 = fmaxf(eloq12, MYJ_EPS1);
            float eloq22 = eloq12 * eloq12;
            float eloq32 = eloq22 * eloq12;
            float eloq42 = eloq22 * eloq22;
            float eloq52 = eloq22 * eloq32;
            float rden2 = 1.0f / (aden * eloq42 + bden * eloq22 + cden);
            float rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + MYJ_RB1;
            float rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2;
            float rhst2 = rhs2 / rhsp2;
            float eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * expf(rhsp2 * dttb);
            eloq13 = fmaxf(eloq13, MYJ_EPS1);
            float eloqn = eloq13;
            if (eloqn > MYJ_EPS1) {
                Q2[k] = EL[k] * EL[k] / (eloqn * eloqn);
                Q2[k] = fmaxf(Q2[k], EPSQ2[k]);
                if (Q2[k] == EPSQ2[k]) EL[k] = EPSL[k];
            } else {
                Q2[k] = EPSQ2[k];
                EL[k] = EPSL[k];
            }
        }
    }
    // Q2(LMH)=AMAX1(B1**(2./3.)*USTAR*USTAR,EPSQ2(LMH))
    Q2[lmh] = fmaxf(powf(MYJ_B1, 2.0f / 3.0f) * ustar * ustar, EPSQ2[lmh]);
}

// DIFCOF (Fortran 1178-1337). LMXL is not read by the live code (see header note).
__device__ void myj_difcof(const int lmh, const float* GM, const float* GH, const float* EL,
                           const float* Q2, const float* Z, float* AKM, float* AKH)
{
    for (int k = 1; k <= lmh - 1; k++) {
        float ell = EL[k];
        float eloq2 = ell * ell / Q2[k];
        float eloq4 = eloq2 * eloq2;
        float gml = GM[k];
        float ghl = GH[k];
        float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
        float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
        float cden = 1.0f;
        float besm = MYJ_BSMH * ghl;
        float besh = MYJ_BSHM * gml + MYJ_BSHH * ghl;
        float rden = 1.0f / (aden * eloq4 + bden * eloq2 + cden);
        float esm = (besm * eloq2 + MYJ_CESM) * rden;
        float esh = (besh * eloq2 + MYJ_CESH) * rden;
        float rdz = 2.0f / (Z[k] - Z[k + 2]);
        float q1l = sqrtf(Q2[k]);
        float elqdz = ell * q1l * rdz;
        AKM[k] = elqdz * esm;
        AKH[k] = elqdz * esh;
    }
}

// VDIFQ (Fortran 1342-1420): implicit vertical diffusion of Q2 (TKE).
__device__ void myj_vdifq(const int lmh, const float dtdif, float* Q2, const float* EL, const float* Z)
{
    float AKQ[MYJ_MAXL + 2], CM[MYJ_MAXL + 2], CR[MYJ_MAXL + 2], DTOZ[MYJ_MAXL + 2], RSQ2[MYJ_MAXL + 2];
    const float esqhf = MYJ_ESQHF;
    for (int k = 1; k <= lmh - 2; k++) {
        DTOZ[k] = (dtdif + dtdif) / (Z[k] - Z[k + 2]);
        AKQ[k] = sqrtf((Q2[k] + Q2[k + 1]) * 0.5f) * (EL[k] + EL[k + 1]) * esqhf / (Z[k + 1] - Z[k + 2]);
        CR[k] = -DTOZ[k] * AKQ[k];
    }
    CM[1] = DTOZ[1] * AKQ[1] + 1.0f;
    RSQ2[1] = Q2[1];
    for (int k = 2; k <= lmh - 2; k++) {
        float cf = -DTOZ[k] * AKQ[k - 1] / CM[k - 1];
        CM[k] = -CR[k - 1] * cf + (AKQ[k - 1] + AKQ[k]) * DTOZ[k] + 1.0f;
        RSQ2[k] = -RSQ2[k - 1] * cf + Q2[k];
    }
    float dtozs = (dtdif + dtdif) / (Z[lmh - 1] - Z[lmh + 1]);
    float akqs = sqrtf((Q2[lmh - 1] + Q2[lmh]) * 0.5f) * (EL[lmh - 1] + MYJ_ELZ0) * esqhf / (Z[lmh] - Z[lmh + 1]);
    float cf = -dtozs * AKQ[lmh - 2] / CM[lmh - 2];
    Q2[lmh - 1] = (dtozs * akqs * Q2[lmh] - RSQ2[lmh - 2] * cf + Q2[lmh - 1])
                  / ((AKQ[lmh - 2] + akqs) * dtozs - CR[lmh - 2] * cf + 1.0f);
    for (int k = lmh - 2; k >= 1; k--) {
        Q2[k] = (-CR[k] * Q2[k + 1] + RSQ2[k]) / CM[k];
    }
}

// VDIFH (Fortran 1425-1522): implicit vertical diffusion of theta, Q, cloud water.
__device__ void myj_vdifh(const float dtdif, const int lmh, const float thz0, const float qz0,
                          const float rkhs, const float chklowq, const float ct,
                          float* TH, float* Q, float* CWM,
                          const float* RKH, const float* Z, const float* RHO)
{
    float CM[MYJ_MAXL + 2], CR[MYJ_MAXL + 2], DTOZ[MYJ_MAXL + 2], RKCT[MYJ_MAXL + 2];
    float RSC[MYJ_MAXL + 2], RSQ[MYJ_MAXL + 2], RST[MYJ_MAXL + 2];
    const float cthf = 0.5f * ct;
    for (int k = 1; k <= lmh - 1; k++) {
        DTOZ[k] = dtdif / (Z[k] - Z[k + 1]);
        CR[k] = -DTOZ[k] * RKH[k];
        RKCT[k] = RKH[k] * (Z[k] - Z[k + 2]) * cthf;
    }
    CM[1] = DTOZ[1] * RKH[1] + RHO[1];
    RST[1] = TH[1] * RHO[1] - RKCT[1] * DTOZ[1];
    RSQ[1] = Q[1] * RHO[1];
    RSC[1] = CWM[1] * RHO[1];
    for (int k = 2; k <= lmh - 1; k++) {
        float dtozl = DTOZ[k];
        float cf = -dtozl * RKH[k - 1] / CM[k - 1];
        CM[k] = -CR[k - 1] * cf + (RKH[k - 1] + RKH[k]) * dtozl + RHO[k];
        RST[k] = -RST[k - 1] * cf + (RKCT[k - 1] - RKCT[k]) * dtozl + TH[k] * RHO[k];
        RSQ[k] = -RSQ[k - 1] * cf + Q[k] * RHO[k];
        RSC[k] = -RSC[k - 1] * cf + CWM[k] * RHO[k];
    }
    float dtozs = dtdif / (Z[lmh] - Z[lmh + 1]);
    float rkhh = RKH[lmh - 1];
    float cf = -dtozs * rkhh / CM[lmh - 1];
    float rkqs = rkhs * chklowq;
    float cmb = CR[lmh - 1] * cf;
    float cmtb = -cmb + (rkhh + rkhs) * dtozs + RHO[lmh];
    float cmqb = -cmb + (rkhh + rkqs) * dtozs + RHO[lmh];
    float cmcb = -cmb + (rkhh) * dtozs + RHO[lmh];
    float rstb = -RST[lmh - 1] * cf + RKCT[lmh - 1] * dtozs + TH[lmh] * RHO[lmh];
    float rsqb = -RSQ[lmh - 1] * cf + Q[lmh] * RHO[lmh];
    float rscb = -RSC[lmh - 1] * cf + CWM[lmh] * RHO[lmh];
    TH[lmh] = (dtozs * rkhs * thz0 + rstb) / cmtb;
    Q[lmh] = (dtozs * rkqs * qz0 + rsqb) / cmqb;
    CWM[lmh] = (rscb) / cmcb;
    for (int k = lmh - 1; k >= 1; k--) {
        float rcml = 1.0f / CM[k];
        TH[k] = (-CR[k] * TH[k + 1] + RST[k]) * rcml;
        Q[k] = (-CR[k] * Q[k + 1] + RSQ[k]) * rcml;
        CWM[k] = (-CR[k] * CWM[k + 1] + RSC[k]) * rcml;
    }
}

// VDIFV (Fortran 1527-1607): implicit vertical diffusion of U and V.
__device__ void myj_vdifv(const int lmh, const float dtdif, const float uz0, const float vz0,
                          const float rkms, const float* D, float* U, float* V,
                          const float* RKM, const float* Z, const float* RHO)
{
    float CM[MYJ_MAXL + 2], CR[MYJ_MAXL + 2], DTOZ[MYJ_MAXL + 2], RSU[MYJ_MAXL + 2], RSV[MYJ_MAXL + 2];
    for (int k = 1; k <= lmh - 1; k++) {
        DTOZ[k] = dtdif / (Z[k] - Z[k + 1]);
        CR[k] = -DTOZ[k] * RKM[k];
    }
    float rhok = RHO[1];
    CM[1] = DTOZ[1] * RKM[1] + rhok;
    RSU[1] = U[1] * rhok;
    RSV[1] = V[1] * rhok;
    for (int k = 2; k <= lmh - 1; k++) {
        float dtozl = DTOZ[k];
        float cf = -dtozl * RKM[k - 1] / CM[k - 1];
        rhok = RHO[k];
        CM[k] = -CR[k - 1] * cf + (RKM[k - 1] + RKM[k] + D[k] * rkms) * dtozl + rhok;
        RSU[k] = -RSU[k - 1] * cf + U[k] * rhok;
        RSV[k] = -RSV[k - 1] * cf + V[k] * rhok;
    }
    float dtozs = dtdif / (Z[lmh] - Z[lmh + 1]);
    float rkmh = RKM[lmh - 1];
    float cf = -dtozs * rkmh / CM[lmh - 1];
    rhok = RHO[lmh];
    float rcmvb = 1.0f / ((rkmh + rkms) * dtozs - CR[lmh - 1] * cf + rhok);
    float dtozak = dtozs * rkms;
    U[lmh] = (dtozak * uz0 - RSU[lmh - 1] * cf + U[lmh] * rhok) * rcmvb;
    V[lmh] = (dtozak * vz0 - RSV[lmh - 1] * cf + V[lmh] * rhok) * rcmvb;
    for (int k = lmh - 1; k >= 1; k--) {
        float rcml = 1.0f / CM[k];
        U[k] = (-CR[k] * U[k + 1] + RSU[k]) * rcml;
        V[k] = (-CR[k] * V[k + 1] + RSV[k]) * rcml;
    }
}

// ---- MYJPBL driver (Fortran 180-753), one thread per column ---------------------------------
// Global 3-D arrays: element for Fortran (I,J,K) at (K-1)*ncol + col. 2-D arrays: element col.
#define MYJ_G3(p, K) ((p)[(size_t)((K) - 1) * (size_t)ncol + (size_t)col])

extern "C" __global__ void phys_myj_myjpbl_column(
    const int im, const int jm, const int lm,
    const float dt, const int nphs,
    const float rdth, const float rdq,
    const float* __restrict__ ptbl,            // PTBL(ITBL,JTBL): C shape [JTBL, ITBL]
    const float* __restrict__ qs0,             // QS0(JTBL)
    const float* __restrict__ sqs,             // SQS(JTBL)
    const float* __restrict__ epsl_g,          // EPSL(LM-1) (1-D)
    const float* __restrict__ epsq2_g,         // EPSQ2(LM)
    const float* __restrict__ dz_g, const float* __restrict__ exner_g,
    const float* __restrict__ pmid_g, const float* __restrict__ q_g,
    const float* __restrict__ cwm_g, const float* __restrict__ u_g,
    const float* __restrict__ v_g, const float* __restrict__ t_g,
    const float* __restrict__ th_g, const float* __restrict__ pinh_g,   // PINH: LM+1 levels
    float* __restrict__ q2_g, float* __restrict__ exch_h_g,
    float* __restrict__ el_myj_g,
    float* __restrict__ rublten_g, float* __restrict__ rvblten_g,
    float* __restrict__ rthblten_g, float* __restrict__ rqblten_g, float* __restrict__ rqcblten_g,
    const float* __restrict__ ht_g, const float* __restrict__ stdh_g,
    const float* __restrict__ tsk_g, const float* __restrict__ xland_g,
    const float* __restrict__ sice_g, const float* __restrict__ snow_g,
    const float* __restrict__ chklowq_g, const float* __restrict__ elflx_g,
    const float* __restrict__ akhs_g, const float* __restrict__ akms_g,
    const float* __restrict__ ustar_g, const float* __restrict__ uz0_g,
    const float* __restrict__ vz0_g,
    float* __restrict__ thz0_g, float* __restrict__ qz0_g, float* __restrict__ qsfc_g,
    float* __restrict__ ct_g,
    float* __restrict__ pblh_g, float* __restrict__ mixht_g, int* __restrict__ kpbl_g)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncol = im * jm;
    if (col >= ncol) return;
    // Fortran ITS_B1..ITE_B1 x JTS_B1..JTE_B1 (module_TURBULENCE.F90:1013-1025, module_SOLVER_GRID_COMP
    // .F90:523-535): MYJPBL runs i=2..IM-1 and j=2..JM-1 (1-based) only. Edge rows and columns are
    // left untouched, including the THSK and THZ0 pre-loops and every tendency.
    const int ii = col % im + 1;                         // 1-based I (I fastest, [JM, IM] layout)
    const int jj = col / im + 1;                         // 1-based J
    if (ii < 2 || ii > im - 1 || jj < 2 || jj > jm - 1) return;
    if (lm > MYJ_MAXL || lm < 4) return;
    const int lmh = lm;
    const float dttb = dt * (float)nphs;                 // DTTURBL=DT*NPHS
    const float rdtturbl = 1.0f / dttb;                  // RDTTURBL=1./DTTURBL
    const float rsqdt = sqrtf(rdtturbl);                 // RSQDT=SQRT(RDTTURBL)
    const float dtdif = dttb;                            // DTDIF=DTTURBL

    // 1-based per-column arrays; index 0 unused
    float zint[MYJ_MAXL + 2], rxner[MYJ_MAXL + 2], thv[MYJ_MAXL + 2];
    float pk[MYJ_MAXL + 2], tk[MYJ_MAXL + 2], qk[MYJ_MAXL + 2], thvk[MYJ_MAXL + 2];
    float rxnerk[MYJ_MAXL + 2], uk[MYJ_MAXL + 2], vk[MYJ_MAXL + 2], q2k[MYJ_MAXL + 2];
    float thek[MYJ_MAXL + 2], psk[MYJ_MAXL + 2];
    float gm[MYJ_MAXL + 2], gh[MYJ_MAXL + 2], el[MYJ_MAXL + 2];
    float akmk[MYJ_MAXL + 2], akhk[MYJ_MAXL + 2], akm[MYJ_MAXL + 2], akh[MYJ_MAXL + 2];
    float dcol[MYJ_MAXL + 2], rhok[MYJ_MAXL + 2], cwmk[MYJ_MAXL + 2], thk[MYJ_MAXL + 2];
    float epsl[MYJ_MAXL + 2], epsq2[MYJ_MAXL + 2];

    for (int k = 1; k <= lmh; k++) epsq2[k] = epsq2_g[k - 1];
    for (int k = 1; k <= lmh - 1; k++) epsl[k] = epsl_g[k - 1];

    // preparations: ZINT, RXNER, THV; AKM=0; ZINT(LM+1)=HT
    for (int k = 1; k <= lmh - 1; k++) akm[k] = 0.0f;
    zint[lmh + 1] = ht_g[col];                           // ZINT(I,J,LM+1)=HT(I,J)
    for (int k = lmh; k >= 1; k--) {
        zint[k] = zint[k + 1] + MYJ_G3(dz_g, k);         // ZINT(K)=ZINT(K+1)+DZ(K)
        rxner[k] = 1.0f / MYJ_G3(exner_g, k);            // RXNER=1./EXNER
        thv[k] = (MYJ_G3(q_g, k) * 0.608f + (1.0f - MYJ_G3(cwm_g, k))) * MYJ_G3(th_g, k);
    }
    MYJ_G3(el_myj_g, lmh) = 0.0f;                        // EL_MYJ(I,J,LM)=0.

    // setup_integration (per column)
    for (int k = lmh; k >= 1; k--) {
        pk[k] = MYJ_G3(pmid_g, k);
        tk[k] = MYJ_G3(t_g, k);
        qk[k] = MYJ_G3(q_g, k);
        thvk[k] = thv[k];
        rxnerk[k] = rxner[k];
        uk[k] = MYJ_G3(u_g, k);
        vk[k] = MYJ_G3(v_g, k);
        q2k[k] = MYJ_G3(q2_g, k);
    }
    // POTENTIAL INSTABILITY: equivalent potential temperature from the saturation tables
    const float pelevfc = MYJ_G3(pmid_g, lmh) * MYJ_ELEVFC;
    for (int k = lmh; k >= 1; k--) {
        if (k == lmh || MYJ_G3(pmid_g, k) > pelevfc) {
            float qbt = qk[k];
            float thbt = MYJ_G3(th_g, k);
            float tth = (thbt - MYJ_THL) * rdth;
            float qq1 = tth - truncf(tth);
            int ittb = (int)tth + 1;
            if (ittb < 1) {
                ittb = 1;
                qq1 = 0.0f;
            } else if (ittb >= MYJ_JTBL) {
                ittb = MYJ_JTBL - 1;
                qq1 = 0.0f;
            }
            float bqs00k = qs0[ittb - 1];
            float sqs00k = sqs[ittb - 1];
            float bqs10k = qs0[ittb];
            float sqs10k = sqs[ittb];
            float bq = (bqs10k - bqs00k) * qq1 + bqs00k;
            float sq = (sqs10k - sqs00k) * qq1 + sqs00k;
            float tq = (qbt - bq) / sq * rdq;
            float pp1 = tq - truncf(tq);
            int iqtb = (int)tq + 1;
            if (iqtb < 1) {
                iqtb = 1;
                pp1 = 0.0f;
            } else if (iqtb >= MYJ_ITBL) {
                iqtb = MYJ_ITBL - 1;
                pp1 = 0.0f;
            }
            // PTBL(IQTB,ITTB) -> ptbl[(ITTB-1)*ITBL + (IQTB-1)]
            float p00k = ptbl[(size_t)(ittb - 1) * MYJ_ITBL + (iqtb - 1)];
            float p10k = ptbl[(size_t)(ittb - 1) * MYJ_ITBL + iqtb];
            float p01k = ptbl[(size_t)ittb * MYJ_ITBL + (iqtb - 1)];
            float p11k = ptbl[(size_t)ittb * MYJ_ITBL + iqtb];
            float psp = p00k + (p10k - p00k) * pp1 + (p01k - p00k) * qq1
                        + (p00k - p10k - p01k + p11k) * pp1 * qq1;
            float rxners = powf(1.0e5f / psp, NMMB_CAPPA);
            thek[k] = thbt * expf(MYJ_ELOCP * qbt * rxners / thbt);
            psk[k] = psp;
        } else {
            thek[k] = thek[k + 1];
            psk[k] = MYJ_G3(pinh_g, 1);                   // PSK(K)=PINH(I,J,1)
        }
    }

    // MIXING LENGTH
    float pblh_c = 0.0f, mixht_c = 0.0f, ct_c = 0.0f;
    int lpbl = 0, lmxl = 0;
    myj_mixlen(lmh, rsqdt, uk, vk, thvk, thek, q2k, epsl, epsq2, zint, pk, psk,
               gm, gh, el, &pblh_c, &lpbl, &lmxl, &ct_c, &mixht_c, lm);
    pblh_g[col] = pblh_c;
    mixht_g[col] = mixht_c;
    ct_g[col] = ct_c;                                    // CT (INOUT), zeroed by MIXLEN

    // TKE production/dissipation
    myj_prodq2(lmh, dttb, ustar_g[col], gm, gh, el, q2k, epsl, epsq2);

    kpbl_g[col] = lpbl;                                  // KPBL(I,J)=LPBL(I,J)

    // Free-atmosphere exchange coefficients
    myj_difcof(lmh, gm, gh, el, q2k, zint, akmk, akhk);
    for (int k = 1; k <= lmh - 1; k++) {
        akh[k] = akhk[k];
        akm[k] = akmk[k];
        float deltaz = 0.5f * (zint[k] - zint[k + 2]);
        MYJ_G3(exch_h_g, k) = akhk[k] * deltaz;          // EXCH_H on layer bottoms, K<=LM-1
    }

    // Vertical diffusion of Q2 (TKE)
    myj_vdifq(lmh, dtdif, q2k, el, zint);
    for (int k = 1; k <= lmh; k++) {
        MYJ_G3(q2_g, k) = fmaxf(q2k[k], epsq2[k]);
        if (k < lmh) MYJ_G3(el_myj_g, k) = el[k];        // EL is not defined at LM
    }

    // main_integration (per column)
    const float thsk = tsk_g[col] * powf(1.0e5f / MYJ_G3(pinh_g, lmh + 1), NMMB_CAPPA);   // THSK
    for (int k = lmh; k >= 1; k--) {
        thk[k] = MYJ_G3(th_g, k);
        qk[k] = MYJ_G3(q_g, k);
        cwmk[k] = MYJ_G3(cwm_g, k);
        rhok[k] = MYJ_G3(pmid_g, k) / (NMMB_R_D * MYJ_G3(t_g, k) * (1.0f + NMMB_P608 * qk[k] - cwmk[k]));
    }
    for (int k = 1; k <= lmh - 1; k++) {
        akhk[k] = akh[k] * 0.5f * (rhok[k] + rhok[k + 1]);
    }

    const float seamask = xland_g[col] - 1.0f;
    thz0_g[col] = (1.0f - seamask) * thsk + seamask * thz0_g[col];   // THZ0 (INOUT)
    const float akhs_dens = akhs_g[col] * rhok[lmh];
    if (seamask < 0.5f) {
        float qfc1 = NMMB_ELWV * chklowq_g[col] * akhs_dens;
        if (snow_g[col] > 0.0f || sice_g[col] > 0.5f) qfc1 = qfc1 * MYJ_RLIVWV;
        if (qfc1 > 0.0f) {
            float qlow = qk[lmh];
            qsfc_g[col] = qlow + elflx_g[col] / qfc1;
        }
    } else {
        float psfc = MYJ_G3(pinh_g, lmh + 1);
        float rxnsfc = powf(1.0e5f / psfc, NMMB_CAPPA);
        qsfc_g[col] = MYJ_PQ0SEA / psfc
                      * expf(NMMB_A2 * (thsk - NMMB_A3 * rxnsfc) / (thsk - NMMB_A4 * rxnsfc));
    }
    qz0_g[col] = (1.0f - seamask) * qsfc_g[col] + seamask * qz0_g[col];   // QZ0 (INOUT)

    // Vertical diffusion of T and water vapor (VDIFH)
    myj_vdifh(dtdif, lmh, thz0_g[col], qz0_g[col], akhs_dens, chklowq_g[col], ct_g[col],
              thk, qk, cwmk, akhk, zint, rhok);

    // Primary variable tendencies (theta, Q, QC)
    for (int k = 1; k <= lmh; k++) {
        MYJ_G3(rthblten_g, k) = (thk[k] - MYJ_G3(th_g, k)) * rdtturbl;
        MYJ_G3(rqblten_g, k) = (qk[k] - MYJ_G3(q_g, k)) * rdtturbl;
        MYJ_G3(rqcblten_g, k) = (cwmk[k] - MYJ_G3(cwm_g, k)) * rdtturbl;
    }

    // Momentum: RSTDH/ZSF feed only DCOL, which is zero (see header note).
    for (int k = 1; k <= lmh - 1; k++) {
        dcol[k] = 0.0f;                                   // DCOL(K)=0. (ZJ override)
        akmk[k] = akm[k];
        akmk[k] = akmk[k] * (rhok[k] + rhok[k + 1]) * 0.5f;
    }
    const float akms_dens = akms_g[col] * rhok[lmh];
    myj_vdifv(lmh, dtdif, uz0_g[col], vz0_g[col], akms_dens, dcol, uk, vk, akmk, zint, rhok);

    for (int k = 1; k <= lmh; k++) {
        MYJ_G3(rublten_g, k) = (uk[k] - MYJ_G3(u_g, k)) * rdtturbl;
        MYJ_G3(rvblten_g, k) = (vk[k] - MYJ_G3(v_g, k)) * rdtturbl;
    }
}
#undef MYJ_G3
