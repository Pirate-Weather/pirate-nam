// phys_myj_prodq2.cu -- MYJ level 2.5 TKE production/dissipation (PRODQ2), one thread per column.
//
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:969-1173 (nam.v4.2.11, 83df4e13c1).
// Start point: ArWen third_party/arwen/core/kernels/myjpbl.cu (PRODQ2 block, Apache-2.0, see
// third_party/arwen/LICENSE), brought to the NAM file: top-down levels (NAM K = 1 is the model top,
// zero-based k0 = K-1), per-level EPSL / EPSQ2 arrays instead of ArWen scalars, the NAM lower
// boundary Q2(LMH) = AMAX1(B1**(2./3.)*USTAR*USTAR, EPSQ2(LMH)), and the NAM constants below.
// The NOAA source file carries no licence header (US government work, NOTICE).
//
// Fortran -> CUDA map (every term of the routine):
//   K=1,LMH-1 loop                         -> k0 = 0 .. lmh-2 inside one thread (column recurrence)
//   AEQU, BEQU, EQOL2                      -> aequ, bequ, eqol2 (AEQM/AEQH/BEQM/BEQH constants)
//   no-turbulence test (3 terms)           -> same three terms, sets Q2=EPSQ2(K), EL=EPSL(K)
//   ANUM, BNUM, ADEN, BDEN, CDEN           -> anum, bnum, aden, bden, cden
//   ARHS, BRHS, CRHS                       -> arhs, brhs, crhs
//   DLOQ1 = EL/SQRT(Q2)                    -> dloq1
//   ELOQ21..ELOQ51, RDEN1, RHSP1           -> eloq21..eloq51, rden1, rhsp1
//   first guess ELOQ12 (EXP, EPS1 floor)   -> eloq12
//   second iteration ELOQ22..ELOQ52, RDEN2,
//   RHS2 (+RB1), RHSP2, RHST2              -> eloq22..eloq52, rden2, rhs2, rhsp2, rhst2
//   corrected ELOQ13 (EXP, AMAX1 EPS1)     -> eloq13
//   ELOQN>EPS1 -> Q2 = EL*EL/ELOQN**2, AMAX1 with EPSQ2, EL=EPSL if Q2==EPSQ2 (else both floors)
//   Q2(LMH) lower boundary                 -> last statement of the thread
// Nothing is dropped: no MPI, OpenMP, print or halo code exists in this routine.
//
// Layout: [LM, JM, IM] float32 (CONTRACT section 1), level index k0 = K-1, zero-based i0, j0.
// USTAR is [JM, IM]. EPSL and EPSQ2 are the MYJPBL floor arrays: 1-D per level [LM] (the state
// table ranks them 1-D, same for every column), read at k0 only.
#include "nmmb_common.cuh"

// ---- module_BL_MYJPBL.F90 PARAMETER block, lines 36-122 (NAM values) ----
#ifndef MYJ_A1
#define MYJ_A1      0.659888514560862645f
#define MYJ_A2X     0.6574209922667784586f
#define MYJ_B1      11.87799326209552761f
#define MYJ_B2      7.226971804046074028f
#define MYJ_C1      0.000830955950095854396f
#define MYJ_BETA    (1.0f / 273.0f)
#define MYJ_BTG     (MYJ_BETA * NMMB_G)
#define MYJ_EPS1    1.0e-12f
#define MYJ_EPS2    0.0f
#define MYJ_EPSTRB  1.0e-24f
#define MYJ_EPSGH   1.0e-9f
#define MYJ_RB1     (1.0f / MYJ_B1)
#define MYJ_ADNH    (9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*(12.0f*MYJ_A1+3.0f*MYJ_B2)*MYJ_BTG*MYJ_BTG)
#define MYJ_ADNM    (18.0f*MYJ_A1*MYJ_A1*MYJ_A2X*(MYJ_B2-3.0f*MYJ_A2X)*MYJ_BTG)
#define MYJ_ANMH    (-9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*MYJ_BTG*MYJ_BTG)
#define MYJ_ANMM    (-3.0f*MYJ_A1*MYJ_A2X*(3.0f*MYJ_A2X+3.0f*MYJ_B2*MYJ_C1+18.0f*MYJ_A1*MYJ_C1-MYJ_B2)*MYJ_BTG)
#define MYJ_BDNH    (3.0f*MYJ_A2X*(7.0f*MYJ_A1+MYJ_B2)*MYJ_BTG)
#define MYJ_BDNM    (6.0f*MYJ_A1*MYJ_A1)
#define MYJ_BEQH    (MYJ_A2X*MYJ_B1*MYJ_BTG+3.0f*MYJ_A2X*(7.0f*MYJ_A1+MYJ_B2)*MYJ_BTG)
#define MYJ_BEQM    (-MYJ_A1*MYJ_B1*(1.0f-3.0f*MYJ_C1)+6.0f*MYJ_A1*MYJ_A1)
#define MYJ_BNMH    (-MYJ_A2X*MYJ_BTG)
#define MYJ_BNMM    (MYJ_A1*(1.0f-3.0f*MYJ_C1))
#define MYJ_AEQH    (9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*MYJ_B1*MYJ_BTG*MYJ_BTG \
                     + 9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*(12.0f*MYJ_A1+3.0f*MYJ_B2)*MYJ_BTG*MYJ_BTG)
#define MYJ_AEQM    (3.0f*MYJ_A1*MYJ_A2X*MYJ_B1*(3.0f*MYJ_A2X+3.0f*MYJ_B2*MYJ_C1+18.0f*MYJ_A1*MYJ_C1-MYJ_B2)*MYJ_BTG \
                     + 18.0f*MYJ_A1*MYJ_A1*MYJ_A2X*(MYJ_B2-3.0f*MYJ_A2X)*MYJ_BTG)
#define MYJ_REQU    (-MYJ_AEQH/MYJ_AEQM)
#endif

extern "C" __global__ void phys_myj_prodq2(
    const int im, const int jm, const int lm, const int lmh, const float dtturbl,
    const float* __restrict__ ustar,     // [JM, IM]     USTAR(I,J)
    const float* __restrict__ gm,        // [LM, JM, IM] GM(K)
    const float* __restrict__ gh,        // [LM, JM, IM] GH(K)
    const float* __restrict__ epsl,      // [LM]         EPSL(K), per level (state table, rank 1)
    const float* __restrict__ epsq2,     // [LM]         EPSQ2(K), per level (state table, rank 1)
    float* __restrict__ el,              // [LM, JM, IM] EL(K)   in/out
    float* __restrict__ q2)              // [LM, JM, IM] Q2(K)   in/out
{
    COLUMN_IJ(im, jm)
    for (int k0 = 0; k0 < lmh - 1; ++k0) {
        const size_t p = IDX3(i0, j0, k0, im, jm);
        const float gml = gm[p];
        const float ghl = gh[p];
        const float aequ = (MYJ_AEQM * gml + MYJ_AEQH * ghl) * ghl;
        const float bequ = MYJ_BEQM * gml + MYJ_BEQH * ghl;
        const float eqol2 = -0.5f * bequ + sqrtf(bequ * bequ * 0.25f - aequ);

        if ((gml + ghl * ghl <= MYJ_EPSTRB)
            || (ghl >= MYJ_EPSGH && gml / ghl <= MYJ_REQU)
            || (eqol2 <= MYJ_EPS2)) {
            // NO TURBULENCE
            q2[p] = epsq2[k0];
            el[p] = epsl[k0];
            continue;
        }

        // TURBULENCE: coefficients of numerator and denominator
        const float anum = (MYJ_ANMM * gml + MYJ_ANMH * ghl) * ghl;
        const float bnum = MYJ_BNMM * gml + MYJ_BNMH * ghl;
        const float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
        const float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
        const float cden = 1.0f;

        // coefficients of the linearised equation numerator
        const float arhs = -(anum * bden - bnum * aden) * 2.0f;
        const float brhs = -anum * 4.0f;
        const float crhs = -bnum * 2.0f;

        // initial value of L/Q
        const float dloq1 = el[p] / sqrtf(q2[p]);

        // first iteration for L/Q, RHS = 0
        const float eloq21 = 1.0f / eqol2;
        const float eloq11 = sqrtf(eloq21);
        const float eloq31 = eloq21 * eloq11;
        const float eloq41 = eloq21 * eloq21;
        const float eloq51 = eloq21 * eloq31;
        const float rden1 = 1.0f / (aden * eloq41 + bden * eloq21 + cden);
        const float rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1;
        float eloq12 = eloq11 + (dloq1 - eloq11) * expf(rhsp1 * dtturbl);
        eloq12 = fmaxf(eloq12, MYJ_EPS1);

        // second iteration for L/Q
        const float eloq22 = eloq12 * eloq12;
        const float eloq32 = eloq22 * eloq12;
        const float eloq42 = eloq22 * eloq22;
        const float eloq52 = eloq22 * eloq32;
        const float rden2 = 1.0f / (aden * eloq42 + bden * eloq22 + cden);
        const float rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + MYJ_RB1;
        const float rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2;
        const float rhst2 = rhs2 / rhsp2;
        float eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * expf(rhsp2 * dtturbl);
        eloq13 = fmaxf(eloq13, MYJ_EPS1);

        // two iterations are enough in most cases
        const float eloqn = eloq13;
        if (eloqn > MYJ_EPS1) {
            float qq = el[p] * el[p] / (eloqn * eloqn);
            qq = fmaxf(qq, epsq2[k0]);
            q2[p] = qq;
            if (q2[p] == epsq2[k0]) {
                el[p] = epsl[k0];
            }
        } else {
            q2[p] = epsq2[k0];
            el[p] = epsl[k0];
        }
    }
    // LOWER BOUNDARY CONDITION FOR Q2: Q2(LMH) = AMAX1(B1**(2./3.)*USTAR*USTAR, EPSQ2(LMH))
    const float us = ustar[IDX2(i0, j0, im)];
    const size_t pb = IDX3(i0, j0, lmh - 1, im, jm);
    q2[pb] = fmaxf(powf(MYJ_B1, 2.0f / 3.0f) * us * us, epsq2[lmh - 1]);
}
