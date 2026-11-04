// phys_myj_difcof.cu -- MYJ level 2.5 exchange coefficients (DIFCOF), one thread per column.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). The module
// header of module_BL_MYJPBL is kept as the notice:
//   MODULE MODULE_BL_MYJPBL, "THE MYJ PBL SCHEME" (NMMB, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:1178-1337 (nam.v4.2.11, 83df4e13c1).
// Start point: ArWen third_party/arwen/core/kernels/myjpbl.cu (DIFCOF block, Apache-2.0, see
// third_party/arwen/LICENSE), brought to the NAM file: top-down levels (Fortran K = 1 is the model
// top, zero-based k0 = K-1), NAM constants below (module_BL_MYJPBL.F90:36-122), float32.
//
// Fortran -> CUDA map (every term of the routine):
//   DO K=1,LMH-1                                  -> k0 = 0 .. lmh-2 inside one thread
//   ELL=EL(K), ELOQ2, ELOQ4                       -> ell, eloq2, eloq4
//   GML=GM(K), GHL=GH(K)                          -> gml, ghl
//   ADEN, BDEN, CDEN=1.                           -> aden, bden, cden
//   BESM=BSMH*GHL, BESH=BSHM*GML+BSHH*GHL         -> besm, besh
//   RDEN=1./(ADEN*ELOQ4+BDEN*ELOQ2+CDEN)          -> rden
//   ESM=(BESM*ELOQ2+CESM)*RDEN, ESH likewise      -> esm, esh
//   RDZ=2./(Z(K)-Z(K+2)), Q1L=SQRT(Q2(K))        -> rdz, q1l (Z is the LM+1 interface array)
//   ELQDZ=ELL*Q1L*RDZ                             -> elqdz
//   AKM(K)=ELQDZ*ESM, AKH(K)=ELQDZ*ESH           -> akm[p], akh[p]
//   the AMAX1 floors and the INVERSION block (:1268-1316) are commented out in the Fortran
//   (WWW and MMM lines, and the IF(LMXL==LMH) block) and are not run; the INVERSION debug prints
//   are not in the run path (PRINT_DIAG is ignored).
//   T is an argument of the Fortran routine but is read only inside the commented-out inversion
//   block, so the wrapper does not pass it.
// AKM and AKH are defined for K = 1..LMH-1; level LM is written as an explicit zero (Fortran leaves
// it undefined, and no consumer reads it). Nothing is dropped: no MPI, OpenMP, print or halo code exists in this routine.
//
// Layout: [LM, JM, IM] float32 (CONTRACT section 1), level index k0 = K-1, zero-based i0, j0.
// GM, GH, EL, Q2 and AKM/AKH use the same flat stride as the CONTRACT layout. Z is [LM+1, JM, IM].
#include "nmmb_common.cuh"

// ---- module_BL_MYJPBL.F90 PARAMETER block, lines 36-122 (NAM values) ----
#define MYJ_A1      0.659888514560862645f
#define MYJ_A2X     0.6574209922667784586f
#define MYJ_B2      7.226971804046074028f
#define MYJ_C1      0.000830955950095854396f
#define MYJ_BETA    (1.0f / 273.0f)
#define MYJ_BTG     (MYJ_BETA * NMMB_G)
#define MYJ_ADNH    (9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*(12.0f*MYJ_A1+3.0f*MYJ_B2)*MYJ_BTG*MYJ_BTG)
#define MYJ_ADNM    (18.0f*MYJ_A1*MYJ_A1*MYJ_A2X*(MYJ_B2-3.0f*MYJ_A2X)*MYJ_BTG)
#define MYJ_BDNH    (3.0f*MYJ_A2X*(7.0f*MYJ_A1+MYJ_B2)*MYJ_BTG)
#define MYJ_BDNM    (6.0f*MYJ_A1*MYJ_A1)
#define MYJ_BSHH    (9.0f*MYJ_A1*MYJ_A2X*MYJ_A2X*MYJ_BTG)
#define MYJ_BSHM    (18.0f*MYJ_A1*MYJ_A1*MYJ_A2X*MYJ_C1)
#define MYJ_BSMH    (-3.0f*MYJ_A1*MYJ_A2X*(3.0f*MYJ_A2X+3.0f*MYJ_B2*MYJ_C1+12.0f*MYJ_A1*MYJ_C1-MYJ_B2)*MYJ_BTG)
#define MYJ_CESH    (MYJ_A2X)
#define MYJ_CESM    (MYJ_A1*(1.0f-3.0f*MYJ_C1))

extern "C" __global__ void phys_myj_difcof(
    const int im, const int jm, const int lm, const int lmh,
    const float* __restrict__ gm,    // [LM, JM, IM] GM(K)   from MIXLEN
    const float* __restrict__ gh,    // [LM, JM, IM] GH(K)   from MIXLEN
    const float* __restrict__ el,    // [LM, JM, IM] EL(K)   (EL_PBL, after PRODQ2)
    const float* __restrict__ q2,    // [LM, JM, IM] Q2(K)
    const float* __restrict__ z,     // [LM+1, JM, IM] Z(K) interface heights, Z(LM+1) = ground
    float* __restrict__ akm,         // [LM, JM, IM] AKM(K), written for K = 1..LMH-1
    float* __restrict__ akh)         // [LM, JM, IM] AKH(K), written for K = 1..LMH-1
{
    COLUMN_IJ(im, jm)
    (void)lm;
    for (int k0 = 0; k0 < lmh - 1; ++k0) {
        const size_t p = IDX3(i0, j0, k0, im, jm);
        const float ell = el[p];
        const float eloq2 = ell * ell / q2[p];
        const float eloq4 = eloq2 * eloq2;
        const float gml = gm[p];
        const float ghl = gh[p];

        const float aden = (MYJ_ADNM * gml + MYJ_ADNH * ghl) * ghl;
        const float bden = MYJ_BDNM * gml + MYJ_BDNH * ghl;
        const float cden = 1.0f;

        const float besm = MYJ_BSMH * ghl;
        const float besh = MYJ_BSHM * gml + MYJ_BSHH * ghl;

        const float rden = 1.0f / (aden * eloq4 + bden * eloq2 + cden);

        const float esm = (besm * eloq2 + MYJ_CESM) * rden;
        const float esh = (besh * eloq2 + MYJ_CESH) * rden;

        const float rdz = 2.0f / (z[IDX3(i0, j0, k0, im, jm)] - z[IDX3(i0, j0, k0 + 2, im, jm)]);
        const float q1l = sqrtf(q2[p]);
        const float elqdz = ell * q1l * rdz;
        akm[p] = elqdz * esm;
        akh[p] = elqdz * esh;
    }
    // Level LM (zero-based k0 = LMH-1) is outside the Fortran K = 1..LMH-1 loop and is never read by
    // the MYJPBL driver (AKM/AKH consumers loop K = 1..LM-1). Write an explicit zero, not stale data.
    if (lmh >= 1 && lmh - 1 < lm) {
        const size_t pt = IDX3(i0, j0, lmh - 1, im, jm);
        akm[pt] = 0.0f;
        akh[pt] = 0.0f;
    }
}
