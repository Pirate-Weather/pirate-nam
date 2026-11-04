// phys_noah_sfcdiags.cu -- SFCDIAGS for nam-gpu: 2 m temperature, 2 m theta and 2 m specific
// humidity from the surface sensible and latent heat fluxes and the surface exchange coefficients.
//
// Notice: module_LS_NOAHLSM.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1, NMMB
// physics, NOAA/NCEP/EMC) keeps its author headers. There is no licence file in the NAM tree;
// the NOAA source notice in NOTICE applies. NAM is US Government work.
//
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:5609-5656 (nam.v4.2.11, 83df4e13c1),
// the SFCDIAGS subroutine. Every executable statement of that routine is here.
//
// Caller: nmm/module_TURBULENCE.F90:925 (after NOAHLSM, on the NPHS physics cadence). Actual ->
// dummy: TWBS->HFX, QWBS->QFX, TSFC->TSK, QSH->QSFC, CHS2->CHS2, CQS2->CQS2, T2->T2,
// TH2X->TH2, Q2X->Q2, PSFC_OUT->PSFC, CP->cp, R_d->r_d, CAPPA->rovcp.
//
// Limits: the caller passes ITS_B1..ITE_B1 = 2..IM-1 and JTS_B1..JTE_B1 = 2..JM-1 (1-based). The
// kernel takes the same 1-based limits and runs one thread per H point inside them.
// KTS..KTE (1..LM+1) are unused: the routine is 2-D. The dummy dimension bounds (ids, ims, ...)
// and INTENT declarations are halo bookkeeping and are dropped (single card, no halo exchange).
//
// Values per point (Fortran order kept):
//   RHO   = PSFC / (R_d * TSK)
//   Q2    = QSFC                              if CQS2 < 1.0e-5
//         = QSFC - QFX / (RHO * CQS2)         otherwise
//   T2    = TSK                               if CHS2 < 1.0e-5
//         = TSK - HFX / (RHO * CP * CHS2)     otherwise
//   TH2   = T2 * (1.0e5 / PSFC) ** ROVCP
// Q2 is only written (INTENT(INOUT) in the Fortran, never read); T2 is written before TH2 reads it.
//
// Single precision throughout (kfpt = single). powf is the IEEE-accurate power (fast numerics,
// --fmad=true per CONTRACT section 3; no fast-math intrinsics).

#include "nmmb_common.cuh"

extern "C" __global__ void phys_noah_sfcdiags(
    const float* __restrict__ HFX,
    const float* __restrict__ QFX,
    const float* __restrict__ TSK,
    const float* __restrict__ QSFC,
    const float* __restrict__ CHS2,
    const float* __restrict__ CQS2,
    const float* __restrict__ PSFC,
    float* __restrict__ T2,
    float* __restrict__ TH2,
    float* __restrict__ Q2,
    const float cp,
    const float r_d,
    const float rovcp,
    const int its, const int ite,
    const int jts, const int jte,
    const int im, const int jm)
{
    // Thread -> zero-based point. Fortran limits are 1-based: i in [its, ite], j in [jts, jte].
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 < its - 1 || i0 > ite - 1) return;
    if (j0 < jts - 1 || j0 > jte - 1) return;
    if (i0 >= im || j0 >= jm) return;

    const size_t p = IDX2(i0, j0, im);

    const float psfc = PSFC[p];
    const float tsk  = TSK[p];
    const float rho  = psfc / (r_d * tsk);

    // Q2 (2 m specific humidity), SFCDIAGS lines 5642-5646 (Fortran CQS2 < 1.E-5 branch).
    if (CQS2[p] < 1.0e-5f) {
        Q2[p] = QSFC[p];
    } else {
        Q2[p] = QSFC[p] - QFX[p] / (rho * CQS2[p]);
    }

    // T2 (2 m temperature), SFCDIAGS lines 5647-5651 (Fortran CHS2 < 1.E-5 branch).
    float t2v;
    if (CHS2[p] < 1.0e-5f) {
        t2v = tsk;
    } else {
        t2v = tsk - HFX[p] / (rho * cp * CHS2[p]);
    }
    T2[p] = t2v;

    // TH2 (2 m potential temperature), SFCDIAGS line 5652, uses the T2 just written.
    TH2[p] = t2v * powf(1.0e5f / psfc, rovcp);
}
