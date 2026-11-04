// phys_fer_deposit.cu -- nam-gpu unit phys-fer-deposit: DEPOSIT, ice deposition at T < T_ICE.
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). Module header block kept as in phys_fer_growth_rates.cu (NOAA "Modified" fer_hires,
// 11 July 2016 and later revisions). The routine comment ("Also uses the Asai (1965) algorithm, but
// uses a different target vapor pressure for the adjustment") is reproduced in the device function.
//
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:2231-2289 (nam.v4.2.11, 83df4e13c1).
//
// Statement map (Fortran -> CUDA):
//   2231       REAL FUNCTION DEPOSIT (PP,Tdum,WVdum,RHgrd,I,J,L) -> nam_deposit(), __device__. Tdum and WVdum are
//                                                                   INOUT by reference, as in Fortran; the
//                                                                   FER_HIRES caller passes scratch copies.
//   2236-2238  RHLIMIT, RHLIMIT1 (HIGH_PRES = double)            -> RHLIMIT is double of the SINGLE literal
//                                                                   .001 (Fortran REAL literal assigned to kind 15)
//   2239       DEP, SSAT REAL(KIND=HIGH_PRES)                    -> double DEP and double SSAT (Fortran kdbl)
//   2242-2243  ESI, WS, DWV (REAL)                               -> float
//   2246-2247  I, J, L integers, Tinp/DWVinp/SSATinp             -> I, J, L feed only the WARN2 write, which is
//                                                                   host-side; DWVinp kept (it is the result)
//   2251       ESI=MIN(1000.*FPVS(Tdum),0.99*PP)                 -> fminf(1000.f*nam_fpvs(...), 0.99f*PP)
//                                                                   (nam_fpvs: phys_fer_fpvs.cu, lines 2840-2895)
//   2252       WS=RHgrd*EPS*ESI/(PP-ESI)                         -> EPS = EP_2 = FER_EP2 (fer_hires.cuh)
//   2253-2255  DWV, SSAT, DEPOSIT=0.                             -> same
//   2261-2268  DO NITER=1,MAX_ITERATIONS: DEP, Tdum, WVdum, DEPOSIT -> loop 1..FER_MAX_ITERATIONS (10). DEP is
//                                                                   double, so the Tdum, WVdum and DEPOSIT updates
//                                                                   are double arithmetic rounded to float on store
//   2269-2273  ESI, WS, DWV, SSAT refresh; EXIT test            -> same; the test compares double SSAT with the
//                                                                   double limits
//   2277-2284  IF (NITER>MAX_ITERATIONS): WARN2 write, SSAT=0, DEPOSIT=DWVinp
//                                                               -> atomicAdd on *nonconv (no printf on device);
//                                                                   SSAT=0 is dead (SSAT unused after), DEPOSIT=DWVinp kept
//   2288       END FUNCTION DEPOSIT                              -> return
//
// Fortran's DO variable after normal completion is MAX_ITERATIONS+1, so "NITER > MAX" is exactly "the loop ran
// out without EXIT"; the C loop leaves niter = MAX+1 in that same case.
//
// Eval kernel fer_deposit_eval: one thread per cell (POINT_IJL). The FER_HIRES column unit calls nam_deposit in
// its ice branch; this kernel runs the same function on every cell for the unit wrapper.
#include "nmmb_common.cuh"
#include "fer_hires.cuh"
#include "phys_fer_fpvs.cu"

// Constants checked against the NAM source (review gap 3): nmm/module_CONSTANTS.F90 (md5 identical to
// share/module_CONSTANTS.F90) sets r=287.04, r_v=461.6, ep_2=r/r_v (so FER_EP2 = 287.04/461.6), xlv=2.5E6,
// cp=1004.6, g=9.8060226, tiw=273.15, cliq=4190, cice=2106, rhowater=1000, epsq=1e-12. The FER module's own
// XLF=3.3358e5 (module-local, not imported from the constants module, whose xlf is 3.50E5), so XLS=XLV+XLF
// with XLF=3.3358e5 as in the Fortran.

// RHLIMIT=.001 in Fortran is a SINGLE literal assigned to a KIND=HIGH_PRES parameter.
#define FER_DEPOSIT_RHLIMIT ((double)0.001f)

// DEPOSIT(PP,Tdum,WVdum,RHgrd): returns the total ice deposition; Tdum and WVdum are updated.
// tbpvs / c1xpvs / c2xpvs: TBPVS and C1XPVS, C2XPVS of GPVS_hr (phys-fer-gpvs).
__device__ inline float nam_deposit(float PP, float& Tdum, float& WVdum, float RHgrd,
                                    const float* __restrict__ tbpvs,
                                    float c1xpvs, float c2xpvs, int* __restrict__ nonconv,
                                    int* __restrict__ warn2, int i1, int j1, int l1)
{
    const double RHLIMIT  = FER_DEPOSIT_RHLIMIT;
    const double RHLIMIT1 = -RHLIMIT;

    float esi  = fminf(1000.0f * nam_fpvs(Tdum, tbpvs, c1xpvs, c2xpvs), 0.99f * PP);  // Sat vapor press w/r/t ice
    float ws   = RHgrd * FER_EP2 * esi / (PP - esi);                                 // Saturation mixing ratio
    float dwv  = WVdum - ws;                                                         // Deficit mixing ratio
    float ssat = dwv / ws;                                                           // Supersaturation ratio
    float deposit = 0.0f;

    const float dwv_inp = dwv;                                                       // DWVinp
    const float tinp    = Tdum;                                                      // Tinp (WARN2 record)
    const float ssatinp = ssat;                                                      // SSATinp (WARN2 record)

    int niter;
    for (niter = 1; niter <= FER_MAX_ITERATIONS; ++niter) {
        const double dep = (double)(dwv / (1.0f + FER_XLS3 * ws / (Tdum * Tdum)));   // Asai (1965)
        Tdum    = (float)((double)Tdum + (double)FER_XLS1 * dep);                    // Updated temperature
        WVdum   = (float)((double)WVdum - dep);                                      // Updated ice mixing ratio
        deposit = (float)((double)deposit + dep);                                    // Total ice deposition

        esi  = fminf(1000.0f * nam_fpvs(Tdum, tbpvs, c1xpvs, c2xpvs), 0.99f * PP);   // Updated sat vapor press
        ws   = RHgrd * FER_EP2 * esi / (PP - esi);                                   // Updated sat mixing ratio
        dwv  = WVdum - ws;                                                           // Updated deficit
        ssat = dwv / ws;                                                             // Grid-scale supersaturation

        const double ssat_d = (double)ssat;
        if (ssat_d >= RHLIMIT1 && ssat_d <= RHLIMIT) break;                          // Exit if SSAT is near 0
    }

    if (niter > FER_MAX_ITERATIONS) {
        // Too many iterations: Fortran WARN2 (written once per run), SSAT=0 and DEPOSIT=DWVinp.
        // Every overflow is counted in *nonconv. The first overflow in the run claims warn2 (CAS on word 0)
        // and stores its record (1-based I,J,L; TC; SSATinp; DWVinp) for the host report, which reads it
        // outside the step path (phys_fer_deposit.report_warn2).
        atomicAdd(nonconv, 1);
        if (atomicCAS(warn2, 0, 1) == 0) {
            warn2[1] = i1;
            warn2[2] = j1;
            warn2[3] = l1;
            warn2[4] = __float_as_int(tinp - FER_T0C);
            warn2[5] = __float_as_int(ssatinp);
            warn2[6] = __float_as_int(dwv_inp);
        }
        deposit = dwv_inp;
    }
    return deposit;
}

// One thread per cell (grid_2d, level in blockIdx.z). pp: PP in Pa [LM,JM,IM]; tdum/wvdum: scratch copies of
// T and Q, updated in place; pidep: DEPOSIT result; coef = {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0}, device, read only.
extern "C" __global__ void fer_deposit_eval(const float* __restrict__ pp,
                                            float* __restrict__ tdum,
                                            float* __restrict__ wvdum,
                                            float* __restrict__ pidep,
                                            const float* __restrict__ tbpvs,
                                            const float* __restrict__ coef,
                                            float rhgrd,
                                            int* __restrict__ nonconv,
                                            int* __restrict__ warn2,
                                            int im, int jm, int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t k = IDX3(i0, j0, l0, im, jm);
    float t  = tdum[k];
    float wv = wvdum[k];
    pidep[k] = nam_deposit(pp[k], t, wv, rhgrd, tbpvs, coef[0], coef[1], nonconv,
                           warn2, (int)i0 + 1, (int)j0 + 1, (int)l0 + 1);
    tdum[k]  = t;
    wvdum[k] = wv;
}
