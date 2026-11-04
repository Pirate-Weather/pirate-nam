// phys_noah_alcalc.cu -- NOAHLSM snow-cover and albedo caller branch plus ALCALC (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:1976-2001 (the albedo and snow-cover
// branch of NOAHLSM, which calls SNFRAC and ALCALC) and SUBROUTINE ALCALC at 2281-2328.
//
// Fortran, every executed statement, per land point in the NOAHLSM B1 loop:
//   IF (XICE > 0.5) THEN                          ! sea ice (ICE = 1)
//     SNCOVR = 1.0 ; ALBEDO = 0.65                (lines 2000-2001)
//   ELSE                                          ! land (ICE = 0)
//     IF (SNEQV == 0.0) THEN
//       SNCOVR = 0.0 ; ALBEDO = ALB               (lines 1982-1983)
//     ELSE
//       SNUP = 4*SNUP ; CALL SNFRAC               (lines 1989-1991; SNFRAC output arrives in SNCOVR)
//       SNCOVR = MIN(SNCOVR, 0.98)                (line 1993)
//       CALL ALCALC: ALBEDO = ALB + SNCOVR*(SNOALB-ALB); IF (ALBEDO > SNOALB) ALBEDO = SNOALB
//                                                  (lines 2309, 2326)
//   END IF
// SNEQV is the NOAHLSM snow water equivalent in metres; the port's SNOW state is SNEQV*1000 (mm),
// and SNEQV == 0 is tested on the mm value (exactly equivalent under the x0.001 scaling).
// SNUP scaling and SNFRAC run in the producer (phys_noah_snfrac); SNCOVR arrives as its output.
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1. Arrays are float32 [JM, IM]
// (C order, IDX2). Fortran REAL = float (kfpt=single). Points outside the B1 limits are not
// written (the NOAHLSM caller computes only B1 points: i in [1, IM-2], j in [1, JM-2] 0-based).

#include "nmmb_common.cuh"

#define ALCALC_ICE_THRESHOLD 0.5f   // XICE > 0.5 (line 1976 caller, ICE = 1)
#define ALCALC_SNCOVR_MAX    0.98f  // MIN(SNCOVR, 0.98) (line 1993)
#define ALCALC_SEAICE_ALB    0.65f  // sea-ice albedo (line 2001)

extern "C" __global__ void phys_noah_alcalc(
    const float* __restrict__ alb,     // ALB     snow-free albedo                         [JM, IM]
    const float* __restrict__ snoalb,  // SNOALB  maximum deep-snow albedo                 [JM, IM]
    const float* __restrict__ snow,    // SNOW    snow water equivalent (mm), SNEQV*1000   [JM, IM]
    const float* __restrict__ sice,    // SICE    sea-ice fraction (XICE)                  [JM, IM]
    float* __restrict__ sncovr,        // in: SNFRAC output; out: SNCOVR after the branch  [JM, IM]
    float* __restrict__ albedo,        // ALBEDO  surface albedo with snow (output)        [JM, IM]
    int its, int ite, int jts, int jte,  // inclusive 0-based B1 limits
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    if (i0 < its || i0 > ite || j0 < jts || j0 > jte) return;
    const size_t n = IDX2(i0, j0, im);

    if (sice[n] > ALCALC_ICE_THRESHOLD) {          // sea ice: no ALCALC call in NOAHLSM
        sncovr[n] = 1.0f;
        albedo[n] = ALCALC_SEAICE_ALB;
        return;
    }
    if (snow[n] == 0.0f) {                         // no snow: SNFRAC not called
        sncovr[n] = 0.0f;
        albedo[n] = alb[n];
        return;
    }
    const float sc = fminf(sncovr[n], ALCALC_SNCOVR_MAX);   // line 1993
    sncovr[n] = sc;
    const float a = alb[n];
    const float sa = snoalb[n];
    float ab = a + sc * (sa - a);                  // line 2309
    if (ab > sa) ab = sa;                          // line 2326
    albedo[n] = ab;
}
