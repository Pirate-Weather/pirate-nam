// phys_noah_snfrac.cu -- SNFRAC: snow cover fraction (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The NOAA author/history
// header block of the source module (module_LS_NOAHLSM.F90, lines 12-25) and the header of
// SUBROUTINE SNFRAC (lines 3931-3942) are reproduced verbatim as the notice:
//
// !-----------------------------------------------------------------------
// !
//       MODULE MODULE_LS_NOAHLSM
// !
// !-----------------------------------------------------------------------
// !***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
// !-----------------------------------------------------------------------
// !
// ! HISTORY LOG:
// !
// !   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
// !               Vasic - SOILW limitted to positive values.
// !
// !-----------------------------------------------------------------------
//
//       SUBROUTINE SNFRAC (SNEQV,SNUP,SALP,SNOWH,SNCOVR)
//
// ! ----------------------------------------------------------------------
// ! SUBROUTINE SNFRAC
// ! ----------------------------------------------------------------------
// ! CALCULATE SNOW FRACTION (0 -> 1)
// ! SNEQV   SNOW WATER EQUIVALENT (M)
// ! SNUP    THRESHOLD SNEQV DEPTH ABOVE WHICH SNCOVR=1
// ! SALP    TUNING PARAMETER
// ! SNCOVR  FRACTIONAL SNOW COVER
// ! ----------------------------------------------------------------------
//       IMPLICIT NONE
//
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:3931-3968 (nam.v4.2.11, 83df4e13c1).
//
// Start point and notice: ArWen (Apache-2.0, namgpu/third_party/arwen/LICENSE, Copyright 2026 ArWen
// contributors), core/kernels/noah.cu, function noah_snfrac. Changes from that copy:
//   * Written as a point kernel (one thread per (i, j), CONTRACT section 3) with per-column
//     arrays, because SNUP is vegetation-dependent (SNUP = SNUPTBL(VEGTYP), module_LS_NOAHLSM
//     REDPRM) and SALP is per point in SFLX. The Fortran is a SUBROUTINE; its caller is
//     SFLX (module_LS_NOAHLSM.F90:1991, after SNUP = 4*SNUP at 1989).
//   * Fortran REAL = float (kfpt=single). EXP is expf.
//
// Fortran (SUBROUTINE SNFRAC, lines 3931-3968), every executed statement in order:
//   IF (SNEQV < SNUP) THEN
//      RSNOW  = SNEQV / SNUP
//      SNCOVR = 1. - ( EXP( - SALP * RSNOW) - RSNOW * EXP( - SALP ) )
//   ELSE
//      SNCOVR = 1.0
//   END IF
// The Fortran dummy SNOWH is INTENT(IN) and is not referenced by any executed statement (the
// Dickinson and Marshall alternatives are commented out in the source), so the kernel takes it
// for the calling convention but does not read it. The Z0N local is never assigned in executed
// code and has no counterpart. No MPI, halo, OpenMP or print statements in this range.
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1. All arrays float32 [JM, IM].

#include "nmmb_common.cuh"

extern "C" __global__ void phys_noah_snfrac(
    const float* __restrict__ sneqv,   // SNEQV  snow water equivalent (m)                  [JM, IM]
    const float* __restrict__ snup,    // SNUP   threshold snow depth for 100% cover (m)    [JM, IM]
    const float* __restrict__ salp,    // SALP   snow cover tuning parameter                [JM, IM]
    const float* __restrict__ snowh,   // SNOWH  snow depth (m); not referenced (see above) [JM, IM]
    float* __restrict__ sncovr,        // SNCOVR fractional snow cover, output (0..1)       [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    (void)snowh;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);

    const float sneqv_c = sneqv[n];
    const float snup_c = snup[n];
    const float salp_c = salp[n];

    float sncovr_c;
    if (sneqv_c < snup_c) {
        const float rsnow = sneqv_c / snup_c;
        sncovr_c = 1.0f - (expf(-salp_c * rsnow) - rsnow * expf(-salp_c));
    } else {
        sncovr_c = 1.0f;
    }
    sncovr[n] = sncovr_c;
}
