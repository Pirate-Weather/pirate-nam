// nam-gpu unit drv-cltend: CLTEND, spread the microphysics/convection temperature change over
// NPRECIP steps (Told, Tadj).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   module_SOLVER_GRID_COMP.F90, SUBROUTINE CLTEND (documentation block: PRGRMMR FERRIER,
//   ORG W/NP22, DATE 01-09-26). Author headers kept as the notice.
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_GRID_COMP.F90:11597-11681 (nam.v4.2.11, 83df4e13c1).
//
// Fortran loop coverage: ITS..ITE x JTS..JTE x 1..LM. On one card the tile is the whole domain
// (CONTRACT.md section 1), so the kernel covers i0 in [0, IM), j0 in [0, JM), l0 in [0, LM).
// Mode ICLTEND (CONTRACT section 2 call sites, SOLVER_RUN 5708 / 5928 / 6008), Fortran branches:
//   icltend < 0  : Told = T                                    (IF ICLTEND<0, lines 11652-11659)
//   icltend == 0 : Tadj = RDTPH*(T - Told); T = Told           (ELSE IF ICLTEND==0, lines 11660-11669)
//   icltend > 0  : T = T + Tadj                                (ELSE, lines 11670-11679)
// RDTPH = 1./REAL(NPRECIP) (line 11661) is computed once on the host in single precision and passed in.
// No Fortran term is dropped: every assignment in CLTEND has its counterpart below.

#include "nmmb_common.cuh"

extern "C" __global__ void cltend(const int im, const int jm, const int lm,
                                  const int icltend, const float rdtph,
                                  float* __restrict__ T,
                                  float* __restrict__ Told,
                                  float* __restrict__ Tadj)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);

    if (icltend < 0) {
        Told[n] = T[n];
    } else if (icltend == 0) {
        Tadj[n] = rdtph * (T[n] - Told[n]);
        T[n] = Told[n];
    } else {
        T[n] = T[n] + Tadj[n];
    }
}
