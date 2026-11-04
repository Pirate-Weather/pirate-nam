// nam-gpu unit drv-h-to-v-tend: H_TO_V_TEND_3D, update the 3-D V-point arrays (U, V) with the
// H-point wind tendencies (DUDT, DVDT) accumulated over NPHS steps.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   module_H_TO_V.F90, MODULE MODULE_H_TO_V (NCEP NMMB, nam.v4.2.11). Author headers kept as the notice.
// Ported to CUDA for nam-gpu from nmm/module_H_TO_V.F90:128-224 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (H_TO_V_TEND_3D, lines 185-221):
//   DELTA_T = DT*NSTEPS                                   (wrapper, single precision, passed in)
//   DO K=1,LM; DO J=JTS,JTE_B1; DO I=ITS,ITE_B1
//     ARRAY_V(I,J,K) = ARRAY_V(I,J,K) + 0.25*(TEND(I,J,K)+TEND(I+1,J,K)+TEND(I,J+1,K)+TEND(I+1,J+1,K))*DELTA_T
// Called from SOLVER_RUN at 5586/5588 (NPHS, DELTA_T = DT*NPHS) and 5827/5830 (NPRECIP, bmj,
// DELTA_T = DT*NPRECIP). The GFS-physics call at 6998/7000 (NPHS) is a scope exclusion: gfs is false.
// The 2-D variant (H_TO_V_TEND_2D, lines 150-171) is not called in SOLVER_RUN and is not part of this kernel.
//
// Loop limits: ITS_B1 = max(its, ids+1) = 2 and ITE_B1 = min(ite, ide-1) = IM-1 (1-based), so the
// zero-based interior is i0 in [1, IM-2], j0 in [1, JM-2] (module_DM_PARALLEL.F90 looplimits, 792-822).
// Every level l0 in [0, LM) is updated.
// Summation order follows the Fortran expression left to right, as the source states it.

#include "nmmb_common.cuh"

extern "C" __global__ void h_to_v_tend_3d(const int im, const int jm, const int lm,
                                          const float delta_t,
                                          const float* __restrict__ tend,
                                          float* __restrict__ array_v)
{
    POINT_IJL
    if (i0 < 1 || i0 > im - 2 || j0 < 1 || j0 > jm - 2 || l0 >= lm) return;

    const size_t n00 = IDX3(i0,     j0,     l0, im, jm);
    const size_t n10 = IDX3(i0 + 1, j0,     l0, im, jm);
    const size_t n01 = IDX3(i0,     j0 + 1, l0, im, jm);
    const size_t n11 = IDX3(i0 + 1, j0 + 1, l0, im, jm);
    const size_t nv  = IDX3(i0,     j0,     l0, im, jm);

    array_v[nv] = array_v[nv]
                + 0.25f * (tend[n00] + tend[n10] + tend[n01] + tend[n11]) * delta_t;
}
