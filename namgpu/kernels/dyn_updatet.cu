// dyn_updatet.cu -- apply the temperature tendency: T = T + TCT, then TCT = 0.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:2410-2456 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine UPDATET(lm, t, tct): the whole body is
//     t(i,j,l)   = t(i,j,l) + tct(i,j,l)
//     tct(i,j,l) = 0.
// over l = 1..LM, j = jts_b1..jte_b1, i = its_b1..ite_b1.
//
// Mapping (CONTRACT.md section 1): Fortran A(i,j,l) -> A[(l-1), (j-1), (i-1)] with IDX3.
//   T and TCT: LM levels, [LM, JM, IM] float32.
// Loop limits: NMMB _b1 limits on a single full domain (module_DM_PARALLEL.F90 looplimits,
// ids = 1, ide = IM): its_b1 = max(its, ids+1) = 2, ite_b1 = min(ite, ide-1) = IM-1, and the same
// in j. So the kernel updates only 1-based i in [2, IM-1], j in [2, JM-1] (zero-based [1, IM-2]).
// Points on the outer rows keep their T and TCT, exactly as in the Fortran.
//
// Levels are independent (no recurrence along l), so one thread per (i, j, l) point, with the
// level in blockIdx.z (grid_2d). Both statements are one-to-one with the Fortran body.
// Fortran lines not mapped to a term: the ENABLE_SMP OpenMP block (2415-2426 and its end) is
// host threading only (jstart/jstop = jts_b1/jte_b1 when OpenMP is off).
//
// The source has no multiply and no divide, so FMA contraction cannot change the result.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_updatet(
    const int lm,                          // total number of levels (Fortran LM)
    const int im,                          // IM
    const int jm,                          // JM
    float* __restrict__ t,                 // [LM, JM, IM] temperature, updated in place
    float* __restrict__ tct)               // [LM, JM, IM] temperature time change, reset to 0
{
    POINT_IJL
    // NMMB _b1 limits for the full domain: 1-based i, j in [2, IM-1], [2, JM-1].
    if (i0 < 1 || i0 > im - 2 || j0 < 1 || j0 > jm - 2 || l0 >= lm) return;

    const size_t ix = IDX3(i0, j0, l0, im, jm);
    t[ix] = t[ix] + tct[ix];               // t(i,j,l) = t(i,j,l) + tct(i,j,l)
    tct[ix] = 0.0f;                        // tct(i,j,l) = 0.
}
