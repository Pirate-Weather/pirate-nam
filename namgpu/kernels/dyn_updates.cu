// dyn_updates.cu -- apply the tracer tendency to the tracers (passive scalar update).
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:2353-2405 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine UPDATES(lm, ntracers, kss, kse, s, tcs):
//     do k = kss, kse ; do l = 1, lm ; do j = jts_b1, jte_b1 ; do i = its_b1, ite_b1
//         s(i,j,l,k)   = s(i,j,l,k) + tcs(i,j,l,k)
//         tcs(i,j,l,k) = 0.
//
// Mapping (CONTRACT.md sections 1 and 3): s = TRACERS[n, l, j, i] (n = k-1), tcs = TRACERS_TEND
// with the same shape. Fortran (i,j,l,k) -> IDX4(i-1, j-1, l-1, k-1, IM, JM, LM).
// kss and kse are the 1-based tracer range passed by the call sites (SOLVER_RUN 4393 and 4543).
//
// Loop limits: NMMB _b1 limits on a single full domain (module_DM_PARALLEL.F90 looplimits,
// ids = 1, ide = IM): its_b1 = max(its, ids+1) = 2, ite_b1 = min(ite, ide-1) = IM-1, and the same
// in j. So the kernel updates only 1-based i in [2, IM-1], j in [2, JM-1] (zero-based [1, IM-2]).
// Points outside that range (the boundary rows) keep both s and tcs unchanged, as in the Fortran.
//
// Thread mapping: one thread per (i, j) with the level in blockIdx.z (grid_2d); each thread runs
// its own tracer loop over k. Every element is written by exactly one thread, so no ordering or
// reduction is involved.
//
// Fortran lines not mapped to a term: the ENABLE_SMP OpenMP directives (2355-2357, 2390-2392)
// are host threading only (jts_b1/jte_b1 loop over j when OpenMP is off). The unused dummy
// declarations and the "temporary arguments" comments carry no code.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_updates(
    const int lm,                          // total number of levels (Fortran LM)
    const int im,                          // IM
    const int jm,                          // JM
    const int ntr,                         // total number of tracers (Fortran NUM_TRACERS_TOTAL)
    const int kss,                         // first tracer updated, 1-based
    const int kse,                         // last tracer updated, 1-based
    float* __restrict__ s,                 // TRACERS [NTR, LM, JM, IM], updated in place
    float* __restrict__ tcs)               // TRACERS_TEND [NTR, LM, JM, IM], zeroed in the interior
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    // NMMB _b1 limits for the full domain: 1-based i, j in [2, IM-1], [2, JM-1].
    if (i0 < 1 || i0 > im - 2 || j0 < 1 || j0 > jm - 2) return;

    for (int k0 = kss - 1; k0 <= kse - 1; ++k0) {
        if (k0 < 0 || k0 >= ntr) continue;
        const size_t ix = IDX4(i0, j0, l0, k0, im, jm, lm);
        s[ix] = s[ix] + tcs[ix];           // s(i,j,l,k) = s(i,j,l,k) + tcs(i,j,l,k)
        tcs[ix] = 0.0f;                    // tcs(i,j,l,k) = 0.
    }
}
