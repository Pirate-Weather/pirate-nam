// WOOF-NAM unit dyn-mono-q2-back: after the MONO species loop, interpolate the Q2 tendency and Q2
// itself from the mid-layers back to the interfaces (the q2_check block at the end of MONO).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module and subroutine header is kept here
// as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
//   subroutine mono (monotonisation of the species tracer arrays)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:5204-5235 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (1-based), inside q2_check (kss <= indx_q2 <= kse; the wrapper tests this on the host,
// it is a configuration constant, so no device branch is needed):
//   [A] l=1..lm, j=jts_b1..jte_b1, i=its_b1..ite_b1:
//         tcs(i,j,l,q2) = (dsg2(l)*pd(i,j) + pdsg1(l)) * tcs(i,j,l,q2)
//   [B] l=1..lm-1, same i,j:
//         tcs(i,j,l,q2) = (tcs(i,j,l,q2) + tcs(i,j,l+1,q2))
//                         / ((dsg2(l)*pd(i,j)+pdsg1(l)) + (dsg2(l+1)*pd(i,j)+pdsg1(l+1)))
//         s(i,j,l,q2)   = max(0.5*(s(i,j,l,q2) + s(i,j,l+1,q2)), epsq2(l))
//   [C] tcs(i,j,lm,q2) = 0 ;  s(i,j,lm,q2) = epsq2(lm)
// [A] must finish for every level before [B] reads tcs(l+1): with one thread per column the two
// passes run in order inside the thread, and [B] at level l reads s(l+1) and tcs(l+1) before they
// are overwritten (ascending l), exactly as the Fortran sequence does. Each column is independent,
// so one thread per column with no halo is exact.
//
// The Fortran s(...,indx_q2) is the Q2 species slot of int_state%TRACERS, and tcs(...,indx_q2) is the
// same slot of int_state%TRACERS_TEND. The wrapper passes those two slabs (TRACERS[indx_q2-1] and
// TRACERS_TEND[indx_q2-1]), so the kernel writes the tracer arrays the rest of the step reads.
//
// Index ranges: its_b1 = 2 and ite_b1 = IM-1 (1-based), so zero-based i0 in [1, IM-2]; the same for
// j0. Other columns are left untouched, as in the Fortran (their tendencies are outside b1).
//
// One thread per column (grid_cols launch). No host sync, no device allocation.
// Fast numerics: fmad contraction allowed (NVRTC --fmad=true); single precision throughout.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_mono_q2_back(
    float* __restrict__ tcs_q2,          // TCS (Q2 species tendency)  [LM,JM,IM]  in/out
    float* __restrict__ q2,              // S   (Q2 species, Q2 itself) [LM,JM,IM] in/out
    const float* __restrict__ pd,        // PD     [JM,IM]
    const float* __restrict__ dsg2,      // DSG2   [LM]  sigma-range layer thickness
    const float* __restrict__ pdsg1,     // PDSG1  [LM]  pressure-range layer thickness
    const float* __restrict__ epsq2,     // EPSQ2  [LM]  2tke floor per level
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)

    // loop limits its_b1..ite_b1, jts_b1..jte_b1 (1-based [2, IM-1], [2, JM-1])
    if (i0 < 1 || i0 > im - 2 || j0 < 1 || j0 > jm - 2) return;

    const float pdv = pd[IDX2(i0, j0, im)];

    // [A] tendency scaling, every level
    for (int l0 = 0; l0 < lm; ++l0) {
        const size_t c = IDX3(i0, j0, l0, im, jm);
        const float f = dsg2[l0] * pdv + pdsg1[l0];
        tcs_q2[c] = f * tcs_q2[c];
    }

    // [B] mid-layer to interface interpolation, ascending l (reads level l+1 before it is written)
    for (int l0 = 0; l0 < lm - 1; ++l0) {
        const size_t c  = IDX3(i0, j0, l0,     im, jm);
        const size_t cu = IDX3(i0, j0, l0 + 1, im, jm);
        const float den = (dsg2[l0] * pdv + pdsg1[l0]) + (dsg2[l0 + 1] * pdv + pdsg1[l0 + 1]);
        tcs_q2[c] = (tcs_q2[c] + tcs_q2[cu]) / den;
        q2[c] = fmaxf(0.5f * (q2[c] + q2[cu]), epsq2[l0]);
    }

    // [C] top-of-column boundary values (level lm is the lowest model layer)
    const size_t cl = IDX3(i0, j0, lm - 1, im, jm);
    tcs_q2[cl] = 0.0f;
    q2[cl] = epsq2[lm - 1];
}
