// nam-gpu unit dyn-adv2-q2-mid: ADV2 setup for a tracer group.
//   (1) floor every species of the group at the ADV2 epsq (1e-20), every level;
//   (2) Q2 (2*TKE) from interfaces to midlayers, in place, level LM down to 2, then level 1;
//   (3) s1 = sqrt(s) for every species of the group (sqrt transform for positive definiteness);
//   (4) pdop = (pd + pdo) * 0.5 on H points (hydrostatic pressure difference read by the CN section).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:4238-4461 (nam.v4.2.11, 83df4e13c1).
//
// Index ranges: its_h1 = max(its-1, ids) = 1 and ite_h1 = min(ite+1, ide) = IM
// (module_DM_PARALLEL.F90:808-809); the same for j. The single-card regional layout covers every
// column. The MPI/OpenMP looplimits split (ENABLE_SMP branch) is dropped.
//
// Everything in this range is column-local (no horizontal stencil), so one thread per column
// (grid_cols) runs the whole range for its column. The Q2 interpolation is an in-place descending
// recurrence over level; the thread runs it in the same order, so level L-1 is still the old value
// when level L is written, as in the Fortran.
//
// Not in this unit (later units): TCS, the Adams-Bashforth and Crank-Nicholson vertical advection,
// the horizontal advection, the boundary branch, MONO and UPDATES.

#include "nmmb_common.cuh"

// ADV2 local parameter epsq = 1.e-20 (the routine's parameter block). Not NMMB_EPSQ from nmmb_common.cuh.
#define ADV2_EPSQ 1.0e-20f

extern "C" __global__ void adv2_q2_mid_col(
    const int IM, const int JM, const int LM,
    const int kss, const int kse,            // tracer group, Fortran 1-based slots
    const int indx_q2,                       // Fortran 1-based slot of Q2 (INDX_Q2 = 3)
    const float* __restrict__ epsq2,         // EPSQ2 [LM]            in
    float* __restrict__ s,                   // TRACERS [NT,LM,JM,IM] inout
    float* __restrict__ s1,                  // S1 [NT,LM,JM,IM]      out
    const float* __restrict__ pd,            // PD [JM,IM]            in
    const float* __restrict__ pdo,           // PDO [JM,IM]           in
    float* __restrict__ pdop)                // PDOP [JM,IM]          out
{
    COLUMN_IJ(IM, JM)

    // (1) floor: s = max(s, epsq) for every species of the group, every level (Fortran 4388-4396)
    for (int ks = kss; ks <= kse; ++ks) {
        const int n0 = ks - 1;
        for (int l0 = 0; l0 < LM; ++l0) {
            const size_t c = IDX4(i0, j0, l0, n0, IM, JM, LM);
            s[c] = fmaxf(s[c], ADV2_EPSQ);
        }
    }

    // (2) Q2 from interfaces to midlayers (Fortran 4400-4414). Only when Q2 is in the group.
    if (kss <= indx_q2 && indx_q2 <= kse) {
        const int n2 = indx_q2 - 1;
        // Fortran L = LM..2 (descending). L0 = L-1 zero-based, so L0 = LM-1..1.
        for (int L0 = LM - 1; L0 >= 1; --L0) {
            const size_t a = IDX4(i0, j0, L0,     n2, IM, JM, LM);
            const size_t b = IDX4(i0, j0, L0 - 1, n2, IM, JM, LM);
            s[a] = fmaxf((s[a] + s[b]) * 0.5f, (epsq2[L0] + epsq2[L0 - 1]) * 0.5f);
        }
        // Fortran L = 1
        const size_t a1 = IDX4(i0, j0, 0, n2, IM, JM, LM);
        s[a1] = fmaxf((s[a1] + epsq2[0]) * 0.5f, epsq2[0]);
    }

    // (3) s1 = sqrt(s) for every species of the group (Fortran 4416-4427)
    for (int ks = kss; ks <= kse; ++ks) {
        const int n0 = ks - 1;
        for (int l0 = 0; l0 < LM; ++l0) {
            const size_t c = IDX4(i0, j0, l0, n0, IM, JM, LM);
            s1[c] = sqrtf(s[c]);
        }
    }

    // (4) pdop = (pd + pdo) * 0.5 on H points (Fortran 4439-4443)
    const size_t c2 = IDX2(i0, j0, IM);
    pdop[c2] = (pd[c2] + pdo[c2]) * 0.5f;
}
