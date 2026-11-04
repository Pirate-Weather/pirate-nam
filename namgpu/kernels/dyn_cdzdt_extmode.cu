// nam-gpu unit dyn-cdzdt-extmode: remove the external (barotropic) mode from W.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3236-3273 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine cdzdt, section "taking external mode out"; lm = LM, dsg2 = DSG2, pdsg1 = PDSG1,
// pdo = PDO, w = W, baro = BARO):
//   [A] b1 range:           baro(i,j) = 0
//   [B] l=1..lm, b1 range:  baro(i,j) = (dsg2(l)*pdo(i,j) + pdsg1(l))*w(i,j,l) + baro(i,j)
//   [C] b1 range:           baro(i,j) = baro(i,j)/pdo(i,j)
//   [D] l=1..lm, b1 range:  w(i,j,l) = w(i,j,l) - baro(i,j)
// BARO is the output the later cdwdt (module_DYNAMICS_ROUTINES, w_tot = w + baro) reads, so it is
// written back to the state.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   jts_b1 = 2 .. jte_b1 = JM-1 and its_b1 = 2 .. ite_b1 = IM-1 (1-based)
//   -> zero-based i0 in [1, IM-2], j0 in [1, JM-2]. Boundary columns are not touched (as in Fortran).
//
// Layout: W [LM,JM,IM], PDO [JM,IM], BARO [JM,IM], DSG2 [LM], PDSG1 [LM] (CONTRACT section 1).
// The four Fortran loops are all over the same column (i,j) and the level sum (l = 1..lm) runs in
// Fortran order, so one thread per column reproduces them exactly: column sum, divide, then the
// subtraction over the same levels. No cross-column dependence, so one launch, no host sync.
// The MPI/OpenMP looplimits split (ENABLE_SMP) is dropped (one card, one thread per column).

#include "nmmb_common.cuh"

extern "C" __global__ void cdzdt_extmode_col(
    const int IM, const int JM, const int LM,
    const float* __restrict__ dsg2,     // DSG2  [LM]          in
    const float* __restrict__ pdsg1,    // PDSG1 [LM]          in
    const float* __restrict__ pdo,      // PDO   [JM,IM]       in
    float* __restrict__ w,              // W     [LM,JM,IM]    inout
    float* __restrict__ baro)           // BARO  [JM,IM]       out (interior only)
{
    COLUMN_IJ(IM, JM)

    // b1 range only: boundary rows and columns are left as they are
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const size_t c2 = IDX2(i0, j0, IM);
    const float p = pdo[c2];

    // [A] + [B] column sum of the barotropic flux, Fortran l=1..lm -> l0=0..LM-1, in order
    float b = 0.0f;
    for (int l0 = 0; l0 < LM; ++l0) {
        b = (dsg2[l0] * p + pdsg1[l0]) * w[IDX3(i0, j0, l0, IM, JM)] + b;
    }

    // [C] divide by PDO
    b = b / p;
    baro[c2] = b;

    // [D] remove the external mode from every level
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t c3 = IDX3(i0, j0, l0, IM, JM);
        w[c3] = w[c3] - b;
    }
}
