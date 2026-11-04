// WOOF-NAM unit dyn-cdwdt-vert-adv: vertical advection of W (column), added into DWDT.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:3593-3652 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine cdwdt, section "vertical advection of w"; lm = LM, dsg2 = DSG2, pdsg1 = PDSG1,
// pdo = PDO, psgdt = PSGDT, w = W, dwdt = DWDT, ttb/tta = 2-D running interface fluxes):
//   [A] all i,j:            ttb(i,j) = 0
//   [B] l=1..lm-1, b1 range: tta(i,j) = (w(i,j,l+1)-w(i,j,l))*psgdt(i,j,l)*0.5
//                            dwdt(i,j,l) = (tta+ttb)/(dsg2(l)*pdo(i,j)+pdsg1(l)) + dwdt(i,j,l)
//                            ttb(i,j) = tta
//   [C] all i,j:            dwdt(i,j,lm) = ttb(i,j)/(dsg2(lm)*pdo(i,j)+pdsg1(lm)) + dwdt(i,j,lm)
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2 .. ite_b1 = IM-1 and jts_b1 = 2 .. jte_b1 = JM-1 (1-based), i.e. zero-based
//   i0 in [1, IM-2], j0 in [1, JM-2]. Section [C] runs over the full domain, but on the boundary
//   ring ttb is still 0 (only the interior ever sets it), so there the update is
//   dwdt = 0/(...) + dwdt = dwdt: the boundary ring is left untouched, as written.
//
// Layout: W [LM,JM,IM], PSGDT [LM+1,JM,IM] (levels 0..LM-2 used), DWDT [LM,JM,IM] (inout),
// PDO [JM,IM], DSG2 [LM], PDSG1 [LM] (CONTRACT section 1).
// The l-loop is a column recurrence (ttb carries level to level), so one thread per column runs
// the l-loop in Fortran order. No cross-column dependence, one launch, no host sync.
// The MPI/OpenMP looplimits split (ENABLE_SMP) is dropped (one card, one thread per column).

#include "nmmb_common.cuh"

extern "C" __global__ void cdwdt_vert_adv_col(
    const int IM, const int JM, const int LM,
    const float* __restrict__ dsg2,     // DSG2  [LM]          in
    const float* __restrict__ pdsg1,    // PDSG1 [LM]          in
    const float* __restrict__ pdo,      // PDO   [JM,IM]       in
    const float* __restrict__ w,        // W     [LM,JM,IM]    in
    const float* __restrict__ psgdt,    // PSGDT [LM+1,JM,IM]  in (levels 0..LM-2)
    float* __restrict__ dwdt)           // DWDT  [LM,JM,IM]    inout
{
    COLUMN_IJ(IM, JM)

    // b1 range only: boundary ring is left as it is (see header)
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const size_t c2 = IDX2(i0, j0, IM);
    const float p = pdo[c2];

    // [A] ttb = 0 ; [B] Fortran l=1..lm-1 -> l0=0..LM-2, in order
    float ttb = 0.0f;
    for (int l0 = 0; l0 < LM - 1; ++l0) {
        const size_t c3 = IDX3(i0, j0, l0, IM, JM);
        const float tta = (w[IDX3(i0, j0, l0 + 1, IM, JM)] - w[c3]) * psgdt[c3] * 0.5f;
        dwdt[c3] = (tta + ttb) / (dsg2[l0] * p + pdsg1[l0]) + dwdt[c3];
        ttb = tta;
    }

    // [C] top-to-bottom closure at the lowest level (l = lm)
    const size_t cl = IDX3(i0, j0, LM - 1, IM, JM);
    dwdt[cl] = ttb / (dsg2[LM - 1] * p + pdsg1[LM - 1]) + dwdt[cl];
}
