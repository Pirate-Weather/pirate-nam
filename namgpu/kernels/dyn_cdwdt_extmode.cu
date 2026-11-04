// nam-gpu unit dyn-cdwdt-extmode: cos^2 attenuation of dw/dt in the top layers, then removal of the
// external (barotropic) mode from dw/dt.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3700-3756 (nam.v4.2.11, 83df4e13c1).
//   3700-3726: cos^2 attenuation of dwdt in the top layers (subroutine cdwdt, dwdt control range)
//   3728-3756: external mode taken out of dwdt
//   3758-3770: global swaphn/polehn and halo_exch are dropped (single card, regional, global=false)
//   3773-...:  spatial filter of dwdt is skipped because wad = 0 (configfile, nam_configfile:3313)
//
// Fortran (cdwdt; lm = LM, dsg2, pdsg1, psgml1, pdo, dwdt; fwhy, slpd, pi as in cdwdt):
//   [A] l=1..lm: why(l) = -99
//   [B] lsltp = 0; sltp = (psgml1(1)+psgml1(2))*0.5
//       do l=1,lm-1: arg = ((psgml1(l)+psgml1(l+1))*0.5 - sltp)/slpd
//                    if (arg > 1) exit
//                    why(l) = 1 - fwhy*cos(arg*pi*0.5)**2 ; lsltp = l
//   [C] l=1..lsltp, b1 range: dwdt(i,j,l) = dwdt(i,j,l)*why(l)
//   [D] b1 range: ttb(i,j) = 0
//   [E] l=1..lm, b1 range: ttb(i,j) = (dsg2(l)*pdo(i,j)+pdsg1(l))*dwdt(i,j,l) + ttb(i,j)
//   [F] b1 range: ttb(i,j) = ttb(i,j)/pdo(i,j)
//   [G] l=1..lm, b1 range: dwdt(i,j,l) = dwdt(i,j,l) - ttb(i,j)
//   ttb is a local array in the Fortran (not a state field), so it stays in a register here.
//
// why(l) and lsltp depend only on psgml1 (vertical coordinate, grid constant), so every thread
// recomputes them from the LM-length psgml1 vector instead of reading a table.
// Fortran pi is double (module_CONSTANTS pi); the cosine here is single precision (kfpt), as the
// rest of the NAM single-precision path. Not bitwise to native; fast numerics.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   jts_b1 = 2 .. jte_b1 = JM-1 and its_b1 = 2 .. ite_b1 = IM-1 (1-based)
//   -> zero-based i0 in [1, IM-2], j0 in [1, JM-2]. Boundary columns are not touched (as in Fortran).
//
// Layout: DWDT [LM,JM,IM] (inout), PDO [JM,IM], PSGML1 [LM], DSG2 [LM], PDSG1 [LM] (CONTRACT 1).
// One thread per column: the attenuation, the column sum, the divide and the subtraction all run
// over the same column in Fortran level order, so no cross-column dependence and no host sync.
// The MPI/OpenMP looplimits split (ENABLE_SMP) is dropped.

#include "nmmb_common.cuh"

extern "C" __global__ void dwdt_extmode_col(
    const int IM, const int JM, const int LM,
    const float fwhy,                   // fwhy  (cdwdt dwdt control factor, configfile fwhy = 1.)
    const float slpd,                   // slpd  (cdwdt control layer depth, Pa, configfile slpd = 1500.)
    const float* __restrict__ psgml1,   // PSGML1 [LM]         in
    const float* __restrict__ dsg2,     // DSG2   [LM]         in
    const float* __restrict__ pdsg1,    // PDSG1  [LM]         in
    const float* __restrict__ pdo,      // PDO    [JM,IM]      in
    float* __restrict__ dwdt)           // DWDT   [LM,JM,IM]   inout
{
    COLUMN_IJ(IM, JM)

    // b1 range only: boundary rows and columns are left as they are
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const size_t c2 = IDX2(i0, j0, IM);
    const float p = pdo[c2];

    // [A][B] lsltp (1-based count of attenuated layers) from the vertical coordinate.
    // Fortran: do l=1,lm-1 ... if (arg > 1) exit ; lsltp = l
    const float sltp = (psgml1[0] + psgml1[1]) * 0.5f;
    int lsltp = 0;
    for (int l0 = 0; l0 < LM - 1; ++l0) {
        const float arg = ((psgml1[l0] + psgml1[l0 + 1]) * 0.5f - sltp) / slpd;
        if (arg > 1.0f) break;
        lsltp = l0 + 1;
    }

    // [C] cos^2 attenuation for l = 1..lsltp (l0 = 0..lsltp-1); why(l) uses psgml1(l), psgml1(l+1)
    for (int l0 = 0; l0 < lsltp; ++l0) {
        const float arg = ((psgml1[l0] + psgml1[l0 + 1]) * 0.5f - sltp) / slpd;
        const float cs = cosf(arg * 3.1415927f * 0.5f);
        const float why = 1.0f - fwhy * cs * cs;
        const size_t c3 = IDX3(i0, j0, l0, IM, JM);
        dwdt[c3] = dwdt[c3] * why;
    }

    // [D][E][F] external mode: column-weighted mean of the attenuated dwdt, divided by pdo
    float ttb = 0.0f;
    for (int l0 = 0; l0 < LM; ++l0) {
        ttb = (dsg2[l0] * p + pdsg1[l0]) * dwdt[IDX3(i0, j0, l0, IM, JM)] + ttb;
    }
    ttb = ttb / p;

    // [G] remove the external mode from every level
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t c3 = IDX3(i0, j0, l0, IM, JM);
        dwdt[c3] = dwdt[c3] - ttb;
    }
}
