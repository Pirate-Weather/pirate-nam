// nam-gpu unit dyn-pdtsdt: surface-pressure tendency, vertical mass flux and PD update.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:1045-1228 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine pdtsdt, lm = LM, dt = DT, sg2 = SG2(1:lm+1), temporaries div/tdiv):
//   [A] l=2..lm, b1 range:   div(i,j,l) = div(i,j,l-1) + div(i,j,l)      (cumulative, in place)
//   [B] h2 range:            psdt(i,j) = 0.;  pdo(i,j) = pd(i,j)
//   [C] b1 range:            psdt(i,j) = -div(i,j,lm);  pd(i,j) = psdt(i,j)*dt + pd(i,j)
//   [D] l=1..lm-1, b1 range: psgdt(i,j,l) = -(-tdiv(i,j,lm)*sg2(l+1) + tdiv(i,j,l))
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, regional, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = max(its, ids+1) = 2 .. ite_b1 = min(ite, ide-1) = IM-1   (1-based)  -> i0 in [1, IM-2]
//   jts_b1 = 2 .. jte_b1 = JM-1                                          (1-based)  -> j0 in [1, JM-2]
//   its_h2 = max(its-2, ids) = 1 .. ite_h2 = min(ite+2, ide) = IM        (1-based)  -> full domain
//   The MPI/OpenMP looplimits split is dropped (one card, one thread per column).
//
// Staggering: PD, PDO, PSDT on H points; DIV, TDIV, PSGDT on H points with LM levels (PSGDT LM+1 allocated,
// levels 1..LM-1 written). Layout per CONTRACT section 1: X[l0, j0, i0] with IDX3.
//
// One thread per column (grid_cols). The level scan in [A] and the top-level read in [C]/[D] are per column.
// Fortran `div` is modified in place by [A] and is an output state (DIV) used later, so it is written back.
// Fortran `tdiv` is read only here (it is produced by DHT, dyn-dht).
// No line of the Fortran is skipped: the OpenMP block (ENABLE_SMP) and the commented-out MPI diagnostic
// reductions (task_change, global_change, loc_npts, glb_npts) are not ported (no effect on the state).

#include "nmmb_common.cuh"

extern "C" __global__ void pdtsdt_col(
    const int IM, const int JM, const int LM,
    const float dt,
    const float* __restrict__ sg2,      // SG2  [LM+1]        Fortran sg2(1:lm+1)
    float* __restrict__ pd,             // PD   [JM,IM]       inout
    float* __restrict__ pdo,            // PDO  [JM,IM]       out
    float* __restrict__ psdt,           // PSDT [JM,IM]       out
    float* __restrict__ psgdt,          // PSGDT[LM+1,JM,IM]  out, levels 0..LM-2
    float* __restrict__ div,            // DIV  [LM,JM,IM]    inout (cumulative sum in place)
    const float* __restrict__ tdiv)     // TDIV [LM,JM,IM]    in
{
    COLUMN_IJ(IM, JM)

    const size_t c2 = IDX2(i0, j0, IM);

    // [B] h2 range (full domain in the single-process regional layout)
    pdo[c2]  = pd[c2];
    psdt[c2] = 0.0f;

    // b1 range: the boundary rows and columns are not updated
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    // [A] cumulative mass divergence down the column, Fortran l=2..lm -> l0=1..LM-1
    for (int l0 = 1; l0 < LM; ++l0) {
        div[IDX3(i0, j0, l0, IM, JM)] = div[IDX3(i0, j0, l0 - 1, IM, JM)] + div[IDX3(i0, j0, l0, IM, JM)];
    }

    // [C] surface pressure tendency and PD update (uses the cumulative total at level lm)
    const float ps = -div[IDX3(i0, j0, LM - 1, IM, JM)];
    psdt[c2] = ps;
    pd[c2]   = ps * dt + pd[c2];

    // [D] vertical mass flux on the interfaces 1..lm-1, Fortran l=1..lm-1 -> l0=0..LM-2
    const float tdiv_top = tdiv[IDX3(i0, j0, LM - 1, IM, JM)];
    for (int l0 = 0; l0 < LM - 1; ++l0) {
        const float tdiv_l = tdiv[IDX3(i0, j0, l0, IM, JM)];
        psgdt[IDX3(i0, j0, l0, IM, JM)] = -(-tdiv_top * sg2[l0 + 1] + tdiv_l);   // sg2(l+1), Fortran 1-based
    }
}
