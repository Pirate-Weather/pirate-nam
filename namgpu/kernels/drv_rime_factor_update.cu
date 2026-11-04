// drv_rime_factor_update.cu -- RIME_FACTOR_UPDATE: rime factor around species advection.
//
// NOTICE: NOAA/NCEP NMMB source, nmm/module_SOLVER_GRID_COMP.F90 (US Government work, no licence
// file in the NAM clone; see NOTICE at the repo root). Routine header kept as the notice:
//   SUBPROGRAM:  RIME_FACTOR_UPDATE
//   PRGRMMR: FERRIER         ORG: W/NP22     DATE: 2013-06-14
//   ABSTRACT: UPDATES THE RIME FACTOR ARRAY AFTER 3D ADVECTION
// Ported to CUDA for WOOF-NAM from nmm/module_SOLVER_GRID_COMP.F90:11688-11771 (nam.v4.2.11, 83df4e13c1).
//
// Two branches, selected by RIME_FACTOR_INPUT (Fortran logical), both called from SOLVER_RUN:
//   RIME_FACTOR_INPUT = .TRUE.  (module_SOLVER_GRID_COMP.F90:4448, before species advection)
//       QG(i,j,k) = QS(i,j,k) * F_RIMEF(i,j,k)           over ITS..ITE, JTS..JTE, K = 1..LM
//       CALL HALO_EXCH(QG,LM,2,2)                          (11761)  -- dropped: single card, no halo
//   RIME_FACTOR_INPUT = .FALSE. (module_SOLVER_GRID_COMP.F90:4590, after species advection)
//       over IMS..IME, JMS..JME, K = 1..LM:
//         if (QG > EPSQ .and. QS > EPSQ)  RIMEF = QG/QS;  F_RIMEF = MIN(50, MAX(1, RIMEF))
//         else                            F_RIMEF = 1
//
// Mapping (CONTRACT.md section 1): Fortran A(i,j,k) -> A[(k-1), (j-1), (i-1)] with IDX3.
// Levels: K = 1 is the model top, exactly as in NMMB; the loop is over every level.
// Loop limits: single card, halo-free (CONTRACT.md section 1). Both kernels cover the domain
// 1..IM x 1..JM x 1..LM. Fortran's after-branch loops IMS..IME, JMS..JME (line 11688-11771), which
// is the memory domain including the halo ring (int_state IMS, IME set at
// module_SOLVER_INTERNAL_STATE.F90:1516-1517; the halo ring is ITS-IHALO .. ITE+IHALO). The port
// stores no halo ring, so the Fortran ring cells of F_RIMEF are NOT written here. Declared residue:
// the port's readers of F_RIMEF are domain-only; Fortran readers of the ring cells are not all
// traced, and the single-card equality ITS=IDS, ITE=IDE is not traced to DM_PARALLEL looplimits.
// No _b1 trimming (the routine uses no _b1 limits).
// EPSQ: nmm/module_CONSTANTS.F90:27 `epsq=1.e-12` (real(kind=kfpt)), NMMB_EPSQ = 1.0e-12f.
// Fortran lines not mapped to a term: 11751-11754 HALO_EXCH (MPI halo, global-only path dropped).
// The MIN/MAX are evaluated in single precision as Fortran REAL (50. and 1. are REAL constants).
// The before-branch has one product, so FMA contraction cannot change it; the after-branch has
// one division and two comparisons, all IEEE (no fast-math).
//
// Launch convention: one thread per (i, j) with the level in blockIdx.z (grid_2d). One launch per
// branch. No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

// Before species advection: QG = QS * F_RIMEF (every point of the domain).
extern "C" __global__ void rime_factor_before(
    const int lm,                               // LM levels
    const int im,                               // IM
    const int jm,                               // JM
    const float* __restrict__ qs,               // [LM, JM, IM] snow mixing ratio (read)
    const float* __restrict__ f_rimef,          // [LM, JM, IM] rime factor (read)
    float* __restrict__ qg)                     // [LM, JM, IM] graupel mixing ratio (written)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t ix = IDX3(i0, j0, l0, im, jm);
    qg[ix] = qs[ix] * f_rimef[ix];              // QG(I,J,K) = QS(I,J,K)*F_RIMEF(I,J,K)
}

// After species advection: F_RIMEF from QG and QS (every point of the domain).
extern "C" __global__ void rime_factor_after(
    const int lm,                               // LM levels
    const int im,                               // IM
    const int jm,                               // JM
    const float* __restrict__ qs,               // [LM, JM, IM] snow mixing ratio (read)
    const float* __restrict__ qg,               // [LM, JM, IM] graupel mixing ratio (read)
    float* __restrict__ f_rimef)                // [LM, JM, IM] rime factor (written)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t ix = IDX3(i0, j0, l0, im, jm);
    const float qgv = qg[ix];
    const float qsv = qs[ix];
    if (qgv > NMMB_EPSQ && qsv > NMMB_EPSQ) {
        const float rimef = qgv / qsv;          // RIMEF = QG/QS
        f_rimef[ix] = fminf(50.0f, fmaxf(1.0f, rimef));   // MIN(50., MAX(1.,RIMEF))
    } else {
        f_rimef[ix] = 1.0f;
    }
}
