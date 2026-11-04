// WOOF-NAM unit drv-loop-limits: looplimits and the tile loop limits as a device table.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   module module_dm_parallel  (NMMB distributed-memory parallel layer, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DM_PARALLEL.F90:1941-1969 (looplimits)
// and nmm/module_DM_PARALLEL.F90:777-822 (ihalo/jhalo use, ims..jme and ids_p1 ... jte_b1_h2)
// (nam.v4.2.11, 83df4e13c1).
//
// Fortran looplimits(taskid, ntasks, lb, ub, i1, i2): block partition, the first nt1 tasks
// get chunk+1 points. Implemented exactly as nmmb_looplimits() in nmmb_common.cuh.
// Task ids: 0 at the lower left, increasing in I then in J (module_DM_PARALLEL.F90 ~650-660).
//
// Multi-task tiles: the Fortran tile of each task is its ESMF computationalCount, which the
// NMM setup takes from ESMF (module_DOMAIN_GRID_COMP.F90, CALL DECOMP at ~line 4815, ijcount =
// I1). looplimits() is the same rule only when ntasks = 1. The wrapper refuses ntasks > 1 until
// the ESMF decomposition is ported (namgpu/steps/drv_loop_limits.py). On one card (inpes =
// jnpes = 1) the tile is the full domain and every entry is exact.
//
// The routine has no per-cell work. It is one thread that writes the table once at init;
// it is not on the step path. Output: int32 out[NLIM], 1-based Fortran values, indices DL_*
// in nmmb_common.cuh.

#include "nmmb_common.cuh"

extern "C" __global__ void drv_loop_limits(
    const int IM, const int JM,
    const int global_flag,   /* 1 when configfile global = .true. */
    const int mype,          /* this task (0 on one card)          */
    const int inpes, const int jnpes,
    int* __restrict__ out)   /* [DL_NLIM] */
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int ids = 1;
    const int jds = 1;
    const int ide = global_flag ? IM + 2 : IM;
    const int jde = global_flag ? JM + 2 : JM;

    // Task position: IDs increase in I, then in J.
    const int mi = mype % inpes;
    const int mj = mype / inpes;

    int its, ite, jts, jte;
    nmmb_looplimits(mi, inpes, ids, ide, &its, &ite);
    nmmb_looplimits(mj, jnpes, jds, jde, &jts, &jte);

    out[DL_IDS] = ids;     out[DL_IDE] = ide;
    out[DL_JDS] = jds;     out[DL_JDE] = jde;
    out[DL_ITS] = its;     out[DL_ITE] = ite;
    out[DL_JTS] = jts;     out[DL_JTE] = jte;

    // Fortran lines 792-822, verbatim formulas.
    out[DL_IDS_P1] = max(its, ids + 1);
    out[DL_IDE_M1] = min(ite, ide - 1);
    out[DL_IDE_M2] = min(ite, ide - 2);
    out[DL_JDS_P1] = max(jts, jds + 1);
    out[DL_JDE_M1] = min(jte, jde - 1);
    out[DL_JDE_M2] = min(jte, jde - 2);

    out[DL_ITS_B1] = max(its, ids + 1);
    out[DL_ITE_B1] = min(ite, ide - 1);
    out[DL_ITS_B2] = max(its, ids + 2);
    out[DL_ITE_B2] = min(ite, ide - 2);
    out[DL_JTS_B1] = max(jts, jds + 1);
    out[DL_JTE_B1] = min(jte, jde - 1);
    out[DL_JTS_B2] = max(jts, jds + 2);
    out[DL_JTE_B2] = min(jte, jde - 2);

    out[DL_ITS_H1] = max(its - 1, ids);
    out[DL_ITE_H1] = min(ite + 1, ide);
    out[DL_ITS_H2] = max(its - 2, ids);
    out[DL_ITE_H2] = min(ite + 2, ide);
    out[DL_JTS_H1] = max(jts - 1, jds);
    out[DL_JTE_H1] = min(jte + 1, jde);
    out[DL_JTS_H2] = max(jts - 2, jds);
    out[DL_JTE_H2] = min(jte + 2, jde);

    out[DL_ITS_B1_H1] = max(its - 1, ids + 1);
    out[DL_ITE_B1_H1] = min(ite + 1, ide - 1);
    out[DL_ITE_B1_H2] = min(ite + 2, ide - 1);
    out[DL_JTS_B1_H1] = max(jts - 1, jds + 1);
    out[DL_JTE_B1_H1] = min(jte + 1, jde - 1);
    out[DL_JTE_B1_H2] = min(jte + 2, jde - 1);

    // Halo-inclusive memory bounds (module_DM_PARALLEL.F90:777-780). Unclamped, as in the
    // active Fortran code (the jms/jme clamp is commented out there).
    out[DL_IHALO] = NMMB_IHALO;
    out[DL_JHALO] = NMMB_JHALO;
    out[DL_IMS] = its - NMMB_IHALO;
    out[DL_IME] = ite + NMMB_IHALO;
    out[DL_JMS] = jts - NMMB_JHALO;
    out[DL_JME] = jte + NMMB_JHALO;
}
