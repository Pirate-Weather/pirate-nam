// nmmb_common.cuh -- shared index macros and constants for every nam-gpu kernel.
//
// Layout (CONTRACT.md): Fortran A(i,j,l) with i=1..IM, j=1..JM, l=1..LM (l=1 = model top)
// lives in a CuPy C-ordered float32 array a[l][j][i]  (shape [LM, JM, IM]).
// Zero-based in CUDA: i0 = i-1, j0 = j-1, l0 = l-1.
#pragma once

#define IDX2(i0, j0, IM)            ((size_t)(j0) * (IM) + (i0))
#define IDX3(i0, j0, l0, IM, JM)    (((size_t)(l0) * (JM) + (j0)) * (IM) + (i0))
#define IDX4(i0, j0, l0, n0, IM, JM, LM) \
    ((((size_t)(n0) * (LM) + (l0)) * (JM) + (j0)) * (IM) + (i0))

// Standard thread mapping for a 2-D/3-D point kernel launched with namgpu.kernels.grid_2d:
#define POINT_IJL                                   \
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x; \
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y; \
    const int l0 = blockIdx.z;

// Standard mapping for a column kernel launched with namgpu.kernels.grid_cols:
#define COLUMN_IJ(IM, JM)                                           \
    const int col_ = blockIdx.x * blockDim.x + threadIdx.x;         \
    if (col_ >= (IM) * (JM)) return;                                \
    const int i0 = col_ % (IM);                                     \
    const int j0 = col_ / (IM);

// NMMB module_CONSTANTS.F90 values (single precision, as kfpt=single), copied 2026-11-04.
#define NMMB_A        6376000.0f
#define NMMB_G        9.8060226f
#define NMMB_R_D      287.04f
#define NMMB_R_V      461.6f
#define NMMB_CP       1004.6f
#define NMMB_CAPPA    (NMMB_R_D / NMMB_CP)
#define NMMB_EPSQ     1.0e-12f
#define NMMB_EPSQ2    0.02f      /* module_DYNAMICS_ROUTINES epsq2; check per unit */
#define NMMB_P608     (NMMB_R_V / NMMB_R_D - 1.0f)
#define NMMB_ELWV     2.501e6f
#define NMMB_ELIV     2.850e6f
#define NMMB_ELIWV    2.683e6f
#define NMMB_TIW      273.15f
#define NMMB_TI0      271.15f
#define NMMB_TWOM     0.00014584f
#define NMMB_STBOLT   5.67051e-8f
#define NMMB_PQ0      379.90516f
#define NMMB_A2       17.2693882f
#define NMMB_A3       273.15f
#define NMMB_A4       35.86f
#define NMMB_DTR      (3.141592653589793f / 180.0f)
// Unit owners: if a module uses a value not here, add it HERE (one place), with its source line.

// ---------------------------------------------------------------------------
// Loop limits (unit drv-loop-limits). Reference: nmm/module_DM_PARALLEL.F90
//   looplimits      lines 1941-1969 (block partition of lb..ub over ntasks tasks)
//   tile limits     lines 792-822   (ids_p1 ... jte_b1_h2 from its/ite/ids/ide)
// Regional single-card case (global = .false., one task): ids = jds = 1,
// ide = IM, jde = JM, and the tile is the whole domain, so every limit below is
// the 1-based Fortran value. A zero-based loop over [a, b] runs i0 = a-1 .. b-1.
// ---------------------------------------------------------------------------

// Fortran looplimits(taskid, ntasks, lb, ub, i1, i2), 0-based taskid, exact:
// counts(itask) = chunk+1 for the first nt1 tasks, chunk for the rest; the
// displacement of task t is the running netdisp, capped at ub.
__device__ inline void nmmb_looplimits(int taskid, int ntasks, int lb, int ub,
                                       int* i1, int* i2)
{
    const int nwork = ub - lb + 1;
    const int chunk = nwork / ntasks;
    const int nt1 = nwork - ntasks * chunk;
    int netdisp = lb;
    int disp = lb;
    int cnt = chunk;
    for (int itask = 1; itask <= ntasks; ++itask) {
        const int c = (itask <= nt1) ? chunk + 1 : chunk;
        if (itask == taskid + 1) { disp = netdisp; cnt = c; }
        netdisp = min(ub, netdisp + c);
    }
    *i1 = disp;
    *i2 = min(ub, disp + cnt - 1);
}

// Tile limits are not macros (they were wrong for any multi-task tile: its was fixed at 1).
// Every kernel that needs a tile limit reads it from the drv-loop-limits device table
// (ctx.tables["NMMB_LIMITS"], built by namgpu/steps/drv_loop_limits.py) at the DL_* index below.
// Values are 1-based Fortran values, exactly as module_DM_PARALLEL.F90 computes them.
#define NMMB_IHALO 3   /* module_DM_PARALLEL.F90:23 */
#define NMMB_JHALO 3   /* module_DM_PARALLEL.F90:24 */

#define DL_IDS 0
#define DL_IDE 1
#define DL_JDS 2
#define DL_JDE 3
#define DL_ITS 4
#define DL_ITE 5
#define DL_JTS 6
#define DL_JTE 7
#define DL_ITS_B1 8
#define DL_ITE_B1 9
#define DL_ITS_B2 10
#define DL_ITE_B2 11
#define DL_JTS_B1 12
#define DL_JTE_B1 13
#define DL_JTS_B2 14
#define DL_JTE_B2 15
#define DL_ITS_H1 16
#define DL_ITE_H1 17
#define DL_ITS_H2 18
#define DL_ITE_H2 19
#define DL_JTS_H1 20
#define DL_JTE_H1 21
#define DL_JTS_H2 22
#define DL_JTE_H2 23
#define DL_ITS_B1_H1 24
#define DL_ITE_B1_H1 25
#define DL_ITE_B1_H2 26
#define DL_JTS_B1_H1 27
#define DL_JTE_B1_H1 28
#define DL_JTE_B1_H2 29
#define DL_IDS_P1 30
#define DL_IDE_M1 31
#define DL_IDE_M2 32
#define DL_JDS_P1 33
#define DL_JDE_M1 34
#define DL_JDE_M2 35
#define DL_IHALO 36
#define DL_JHALO 37
#define DL_IMS 38
#define DL_IME 39
#define DL_JMS 40
#define DL_JME 41
#define DL_NLIM 42
