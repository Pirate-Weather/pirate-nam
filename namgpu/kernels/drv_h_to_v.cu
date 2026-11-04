// drv_h_to_v.cu -- H_TO_V_2D and H_TO_V_3D: move a field from B-grid H points to V points.
//
// NOTICE: NOAA/NCEP NMMB source, nmm/module_H_TO_V.F90 (US Government work, no licence file in
// the NAM clone; see NOTICE at the repo root). Module header kept as the notice:
//   MODULE MODULE_H_TO_V
//   MOVE DATA RESIDING ON B-GRID H POINTS TO V POINTS.
// Ported to CUDA for nam-gpu from nmm/module_H_TO_V.F90:42-122 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (both routines, identical stencil):
//   ARRAY_V(I,J[,K]) = (ARRAY_H(I,J,K) + ARRAY_H(I+1,J,K)
//                     + ARRAY_H(I,J+1,K) + ARRAY_H(I+1,J+1,K)) * 0.25
// V(i,j) sits NE of H(i,j); its four H neighbours are H(i,j), H(i+1,j), H(i,j+1), H(i+1,j+1).
//
// Loop limits: single tile, halo-free (CONTRACT.md section 1). The Fortran loops are
//   DO J = JTS, JTE_B1   and   DO I = ITS, ITE_B1
// with ITS, JTS, ITE_B1, JTE_B1 read from the drv-loop-limits table (lim, DL_* in
// nmmb_common.cuh, 1-based). Zero-based that is i0 in [0, IM-2], j0 in [0, JM-2]: the last
// column and last row of V are not written (the Fortran does not write them either; they are
// unused). The 3-D routine loops K = 1..LM (level l0 = blockIdx.z, grid_2d).
//
// Mapping: 2-D arrays are [JM, IM] (IDX2); 3-D arrays are [LM, JM, IM] (IDX3), level 0 = top.
// Summation order is the Fortran left-to-right order ((a+b)+c)+d, then the *0.25 scale;
// the multiply is not an input to an add, so FMA contraction cannot change the result.
// Fortran lines not mapped to a term: none. The !$omp parallel do directives are ignored (no
// effect on the result); the MPI-free single tile needs no halo.
//
// Launch: one thread per (i, j[, l]); no host sync, no allocation, no printf.

#include "nmmb_common.cuh"

// H_TO_V_2D: ARRAY_H, ARRAY_V are [JM, IM].
extern "C" __global__ void h_to_v_2d(
    const int im,                               // IM
    const int jm,                               // JM
    const int* __restrict__ lim,                // [DL_NLIM] drv-loop-limits table
    const float* __restrict__ array_h,          // [JM, IM] H-point field (read)
    float* __restrict__ array_v,                // [JM, IM] V-point field (written)
    const int JOFF)                             // row slab: lim holds GLOBAL rows, jg = j0 + JOFF
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int jg = j0 + JOFF;
    if (i0 < lim[DL_ITS] - 1 || i0 >= lim[DL_ITE_B1] ||
        jg < lim[DL_JTS] - 1 || jg >= lim[DL_JTE_B1] || j0 + 1 >= jm) return;
    const float a = array_h[IDX2(i0,     j0,     im)];
    const float b = array_h[IDX2(i0 + 1, j0,     im)];
    const float c = array_h[IDX2(i0,     j0 + 1, im)];
    const float d = array_h[IDX2(i0 + 1, j0 + 1, im)];
    array_v[IDX2(i0, j0, im)] = (a + b + c + d) * 0.25f;
}

// H_TO_V_3D: ARRAY_H, ARRAY_V are [LM, JM, IM]; one thread per (i, j, level).
extern "C" __global__ void h_to_v_3d(
    const int lm,                               // LM levels
    const int im,                               // IM
    const int jm,                               // JM
    const int* __restrict__ lim,                // [DL_NLIM] drv-loop-limits table
    const float* __restrict__ array_h,          // [LM, JM, IM] H-point field (read)
    float* __restrict__ array_v,                // [LM, JM, IM] V-point field (written)
    const int JOFF)                             // row slab: lim holds GLOBAL rows, jg = j0 + JOFF
{
    POINT_IJL
    const int jg = j0 + JOFF;
    if (i0 < lim[DL_ITS] - 1 || i0 >= lim[DL_ITE_B1] ||
        jg < lim[DL_JTS] - 1 || jg >= lim[DL_JTE_B1] || j0 + 1 >= jm || l0 >= lm) return;
    const float a = array_h[IDX3(i0,     j0,     l0, im, jm)];
    const float b = array_h[IDX3(i0 + 1, j0,     l0, im, jm)];
    const float c = array_h[IDX3(i0,     j0 + 1, l0, im, jm)];
    const float d = array_h[IDX3(i0 + 1, j0 + 1, l0, im, jm)];
    array_v[IDX3(i0, j0, l0, im, jm)] = (a + b + c + d) * 0.25f;
}
