// WOOF-NAM unit bnd-boundary-init: boundary-zone working arrays filled from the full-domain
// fields at the four lateral edges (S, N, W, E), for the H-point and V-point variables.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_CONTROL.F90:1145-1774 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine boundary_init (1-based indices, full domain IM x JM, single process, all four
// edge flags true: w_bdy=(its==ids), e_bdy=(ite==ide), s_bdy=(jts==jds), n_bdy=(jte==jde)).
// Every block is a strip copy from the full-domain variable into the boundary working array,
// index 1 of the time/tendency dimension (the value slot; index 2 is written by bnd-read-bc):
//   S (H, lnsh): bnd(i,n,k,1)   = var(i, jds-1+n, k)          n=1..lnsh   -> src row  s
//   N (H, lnsh): bnd(i,n,k,1)   = var(i, jde-lnsh+n, k)                     -> src row  JM-lnsh+s
//   W (H, lnsh): bnd(n,j,k,1)   = var(ids-1+n, j, k)          n=1..lnsh   -> src col  s
//   E (H, lnsh): bnd(n,j,k,1)   = var(ide-lnsh+n, j, k)                     -> src col  IM-lnsh+s
//   S/N/W/E (V, lnsv): identical, with lnsv, and N/E offsets JM-1-lnsv+s, IM-1-lnsv+s
//                      (module_CONTROL.F90 V-point blocks, same routine).
// Here s = n-1 is the zero-based strip position. Level loop k=1..lm is grid.z (l0). The 2-D field
// PD is the same copy with nlev = 1 (its loops over k are absent in the Fortran).
// Nothing of 1145-1774 is skipped except: the 4-D branches (no 4-D boundary variable exists in the
// 12 km parent: nvars_4d_h = 0, module_SOLVER_GRID_COMP.F90:7662), the OpenMP and timer
// directives, and the MPI-only bounds (ims/ime, jms/jme equal ids/ide, jds/jde in one process).
// The routine computes no blending weights; the lnsh/lnsv blending is in bnd-bocoh-blend and
// bnd-bocov-blend.
//
// Layout: the full field is CuPy [nlev, JM, IM] (Fortran (i,j,l) -> [l,j,i], zero-based).
// The strip buffer passed as dst is the value slot of ctx.bc["<SIDE>_<NAME>"], a [2, ...] array
// (slot 0 = value, slot 1 = tendency, left to bnd-read-bc). Its value-slot shape is
// [nlev, width, IM] for S/N (position s, then i) and [nlev, JM, width] for W/E (j, then position s);
// for 2-D PD the nlev axis is absent (the wrapper passes [width, IM] or [JM, width]), and the
// flat index is the same with level 0. The wrapper keeps one buffer per (variable, side).
//
// One thread per (strip position s, edge cell b, level): blockIdx.x/threadIdx.x -> s,
// blockIdx.y/threadIdx.y -> i (S/N) or j (W/E), blockIdx.z -> level. One launch per
// (variable, side). No host sync, no device allocation.

#include "nmmb_common.cuh"

extern "C" __global__ void bnd_boundary_init_strip(
    const float* __restrict__ src,   // full field [nlev, JM, IM]
    float* __restrict__ dst,         // strip buffer, see header
    const int side,                  // 0 = S, 1 = N, 2 = W, 3 = E
    const int nlev,
    const int im,
    const int jm,
    const int width,                 // lnsh (H variables) or lnsv (V variables)
    const int src0)                  // zero-based full-domain index of strip position s = 0
{
    const int s  = blockIdx.x * blockDim.x + threadIdx.x;   // strip position, n-1
    const int b  = blockIdx.y * blockDim.y + threadIdx.y;   // i (S/N) or j (W/E)
    const int l0 = blockIdx.z;                              // level, 0 = model top
    if (l0 >= nlev || s >= width) return;

    if (side < 2) {
        // S / N: bnd(i, s+1, k) = var(i, src0 + s, k)
        if (b >= im) return;
        const int i0 = b;
        const int j_src = src0 + s;
        dst[((size_t)l0 * width + s) * im + i0] = src[IDX3(i0, j_src, l0, im, jm)];
    } else {
        // W / E: bnd(s+1, j, k) = var(src0 + s, j, k)
        if (b >= jm) return;
        const int j0 = b;
        const int i_src = src0 + s;
        dst[((size_t)l0 * jm + j0) * width + s] = src[IDX3(i_src, j0, l0, im, jm)];
    }
}
