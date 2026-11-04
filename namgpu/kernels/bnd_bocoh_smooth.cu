// bnd_bocoh_smooth.cu -- optional spatial smoothing of the H boundary rows (S, N, W, E).
//
// NOTICE: NAM source file nmm/module_FLTBNDS.F90 (NOAA/NCEP/EMC, US Government work, no licence
// file). The BOCOH subroutine carries NOAA author/history headers; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_FLTBNDS.F90:5963-6387 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (BOCOH, smoothing block, `if(nsmud>=1)`): for ks = 1..nsmud, for each side in the order
// South, North, West, East, for each H boundary working array (var_2d, then var_3d, then var_4d
// slice by slice), then PINT levels 2..LM+1, the strip stencil
//     hbc(i,j) = (v(i,j-1) + v(i-1,j) + v(i+1,j) + v(i,j+1)) * w1 + w2 * v(i,j)
// is computed into a scratch hbc over the whole strip, and only then copied back into v.
// w1 = wa*0.25 = 0.125, w2 = 1-wa = 0.5 (wa = 0.5, module_FLTBNDS parameters).
//
// Two kernels per (side, array slab), so the Fortran read-all-then-write order holds across threads:
//   bocoh_smooth_ww    : hbc -> ww for the strip box (one thread per strip cell, z = slab)
//   bocoh_smooth_copy  : ww -> v for the same box
// A "slab" is one [JM,IM] level. The wrapper passes a base pointer and nslab; slab l0 starts at
// src + l0*JM*IM, so 2-D arrays (nslab=1), 3-D arrays and PINT(:,:,2:LM+1) use the same kernels.
//
// Strip boxes (zero-based, inclusive; single full domain, module_DM_PARALLEL looplimits with
// ids=1 ide=IM, jds=1 jde=JM, lines = LNSH; boxes are computed by the wrapper):
//   south  i0 in [1, IM-2]        j0 in [1, lines-1]         (Fortran j = jts+1 .. jts-1+lines)
//   north  i0 in [1, IM-2]        j0 in [JM-lines, JM-2]     (Fortran j = jte-lines+1 .. jte-1)
//   west   i0 in [1, lines-1]     j0 in [lines, JM-lines-1]  (Fortran i = its+1 .. its-1+lines)
//   east   i0 in [IM-lines, IM-2] j0 in [lines, JM-lines-1]  (Fortran i = ite-lines+1 .. ite-1)
//   j limits for west/east: max(jts_h2, jds+lines) .. min(jte_h2, jde-lines), 1-based.
//
// Sides are launched in Fortran order (S, N, W, E): the south strip reads row 6 and the west strip
// reads row 5 of the south strip, so a later side sees the values an earlier side wrote, as in the
// Fortran.
//
// Not mapped (no term in this kernel):
//   - halo_exch calls after the smoothing loop (single process, halo-free layout, CONTRACT.md s.1);
//   - OpenMP / ENABLE_SMP directives;
//   - the local allocate/deallocate of hbc_2d (the scratch is the ww argument instead).
//
// Launch: no host sync, no device malloc. ww is [LM, JM, IM] scratch allocated by the wrapper.

#include "nmmb_common.cuh"

#define BOCOH_W1 (0.5f * 0.25f)   // wa*0.25, Fortran w1 (kfpt single)
#define BOCOH_W2 (1.0f - 0.5f)    // 1-wa,    Fortran w2

// Stencil into ww over the box [i0lo..i0hi] x [j0lo..j0hi], one slab per z.
extern "C" __global__ void bocoh_smooth_ww(
    const int IM, const int JM, const int nslab,
    const int i0lo, const int i0hi, const int j0lo, const int j0hi,
    const float* __restrict__ src,    // [nslab, JM, IM] (slab stride JM*IM)
    float* __restrict__ ww)           // [>= nslab, JM, IM] scratch
{
    const int i0 = i0lo + blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = j0lo + blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 > i0hi || j0 > j0hi || l0 >= nslab) return;
    const size_t slab = (size_t)JM * (size_t)IM;
    const float* s = src + (size_t)l0 * slab;
    float* w = ww + (size_t)l0 * slab;
    // Fortran order: (j-1) + (i-1) + (i+1) + (j+1), then *w1, then + w2*v(i,j).
    const float sum = ((s[IDX2(i0, j0 - 1, IM)] + s[IDX2(i0 - 1, j0, IM)])
                        + s[IDX2(i0 + 1, j0, IM)]) + s[IDX2(i0, j0 + 1, IM)];
    w[IDX2(i0, j0, IM)] = sum * BOCOH_W1 + BOCOH_W2 * s[IDX2(i0, j0, IM)];
}

// Copy ww back into the strip of the working array.
extern "C" __global__ void bocoh_smooth_copy(
    const int IM, const int JM, const int nslab,
    const int i0lo, const int i0hi, const int j0lo, const int j0hi,
    float* __restrict__ dst,          // [nslab, JM, IM] (slab stride JM*IM)
    const float* __restrict__ ww)
{
    const int i0 = i0lo + blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = j0lo + blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 > i0hi || j0 > j0hi || l0 >= nslab) return;
    const size_t slab = (size_t)JM * (size_t)IM;
    dst[l0 * slab + IDX2(i0, j0, IM)] = ww[l0 * slab + IDX2(i0, j0, IM)];
}
