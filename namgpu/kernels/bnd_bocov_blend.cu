// bnd_bocov_blend.cu -- write the V-point boundary values into the domain V fields (U, V and any
// other 2-D/3-D V-point boundary variable), blended over the LNSV outer rows and columns, then the
// optional boundary smoothing passes (nsmud >= 1).
//
// NOTICE: NAM source file nmm/module_FLTBNDS.F90 (NOAA/NCEP/EMC, US Government work, no licence
// file). Keep the NOAA author headers of that file as notice; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_FLTBNDS.F90:6590-7010 (nam.v4.2.11, 83df4e13c1).
//
// Fortran: the blend block of subroutine BOCOV (lines 6590-7010), which follows the update block
// (bnd-bocov-update, lines 6392-6589). wv(1:lnsv) is computed by BOCOV at lines 6461-6464 and is
// passed in here as a device vector (namgpu/boundary/bnd_bocov_blend.py builds it with the same
// single-precision formula). wa, w1, w2 are the BOCOV parameters (wa = 0.5, w1 = wa*0.25 = 0.125,
// w2 = 1-wa = 0.5), exact in single precision.
//
// Layouts (CONTRACT section 1; the same as bnd_bocov_update.cu):
//   full (V-point domain field, Fortran full_var):  2-D [JM][IM], 3-D [LM][JM][IM]  (IDX3 macro)
//   strips (Fortran bnd_vars_v%var_xd%south/north/west/east), slot k = 0 is Fortran slot 1
//   (value; the blend reads only this slot, Fortran index 1 in the time dimension):
//     S/N  2-D (2, LNSV, IM)      3-D (2, LM, LNSV, IM)       index [l][jb][ib]
//     W/E  2-D (2, JM, LNSV)      3-D (2, LM, JM, LNSV)       index [l][jb][ib]
//   The bdy pointer passed to the kernels is the slot-0 base (the first half of the array).
//
// Blend, Fortran (1-based i, j; single process: its_h1 = its_h2 = ids = 1, ite_h1 = ite_h2 = ide = IM,
// jts_h1 = jts_h2 = jds = 1, jte_h1 = jte_h2 = jde = JM; looplimits, module_DM_PARALLEL.F90):
//   South (s_bdy), 2-D and 3-D identical, j = jb = 1..LNSV:
//       ivs = max(its_h1, jb); ive = min(ite_h1, ide-jb);  i = ivs..ive  (ib = i)
//       full(i,j[,l]) = south(ib,jb[,l],1)*wv(jb) + full(i,j[,l])*(1.-wv(jb))
//   North (n_bdy), j = jte-lnsv .. jte-1 with jb = j-jte+lnsv+1 = 1..LNSV:
//       ivs = max(its_h1, lnsv-jb+1); ive = min(ite_h1, ide+jb-lnsv-1)   (3-D uses its_h2/ite_h2,
//       identical here); ib = i
//       full(i,j[,l]) = north(ib,jb[,l],1)*wv(lnsv+1-jb) + full(i,j[,l])*(1.-wv(lnsv+1-jb))
//   West (w_bdy), i = its..its-1+lnsv with ib = i = 1..LNSV:
//       jvs = max(jts_h1, 1+ib); jve = min(jte_h1, jde-1-ib);  j = jvs..jve  (jb = j)
//       full(i,j[,l]) = west(ib,jb[,l],1)*wv(ib) + full(i,j[,l])*(1.-wv(ib))
//   East (e_bdy), i = ite-lnsv .. ite-1 with ib = i-ide+lnsv+1 = 1..LNSV:
//       jvs = max(jts_h1, lnsv-ib+2); jve = min(jte_h1, jde-lnsv+ib-2);  j = jvs..jve  (jb = j)
//       full(i,j[,l]) = east(ib,jb[,l],1)*wv(lnsv+1-ib) + full(i,j[,l])*(1.-wv(lnsv+1-ib))
// Order: South (2-D, then 3-D), North, West, East. The four sides overlap only at corners and the
// wrapper launches them in exactly this order, so the corner values come out as in the Fortran.
// Every (side, variable) launch writes each element at most once and reads only its own element,
// so one thread per element inside a launch is exact and race free.
//
// Smoothing (Fortran `if(nsmud>=1)`, smooth loop ks = 1..nsmud): NOTE nsmud is set to 0 by BOCOV
// (line 6459, `nsmud=0`, the `nsmud=lines-1` line is commented out), so in NAM this branch is never
// taken. It is ported in full and runs only when the wrapper is called with nsmud >= 1.
//   Per ks, per region (South, North, West, East, in that order), per variable and level:
//     vbc(i,j) = (full(i,j-1)+full(i-1,j)+full(i+1,j)+full(i,j+1))*w1 + w2*full(i,j)   (gather)
//     full(i,j) = vbc(i,j)                                                             (scatter)
//   Regions (1-based, inclusive; single process, lines = lnsv, ids = jds = 1):
//     South: i = max(its_h1,ids+1)..min(ite_h1,ide-2) = 2..IM-2;  j = jts+1..jts-1+lines = 2..LNSV
//     North: i = 2..IM-2;  j = jte-lines..jte-2 = JM-LNSV..JM-2
//     West:  i = ite-lines..ite-2 = IM-LNSV..IM-2;  j = max(jts_h1,jds+lines)..min(jte_h1,jde-lines-1)
//            = (1+LNSV)..(JM-LNSV-1)
//     East:  identical to West in the Fortran (i = ite-lines..ite-2, j as West). Ported as written.
//   Note: the Fortran West and East smoothing blocks both use i = ite-lines..ite-2, i.e. the EAST
//   edge; the West block never touches the west edge. Ported unchanged; flagged for review.
//   The halo_exch calls after each ks pass are dropped (single process, CONTRACT section 1).
//   The `vbc_2d` temporary is replaced by a device scratch [LM][JM][IM] (wrapper), written by the
//   gather and read by the scatter; gather and scatter are separate launches so no thread reads a
//   value another thread has already overwritten within the same pass.
//
// Launches: blend, one per (side, variable); smoothing only when nsmud >= 1. No host sync, no device
// allocation in the kernels, pointer and vector tables are built once by the wrapper.
// The dt argument of BOCOV is not used in the Fortran body and is not passed.

#include "nmmb_common.cuh"

// ---------------------------------------------------------------------------------------------
// Blend one side for one variable. Thread (p, q, l):
//   side 0 (S), 1 (N):  p = i0 in [0, IM),  q = jb0 in [0, LNSV)   (jb = q+1)
//   side 2 (W), 3 (E):  p = ib0 in [0, LNSV), q = j0 in [0, JM)    (ib = p+1)
// Launch the grid over (P, Q, nl) with P = IM or LNSV and Q = LNSV or JM as above.
// full: domain field [nl][JM][IM]; bdy: strip slot 0, [nl][LNSV][IM] (S/N) or [nl][JM][LNSV] (W/E).
// nl = 1 for 2-D variables, LM for 3-D.
extern "C" __global__ void bocov_blend_side(
    float* __restrict__ full,
    const float* __restrict__ bdy,
    const float* __restrict__ wv,      // [LNSV]: wv[k-1] is Fortran wv(k)
    int side, int im, int jm, int nl, int lnsv)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z;
    if (l >= nl) return;

    if (side == 0) {                                   // South
        if (p >= im || q >= lnsv) return;
        const int jb = q + 1;
        const int i = p + 1;
        const int ivs = max(1, jb);                    // max(its_h1, jb)
        const int ive = min(im, im - jb);              // min(ite_h1, ide - jb)
        if (i < ivs || i > ive) return;
        const float w = wv[jb - 1];                    // wv(jb)
        const float b = bdy[((size_t)l * lnsv + (jb - 1)) * im + (i - 1)];
        const size_t fi = IDX3(i - 1, jb - 1, l, im, jm);
        full[fi] = b * w + full[fi] * (1.0f - w);
        return;
    }
    if (side == 1) {                                   // North
        if (p >= im || q >= lnsv) return;
        const int jb = q + 1;                          // jb = j - jte + lnsv + 1
        const int j0 = jm - lnsv + q - 1;              // zero-based j = (jte-lnsv+jb-1) - 1
        const int i = p + 1;
        const int ivs = max(1, lnsv - jb + 1);         // max(its_h1, lnsv-jb+1)
        const int ive = min(im, im + jb - lnsv - 1);   // min(ite_h1, ide+jb-lnsv-1)
        if (i < ivs || i > ive) return;
        const float w = wv[lnsv - jb];                 // wv(lnsv+1-jb)
        const float b = bdy[((size_t)l * lnsv + (jb - 1)) * im + (i - 1)];
        const size_t fi = IDX3(i - 1, j0, l, im, jm);
        full[fi] = b * w + full[fi] * (1.0f - w);
        return;
    }
    if (side == 2) {                                   // West
        if (p >= lnsv || q >= jm) return;
        const int ib = p + 1;                          // i = ib
        const int j = q + 1;
        const int jvs = max(1, 1 + ib);                // max(jts_h1, 1+ib)
        const int jve = min(jm, jm - 1 - ib);          // min(jte_h1, jde-1-ib)
        if (j < jvs || j > jve) return;
        const float w = wv[ib - 1];                    // wv(ib)
        const float b = bdy[((size_t)l * jm + (j - 1)) * lnsv + (ib - 1)];
        const size_t fi = IDX3(ib - 1, j - 1, l, im, jm);
        full[fi] = b * w + full[fi] * (1.0f - w);
        return;
    }
    // side == 3: East
    if (p >= lnsv || q >= jm) return;
    const int ib = p + 1;                              // ib = i - ide + lnsv + 1
    const int i0 = ib + im - lnsv - 2;                 // zero-based i = ib + ide - lnsv - 1
    const int j = q + 1;
    const int jvs = max(1, lnsv - ib + 2);             // max(jts_h1, lnsv-ib+2)
    const int jve = min(jm, jm - lnsv + ib - 2);       // min(jte_h1, jde-lnsv+ib-2)
    if (j < jvs || j > jve) return;
    const float w = wv[lnsv - ib];                     // wv(lnsv+1-ib)
    const float b = bdy[((size_t)l * jm + (j - 1)) * lnsv + (ib - 1)];
    const size_t fi = IDX3(i0, j - 1, l, im, jm);
    full[fi] = b * w + full[fi] * (1.0f - w);
}

// ---------------------------------------------------------------------------------------------
// Smoothing gather: vbc over the 1-based inclusive region [i1..i2] x [j1..j2], all levels.
// Thread (x, y, l): i = i1 + x, j = j1 + y. Writes tmp (same [nl][JM][IM] layout as full).
extern "C" __global__ void bocov_smooth_gather(
    float* __restrict__ tmp,
    const float* __restrict__ full,
    int i1, int i2, int j1, int j2, int im, int jm, int nl)
{
    const int i = i1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int j = j1 + blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z;
    if (i > i2 || j > j2 || l >= nl) return;
    const int i0 = i - 1, j0 = j - 1;
    const float a = full[IDX3(i0, j0 - 1, l, im, jm)];   // full(i, j-1)
    const float b = full[IDX3(i0 - 1, j0, l, im, jm)];   // full(i-1, j)
    const float c = full[IDX3(i0 + 1, j0, l, im, jm)];   // full(i+1, j)
    const float d = full[IDX3(i0, j0 + 1, l, im, jm)];   // full(i, j+1)
    const float ctr = full[IDX3(i0, j0, l, im, jm)];
    const float w1 = 0.125f;                             // wa*0.25, wa = 0.5
    const float w2 = 0.5f;                               // 1 - wa
    tmp[IDX3(i0, j0, l, im, jm)] = (a + b + c + d) * w1 + w2 * ctr;
}

// Smoothing scatter: full(i,j,l) = tmp(i,j,l) over the same region.
extern "C" __global__ void bocov_smooth_scatter(
    float* __restrict__ full,
    const float* __restrict__ tmp,
    int i1, int i2, int j1, int j2, int im, int jm, int nl)
{
    const int i = i1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int j = j1 + blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z;
    if (i > i2 || j > j2 || l >= nl) return;
    const size_t k = IDX3(i - 1, j - 1, l, im, jm);
    full[k] = tmp[k];
}
