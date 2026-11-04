// bnd_bocoh_blend.cu -- H-point lateral boundary blend (final write of BOCOH) for the NAM parent.
//
// NOTICE: NAM source file nmm/module_FLTBNDS.F90 (NOAA/NCEP/EMC, US Government work, no licence
// file). Its NOAA header is the notice; see NOTICE at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_FLTBNDS.F90:5355-5962 (nam.v4.2.11, 83df4e13c1).
// The blend sits inside subroutine BOCOH (module_FLTBNDS.F90:5064-6387). Its blending weights
// wh(k) = 1 - (0.9/(lnsh-1))*(k-1) are set at module_FLTBNDS.F90:5156-5159.
//
// Fortran section mapping (1-based Fortran line numbers, module_FLTBNDS.F90):
//   S boundary : 5364-5419 (PD, T, Q, CW blend),   5420-5452 (PINT, j = 1..lnsh)
//   N boundary : 5453-5508 (blend),                5509-5541 (PINT, j = JDE-lnsh+1..JDE)
//   W boundary : 5542-5597 (blend),                5598-5630 (PINT, i = 1..lnsh)
//   E boundary : 5631-5686 (blend),                5687-5720 (PINT, i = IDE+1-lnsh..IDE)
//   corners    : SW 5721, SE 5765, NW 5809, NE 5853 (four 4-point averages, 0.25 weight)
//   edge PINT  : 5898-5961 (pint(i,j,1) = pt and the dsg2/pdsg1 recursion on the outer two
//                rows and columns, after the corners, no blend weight)
// Launch order on one stream, as the Fortran: strip(S), strip(N), strip(W), strip(E), corners,
// pint_edges. Strip launches each touch only their own cells; the corner launch reads only
// non-corner cells written by the strips; pint_edges runs last.
//
// H boundary variables are the parent's hardwired list (module_SOLVER_GRID_COMP.F90:7658-7664,
// MY_DOMAIN_ID == 1): NVARS_BC_2D_H = 1 (PD), NVARS_BC_3D_H = 3 (T, Q, CW), NVARS_BC_4D_H = 0.
// Q2 is not a parent H boundary variable and is not blended (module_FLTBNDS.F90 has no Q2 term).
//
// Single-process regional layout (CONTRACT.md section 1): ids = its = 1, ide = ite = IM,
// jds = jts = 1, jde = jte = JM, so its_h2 = 1, ite_h2 = IM, jts_h2 = 1, jte_h2 = JM
// (module_DM_PARALLEL.F90 looplimits). s_bdy, n_bdy, w_bdy, e_bdy are all true.
//
// Boundary strip layout (value slot only, the wrapper passes the slot-0 array; CONTRACT.md 6):
//   S and N : 2-D PD [lnsh, IM] index [jb0, ib0];     3-D [LM, lnsh, IM]  index [l0, jb0, ib0]
//   W and E : 2-D PD [JM, lnsh] index [j0, ib0];      3-D [LM, JM, lnsh]  index [l0, j0, ib0]
// jb0 / ib0 are zero-based strip indices (jb - 1 or ib - 1). H fields are [LM, JM, IM] (PD is
// [JM, IM]); PINT is [LM+1, JM, IM].
//
// Thread mapping of the strip kernel: one thread per (strip cell, along-edge index). Each thread
// owns one H column (i, j), so the PD blend, the 3-D blends and the PINT recursion of that
// column run in Fortran order inside one thread (PD first, then PINT, because PINT reads the
// blended PD).
//
// Fortran lines not mapped to a term: none. The ENABLE_SMP/OpenMP paths are host-only and do not
// apply. The 4-D H boundary loops (nvars_bc_4d_h) are not ported: the parent has no 4-D H variable
// (NVARS_BC_4D_H = 0), so the loops are dead for NAM.

#include "nmmb_common.cuh"

// Blend weight wh(k), k = 1..lnsh (Fortran 1-based index), single precision (kfpt = single).
__device__ __forceinline__ float bocoh_wh(int k, int lnsh) {
    return 1.0f - (0.9f / (float)(lnsh - 1)) * (float)(k - 1);
}

// side: 0 = S, 1 = N, 2 = W, 3 = E.
extern "C" __global__ void bnd_bocoh_blend_strip(
    const int side, const int im, const int jm, const int lm, const int lnsh,
    const float pt,
    const float* __restrict__ dsg2,        // [LM]
    const float* __restrict__ pdsg1,       // [LM]
    const float* __restrict__ bc_pd,       // S/N [lnsh, IM]  W/E [JM, lnsh]
    const float* __restrict__ bc_t,        // S/N [LM, lnsh, IM]  W/E [LM, JM, lnsh]
    const float* __restrict__ bc_q,
    const float* __restrict__ bc_cw,
    float* __restrict__ pd,                // [JM, IM]
    float* __restrict__ t,                 // [LM, JM, IM]
    float* __restrict__ q,
    float* __restrict__ cw,
    float* __restrict__ pint,              // [LM+1, JM, IM]
    const int JOFF, const int JMG)         // row slab: W/E domain tests on the global row (S only on
                                           // the first rank, N only on the last; their local rows are exact)
{
    const bool sn = (side == 0 || side == 1);
    const int nb = sn ? im : jm;                          // along-edge length
    const int b  = blockIdx.x * blockDim.x + threadIdx.x; // along-edge index, zero-based
    const int a  = blockIdx.y * blockDim.y + threadIdx.y; // strip cell, zero-based
    if (b >= nb || a >= lnsh) return;

    const int jb = a + 1;   // Fortran jb (S, N) or ib (W, E), 1-based
    const int bb = b + 1;   // Fortran along-edge index, 1-based
    int i1 = 0, j1 = 0, k = 0;
    bool ok = false;

    if (side == 0) {                          // S: j = jb, i in [jb, IDE+1-jb]
        j1 = jb; i1 = bb; k = jb;
        ok = (i1 >= jb) && (i1 <= im + 1 - jb);
    } else if (side == 1) {                   // N: j = jb + JDE - lnsh, i in [1-jb+lnsh, IDE+jb-lnsh]
        j1 = jb + jm - lnsh; i1 = bb; k = lnsh + 1 - jb;
        ok = (i1 >= 1 - jb + lnsh) && (i1 <= im + jb - lnsh);
    } else if (side == 2) {                   // W: i = ib, j in [1+ib, JDE-ib]
        i1 = jb; j1 = bb; k = jb;
        ok = (j1 + JOFF >= 1 + jb) && (j1 + JOFF <= JMG - jb);
    } else {                                  // E: i = IDE-lnsh+ib, j in [2+lnsh-ib, JDE-lnsh-1+ib]
        i1 = im - lnsh + jb; j1 = bb; k = lnsh + 1 - jb;
        ok = (j1 + JOFF >= 2 + lnsh - jb) && (j1 + JOFF <= JMG - lnsh - 1 + jb);
    }
    if (!ok) return;

    const int i0 = i1 - 1, j0 = j1 - 1;
    const float w  = bocoh_wh(k, lnsh);
    const float wc = 1.0f - w;             // (1.-wh(k)) as written in the Fortran
    const size_t cell = IDX2(i0, j0, im);

    // Strip index of this H point (value slot). bi3(l0) = bi3_base + l0 * lstride.
    size_t bi2, bi3_base, lstride;
    if (sn) {
        const size_t jb0 = (size_t)(jb - 1);
        bi2 = jb0 * im + i0;                   // [lnsh, IM]
        bi3_base = jb0 * im + i0;              // [LM, lnsh, IM]
        lstride = (size_t)lnsh * im;
    } else {
        const size_t ib0 = (size_t)(jb - 1);
        bi2 = (size_t)j0 * lnsh + ib0;         // [JM, lnsh]
        bi3_base = (size_t)j0 * lnsh + ib0;    // [LM, JM, lnsh]
        lstride = (size_t)jm * lnsh;
    }

    // 2-D PD first: the PINT recursion below reads the blended PD.
    pd[cell] = bc_pd[bi2] * w + pd[cell] * wc;

    // 3-D H variables T, Q, CW, every level.
    for (int l0 = 0; l0 < lm; ++l0) {
        const size_t c3 = IDX3(i0, j0, l0, im, jm);
        const size_t bi = bi3_base + (size_t)l0 * lstride;
        t[c3]  = bc_t[bi]  * w + t[c3]  * wc;
        q[c3]  = bc_q[bi]  * w + q[c3]  * wc;
        cw[c3] = bc_cw[bi] * w + cw[c3] * wc;
    }

    // PINT on this column: pint(i,j,1) = pt, then pint(l+1) = blend of (pint(l) + dsg2*pd + pdsg1).
    const float pdc = pd[cell];
    pint[IDX3(i0, j0, 0, im, jm)] = pt;
    for (int l = 1; l <= lm; ++l) {
        const float pl  = pint[IDX3(i0, j0, l - 1, im, jm)];
        const float pl1 = pint[IDX3(i0, j0, l, im, jm)];
        pint[IDX3(i0, j0, l, im, jm)] = (pl + (dsg2[l - 1] * pdc + pdsg1[l - 1])) * w + pl1 * wc;
    }
}

// Four corner points (Fortran 5721-5897). Corner centres (1-based): (2,2), (IM-1,2), (2,JM-1),
// (IM-1,JM-1). Each is the average of the neighbours (x,y-1), (x-1,y), (x+1,y), (x,y+1), summed
// in that order and multiplied by 0.25 (the Fortran (...)*0.25). Threads 0..3 = SW, SE, NW, NE.
extern "C" __global__ void bnd_bocoh_blend_corners(
    const int im, const int jm, const int lm,
    float* __restrict__ pd,
    float* __restrict__ t,
    float* __restrict__ q,
    float* __restrict__ cw,
    const int SN_MASK)                             // row slab: bit 0 south corners here, bit 1 north (one card: 3)
{
    const int c = threadIdx.x;
    if (c > 3 || blockIdx.x != 0) return;
    if (!(SN_MASK & ((c & 2) ? 2 : 1))) return;
    const int x0 = ((c & 1) ? (im - 1) : 2) - 1;   // zero-based centre column
    const int y0 = ((c & 2) ? (jm - 1) : 2) - 1;   // zero-based centre row

    {
        const float s = ((pd[IDX2(x0, y0 - 1, im)] + pd[IDX2(x0 - 1, y0, im)])
                         + pd[IDX2(x0 + 1, y0, im)]) + pd[IDX2(x0, y0 + 1, im)];
        pd[IDX2(x0, y0, im)] = s * 0.25f;
    }
    float* fields[3] = { t, q, cw };               // T, Q, CW (NVARS_BC_3D_H = 3)
    for (int l0 = 0; l0 < lm; ++l0) {
        for (int f = 0; f < 3; ++f) {
            float* a = fields[f];
            const float s = ((a[IDX3(x0, y0 - 1, l0, im, jm)] + a[IDX3(x0 - 1, y0, l0, im, jm)])
                             + a[IDX3(x0 + 1, y0, l0, im, jm)]) + a[IDX3(x0, y0 + 1, l0, im, jm)];
            a[IDX3(x0, y0, l0, im, jm)] = s * 0.25f;
        }
    }
}

// PINT on the outer two rows and columns (Fortran 5898-5961): pint(i,j,1) = pt and
// pint(i,j,l+1) = pint(i,j,l) + (dsg2(l)*pd(i,j) + pdsg1(l)), no blend weight. Those columns are
// written by up to two Fortran blocks with identical values, so one thread per column is
// equivalent. Runs after the corners, so it sees the corner-blended PD.
extern "C" __global__ void bnd_bocoh_blend_pint_edges(
    const int im, const int jm, const int lm, const float pt,
    const float* __restrict__ dsg2,
    const float* __restrict__ pdsg1,
    const float* __restrict__ pd,
    float* __restrict__ pint,
    const int JOFF, const int JMG)         // row slab: domain tests on the global row
{
    COLUMN_IJ(im, jm)
    const int jg = j0 + JOFF;
    // Zero-based tests: rows j0 = 0, 1 and jm-2, jm-1; columns i0 = 0, 1 and im-2, im-1.
    if (!(jg < 2 || jg >= JMG - 2 || i0 < 2 || i0 >= im - 2)) return;
    const float pdc = pd[IDX2(i0, j0, im)];
    pint[IDX3(i0, j0, 0, im, jm)] = pt;
    for (int l = 1; l <= lm; ++l) {
        pint[IDX3(i0, j0, l, im, jm)] =
            pint[IDX3(i0, j0, l - 1, im, jm)] + (dsg2[l - 1] * pdc + pdsg1[l - 1]);
    }
}
