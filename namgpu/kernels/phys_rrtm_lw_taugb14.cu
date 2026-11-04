// phys_rrtm_lw_taugb14.cu -- unit phys-rrtm-lw-taugb14: RRTM LW band 14 optical depths, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). The
//   lw-rrtm3 header of phys/radlw_main_nmmb.f and the module header of phys/radlw_datatb.f
//   (module_radlw_kgb14) are kept in the NAM clone and apply to this port.
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:5912-5976 (subroutine taugb14, internal to
//   taumol; nam.v4.2.11, 83df4e13c1). Its tables are module_radlw_kgb14 from phys/radlw_datatb.f
//   (the module this routine USEs at radlw_main_nmmb.f:5919), extracted to
//   namgpu/data/rrtm_lw/kgb14.npz by tools/extract-rrtm-lw-kgb14.py.
//
// Band 14: 2250-2380 cm-1 (low - CO2; high - CO2). NG14 = 2 g-points, NS14 = 134 (radlw_param.f:116).
//
// Fortran -> CUDA mapping:
//   * Fortran 1-based indices are kept as values; 0-based offsets are taken where the array is read.
//   * nspa(14) = 1 (radlw_main_nmmb.f:284) and nspb(14) = 1 (radlw_main_nmmb.f:285), so
//       lower: ind0 = (jp-1)*5 + (jt-1) + 1,   ind1 = jp*5 + (jt1-1) + 1          (into absa, 1..65)
//       upper: ind0 = (jp-13)*5 + (jt-1) + 1,  ind1 = (jp-12)*5 + (jt1-1) + 1     (into absb, 1..235)
//   * colamt(k,2) is the CO2 column: colamt[gas 1 (0-based)][k][col].
//   * Lower loop (k = 1..laytrop): self-continuum and foreign-continuum terms (forref uses indfor), with
//     the Fortran term order: colamt*(fac00*absa(ig,ind0) + fac10*absa(ig,ind0p) + fac01*absa(ig,ind1)
//     + fac11*absa(ig,ind1p)) + tauself + taufor. fracs = fracrefa(ig).
//   * Upper loop (k = laytrop+1..nlay): the absb interpolation only (no continuum terms). fracs = fracrefb(ig).
//   * Destination is the Fortran one: taug(ns14+ig,k) and fracs(ns14+ig,k) with NS14 = 134, i.e. 0-based
//     rows 134 + ig0 of the full 140-row taug/fracs arrays (the wrapper allocates those arrays).
//   * Every (ig, k) element of the slab is written by exactly one of the two loops.
//   * Fortran reals are single here (kind_phys); no double precision.
//
// Layouts (device, C order; col = j*IM + i, zero-based):
//   per-layer per-column inputs      [nlay][ncol]                 element (k, col) at k*ncol + col
//   colamt                           [MAXGAS=6][nlay][ncol]       element (m, k, col) at (m*nlay + k)*ncol + col
//   taug, fracs (full g-point, tile-local t = col - c0)  [140][nlay][ntile]  element (g, k, t) at (g*nlay + k)*ntile + t
//                                    band 14 writes g = 134, 135 (0-based) only
//   absa   [65][2]   absa(ig, m)   at (m-1)*2 + (ig-1)
//   absb   [235][2]  absb(ig, m)   at (m-1)*2 + (ig-1)
//   selfref[10][2], forref[4][2]   same convention
//   fracrefa[2], fracrefb[2]
//
// Launch: grid_cols(ntile, 1) with BLOCK_COL, one thread per tile column (the layer loop is the Fortran k-loop).
// Inputs are indexed with the full column count ncol; outputs with the tile width ntile (taugb08 convention).
// No host sync, no allocation, no print. No MPI, halo or OpenMP.

#include "nmmb_common.cuh"

#define NG14_   2
#define NSPA14_ 1      /* radlw_main_nmmb.f:284, nspa(14) = 1 */
#define NSPB14_ 1      /* radlw_main_nmmb.f:285, nspb(14) = 1 */
#define NS14_   134    /* radlw_param.f:116, NS14: 0-based offset of band 14's g-points */

extern "C" __global__ void phys_rrtm_lw_taugb14(
    const int nlay, const int ncol, const int c0, const int ntile,
    const int* __restrict__ laytrop,          /* [ncol]          */
    const int* __restrict__ jp,               /* [nlay][ncol]    */
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const float* __restrict__ colamt,         /* [6][nlay][ncol] */
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ absa,           /* [65][2]         */
    const float* __restrict__ absb,           /* [235][2]        */
    const float* __restrict__ selfref,        /* [10][2]         */
    const float* __restrict__ forref,         /* [4][2]          */
    const float* __restrict__ fracrefa,       /* [2]             */
    const float* __restrict__ fracrefb,       /* [2]             */
    float* __restrict__ taug,                 /* [140][nlay][ntile] full g-point, tile-local */
    float* __restrict__ fracs)                /* [140][nlay][ntile] full g-point, tile-local */
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;   /* tile-local column */
    if (t >= ntile) return;
    const int col = c0 + t;                                  /* global column */

    const int lt = laytrop[col];              /* Fortran laytrop, a count of lower layers */

    /* ---- lower atmosphere: Fortran do k = 1, laytrop ---- */
    for (int k0 = 0; k0 < lt; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;          /* [nlay][ncol] element */
        const int jp_k  = jp[kc];
        const int jt_k  = jt[kc];
        const int jt1_k = jt1[kc];

        /* Fortran: ind0 = ((jp-1)*5 + (jt-1)) * nspa(14) + 1 ; ind1 = (jp*5 + (jt1-1)) * nspa(14) + 1 */
        const int ind0  = ((jp_k - 1) * 5 + (jt_k - 1)) * NSPA14_ + 1;
        const int ind1  = (jp_k * 5 + (jt1_k - 1)) * NSPA14_ + 1;

        const int inds  = indself[kc];
        const int indf  = indfor[kc];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;

        const float colco2 = colamt[(size_t)(1 * nlay + k0) * ncol + col];   /* colamt(k,2) */
        const float sfac   = selffac[kc];
        const float sfrac  = selffrac[kc];
        const float ffac   = forfac[kc];
        const float ffrac  = forfrac[kc];
        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];

        for (int ig0 = 0; ig0 < NG14_; ++ig0) {
            /* 1-based table indices: selfref(ig, inds), absa(ig, ind0) etc. are stored [m-1][ig0] */
            const float selfA = selfref[(size_t)(inds - 1) * NG14_ + ig0];
            const float selfB = selfref[(size_t)(indsp - 1) * NG14_ + ig0];
            const float tauself = sfac * (selfA + sfrac * (selfB - selfA));

            const float forA = forref[(size_t)(indf - 1) * NG14_ + ig0];
            const float forB = forref[(size_t)(indfp - 1) * NG14_ + ig0];
            const float taufor = ffac * (forA + ffrac * (forB - forA));

            const float a00 = absa[(size_t)(ind0 - 1) * NG14_ + ig0];
            const float a10 = absa[(size_t)(ind0p - 1) * NG14_ + ig0];
            const float a01 = absa[(size_t)(ind1 - 1) * NG14_ + ig0];
            const float a11 = absa[(size_t)(ind1p - 1) * NG14_ + ig0];

            const float tau = colco2 * (f00 * a00 + f10 * a10 + f01 * a01 + f11 * a11)
                            + tauself + taufor;

            const size_t oc = ((size_t)(NS14_ + ig0) * nlay + k0) * ntile + t;   /* taug(ns14+ig, k) */
            taug[oc]  = tau;
            fracs[oc] = fracrefa[ig0];
        }
    }

    /* ---- upper atmosphere: Fortran do k = laytrop+1, nlay ---- */
    for (int k0 = lt; k0 < nlay; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;
        const int jp_k  = jp[kc];
        const int jt_k  = jt[kc];
        const int jt1_k = jt1[kc];

        /* Fortran: ind0 = ((jp-13)*5 + (jt-1)) * nspb(14) + 1 ; ind1 = ((jp-12)*5 + (jt1-1)) * nspb(14) + 1 */
        const int ind0  = ((jp_k - 13) * 5 + (jt_k - 1)) * NSPB14_ + 1;
        const int ind1  = ((jp_k - 12) * 5 + (jt1_k - 1)) * NSPB14_ + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;

        const float colco2 = colamt[(size_t)(1 * nlay + k0) * ncol + col];   /* colamt(k,2) */
        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];

        for (int ig0 = 0; ig0 < NG14_; ++ig0) {
            const float b00 = absb[(size_t)(ind0 - 1) * NG14_ + ig0];
            const float b10 = absb[(size_t)(ind0p - 1) * NG14_ + ig0];
            const float b01 = absb[(size_t)(ind1 - 1) * NG14_ + ig0];
            const float b11 = absb[(size_t)(ind1p - 1) * NG14_ + ig0];

            const size_t oc = ((size_t)(NS14_ + ig0) * nlay + k0) * ntile + t;   /* taug(ns14+ig, k) */
            taug[oc]  = colco2 * (f00 * b00 + f10 * b10 + f01 * b01 + f11 * b11);
            fracs[oc] = fracrefb[ig0];
        }
    }
}
