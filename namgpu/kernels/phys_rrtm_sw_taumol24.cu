// phys_rrtm_sw_taumol24.cu -- TAUMOL24: SW band 24 (12850-16000 cm-1) optical depths and Rayleigh
// extinction for the NMMB RRTM shortwave (RRTM group). One thread per (i, j) column; the layer loop
// k = 1..nlay runs inside the thread, the g-point loop j = 1..NG24 (8) runs inside the layer loop,
// and every term of the Fortran routine is present.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep's modifications of the rrtm-sw radiation code from aer inc.   !
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4277-4381 (SUBROUTINE taumol24, which USEs
// module_radsw_kgb24_nmmb). The k-tables are module_radsw_kgb24_nmmb (phys/radsw_datatb_nmmb.f:15441-16967),
// extracted by tools/extract_rrtm_sw_kgb24.py into namgpu/data/rrtm_sw/kgb24.npz.
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   colamt[n, l, j, i]     [6, NLAY, JM, IM]  Fortran colamt(i,j,l,n); n = 1 H2O, 3 O3, 6 O2 (0-based 0, 2, 5)
//   colmol, jp, jt, jt1, fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor
//                          [NLAY, JM, IM]  (int32 for jp, jt, jt1, indself, indfor)
//   laytrop                [JM, IM] int32, 1-based layer index (Fortran laytrop(i,j))
//   absa[j, ind]           [NG24=8, 585]   Fortran absa(MSA24, NG24), ind 1-based in Fortran
//   absb[j, ind]           [8, 235]
//   selfref[j, ind]        [8, 10]
//   forref[j, ind]         [8, 3]
//   rayla[js, j]           [MFX24=9, 8]    Fortran rayla(NG24, MFX24) = rayla(j, js)
//   raylb[j]               [8]
//   abso3a[j], abso3b[j]   [8]
//   taug[g, l, j, i], taur[g, l, j, i]    [NGPTSW=112, NLAY, JM, IM]; this unit writes the planes
//                          g = NS24 .. NS24+NG24-1 (0-based 66..73). The other 104 planes belong to the
//                          other band units and are left untouched.
//
// Translation notes:
//   - id0 / id1 (Fortran dispatcher, radsw_main_nmmb.f:3251-3257) are evaluated per layer with
//     rrtm_sw_taumol_index(24, k, laytrop, ...): lower atmosphere (k <= laytrop) with nspa(24), upper
//     (k > laytrop) with nspb(24). The helper is shared with the other taumol units (rrtm_sw_bands.cuh).
//   - The Fortran routine runs two block loops: loop 1 over k = 1..MAXVAL(laytrop) under
//     WHERE(k <= laytrop), and loop 2 over k = MINVAL(laytrop)+1..NLAY under WHERE(k >= laytrop). The
//     two overlap at k == laytrop, so in a block with several columns the value at k == laytrop of a
//     column whose laytrop is larger than the block minimum is taken from loop 2 (the upper-atmosphere
//     formula). That result depends on the other columns of the block. This port uses the per-column
//     rule of the dispatcher (k <= laytrop: lower formula; k > laytrop: upper formula), which is what
//     the Fortran computes for every column of a single-column block and what RRTM defines.
//   - Fortran mod(specmult, f_one) with specmult >= 0 is specmult - int(specmult).
//   - Fortran MIN(oneminus, x) is fminf(SW_ONEMINUS, x) (rrtm_sw_bands.cuh).
//   - Single precision literals; fmad on (NVRTC option); no host sync, no device allocation, no printf.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. One launch covers the whole (i, j) plane.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define SW24_B       8            // band 24 - 16 (0-based band index into rrtm_sw_bands.cuh tables)
#define SW24_NG      8            // NG24 (radsw_param_nmmb.f:162)
#define SW24_NS      66           // NS24 (radsw_param_nmmb.f:174-175), 0-based first g-point of band 24
#define SW24_MSA     585          // MSA24 (radsw_kgb24_nmmb)
#define SW24_MSB     235          // MSB24
#define SW24_MSF     10           // MSF24
#define SW24_MFR     3            // MFR24
#define SW24_NGPTSW  112

// Fortran 1-based table indices -> flat C offsets (table layouts in the header above).
#define SW24_ABSA(j, ind)    absa[(size_t)(j) * SW24_MSA + ((ind) - 1)]
#define SW24_ABSB(j, ind)    absb[(size_t)(j) * SW24_MSB + ((ind) - 1)]
#define SW24_SELFREF(j, ind) selfref[(size_t)(j) * SW24_MSF + ((ind) - 1)]
#define SW24_FORREF(j, ind)  forref[(size_t)(j) * SW24_MFR + ((ind) - 1)]

extern "C" __global__ void phys_rrtm_sw_taumol24(
    const float* __restrict__ colamt,
    const float* __restrict__ colmol,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const int*   __restrict__ laytrop,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const float* __restrict__ rayla,
    const float* __restrict__ raylb,
    const float* __restrict__ abso3a,
    const float* __restrict__ abso3b,
    float* __restrict__ taug,
    float* __restrict__ taur,
    const int nlay, const int jm, const int im)
{
    COLUMN_IJ(im, jm)
    const int lt = laytrop[IDX2(i0, j0, im)];
    const float strrat24 = c_sw_strrat[SW24_B];      // strrat(24) (radsw_datatb_nmmb.f:1108-1110)

    for (int k = 1; k <= nlay; ++k) {
        const size_t lk = IDX3(i0, j0, k - 1, im, jm);
        const float h2o = colamt[IDX4(i0, j0, k - 1, 0, im, jm, nlay)];   // colamt(k,1)
        const float o3  = colamt[IDX4(i0, j0, k - 1, 2, im, jm, nlay)];   // colamt(k,3)
        const float o2  = colamt[IDX4(i0, j0, k - 1, 5, im, jm, nlay)];   // colamt(k,6)
        const float cm  = colmol[lk];

        // id0 / id1 for band 24 at this layer (Fortran radsw_main_nmmb.f:3251-3257).
        int id0, id1;
        rrtm_sw_taumol_index(SW_NBLOW + SW24_B, k, lt, jp[lk], jt[lk], jt1[lk], &id0, &id1);

        if (k <= lt) {
            // ---- lower atmosphere: Fortran loop 1 (lines 4290-4327 region of taumol24) ----
            const float speccomb = h2o + strrat24 * o2;
            const float specmult = 8.0f * fminf(SW_ONEMINUS, h2o / speccomb);
            const int   jsi = (int)specmult;
            const int   js  = 1 + jsi;
            const float fs  = specmult - (float)jsi;      // mod(specmult, f_one)
            const float fs1 = 1.0f - fs;

            const float fac000 = fs1 * fac00[lk];
            const float fac010 = fs1 * fac10[lk];
            const float fac100 = fs  * fac00[lk];
            const float fac110 = fs  * fac10[lk];
            const float fac001 = fs1 * fac01[lk];
            const float fac011 = fs1 * fac11[lk];
            const float fac101 = fs  * fac01[lk];
            const float fac111 = fs  * fac11[lk];

            const int ind01 = max(1, min(SW24_MSA, id0 + js));
            const int ind02 = min(SW24_MSA, ind01 + 1);
            const int ind03 = min(SW24_MSA, ind01 + 9);
            const int ind04 = min(SW24_MSA, ind01 + 10);
            const int ind11 = max(1, min(SW24_MSA, id1 + js));
            const int ind12 = min(SW24_MSA, ind11 + 1);
            const int ind13 = min(SW24_MSA, ind11 + 9);
            const int ind14 = min(SW24_MSA, ind11 + 10);

            const int inds  = indself[lk];
            const int indf  = indfor[lk];
            const int indsp = inds + 1;
            const int indfp = indf + 1;
            const float sfac = selffac[lk];
            const float sfrc = selffrac[lk];
            const float ffac = forfac[lk];
            const float ffrc = forfrac[lk];

            for (int jj = 0; jj < SW24_NG; ++jj) {
                const float absorb =
                      fac000 * SW24_ABSA(jj, ind01) + fac100 * SW24_ABSA(jj, ind02)
                    + fac010 * SW24_ABSA(jj, ind03) + fac110 * SW24_ABSA(jj, ind04)
                    + fac001 * SW24_ABSA(jj, ind11) + fac101 * SW24_ABSA(jj, ind12)
                    + fac011 * SW24_ABSA(jj, ind13) + fac111 * SW24_ABSA(jj, ind14);
                const float selfc = SW24_SELFREF(jj, inds)
                                  + sfrc * (SW24_SELFREF(jj, indsp) - SW24_SELFREF(jj, inds));
                const float forc  = SW24_FORREF(jj, indf)
                                  + ffrc * (SW24_FORREF(jj, indfp) - SW24_FORREF(jj, indf));

                taug[IDX4(i0, j0, k - 1, SW24_NS + jj, im, jm, nlay)] =
                      speccomb * absorb
                    + o3 * abso3a[jj]
                    + h2o * (sfac * selfc + ffac * forc);

                // Rayleigh: colmol * (rayla(j,js) + fs*(rayla(j,js+1) - rayla(j,js)))
                const float ray0 = rayla[(size_t)(js - 1) * SW24_NG + jj];
                const float ray1 = rayla[(size_t)(js)     * SW24_NG + jj];
                taur[IDX4(i0, j0, k - 1, SW24_NS + jj, im, jm, nlay)] =
                    cm * (ray0 + fs * (ray1 - ray0));
            }
        } else {
            // ---- upper atmosphere: Fortran loop 2 (k > laytrop) ----
            const int ind01 = max(1, min(SW24_MSB, id0 + 1));
            const int ind02 = min(SW24_MSB, ind01 + 1);
            const int ind11 = max(1, min(SW24_MSB, id1 + 1));
            const int ind12 = min(SW24_MSB, ind11 + 1);

            const float f00 = fac00[lk], f01 = fac01[lk], f10 = fac10[lk], f11 = fac11[lk];

            for (int jj = 0; jj < SW24_NG; ++jj) {
                taug[IDX4(i0, j0, k - 1, SW24_NS + jj, im, jm, nlay)] =
                      o2 * (f00 * SW24_ABSB(jj, ind01) + f10 * SW24_ABSB(jj, ind02)
                          + f01 * SW24_ABSB(jj, ind11) + f11 * SW24_ABSB(jj, ind12))
                    + o3 * abso3b[jj];

                taur[IDX4(i0, j0, k - 1, SW24_NS + jj, im, jm, nlay)] = cm * raylb[jj];
            }
        }
    }
}
