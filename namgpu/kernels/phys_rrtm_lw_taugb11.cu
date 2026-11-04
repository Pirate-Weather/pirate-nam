// phys_rrtm_lw_taugb11.cu -- unit phys-rrtm-lw-taugb11: LW band 11 optical depths and Planck
// fractions of the NMMB RRTM longwave, nam-gpu.
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone; see NOTICE).
//   The lw-rrtm3 package header of phys/radlw_main_nmmb.f is kept as the notice:
//   "lw-rrtm3 radiation package description ... this package includes ncep modifications of the
//    rrtm-lw radiation code from aer inc."
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:5391-5478 (SUBROUTINE taugb11, nam.v4.2.11,
// 83df4e13c1). Tables: module_radlw_kgb11 of phys/radlw_datatb.f:28426-29089, extracted by
// tools/extract_rrtm_lw_kgb11.py into namgpu/data/rrtm_lw/kgb11.npz. The Fortran routine uses
// module_radlw_kgb11 (radlw_datatb.f), not module_radlw_kgb11_nmmb: its absa(ig,id) indexing needs
// absa(NG11=8, 65) and absb(8, 235), which the _nmmb layout does not provide.
//
// Constants (radlw_param_nmmb.f, literals here):
//   NG11 = 8 g-points in band 11 (radlw_param_nmmb.f:154).
//   nspa(11) = 1 and nspb(11) = 1 (radlw_main.f:284-285), so ind0/ind1 are the plain (jp, jt)
//   table index with no ka(5,13,nspa) stride.
//
// Index conventions. Indices from phys-rrtm-lw-setcoef are 1-based Fortran values (jp 1..58, jt and
// jt1 1..4, indself 0..9 (0 on the upper branch, unused there), indfor 1..3, indminor 1..18). This
// kernel keeps the Fortran 1-based arithmetic and subtracts 1 only at each array access.
//   Fortran (i,j,k) -> [k-1, j-1, i-1] (CONTRACT section 1).
//   Tables are C order reversed: absa(ig,id) -> absa[(id-1)*8 + (ig-1)], and likewise for all tables.
//
// Output rows. taug and fracs are [R, nlay, JM, IM]. The caller passes row0, the 0-based row of the
// first band-11 g-point, so the Fortran taug(ns11+ig,k) (ns11 = NS11 = 114 in a full 140-row array)
// is written to row row0 + ig - 1. A band-only buffer uses row0 = 0 and R = 8.
//
// Launch: one thread per (i, j) column (grid_cols). The layer loop runs inside the thread and covers
// both Fortran branches (lower k = 1..laytrop, upper k = laytrop+1..nlay). Every term of the Fortran
// expressions is present. No host sync, no device allocation, no print.
//
// Fast numerics: fmad on (NVRTC option), single precision (kfpt = single).

#include "nmmb_common.cuh"

#define NG11_BAND 8

// Table accessors, Fortran 1-based (ig, id).
#define ABSA(ig, id)    absa[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]
#define ABSB(ig, id)    absb[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]
#define SELFREF(ig, id) selfref[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]
#define FORREF(ig, id)  forref[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]
#define KA_MO2(ig, id)  ka_mo2[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]
#define KB_MO2(ig, id)  kb_mo2[((size_t)((id) - 1)) * NG11_BAND + ((ig) - 1)]

extern "C" __global__ void phys_rrtm_lw_taugb11(
    const int*   __restrict__ laytrop,     // [JM, IM]
    const int*   __restrict__ jp,          // [nlay, JM, IM]
    const int*   __restrict__ jt,          // [nlay, JM, IM]
    const int*   __restrict__ jt1,         // [nlay, JM, IM]
    const int*   __restrict__ indself,     // [nlay, JM, IM]
    const int*   __restrict__ indfor,      // [nlay, JM, IM]
    const int*   __restrict__ indminor,    // [nlay, JM, IM]
    const float* __restrict__ fac00,       // [nlay, JM, IM]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ minorfrac,
    const float* __restrict__ scaleminor,
    const float* __restrict__ colamt,      // [MAXGAS=6, nlay, JM, IM]
    const float* __restrict__ absa,        // [65, 8]
    const float* __restrict__ absb,        // [235, 8]
    const float* __restrict__ selfref,     // [10, 8]
    const float* __restrict__ forref,      // [4, 8]
    const float* __restrict__ fracrefa,    // [8]
    const float* __restrict__ fracrefb,    // [8]
    const float* __restrict__ ka_mo2,      // [19, 8]
    const float* __restrict__ kb_mo2,      // [19, 8]
    float* __restrict__ taug,              // [R, nlay, JM, IM]
    float* __restrict__ fracs,             // [R, nlay, JM, IM]
    const int nlay,
    const int im,
    const int jm,
    const int row0)
{
    COLUMN_IJ(im, jm)

    const int laytrop_c = laytrop[IDX2(i0, j0, im)];

    // ---- lower atmosphere: Fortran k = 1 .. laytrop ----
    for (int k = 1; k <= laytrop_c; ++k) {
        const size_t lk = IDX3(i0, j0, k - 1, im, jm);
        const int jpk  = jp[lk];
        const int jtk  = jt[lk];
        const int jt1k = jt1[lk];

        // nspa(11) = 1
        const int ind0 = (jpk - 1) * 5 + (jtk - 1) + 1;
        const int ind1 = jpk * 5 + (jt1k - 1) + 1;

        const int inds = indself[lk];
        const int indf = indfor[lk];
        const int indm = indminor[lk];
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;

        const float scaleo2 = colamt[IDX4(i0, j0, k - 1, 5, im, jm, nlay)] * scaleminor[lk];
        const float co1     = colamt[IDX4(i0, j0, k - 1, 0, im, jm, nlay)];

        for (int ig = 1; ig <= NG11_BAND; ++ig) {
            const float tauself = selffac[lk] * (SELFREF(ig, inds) + selffrac[lk]
                                  * (SELFREF(ig, indsp) - SELFREF(ig, inds)));
            const float taufor  = forfac[lk] * (FORREF(ig, indf) + forfrac[lk]
                                  * (FORREF(ig, indfp) - FORREF(ig, indf)));
            const float tauo2   = scaleo2 * (KA_MO2(ig, indm) + minorfrac[lk]
                                  * (KA_MO2(ig, indmp) - KA_MO2(ig, indm)));

            const size_t out = IDX4(i0, j0, k - 1, row0 + ig - 1, im, jm, nlay);
            taug[out] = co1
                        * (fac00[lk] * ABSA(ig, ind0) + fac10[lk] * ABSA(ig, ind0p)
                         + fac01[lk] * ABSA(ig, ind1) + fac11[lk] * ABSA(ig, ind1p))
                        + tauself + taufor + tauo2;

            fracs[out] = fracrefa[ig - 1];
        }
    }

    // ---- upper atmosphere: Fortran k = laytrop+1 .. nlay ----
    for (int k = laytrop_c + 1; k <= nlay; ++k) {
        const size_t lk = IDX3(i0, j0, k - 1, im, jm);
        const int jpk  = jp[lk];
        const int jtk  = jt[lk];
        const int jt1k = jt1[lk];

        // nspb(11) = 1
        const int ind0 = (jpk - 13) * 5 + (jtk - 1) + 1;
        const int ind1 = (jpk - 12) * 5 + (jt1k - 1) + 1;

        const int indf = indfor[lk];
        const int indm = indminor[lk];
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;

        const float scaleo2 = colamt[IDX4(i0, j0, k - 1, 5, im, jm, nlay)] * scaleminor[lk];
        const float co1     = colamt[IDX4(i0, j0, k - 1, 0, im, jm, nlay)];

        for (int ig = 1; ig <= NG11_BAND; ++ig) {
            const float taufor = forfac[lk] * (FORREF(ig, indf) + forfrac[lk]
                                 * (FORREF(ig, indfp) - FORREF(ig, indf)));
            const float tauo2  = scaleo2 * (KB_MO2(ig, indm) + minorfrac[lk]
                                 * (KB_MO2(ig, indmp) - KB_MO2(ig, indm)));

            const size_t out = IDX4(i0, j0, k - 1, row0 + ig - 1, im, jm, nlay);
            taug[out] = co1
                        * (fac00[lk] * ABSB(ig, ind0) + fac10[lk] * ABSB(ig, ind0p)
                         + fac01[lk] * ABSB(ig, ind1) + fac11[lk] * ABSB(ig, ind1p))
                        + taufor + tauo2;

            fracs[out] = fracrefb[ig - 1];
        }
    }
}
