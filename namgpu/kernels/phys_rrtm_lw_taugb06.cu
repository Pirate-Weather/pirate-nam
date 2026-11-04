// phys_rrtm_lw_taugb06.cu -- unit phys-rrtm-lw-taugb06: RRTM LW band 6 optical depths, nam-gpu.
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). The
//   lw-rrtm3 header of phys/radlw_main_nmmb.f and the module header of phys/radlw_datatb.f
//   (module_radlw_kgb06) are kept in the NAM clone and apply to this port.
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:4602-4683 (subroutine taugb06, internal to
//   taumol; nam.v4.2.11, 83df4e13c1). Its tables are module_radlw_kgb06 from phys/radlw_datatb.f
//   (the module this routine USEs), extracted to namgpu/data/rrtm_lw/kgb06.npz.
//
// Band 6: 820-980 cm-1 (low key H2O, low minor CO2, high minor CFC-11 and CFC-12).
//
// Fortran -> CUDA mapping:
//   * Fortran 1-based indices are kept as values; 0-based offsets are taken where the array is read.
//   * ns06 = NS06 = 68 (module_radlw_parameters, radlw_param.f:114), so taug(ns06+ig,k) with ig 1..8
//     is row (68 + ig0) of the 140-row taug. The caller passes the 8-row band slab, so the kernel
//     writes slab row ig0 = ig-1.
//   * nspa(6) = 1 (radlw_main_nmmb.f:284), so ind0 = (jp-1)*5 + (jt-1) + 1 and
//     ind1 = jp*5 + (jt1-1) + 1, both 1-based into absa(ig, 1..65).
//   * chi_mls(2, jp+1) is the CO2 reference mixing ratio at reference level jp+1 (1-based); the caller
//     passes column 2 of chi_mls as chi_co2[59] (0-based), so the value is chi_co2[jp].
//   * The lower loop (k = 1..laytrop) and the upper loop (k = laytrop+1..nlay) are both kept; every
//     (ig, k) element of the slab is written by one of them.
//   * Per column and layer k, the self-continuum, foreign-continuum, minor-gas and band-6 absorption
//     terms are evaluated in the Fortran order and parenthesisation. Fortran reals are single here.
//   * (ratco2-2.0)**0.77 is powf (Fortran real**real).
//
// Layouts (device, C order; col = j*IM + i, zero-based):
//   per-layer per-column inputs      [nlay][ncol]                 element (k, col) at k*ncol + col
//   colamt, wx (multi-gas)           [MAXGAS or 4][nlay][ncol]    element (m, k, col) at (m*nlay + k)*ncol + col
//   taug6, fracs6 (band-6 slab out)  [8][nlay][ncol]              element (ig, k, col) at (ig*nlay + k)*ncol + col
//   absa   [65][8]   absa(ig, m)    at (m-1)*8 + (ig-1)
//   selfref[10][8], forref[4][8], ka_mco2[19][8]   same convention
//   fracrefa[8], cfc11adj[8], cfc12[8], chi_co2[59] (0-based by reference level)
//
// Launch: grid_cols(IM, JM) with BLOCK_COL, one thread per column (the layer loop is the Fortran k-loop).
// No host sync, no allocation, no print. No MPI, halo or OpenMP.

#include "nmmb_common.cuh"

#define NG06_   8
#define NS06_   68     /* radlw_param.f:114, NS06 = 68; slab row = NS06 + ig, caller places the slab */
#define NSPA6_  1      /* radlw_main_nmmb.f:284, nspa(6) = 1 */

extern "C" __global__ void phys_rrtm_lw_taugb06(
    const int nlay, const int ncol,
    const int* __restrict__ laytrop,          /* [ncol]          */
    const int* __restrict__ jp,               /* [nlay][ncol]    */
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const int* __restrict__ indminor,
    const float* __restrict__ coldry,         /* [nlay][ncol]    */
    const float* __restrict__ colamt,         /* [6][nlay][ncol] */
    const float* __restrict__ wx,             /* [4][nlay][ncol] */
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ minorfrac,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ chi_co2,        /* [59]            */
    const float* __restrict__ absa,           /* [65][8]         */
    const float* __restrict__ selfref,        /* [10][8]         */
    const float* __restrict__ forref,         /* [4][8]          */
    const float* __restrict__ ka_mco2,        /* [19][8]         */
    const float* __restrict__ fracrefa,       /* [8]             */
    const float* __restrict__ cfc11adj,       /* [8]             */
    const float* __restrict__ cfc12,          /* [8]             */
    float* __restrict__ taug6,                /* [8][nlay][ncol] */
    float* __restrict__ fracs6)               /* [8][nlay][ncol] */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    /* Fortran laytrop, a count of lower layers. Clamped to 0..nlay: Fortran has no guard and an
     * out-of-range value would write past the band slab here. Valid inputs are unchanged. */
    int lt = laytrop[col];
    lt = lt < 0 ? 0 : (lt > nlay ? nlay : lt);

    /* ---- lower atmosphere: Fortran do k = 1, laytrop ---- */
    for (int k0 = 0; k0 < lt; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;          /* [nlay][ncol] element */
        const int jp_k  = jp[kc];
        const int jt_k  = jt[kc];
        const int jt1_k = jt1[kc];

        /* Fortran: ind0 = ((jp-1)*5 + (jt-1)) * nspa(6) + 1 ; ind1 = (jp*5 + (jt1-1)) * nspa(6) + 1 */
        const int ind0  = ((jp_k - 1) * 5 + (jt_k - 1)) * NSPA6_ + 1;
        const int ind1  = (jp_k * 5 + (jt1_k - 1)) * NSPA6_ + 1;

        const int inds  = indself[kc];
        const int indf  = indfor[kc];
        const int indm  = indminor[kc];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;

        /* CO2 minor-species adjustment: temp = coldry * chi_mls(2, jp+1) */
        const float temp   = coldry[kc] * chi_co2[jp_k];          /* 0-based chi_co2[jp] = chi_mls(2, jp+1) */
        const float colco2 = colamt[(size_t)(1 * nlay + k0) * ncol + col];   /* colamt(k,2) */
        const float ratco2 = colco2 / temp;
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 2.0f + powf(ratco2 - 2.0f, 0.77f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = colco2;
        }

        const float colh2o = colamt[(size_t)(0 * nlay + k0) * ncol + col];   /* colamt(k,1) */
        const float wx2    = wx[(size_t)(1 * nlay + k0) * ncol + col];       /* wx(k,2) */
        const float wx3    = wx[(size_t)(2 * nlay + k0) * ncol + col];       /* wx(k,3) */
        const float sfac   = selffac[kc];
        const float sfrac  = selffrac[kc];
        const float ffac   = forfac[kc];
        const float ffrac  = forfrac[kc];
        const float mfrac  = minorfrac[kc];
        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];

        for (int ig0 = 0; ig0 < NG06_; ++ig0) {
            /* 1-based table indices: selfref(ig, inds), absa(ig, ind0) etc. are stored [m-1][ig0] */
            const float selfA = selfref[(size_t)(inds - 1) * NG06_ + ig0];
            const float selfB = selfref[(size_t)(indsp - 1) * NG06_ + ig0];
            const float tauself = sfac * (selfA + sfrac * (selfB - selfA));

            const float forA = forref[(size_t)(indf - 1) * NG06_ + ig0];
            const float forB = forref[(size_t)(indfp - 1) * NG06_ + ig0];
            const float taufor = ffac * (forA + ffrac * (forB - forA));

            const float coA = ka_mco2[(size_t)(indm - 1) * NG06_ + ig0];
            const float coB = ka_mco2[(size_t)(indmp - 1) * NG06_ + ig0];
            const float absco2 = coA + mfrac * (coB - coA);

            const float a00 = absa[(size_t)(ind0 - 1) * NG06_ + ig0];
            const float a10 = absa[(size_t)(ind0p - 1) * NG06_ + ig0];
            const float a01 = absa[(size_t)(ind1 - 1) * NG06_ + ig0];
            const float a11 = absa[(size_t)(ind1p - 1) * NG06_ + ig0];

            const float tau = colh2o * (f00 * a00 + f10 * a10 + f01 * a01 + f11 * a11)
                            + tauself + taufor + adjcolco2 * absco2
                            + wx2 * cfc11adj[ig0] + wx3 * cfc12[ig0];

            const size_t oc = ((size_t)ig0 * nlay + k0) * ncol + col;   /* taug6(ig, k) */
            taug6[oc]  = tau;
            fracs6[oc] = fracrefa[ig0];
        }
    }

    /* ---- upper atmosphere: Fortran do k = laytrop+1, nlay (nothing else goes on here in band 6) ---- */
    for (int k0 = lt; k0 < nlay; ++k0) {
        const float wx2 = wx[(size_t)(1 * nlay + k0) * ncol + col];
        const float wx3 = wx[(size_t)(2 * nlay + k0) * ncol + col];
        for (int ig0 = 0; ig0 < NG06_; ++ig0) {
            const size_t oc = ((size_t)ig0 * nlay + k0) * ncol + col;
            taug6[oc]  = wx2 * cfc11adj[ig0] + wx3 * cfc12[ig0];
            fracs6[oc] = fracrefa[ig0];
        }
    }
}
