// phys_rrtm_sw_taumol17.cu -- unit phys-rrtm-sw-taumol17: NMMB RRTM shortwave band 17 optical depths, WOOF-NAM.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). The sw-rrtm3
//   header of phys/radsw_main_nmmb.f and the module headers of phys/radsw_datatb.f and
//   phys/radsw_datatb_nmmb.f (module_radsw_kgb17, module_radsw_kgb17_nmmb) are kept in the NAM clone and
//   apply to this port. Band-17 k-tables: namgpu/data/rrtm_sw/kgb17.npz (extracted by
//   tools/extract-rrtm-sw-kgb17.py from module_radsw_kgb17_nmmb; the GFS-named module_radsw_kgb17 that
//   taumol17 USEs is numerically identical, checked 2026-11-04).
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:3495-3627 (subroutine taumol17, internal to
//   taumol; nam.v4.2.11, 83df4e13c1).
//
// Band 17: 3250-4000 cm-1, low key H2O and CO2, high key H2O and CO2. NG17 = 12 g-points at
//   NS17 = 6 (radsw_param_nmmb.f). This kernel writes the 12-row band slab; the caller places it at
//   0-based g-points 6..17 of its taug and taur (NGPT = 112).
//
// Fortran -> CUDA mapping:
//   * Fortran 1-based indices are kept as values; array reads subtract 1 at the access.
//   * colamt(k,1) = H2O, colamt(k,2) = CO2 (taumol17 reads only these two of MAXGAS = 6).
//   * speccomb = colamt(k,1) + strrat(17)*colamt(k,2); strrat(17) is passed as a scalar.
//   * specmult = NSPA-style binary parameter: lower 8.0*min(oneminus, c1/speccomb), upper 4.0*min(...).
//     js = 1 + int(specmult), fs = MOD(specmult, f_one) = fmodf(specmult, 1.0f), fs1 = f_one - fs.
//     oneminus = 1.0 - eps, eps = 1.0e-6 (radsw_main_nmmb.f:291-292).
//   * id0 / id1 are the taumol driver's band indices (radsw_main_nmmb.f:3251-3257) for band 17, computed
//     here from jp, jt, jt1 and the layer's branch: lower (k <= laytrop) uses nspa(17) = 9,
//     upper (k > laytrop) uses nspb(17) = 5. Upper: id0 = ((jp-13)*5 + (jt-1))*5, id1 = ((jp-12)*5 + (jt1-1))*5.
//   * ind01..ind14 clamps, MAX(1, MIN(MSA17 or MSB17, ...)), with the Fortran offsets
//     +1, +9, +10 (lower, MSA17 = 585) and +1, +5, +6 (upper, MSB17 = 1175).
//   * Lower terms: taug = speccomb*(8 absa terms) + colamt(k,1)*(selffac*(selfref(inds,j)
//     + selffrac*(selfref(inds+1,j)-selfref(inds,j))) + forfac*(forref(indf,j)
//     + forfrac*(forref(indf+1,j)-forref(indf,j)))).  Upper terms: taug = speccomb*(8 absb terms)
//     + (colamt(k,1)*forfac)*(forref(indf,j) + forfrac*(forref(indf+1,j)-forref(indf,j))).
//   * taur(k, NS17+j) = colmol(k)*rayl for every k = 1..nlay and every j, with rayl = 6.86e-10
//     (module_radsw_kgb17).
//   * Self term when indself == 0: setcoef sets indself = 0 and selffac = 0 above the self-continuum
//     range. The Fortran reads selfref(0, j) there and multiplies by selffac = 0; this kernel skips the
//     self term, which is the same value for finite table reads and avoids an out-of-range read.
//   * DIVERGENCE FROM THE FORTRAN (declared, not bitwise): the Fortran runs its lower loop over
//     k = 1..MAXVAL(laytrop) with WHERE k <= laytrop, and its upper loop over k = MINVAL(laytrop)+1..NLAY
//     with WHERE k >= laytrop. So at k == laytrop it overwrites the absa value with the absb value
//     (using the lower-branch indices) whenever the chunk minimum of laytrop is below that laytrop. That
//     overlap depends on the CPU chunk (CHK), which the GPU path drops. This kernel uses the split the
//     GFS taumol17 (radsw_main.f) and RRTM use: k <= laytrop is the lower branch, k > laytrop the upper.
//     The value at k == laytrop therefore differs from the NMMB chunked overlap. Physics goes its own way
//     (owner ruling, 2026-08-20); this is recorded for the unit's receipt.
//   * The Fortran computes taumol17 only for the nday daylight columns (taumol, radsw_main_nmmb.f:3340-3365).
//     This kernel computes every column; night columns carry values the caller does not read.
//   * Every assignment of the Fortran routine has a term here: the Rayleigh loop, both species loops, the
//     absa/absb sums, the self and foreign continuum terms.
//
// Layouts (device, C order; col = j*IM + i, zero-based; ncol = IM*JM):
//   per-layer per-column inputs  [nlay][ncol]           element (k, col) at k*ncol + col
//   laytrop                      [ncol]                 int32, Fortran laytrop (count of lower layers)
//   colamt (MAXGAS = 6)          [6][nlay][ncol]        element (m, k, col) at (m*nlay + k)*ncol + col
//   taug17, taur17 (band slab)   [12][nlay][ncol]       element (g, k, col) at (g*nlay + k)*ncol + col
//   absa   [585][12]    absa(ind, g)    at (ind-1)*12 + (g-1)
//   absb   [1175][12]   absb(ind, g)    at (ind-1)*12 + (g-1)
//   selfref[10][12]     selfref(ind, g) at (ind-1)*12 + (g-1)
//   forref [4][12]      forref(ind, g)  at (ind-1)*12 + (g-1)
//
// Launch: grid_cols(IM, JM) with BLOCK_COL, one thread per column (the layer loop is the Fortran k-loop).
// No host sync, no allocation, no print. No MPI, halo or OpenMP.

#include "nmmb_common.cuh"

#define NG17_    12                 /* radsw_param_nmmb.f:161, NG17 = 12 */
#define MSA17_   585                /* module_radsw_kgb17 MSA17 */
#define MSB17_   1175               /* module_radsw_kgb17 MSB17 */
#define MSF17_   10                 /* module_radsw_kgb17 MSF17 (selfref rows) */
#define MFR17_   4                  /* module_radsw_kgb17 MFR17 (forref rows) */
#define NSPA17_  9                  /* radsw_main_nmmb.f nspa(17), band order 16..29 */
#define NSPB17_  5                  /* radsw_main_nmmb.f nspb(17) */
#define RAYL17   6.86e-10f          /* module_radsw_kgb17 rayl */
#define ONEMINUS_SW (1.0f - 1.0e-6f) /* radsw_main_nmmb.f:291-292 */

extern "C" __global__ void phys_rrtm_sw_taumol17(
    const int nlay, const int ncol,
    const float strrat17,                  /* strrat(17) */
    const int* __restrict__ laytrop,       /* [ncol]          */
    const int* __restrict__ jp,            /* [nlay][ncol]    */
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const float* __restrict__ colamt,      /* [6][nlay][ncol] */
    const float* __restrict__ colmol,      /* [nlay][ncol]    */
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ absa,        /* [585][12]       */
    const float* __restrict__ absb,        /* [1175][12]      */
    const float* __restrict__ selfref,     /* [10][12]        */
    const float* __restrict__ forref,      /* [4][12]         */
    float* __restrict__ taug17,            /* [12][nlay][ncol] */
    float* __restrict__ taur17)            /* [12][nlay][ncol] */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    const int lt = laytrop[col];           /* Fortran laytrop: layers 1..lt are the lower atmosphere */

    /* ---- Rayleigh: Fortran do k = 1, nlay; taur(:, k, NS17+j) = colmol(k) * rayl for all j ---- */
    for (int k0 = 0; k0 < nlay; ++k0) {
        const float tauray = colmol[(size_t)k0 * ncol + col] * RAYL17;
        for (int j0 = 0; j0 < NG17_; ++j0)
            taur17[((size_t)j0 * nlay + k0) * ncol + col] = tauray;
    }

    /* ---- lower atmosphere: Fortran do k = 1, laytrop (k0 = 0 .. lt-1) ---- */
    for (int k0 = 0; k0 < lt; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;                  /* [nlay][ncol] element */
        const float c1 = colamt[kc];                                /* colamt(k,1) H2O      */
        const float c2 = colamt[((size_t)1 * nlay + k0) * ncol + col]; /* colamt(k,2) CO2   */
        const float speccomb = c1 + strrat17 * c2;
        const float specmult = 8.0f * fminf(ONEMINUS_SW, c1 / speccomb);
        const int   js  = 1 + (int)specmult;
        const float fs  = fmodf(specmult, 1.0f);
        const float fs1 = 1.0f - fs;

        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];
        const float fac000 = fs1 * f00;
        const float fac010 = fs1 * f10;
        const float fac100 = fs  * f00;
        const float fac110 = fs  * f10;
        const float fac001 = fs1 * f01;
        const float fac011 = fs1 * f11;
        const float fac101 = fs  * f01;
        const float fac111 = fs  * f11;

        const int id0 = ((jp[kc] - 1) * 5 + (jt[kc] - 1)) * NSPA17_;
        const int id1 = (jp[kc] * 5 + (jt1[kc] - 1)) * NSPA17_;

        int ind01 = max(1, min(MSA17_, id0 + js));
        int ind02 = min(MSA17_, ind01 + 1);
        int ind03 = min(MSA17_, ind01 + 9);
        int ind04 = min(MSA17_, ind01 + 10);
        int ind11 = max(1, min(MSA17_, id1 + js));
        int ind12 = min(MSA17_, ind11 + 1);
        int ind13 = min(MSA17_, ind11 + 9);
        int ind14 = min(MSA17_, ind11 + 10);

        const int inds  = indself[kc];
        const int indf  = indfor[kc];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const float cs  = selffac[kc];
        const float csf = selffrac[kc];
        const float cf  = forfac[kc];
        const float cff = forfrac[kc];

        for (int j0 = 0; j0 < NG17_; ++j0) {
            const float absterm =
                  fac000 * absa[(size_t)(ind01 - 1) * NG17_ + j0]
                + fac100 * absa[(size_t)(ind02 - 1) * NG17_ + j0]
                + fac010 * absa[(size_t)(ind03 - 1) * NG17_ + j0]
                + fac110 * absa[(size_t)(ind04 - 1) * NG17_ + j0]
                + fac001 * absa[(size_t)(ind11 - 1) * NG17_ + j0]
                + fac101 * absa[(size_t)(ind12 - 1) * NG17_ + j0]
                + fac011 * absa[(size_t)(ind13 - 1) * NG17_ + j0]
                + fac111 * absa[(size_t)(ind14 - 1) * NG17_ + j0];

            const float forpart = forref[(size_t)(indf - 1) * NG17_ + j0]
                + cff * (forref[(size_t)(indfp - 1) * NG17_ + j0] - forref[(size_t)(indf - 1) * NG17_ + j0]);
            float selfpart = 0.0f;
            if (inds > 0) {   /* indself == 0: selffac is 0 there, the self term is skipped (see header) */
                selfpart = cs * (selfref[(size_t)(inds - 1) * NG17_ + j0]
                    + csf * (selfref[(size_t)(indsp - 1) * NG17_ + j0] - selfref[(size_t)(inds - 1) * NG17_ + j0]));
            }

            taug17[((size_t)j0 * nlay + k0) * ncol + col] =
                speccomb * absterm + c1 * (selfpart + cf * forpart);
        }
    }

    /* ---- upper atmosphere: Fortran do k = laytrop+1, nlay (k0 = lt .. nlay-1) ---- */
    for (int k0 = lt; k0 < nlay; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;
        const float c1 = colamt[kc];
        const float c2 = colamt[((size_t)1 * nlay + k0) * ncol + col];
        const float speccomb = c1 + strrat17 * c2;
        const float specmult = 4.0f * fminf(ONEMINUS_SW, c1 / speccomb);
        const int   js  = 1 + (int)specmult;
        const float fs  = fmodf(specmult, 1.0f);
        const float fs1 = 1.0f - fs;

        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];
        const float fac000 = fs1 * f00;
        const float fac010 = fs1 * f10;
        const float fac100 = fs  * f00;
        const float fac110 = fs  * f10;
        const float fac001 = fs1 * f01;
        const float fac011 = fs1 * f11;
        const float fac101 = fs  * f01;
        const float fac111 = fs  * f11;

        const int id0 = ((jp[kc] - 13) * 5 + (jt[kc] - 1)) * NSPB17_;
        const int id1 = ((jp[kc] - 12) * 5 + (jt1[kc] - 1)) * NSPB17_;

        int ind01 = max(1, min(MSB17_, id0 + js));
        int ind02 = min(MSB17_, ind01 + 1);
        int ind03 = min(MSB17_, ind01 + 5);
        int ind04 = min(MSB17_, ind01 + 6);
        int ind11 = max(1, min(MSB17_, id1 + js));
        int ind12 = min(MSB17_, ind11 + 1);
        int ind13 = min(MSB17_, ind11 + 5);
        int ind14 = min(MSB17_, ind11 + 6);

        const int indf  = indfor[kc];
        const int indfp = indf + 1;
        const float cf  = forfac[kc];
        const float cff = forfrac[kc];

        for (int j0 = 0; j0 < NG17_; ++j0) {
            const float absterm =
                  fac000 * absb[(size_t)(ind01 - 1) * NG17_ + j0]
                + fac100 * absb[(size_t)(ind02 - 1) * NG17_ + j0]
                + fac010 * absb[(size_t)(ind03 - 1) * NG17_ + j0]
                + fac110 * absb[(size_t)(ind04 - 1) * NG17_ + j0]
                + fac001 * absb[(size_t)(ind11 - 1) * NG17_ + j0]
                + fac101 * absb[(size_t)(ind12 - 1) * NG17_ + j0]
                + fac011 * absb[(size_t)(ind13 - 1) * NG17_ + j0]
                + fac111 * absb[(size_t)(ind14 - 1) * NG17_ + j0];

            const float forpart = forref[(size_t)(indf - 1) * NG17_ + j0]
                + cff * (forref[(size_t)(indfp - 1) * NG17_ + j0] - forref[(size_t)(indf - 1) * NG17_ + j0]);

            taug17[((size_t)j0 * nlay + k0) * ncol + col] =
                speccomb * absterm + (c1 * cf) * forpart;
        }
    }
}
