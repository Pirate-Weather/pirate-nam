// phys_rrtm_sw_taumol22.cu -- taumol22: SW band 22 (7700-8050 cm-1) optical depths and the Rayleigh
// optical depth, NMMB RRTM shortwave (physics-rrtm group). Chunk path only (driver seam run_chunk).
// One thread per day column of the chunk; the layer loop k = 1..NLAY runs inside the thread.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NAM source file phys/radsw_main_nmmb.f (NCEP/EMC, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file; see
// NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep's modifications of the rrtm-sw radiation code from aer inc.   !
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4085-4201 (SUBROUTINE taumol22, nam.v4.2.11,
// 83df4e13c1). Tables: module_radsw_kgb22_nmmb (phys/radsw_datatb_nmmb.f:14760-15190), extracted by
// tools/extract-rrtm-sw-kgb22.py into namgpu/data/rrtm_sw/kgb22.npz.
//
// Chunk layout (phys-rrtm-sw-swrad chunk scratch, chunk column t, layer k0 = k - 1):
//   per-column, per-layer arrays [nc, nlay] (colmol, fac00..fac11, selffac, selffrac, forfac, forfrac,
//   indself, indfor, jp, jt, jt1), colamt [nc, nlay, MAXGAS], laytrop [nc], taug and taur [nc, nlay, 112].
//   This band writes planes NS22 = 54 and 55 (0-based) of taug and taur: the Fortran taug(:,k,NS22+j),
//   j = 1..2, and taur(:,k,NS22+j) for both g-points.
//
// The band-22 index arrays id0/id1 are evaluated in the kernel with rrtm_sw_taumol_index(22, ...)
// (rrtm_sw_bands.cuh), which is the taumol dispatcher's WHERE blocks (radsw_main_nmmb.f:3251-3256)
// with nspa(22) = 9 and nspb(22) = 1 (radsw_main_nmmb.f:308-309).
//
// Fortran lines covered:
//   4085-4096  routine header, constants o2adj = 1.6, o2tem = 4.35e-4/(350*2)
//   4097-4105  tauray = colmol*rayl; taur(:,k,NS22+j) = tauray for j = 1..NG22 (all k)
//   4107-4137  lower atmosphere (k <= laytrop): speccomb, specmult, js, fs, the eight fac weights,
//              ind01..ind14, inds/indf, the absa + self + foreign-continuum sum (WHERE k <= laytrop)
//   4138-4160  upper atmosphere: ind01/02/11/12 from absb; taug = c6*o2adj*(...) + o2cont
//
// DECLARED DEVIATIONS (both checked by tools/check-phys-rrtm-sw-taumol22-fortran.py):
//   - k == laytrop. The Fortran upper loop runs from k = MINVAL(laytrop(1:CHK))+1 with mask k >= laytrop,
//     so at k == laytrop it overwrites the absa value with the absb formula whenever laytrop exceeds the
//     minimum of its 8-column chunk. That result depends on the chunk. This port keeps the absa value
//     at k == laytrop (layer split k <= laytrop -> absa, k > laytrop -> absb). One layer per column,
//     about 95 hPa. NEEDS OWNER SIGN-OFF: the chunk-dependent Fortran result is not reproduced.
//   - speccomb == 0. The Fortran ratio colamt(1)/speccomb is 0/0 and MIN(oneminus, NaN) returns oneminus
//     in gfortran (witnessed with gfortran MIN(0.999999, NaN) = 0.999999). The Fortran therefore
//     takes specmult = 8*oneminus and js = 8, multiplied by speccomb = 0. This port takes the ratio as 0,
//     specmult = 0. Both give taug = 0 (+ o2cont = 0) because speccomb = 0 forces colamt(1) = colamt(6) = 0.
//
// Index conventions: colmol, fac00..fac11, selffac, selffrac, forfac, forfrac, indself, indfor, jp, jt,
// jt1: [nc, nlay] with the strides passed in; colamt: [nc, nlay, MAXGAS = 6], Fortran colamt(i,k,n) ->
// [t, k0, n-1]; laytrop: [nc], 1-based layer index as in the Fortran.
// Single precision (kind_taum = 4, the SINGLE_PREC build of machine.f). See the Python wrapper.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation, no printf.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define NG22_SW        2
#define NS22_SW        54
#define MSA22_SW       585
#define MSB22_SW       235
#define ONEMINUS_SW    (1.0f - 1.0e-6f)     // radsw_main_nmmb.f:292 (oneminus = 1.0 - eps, eps = 1e-6)
#define O2ADJ_SW       1.6f                 // radsw_main_nmmb.f:4094
#define O2TEM_SW       (4.35e-4f / (350.0f * 2.0f))  // radsw_main_nmmb.f:4095
#define RAYL_SW        1.54e-08f            // module_radsw_kgb22_nmmb, rayl (radsw_datatb_nmmb.f:14791)
#define STRRAT22_SW    3.63328e-02f         // strrat(22), o2adj-multiplied (radsw_datatb_nmmb.f:1108)

extern "C" __global__ void rrtm_sw_taumol22(
    const float* __restrict__ colmol,     // [nc, nlay]
    const float* __restrict__ colamt,     // [nc, nlay, MAXGAS]
    const int*   __restrict__ laytrop,    // [nc]
    const float* __restrict__ fac00,      // [nc, nlay]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const int*   __restrict__ jp,         // [nc, nlay] setcoef outputs
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const float* __restrict__ absa,       // Fortran absa(585, 2), column-major: absa[(ms-1) + 585*(g-1)]
    const float* __restrict__ absb,       // Fortran absb(235, 2)
    const float* __restrict__ selfref,    // Fortran selfref(10, 2)
    const float* __restrict__ forref,     // Fortran forref(3, 2)
    const int*   __restrict__ ndayp,      // device SW_NDAY; columns at or past it are not computed
    const int cstart,                     // first day column of this chunk
    const int ncols,                      // chunk columns (nc)
    const int nlay,
    const long long lc, const long long lk,                            // per (column, layer) arrays
    const long long ca_c, const long long ca_k, const long long ca_m,  // colamt
    const long long tg_c, const long long tg_k, const long long tg_g,  // taug, taur
    float* __restrict__ taug,             // [nc, nlay, 112]; band planes NS22_SW..NS22_SW+1 written
    float* __restrict__ taur)             // [nc, nlay, 112]; band planes NS22_SW..NS22_SW+1 written
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols) return;
    if (cstart + t >= ndayp[0]) return;

    const int L = laytrop[t];

    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t lij = (size_t)t * (size_t)lc + (size_t)k0 * (size_t)lk;
        const size_t cA  = (size_t)t * (size_t)ca_c + (size_t)k0 * (size_t)ca_k;
        const size_t cT  = (size_t)t * (size_t)tg_c + (size_t)k0 * (size_t)tg_k;
        const int k = k0 + 1;                          // Fortran layer index

        int id0, id1;                                  // band-22 table offsets, Fortran id0/id1(k,22)
        rrtm_sw_taumol_index(22, k, L, jp[lij], jt[lij], jt1[lij], &id0, &id1);

        // ---- Rayleigh (4097-4105): every layer, both g-points share it ----
        const float tauray = colmol[lij] * RAYL_SW;
        taur[cT + (size_t)NS22_SW * (size_t)tg_g]       = tauray;
        taur[cT + (size_t)(NS22_SW + 1) * (size_t)tg_g] = tauray;

        const float c1 = colamt[cA + 0 * (size_t)ca_m];   // colamt(:,k,1)  h2o
        const float c6 = colamt[cA + 5 * (size_t)ca_m];   // colamt(:,k,6)  o2
        const float o2cont = O2TEM_SW * c6;

        if (k <= L) {
            // ---- lower atmosphere: absa table (4107-4137) ----
            const float speccomb = c1 + STRRAT22_SW * c6;
            const float ratio = (speccomb > 0.0f) ? (c1 / speccomb) : 0.0f;   // DECLARED: speccomb == 0
            const float specmult = 8.0f * fminf(ONEMINUS_SW, ratio);
            const int js = 1 + (int)specmult;
            const float fs = fmodf(specmult, 1.0f);
            const float fs1 = 1.0f - fs;

            const float f00 = fac00[lij], f01 = fac01[lij], f10 = fac10[lij], f11 = fac11[lij];
            const float fac000 = fs1 * f00;
            const float fac010 = fs1 * f10;
            const float fac100 = fs * f00;
            const float fac110 = fs * f10;
            const float fac001 = fs1 * f01;
            const float fac011 = fs1 * f11;
            const float fac101 = fs * f01;
            const float fac111 = fs * f11;

            const int ind01 = max(1, min(MSA22_SW, id0 + js));
            const int ind02 = min(MSA22_SW, ind01 + 1);
            const int ind03 = min(MSA22_SW, ind01 + 9);
            const int ind04 = min(MSA22_SW, ind01 + 10);
            const int ind11 = max(1, min(MSA22_SW, id1 + js));
            const int ind12 = min(MSA22_SW, ind11 + 1);
            const int ind13 = min(MSA22_SW, ind11 + 9);
            const int ind14 = min(MSA22_SW, ind11 + 10);

            const int inds = indself[lij];
            const int indf = indfor[lij];
            const int indsp = inds + 1;
            const int indfp = indf + 1;

            for (int g = 0; g < NG22_SW; ++g) {
                const float* ag = absa + MSA22_SW * g;     // absa(:, g+1)
                const float* sg = selfref + 10 * g;        // selfref(:, g+1), MSF22 = 10
                const float* fg = forref + 3 * g;          // forref(:, g+1), MFR22 = 3

                const float taug_v = speccomb
                    * ( fac000 * ag[ind01 - 1] + fac100 * ag[ind02 - 1]
                      + fac010 * ag[ind03 - 1] + fac110 * ag[ind04 - 1]
                      + fac001 * ag[ind11 - 1] + fac101 * ag[ind12 - 1]
                      + fac011 * ag[ind13 - 1] + fac111 * ag[ind14 - 1] )
                    + c1 * ( selffac[lij] * ( sg[inds - 1]
                                + selffrac[lij] * (sg[indsp - 1] - sg[inds - 1]) )
                           + forfac[lij] * ( fg[indf - 1]
                                + forfrac[lij] * (fg[indfp - 1] - fg[indf - 1]) ) )
                    + o2cont;
                taug[cT + (size_t)(NS22_SW + g) * (size_t)tg_g] = taug_v;
            }
        } else {
            // ---- upper atmosphere: absb table (4138-4160) ----
            const int ind01 = max(1, min(MSB22_SW, id0 + 1));
            const int ind02 = min(MSB22_SW, ind01 + 1);
            const int ind11 = max(1, min(MSB22_SW, id1 + 1));
            const int ind12 = min(MSB22_SW, ind11 + 1);

            const float f00 = fac00[lij], f01 = fac01[lij], f10 = fac10[lij], f11 = fac11[lij];

            for (int g = 0; g < NG22_SW; ++g) {
                const float* bg = absb + MSB22_SW * g;     // absb(:, g+1)
                const float taug_v = c6 * O2ADJ_SW
                    * ( f00 * bg[ind01 - 1] + f10 * bg[ind02 - 1]
                      + f01 * bg[ind11 - 1] + f11 * bg[ind12 - 1] )
                    + o2cont;
                taug[cT + (size_t)(NS22_SW + g) * (size_t)tg_g] = taug_v;
            }
        }
    }
}
