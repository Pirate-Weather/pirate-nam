// phys_rrtm_sw_taumol16.cu -- TAUMOL16: SW band 16 (2600-3250 cm-1) gaseous and Rayleigh optical
// depths for the NMMB RRTM shortwave (RRTM group). One thread per column; the layer loop k = 1..nlay
// runs inside the thread, the g-point loop j = 1..NG16 (NG16 = 6) runs inside the layer loop, and every
// term of the Fortran routine is present except the chunk-dependent overlap noted below.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes NCEP modifications of the rrtm-sw radiation code from AER inc.     !
//
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:3384-3490 (SUBROUTINE taumol16, an
// internal procedure of taumol, which USEs module_radsw_kgb16). The k-tables are
// module_radsw_kgb16_nmmb (phys/radsw_datatb_nmmb.f:1384-2528), extracted by
// tools/extract_rrtm_sw_kgb16.py into namgpu/data/rrtm_sw/kgb16.npz.
//
// Two entry layouts share this one kernel. Strides are passed in, so the same body serves both:
//   full domain (one launch, all columns live): [nlay, JM, IM] arrays, column t = j*IM + i,
//       lc = 1, lk = JM*IM; colamt [6, nlay, JM, IM]: ca_c = 1, ca_k = JM*IM, ca_m = nlay*JM*IM;
//       taug/taur [6, nlay, JM, IM]: tg_c = 1, tg_k = JM*IM, tg_g = nlay*JM*IM.
//   chunk (driver seam run_chunk, day columns cstart .. cstart+nc-1 of SW_NDAY): [nc, nlay] arrays,
//       lc = nlay, lk = 1; colamt [nc, nlay, 6]: ca_c = nlay*6, ca_k = 6, ca_m = 1;
//       taug/taur [nc, nlay, NGPTSW]: tg_c = nlay*NGPTSW, tg_k = NGPTSW, tg_g = 1.
// Only g-point planes 0..5 (NS16 = 0, NG16 = 6) are written; the other planes belong to other units.
//
// Index conventions (CONTRACT.md section 1). Integer index values (indself, indfor, id0, id1, laytrop)
// are the Fortran values (1-based):
//   colamt        Fortran colamt(k,m), m = 1..6 -> plane m-1
//   colmol, fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac   per (column, layer)
//   indself, indfor, jp, jt, jt1                                              per (column, layer), int32
//   id0, id1 (band 16) are evaluated per layer by rrtm_sw_taumol_index(16, ...) (rrtm_sw_bands.cuh),
//   which is the Fortran id0(:,k,16) / id1(:,k,16) WHERE blocks of taumol (radsw_main_nmmb.f:3250-3257)
//   laytrop                                   per column, int32
//   absa   (Fortran absa(585,NG16))    [NG16=6, 585]   absa[g*585 + (ind-1)]
//   absb   (Fortran absb(235,NG16))    [NG16, 235]
//   selfref (Fortran selfref(10,NG16)) [NG16, 10]
//   forref  (Fortran forref(3,NG16))   [NG16, 3]
//
// Translation notes:
//   - Loop 1 (the Fortran k = 1..MAXVAL(laytrop) with mask k <= laytrop) is k <= laytrop(column) here.
//   - Loop 2 (the Fortran k = MINVAL(laytrop(1:CHK))+1..nlay with mask k >= laytrop) is k >= laytrop(column)
//     here. DECLARED DIVERGENCE (NAM-GPU, 2026-11-04 review): the Fortran MINVAL is over the native
//     CHNK_RRTM = 8 column tile (and the day-part split inside it), which depends on the process and
//     thread decomposition, so no fixed rule reproduces it. Where both loops write layer k == laytrop
//     (laytrop > tile minimum), the upper form wins, as in the Fortran. Where laytrop equals its tile
//     minimum, the Fortran keeps the lower form at k == laytrop and this kernel writes the upper form.
//     The same divergence is declared in phys_rrtm_sw_taumol17/20/21. No domain-wide minimum is used.
//   - taur is written for every layer and g-point (Fortran loop "do k = 1, nlay" that sets
//     tauray = colmol*rayl).
//   - Fortran MOD(a, f_one) for a >= 0 is fmodf(a, 1.0f). Fortran INT() truncates toward zero.
//   - Fortran MIN(a, oneminus) is fminf(a, oneminus); SW_ONEMINUS from rrtm_sw_bands.cuh.
//   - strrat(16) = 252.131 (radsw_datatb_nmmb.f:1108), passed as strrat16; rayl = 2.91e-10, passed as rayl.
//   - The table-index clamps MAX(1, MIN(MSA16, ...)) and MIN(MSA16, ind+n) are kept as written.
//   - Index range of the self and foreign continuum reads (indsp - 1 and indfp - 1 into selfref and forref
//     rows of length 10 and 3). Within the lower atmosphere (k <= laytrop) setcoef sets indself in 1..9
//     and indfor in 1..2 (clamped in phys_rrtm_sw_setcoef.cu), so indsp <= 10 and indfp <= 3 stay inside
//     their row. indself = 0 is written only in the upper branch (setcoef, plog <= 4.56), which loop 1 does
//     not read for monotone pressure. A non-monotone column can reach it, and the Fortran reads out of
//     bounds there too, so that case is native-undefined. The wrapper's check asserts the range.
//
// Fast numerics: fmad on (NVRTC option), IEEE division, no host sync, no device allocation.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define TAU16_NG16   6
#define TAU16_MSA16  585
#define TAU16_MSB16  235
#define TAU16_MSF16  10
#define TAU16_MFR16  3

extern "C" __global__ void rrtm_sw_taumol16(
    const float* __restrict__ colamt,
    const float* __restrict__ colmol,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const int*   __restrict__ laytrop,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const float  rayl,
    const float  strrat16,
    const int*   __restrict__ ndayp,     // device SW_NDAY (chunk path); read only when use_gate != 0
    const int    use_gate,
    const int    nday_host,              // live column count when use_gate == 0 (full domain)
    const int    cstart,                 // first day column of this launch (chunk path)
    const int    ncols,                  // columns in this launch (nc, or JM*IM)
    const int    nlay,
    const long long lc, const long long lk,                            // per (column, layer) arrays
    const long long ca_c, const long long ca_k, const long long ca_m,  // colamt
    const long long tg_c, const long long tg_k, const long long tg_g,  // taug, taur
    float* __restrict__ taug,
    float* __restrict__ taur)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols) return;
    const int live = use_gate ? ndayp[0] : nday_host;
    if (cstart + t >= live) return;

    const int L = laytrop[t];

    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t lij = (size_t)t * (size_t)lc + (size_t)k0 * (size_t)lk;
        const size_t cA  = (size_t)t * (size_t)ca_c + (size_t)k0 * (size_t)ca_k;
        const size_t cT  = (size_t)t * (size_t)tg_c + (size_t)k0 * (size_t)tg_k;
        const int k = k0 + 1;                 // Fortran layer index
        int id0, id1;                         // band-16 table offsets, Fortran id0/id1(k,16)
        rrtm_sw_taumol_index(16, k, L, jp[lij], jt[lij], jt1[lij], &id0, &id1);

        // Rayleigh optical depth, every layer and g-point (taur(1:CHK,k,NS16+j) = colmol*rayl).
        const float tauray = colmol[lij] * rayl;
        for (int g = 0; g < TAU16_NG16; ++g) {
            taur[cT + (size_t)g * (size_t)tg_g] = tauray;
        }

        // Loop 1: lower atmosphere, k <= laytrop (water vapour and CH4 binary species).
        if (k <= L) {
            const float c1 = colamt[cA + 0 * (size_t)ca_m];   // colamt(k,1) H2O
            const float c5 = colamt[cA + 4 * (size_t)ca_m];   // colamt(k,5) CH4
            const float speccomb = c1 + strrat16 * c5;
            const float specmult = 8.0f * fminf(SW_ONEMINUS, c1 / speccomb);

            const int js = 1 + (int)specmult;
            const float fs = fmodf(specmult, 1.0f);
            const float fs1 = 1.0f - fs;

            const float f00 = fac00[lij];
            const float f01 = fac01[lij];
            const float f10 = fac10[lij];
            const float f11 = fac11[lij];
            const float fac000 = fs1 * f00;
            const float fac010 = fs1 * f10;
            const float fac100 = fs * f00;
            const float fac110 = fs * f10;
            const float fac001 = fs1 * f01;
            const float fac011 = fs1 * f11;
            const float fac101 = fs * f01;
            const float fac111 = fs * f11;

            const int ind01 = max(1, min(TAU16_MSA16, id0 + js));
            const int ind02 = min(TAU16_MSA16, ind01 + 1);
            const int ind03 = min(TAU16_MSA16, ind01 + 9);
            const int ind04 = min(TAU16_MSA16, ind01 + 10);
            const int ind11 = max(1, min(TAU16_MSA16, id1 + js));
            const int ind12 = min(TAU16_MSA16, ind11 + 1);
            const int ind13 = min(TAU16_MSA16, ind11 + 9);
            const int ind14 = min(TAU16_MSA16, ind11 + 10);

            const int inds = indself[lij];
            const int indf = indfor[lij];
            const int indsp = inds + 1;
            const int indfp = indf + 1;
            const float sfac = selffac[lij];
            const float sfrac = selffrac[lij];
            const float ffac = forfac[lij];
            const float ffrac = forfrac[lij];

            for (int g = 0; g < TAU16_NG16; ++g) {
                const float* A = absa + g * TAU16_MSA16;
                const float* S = selfref + g * TAU16_MSF16;
                const float* F = forref + g * TAU16_MFR16;
                const float lower =
                    speccomb * (fac000 * A[ind01 - 1] + fac100 * A[ind02 - 1]
                              + fac010 * A[ind03 - 1] + fac110 * A[ind04 - 1]
                              + fac001 * A[ind11 - 1] + fac101 * A[ind12 - 1]
                              + fac011 * A[ind13 - 1] + fac111 * A[ind14 - 1])
                    + c1 * (sfac * (S[inds - 1] + sfrac * (S[indsp - 1] - S[inds - 1]))
                          + ffac * (F[indf - 1] + ffrac * (F[indfp - 1] - F[indf - 1])));
                taug[cT + (size_t)g * (size_t)tg_g] = lower;
            }
        }

        // Loop 2: upper atmosphere, k >= laytrop (CH4 times absb). Runs after loop 1, so at k == laytrop
        // the upper form is the value kept (see the declared divergence in the header).
        if (k >= L) {
            const float c5 = colamt[cA + 4 * (size_t)ca_m];   // colamt(k,5) CH4
            const float f00 = fac00[lij];
            const float f01 = fac01[lij];
            const float f10 = fac10[lij];
            const float f11 = fac11[lij];

            // nspb(16) = 1 (radsw_main_nmmb.f data nspb); id0/id1 from the helper, then the
            // Fortran MAX/MIN clamps on id0 + 1 and id1 + 1 as written.
            const int ind01 = max(1, min(TAU16_MSB16, id0 + 1));
            const int ind02 = min(TAU16_MSB16, ind01 + 1);
            const int ind11 = max(1, min(TAU16_MSB16, id1 + 1));
            const int ind12 = min(TAU16_MSB16, ind11 + 1);

            for (int g = 0; g < TAU16_NG16; ++g) {
                const float* B = absb + g * TAU16_MSB16;
                taug[cT + (size_t)g * (size_t)tg_g] =
                    c5 * (f00 * B[ind01 - 1] + f10 * B[ind02 - 1]
                        + f01 * B[ind11 - 1] + f11 * B[ind12 - 1]);
            }
        }
    }
}
