// phys_rrtm_sw_taumol19.cu -- TAUMOL19: SW band 19 (4650-5150 cm-1, H2O + CO2) optical depths and
// Rayleigh depths for the NMMB RRTM shortwave (RRTM group). One thread per column; the layer loop
// runs inside the thread; every term of the Fortran routine is present.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
//   ==============================================================
//               sw-rrtm3 radiation package description
//   ==============================================================
//
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:3743-3847 (SUBROUTINE taumol19), with the
// k-tables of module_radsw_kgb19_nmmb (radsw_datatb_nmmb.f:8639-10141) passed in from
// namgpu/data/rrtm_sw/kgb19.npz by the wrapper. Band-19 constants: NG19 = 8, NS19 = 26 (radsw_param.f
// NG and NS data), strrat(19) = 5.49281 (radsw_datatb_nmmb.f:1108, element 19 of the NBLOW:NBHGH
// array), rayl = 2.29e-9 (module_radsw_kgb19_nmmb), oneminus = 1 - 1e-6 (radsw_main_nmmb.f:292).
// The band index helper rrtm_sw_taumol_index() is rrtm_sw_bands.cuh (the Fortran id0/id1 dispatcher
// locals, radsw_main_nmmb.f:3236-3257, evaluated per column here instead of stored for the domain).
//
// LAYER ORDER (verified 2026-11-04 against the Fortran): the layer index k1 = 1 .. nlay here is the
// RADIATION INTERNAL order, surface first (k1 = 1 is the lowest layer). It is the order the Fortran
// taumol19 sees: swrad writes pavel/colamt/fac* in that order (radsw_main_nmmb.f:732-746 for
// ivflip == 0: kk = nlp1 - k, i.e. top-first NMMB input reversed; the ivflip == 1 branch copies
// as-is), setcoef's laytrop is the last internal k with ln(p) > 4.56 (radsw_main_nmmb.f:2055, 2103),
// so laytrop is the count of lower layers counted from the surface. The NAM run uses ivflip = 0
// (rad_initialize_nmmb sets ivflip = iflip = 0). The caller flip is phys_rrtm_sw_swrad's swrad_prep
// (kk0 = nlay - 1 - k0 for ivflip == 0). The NMMB model-top-first convention (CONTRACT section 1)
// does NOT apply to these arrays; they are radiation-internal (CONTRACT section 1 SW exception).
//
// CHUNK MINIMUM (fixed 2026-11-04, review of phys-rrtm-sw-taumol19). The Fortran takes
//   MINVAL(laytrop(1:CHK)) with CHK = CHNK_RRTM = 8 (configure.nems: -DCHNK_RRTM=8)
// over each chunk of 8 consecutive DAY columns (radsw_main_nmmb.f:351; the taumol19 pass-2 loop is at
// 3827-3838). Pass 2 of the Fortran starts at k = kmin + 1 with kmin = that chunk minimum, so a
// column whose laytrop equals its chunk minimum keeps the pass-1 absa value at k = laytrop, and a
// column whose laytrop exceeds it gets the absb overwrite at k = laytrop. This kernel computes the
// minimum over the same 8-column group on the device (group = day index / 8, live columns only; the
// Fortran padding copies laytrop(nday), which is already a live value and so cannot change a minimum).
// No host sync, no per-tile minimum. The group partition needs cstart % 8 == 0 (the wrapper refuses
// other starts): a different partition gives a different kmin and so different columns.
//
// Layouts (strides passed in; one body serves the chunk path and the full-tile path):
//   chunk path (run_chunk, the swrad driver): [nc, nlay] per-column arrays, lc = nlay, lk = 1;
//       colamt [nc, nlay, 6] ca_c = nlay*6, ca_k = 6, ca_m = 1;
//       taug/taur [nc, nlay, NGPTSW] tg_c = nlay*NGPTSW, tg_k = NGPTSW, tg_g = 1.
//   tile path (taumol19(inp, out), the done-rule selftest): [nlay, ncol] arrays with c fastest,
//       lc = 1, lk = ncol; colamt [6, nlay, ncol] ca_c = 1, ca_k = ncol, ca_m = nlay*ncol;
//       taug/taur [NGPTSW, nlay, ncol] tg_c = 1, tg_k = ncol, tg_g = nlay*ncol.
// Only g-point planes NS19 + j, j = 0..7 (g = 27..34), are written; the other planes belong to other units.
//
// Index conventions: integer index values (indself, indfor, id0, id1, jp, jt, jt1, laytrop) are the
// Fortran values (1-based); colamt(k, m) is plane m-1 of the colamt layout above.
//   absa [585, 8], absb [235, 8], selfref [10, 8], forref [3, 8]  float32, Fortran index order.
//
// Translation notes:
//   - The Fortran WHERE blocks are per-column masks. Pass 1 (Fortran k = 1..MAXVAL(laytrop), mask
//     k <= laytrop) is k1 <= laytrop(column). Pass 2 (Fortran k = kmin+1..NLAY, mask k >= laytrop) is
//     k1 >= laytrop(column) and k1 >= kmin + 1. The two passes run in that order per layer, so the
//     pass-2 value wins at k1 == laytrop when laytrop > kmin.
//   - taur is written for every layer and g-point (Fortran loop "do k = 1, nlay").
//   - MIN/MAX on integer indices are the Fortran MIN/MAX; ind01..ind14 are 1-based and read at ind-1.
//   - MOD(specmult, 1.0) is fmodf (specmult >= 0). INT() truncates toward zero, as the C cast does.
//   - Single precision literals (f suffix), as kfpt = single; fmad is on (NVRTC option).
//   - Fortran MIN(oneminus, x) with x NaN (speccomb = 0) is not reproduced bit-for-bit: fminf returns
//     the non-NaN argument. speccomb is colamt(H2O) + strrat*colamt(CO2) and is > 0 for every layer
//     the setcoef writes (the H2O vmr is clamped positive upstream).
//   - No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define TAUMOL19_NG      8      // NG19 (radsw_param.f)
#define TAUMOL19_NS      26     // NS19 = NS18 + NG18 = 18 + 8 (radsw_param.f)
#define TAUMOL19_CHK     8      // CHNK_RRTM (configure.nems -DCHNK_RRTM=8); Fortran MINVAL group size
#define TAUMOL19_MSA     585    // MSA19 (module_radsw_kgb19_nmmb)
#define TAUMOL19_MSB     235    // MSB19
#define TAUMOL19_RAYL    2.29e-09f
#define TAUMOL19_STRRAT  5.49281e+0f
#define TAUMOL19_ONEMIN  0.999999f

extern "C" __global__ void rrtm_sw_taumol19(
    const float* __restrict__ colamt,
    const float* __restrict__ colmol,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const int* __restrict__ jp,
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ laytrop,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const int* __restrict__ ndayp,     // device SW_NDAY (chunk path); read only when use_gate != 0
    const int use_gate,
    const int nday_host,               // live column count when use_gate == 0
    const int cstart,                  // day index of launch column 0 (multiple of TAUMOL19_CHK)
    const int ncols,                   // columns in this launch
    const int nlay,
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

    // Fortran MINVAL(laytrop(1:CHK)) over this column's chunk of CHK day columns (live entries only).
    const int lt = laytrop[t];
    const int g0 = (t / TAUMOL19_CHK) * TAUMOL19_CHK;
    const int g1 = min(g0 + TAUMOL19_CHK, ncols);
    int kmin = lt;
    for (int s = g0; s < g1; ++s) {
        if (cstart + s < live) kmin = min(kmin, laytrop[s]);
    }

    for (int k1 = 1; k1 <= nlay; ++k1) {
        const int l = k1 - 1;
        const size_t lij = (size_t)t * (size_t)lc + (size_t)l * (size_t)lk;
        const size_t cA  = (size_t)t * (size_t)ca_c + (size_t)l * (size_t)ca_k;
        const size_t cT  = (size_t)t * (size_t)tg_c + (size_t)l * (size_t)tg_k;

        // Rayleigh optical depth, every g-point of band 19 (Fortran loop "do k = 1, nlay").
        const float tauray = colmol[lij] * TAUMOL19_RAYL;
        for (int j = 0; j < TAUMOL19_NG; ++j) {
            taur[cT + (size_t)(TAUMOL19_NS + j) * (size_t)tg_g] = tauray;
        }

        // ---- Pass 1 (Fortran: do k = 1, MAXVAL(laytrop); WHERE k .LE. laytrop) ----
        if (k1 <= lt) {
            const float c_h2o = colamt[cA + 0 * (size_t)ca_m];
            const float c_co2 = colamt[cA + 1 * (size_t)ca_m];
            const float speccomb = c_h2o + TAUMOL19_STRRAT * c_co2;
            const float specmult = 8.0f * fminf(TAUMOL19_ONEMIN, c_h2o / speccomb);

            const int js = 1 + (int)specmult;
            const float fs = fmodf(specmult, 1.0f);
            const float fs1 = 1.0f - fs;

            const float fac000 = fs1 * fac00[lij];
            const float fac010 = fs1 * fac10[lij];
            const float fac100 = fs * fac00[lij];
            const float fac110 = fs * fac10[lij];
            const float fac001 = fs1 * fac01[lij];
            const float fac011 = fs1 * fac11[lij];
            const float fac101 = fs * fac01[lij];
            const float fac111 = fs * fac11[lij];

            int id0, id1;
            rrtm_sw_taumol_index(19, k1, lt, jp[lij], jt[lij], jt1[lij], &id0, &id1);

            const int ind01 = max(1, min(TAUMOL19_MSA, id0 + js));
            const int ind02 = min(TAUMOL19_MSA, ind01 + 1);
            const int ind03 = min(TAUMOL19_MSA, ind01 + 9);
            const int ind04 = min(TAUMOL19_MSA, ind01 + 10);
            const int ind11 = max(1, min(TAUMOL19_MSA, id1 + js));
            const int ind12 = min(TAUMOL19_MSA, ind11 + 1);
            const int ind13 = min(TAUMOL19_MSA, ind11 + 9);
            const int ind14 = min(TAUMOL19_MSA, ind11 + 10);

            const int inds = indself[lij];
            const int indf = indfor[lij];
            const int indsp = inds + 1;
            const int indfp = indf + 1;

            for (int j = 0; j < TAUMOL19_NG; ++j) {
                const float a_sum =
                      fac000 * absa[(size_t)(ind01 - 1) * TAUMOL19_NG + j]
                    + fac100 * absa[(size_t)(ind02 - 1) * TAUMOL19_NG + j]
                    + fac010 * absa[(size_t)(ind03 - 1) * TAUMOL19_NG + j]
                    + fac110 * absa[(size_t)(ind04 - 1) * TAUMOL19_NG + j]
                    + fac001 * absa[(size_t)(ind11 - 1) * TAUMOL19_NG + j]
                    + fac101 * absa[(size_t)(ind12 - 1) * TAUMOL19_NG + j]
                    + fac011 * absa[(size_t)(ind13 - 1) * TAUMOL19_NG + j]
                    + fac111 * absa[(size_t)(ind14 - 1) * TAUMOL19_NG + j];

                const float self_ref = selfref[(size_t)(inds - 1) * TAUMOL19_NG + j];
                const float self_refp = selfref[(size_t)(indsp - 1) * TAUMOL19_NG + j];
                const float for_ref = forref[(size_t)(indf - 1) * TAUMOL19_NG + j];
                const float for_refp = forref[(size_t)(indfp - 1) * TAUMOL19_NG + j];

                taug[cT + (size_t)(TAUMOL19_NS + j) * (size_t)tg_g] =
                      speccomb * a_sum
                    + c_h2o * (selffac[lij] * (self_ref + selffrac[lij] * (self_refp - self_ref))
                             + forfac[lij] * (for_ref + forfrac[lij] * (for_refp - for_ref)));
            }
        }

        // ---- Pass 2 (Fortran: do k = MINVAL(laytrop(1:CHK))+1, NLAY; WHERE k .GE. laytrop) ----
        if (k1 >= lt && k1 >= kmin + 1) {
            int id0, id1;
            rrtm_sw_taumol_index(19, k1, lt, jp[lij], jt[lij], jt1[lij], &id0, &id1);

            const int ind01 = max(1, min(TAUMOL19_MSB, id0 + 1));
            const int ind02 = min(TAUMOL19_MSB, ind01 + 1);
            const int ind11 = max(1, min(TAUMOL19_MSB, id1 + 1));
            const int ind12 = min(TAUMOL19_MSB, ind11 + 1);

            const float c_co2 = colamt[cA + 1 * (size_t)ca_m];
            for (int j = 0; j < TAUMOL19_NG; ++j) {
                taug[cT + (size_t)(TAUMOL19_NS + j) * (size_t)tg_g] =
                    c_co2 * (fac00[lij] * absb[(size_t)(ind01 - 1) * TAUMOL19_NG + j]
                           + fac10[lij] * absb[(size_t)(ind02 - 1) * TAUMOL19_NG + j]
                           + fac01[lij] * absb[(size_t)(ind11 - 1) * TAUMOL19_NG + j]
                           + fac11[lij] * absb[(size_t)(ind12 - 1) * TAUMOL19_NG + j]);
            }
        }
    }
}
