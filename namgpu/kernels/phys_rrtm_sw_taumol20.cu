// phys_rrtm_sw_taumol20.cu -- TAUMOL20: SW band 20 (5150-6150 cm-1) optical depths, Rayleigh
// optical depths and the band-20 top-of-atmosphere solar source (RRTM group, sw).
// One thread per column; the layer loop k = 1..nlay runs inside the thread.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep modifications of the rrtm-sw radiation code from aer inc.     !
//
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:3852-3943 (SUBROUTINE taumol20). The band-20
// solar source is the taumol driver case block: case (16, 20, 23, 25, 26, 29) at radsw_main_nmmb.f:3269,
// sfluxzen(:, ns+j) = sfluxref01(j, 1, ibd) at 3272-3273, with ns = ngs(20) = NS20 = 34
// (radsw_param_nmmb.f:140). The id0/id1 index setup is the driver WHERE blocks at 3250-3257.
// The k-tables are module_radsw_kgb20_nmmb (radsw_datatb_nmmb.f:10146-10910), extracted to
// namgpu/data/rrtm_sw/kgb20.npz.
//
// Index layout uses generic strides, the same convention as phys_rrtm_sw_taumol16.cu:
//   per (column t, layer k0):  lij = t*lc + k0*lk         for [nlay, JM, IM] or [nc, nlay] arrays
//                              cA  = t*ca_c + k0*ca_k     colamt; gas m at + m*ca_m
//                              cT  = t*tg_c + k0*tg_k     taug/taur; g-plane at + g*tg_g
//   The caller passes taug, taur and sfluxzen pointers already offset to plane NS20.
//   Tables, Fortran (m, ig) -> C [ig, m]: absa [NG20, 65], absb [NG20, 235], selfref [NG20, 10],
//   forref [NG20, 4], absch4 [NG20], sflux20 [NG20] (= sfluxref01(:, 1, ibx(20))).
//
// Translation notes:
//   - Band 20 has NSPA = NSPB = 1 (radsw_main_nmmb.f:308-309), so id0/id1 reduce to the lower
//     index (jp-1)*5 + (jt-1) and jp*5 + (jt1-1) for the lower atmosphere, and to
//     (jp-13)*5 + (jt-1), (jp-12)*5 + (jt1-1) for the upper atmosphere.
//   - Lower atmosphere (k <= laytrop): absa with self-continuum (selfref; indself in 1..9 from
//     setcoef, radsw_main_nmmb.f:2116) and foreign continuum (forref; indfor in 1..3), plus CH4.
//   - Upper atmosphere (k > laytrop): absb table with foreign continuum only, plus CH4.
//   - DECLARED DIVERGENCE, pending a decision by the project lead: the Fortran upper pass is
//     WHERE(k .GE. laytrop) over k = MINVAL(laytrop over the CHK chunk)+1 .. NLAY. It rewrites layer
//     k == laytrop(column) with the upper formula, indexed by the lower-branch id0/id1, whenever
//     laytrop(column) exceeds the chunk minimum. The result depends on CHK, which CONTRACT section 2
//     drops. This port keeps the per-column rule k > laytrop, the rule the scalar RRTM taumol20 states.
//     The Fortran is not reproduced as written.
//   - Single precision everywhere (kfpt = single; kind_taum = 4 in machine.f).
//   - Day gating: column t of this launch is day column cstart + t and is live when
//     cstart + t < live, where live = ndayp[0] (device SW_NDAY) if use_gate, else nday_host. Dead
//     columns return before writing. The Fortran NDAY..CHK padding is never consumed by the day-list
//     gather (radsw_main_nmmb.f:2324-2342 pads only columns the post-processing ignores).
//   - sfluxzen is written only when do_sflux != 0 (full-domain path). The taumol unit writes the
//     band-20 solar source for every column already, so the chunk path leaves it to taumol.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation.

#include "nmmb_common.cuh"

#define NG20       10
#define MSA20      65
#define MSB20      235
#define MSF20      10
#define MFR20      4
#define NS20       34          // radsw_param_nmmb.f:173 (NS20 = NS19 + NG19 = 26 + 8); used by callers
#define NGPTSW     112         // radsw_param_nmmb.f:150
#define NSPA20     1           // radsw_main_nmmb.f:308, nspa(20) = 1
#define NSPB20     1           // radsw_main_nmmb.f:309, nspb(20) = 1

extern "C" __global__ void rrtm_sw_taumol20(
    const float* __restrict__ colamt,     // gas m of column t, layer k0 at cA + m*ca_m
    const float* __restrict__ colmol,
    const int*   __restrict__ laytrop,    // per column
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
    const float* __restrict__ absa,       // [NG20, MSA20]
    const float* __restrict__ absb,       // [NG20, MSB20]
    const float* __restrict__ selfref,    // [NG20, MSF20]
    const float* __restrict__ forref,     // [NG20, MFR20]
    const float* __restrict__ absch4,     // [NG20]
    const float* __restrict__ sflux20,    // [NG20]
    const float  rayl,
    const int*   __restrict__ ndayp,      // device SW_NDAY; read only when use_gate != 0
    const int    use_gate,
    const int    nday_host,               // live count when use_gate == 0
    const int    cstart,                  // first day column of this launch
    const int    ncols,                   // columns in this launch
    const int    nlay,
    const long long lc, const long long lk,                            // per (column, layer)
    const long long ca_c, const long long ca_k, const long long ca_m,  // colamt
    const long long tg_c, const long long tg_k, const long long tg_g,  // taug, taur
    const int    do_sflux,
    const long long sf_c, const long long sf_g,                        // sfluxzen
    float* __restrict__ taug,             // base at plane NS20
    float* __restrict__ taur,             // base at plane NS20
    float* __restrict__ sfluxzen)         // base at plane NS20 (used when do_sflux)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols) return;
    const int live = use_gate ? ndayp[0] : nday_host;
    if (cstart + t >= live) return;

    const int lt = laytrop[t];

    if (do_sflux) {
        for (int g = 0; g < NG20; ++g) {
            sfluxzen[(size_t)t * sf_c + (size_t)g * sf_g] = sflux20[g];
        }
    }

    for (int k0 = 0; k0 < nlay; ++k0) {
        const int k = k0 + 1;                       // Fortran layer index (1 = model top)
        const size_t lij = (size_t)t * lc + (size_t)k0 * lk;
        const size_t cA  = (size_t)t * ca_c + (size_t)k0 * ca_k;
        const size_t cT  = (size_t)t * tg_c + (size_t)k0 * tg_k;

        // Rayleigh optical depth for every layer (taumol20 first loop).
        const float tauray = colmol[lij] * rayl;
        for (int g = 0; g < NG20; ++g) {
            taur[cT + (size_t)g * tg_g] = tauray;
        }

        const float c_h2o = colamt[cA + 0 * (size_t)ca_m];   // colamt(:,k,1)
        const float c_ch4 = colamt[cA + 4 * (size_t)ca_m];   // colamt(:,k,5)

        const int jpk  = jp [lij];
        const int jtk  = jt [lij];
        const int jt1k = jt1[lij];
        const float f00 = fac00[lij];
        const float f01 = fac01[lij];
        const float f10 = fac10[lij];
        const float f11 = fac11[lij];

        if (k <= lt) {
            // Lower atmosphere: absa table with self and foreign continuum.
            const int id0 = ((jpk - 1) * 5 + (jtk - 1)) * NSPA20;
            const int id1 = (jpk * 5 + (jt1k - 1)) * NSPA20;
            const int ind01 = max(1, min(MSA20, id0 + 1));
            const int ind02 = min(MSA20, ind01 + 1);
            const int ind11 = max(1, min(MSA20, id1 + 1));
            const int ind12 = min(MSA20, ind11 + 1);

            const int inds = indself[lij];
            const int indf = indfor [lij];
            const int indsp = inds + 1;
            const int indfp = indf + 1;
            const float sfac = selffac [lij];
            const float sfrac = selffrac[lij];
            const float ffac = forfac[lij];
            const float ffrac = forfrac[lij];

            for (int g = 0; g < NG20; ++g) {
                const float* ra = absa + g * MSA20;
                const float* rs = selfref + g * MSF20;
                const float* rf = forref + g * MFR20;
                const float absterm = f00 * ra[ind01 - 1] + f10 * ra[ind02 - 1]
                                    + f01 * ra[ind11 - 1] + f11 * ra[ind12 - 1];
                const float selfterm = rs[inds - 1] + sfrac * (rs[indsp - 1] - rs[inds - 1]);
                const float forterm = rf[indf - 1] + ffrac * (rf[indfp - 1] - rf[indf - 1]);
                taug[cT + (size_t)g * tg_g] =
                    c_h2o * ((absterm) + sfac * (selfterm) + ffac * (forterm))
                    + c_ch4 * absch4[g];
            }
        } else {
            // Upper atmosphere: absb table with foreign continuum only.
            const int id0 = ((jpk - 13) * 5 + (jtk - 1)) * NSPB20;
            const int id1 = ((jpk - 12) * 5 + (jt1k - 1)) * NSPB20;
            const int ind01 = max(1, min(MSB20, id0 + 1));
            const int ind02 = min(MSB20, ind01 + 1);
            const int ind11 = max(1, min(MSB20, id1 + 1));
            const int ind12 = min(MSB20, ind11 + 1);

            const int indf = indfor[lij];
            const int indfp = indf + 1;
            const float ffac = forfac [lij];
            const float ffrac = forfrac[lij];

            for (int g = 0; g < NG20; ++g) {
                const float* rb = absb + g * MSB20;
                const float* rf = forref + g * MFR20;
                const float absterm = f00 * rb[ind01 - 1] + f10 * rb[ind02 - 1]
                                    + f01 * rb[ind11 - 1] + f11 * rb[ind12 - 1];
                const float forterm = rf[indf - 1] + ffrac * (rf[indfp - 1] - rf[indf - 1]);
                taug[cT + (size_t)g * tg_g] =
                    c_h2o * ((absterm) + ffac * (forterm))
                    + c_ch4 * absch4[g];
            }
        }
    }
}
