// phys_rrtm_sw_setcoef.cu -- SETCOEF: per-layer interpolation coefficients for the NMMB RRTM
// shortwave (RRTM group). One thread per column; the layer loop k = 1..nlay runs inside the thread;
// every term of the Fortran routine is present.
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
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:1989-2139 (SUBROUTINE setcoef).
// The reference tables are module_radsw_ref preflog and tref (59 entries each), filled by
// phys_rrtm_sw_table_ref from radsw_datatb.f. The constants stpfac, f_one and f_zero are
// radsw_main_nmmb.f lines 294-299, written here as literals.
//
// Two entries share one per-layer body (setcoef_layers):
//   rrtm_sw_setcoef        full domain: pavel, tavel, h2ovmr [nlay, JM, IM]; one thread per (i, j).
//                          Index (k, j, i) -> ((k-1)*JM + j)*IM + i. laytrop [JM, IM].
//   rrtm_sw_setcoef_chunk  driver chunk: pavel, tavel, h2ovmr and every output [nc, nlay] (laytrop
//                          [nc]); one thread per day column t of the chunk. Column t is live only
//                          while cstart + t < *SW_NDAY, as in swrad_zero_cld. The Fortran caller
//                          calls setcoef once per day column j1 inside DO j1 = 1,NDAY
//                          (radsw_main_nmmb.f:1005); the driver chunk runs all its day columns in one
//                          launch.
//
// Units: pavel is the layer pressure in mb, as the Fortran caller passes plyr (radsw_main_nmmb.f:392
// and :523, "layer pressures (mb)"). tavel is in K, h2ovmr is the water vapour volume mixing ratio.
//
// Translation notes:
//   - laytrop: the Fortran sets laytrop = nlay first, then laytrop = k for every layer with
//     plog > 4.56, so the final value is the LAST (largest k) such layer, or nlay when none
//     qualifies. The kernel reproduces that rule.
//   - jp is clamped to 1..58 and jt/jt1 to 1..4 (Fortran max/min). The SW routine does NOT clamp
//     fp, ft or ft1 (the LW routine does), so none are clamped here.
//   - Fortran INT() truncates toward zero; the C (int) casts do the same for these ranges.
//   - Single precision literals (f suffix) everywhere, as kfpt = single.
//   - selffac = h2ovmr * forfac (the forfac of the same layer), as the Fortran has it.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation.

#include "nmmb_common.cuh"

#define STPFAC_SW   (296.0f / 1013.0f)   // radsw_main_nmmb.f:294
#define F_ONE_SW    1.0f                  // radsw_main_nmmb.f:299
#define F_ZERO_SW   0.0f                  // radsw_main_nmmb.f:298

struct SetcoefOut {
    int* jp; int* jt; int* jt1; int* indself; int* indfor;
    float* fac00; float* fac01; float* fac10; float* fac11;
    float* selffac; float* selffrac; float* forfac; float* forfrac;
};

// One column. Layer k (1-based) of this column lives at base + (k-1)*stride in every per-layer array.
// Returns laytrop for the column.
__device__ __forceinline__ int setcoef_layers(
    const float* __restrict__ pavel, const float* __restrict__ tavel, const float* __restrict__ h2ovmr,
    const float* __restrict__ preflog, const float* __restrict__ tref,
    int nlay, size_t base, size_t stride, const SetcoefOut& o)
{
    // Fortran: laytrop = nlay on entry; overwritten with k for each layer with plog > 4.56.
    int laytrop_c = nlay;

    for (int k = 1; k <= nlay; ++k) {
        const size_t lk = base + (size_t)(k - 1) * stride;
        const float pav = pavel[lk];
        const float tav = tavel[lk];
        const float h2o = h2ovmr[lk];

        // forfac(k) = pavel(k)*stpfac / (tavel(k)*(f_one + h2ovmr(k)))
        const float forf = pav * STPFAC_SW / (tav * (F_ONE_SW + h2o));
        o.forfac[lk] = forf;

        // Reference pressures on either side of the layer pressure: jp and jp1 = jp+1.
        const float plog = logf(pav);
        const int jpk = max(1, min(58, (int)(36.0f - 5.0f * (plog + 0.04f))));
        const int jp1 = jpk + 1;
        // fp = 5 * (preflog(jp) - plog): fraction of the ln(p) interval between jp and jp1.
        const float fp = 5.0f * (preflog[jpk - 1] - plog);

        // Reference temperatures nearest the layer temperature (jt for jp, jt1 for jp1), and ft, ft1.
        const float tem1 = (tav - tref[jpk - 1]) / 15.0f;
        const float tem2 = (tav - tref[jp1 - 1]) / 15.0f;
        const int jtk  = max(1, min(4, (int)(3.0f + tem1)));
        const int jt1k = max(1, min(4, (int)(3.0f + tem2)));
        const float ft  = tem1 - (float)(jtk - 3);
        const float ft1 = tem2 - (float)(jt1k - 3);

        o.jp[lk]  = jpk;
        o.jt[lk]  = jtk;
        o.jt1[lk] = jt1k;

        // Interpolation factors for the four reference points (fp1 = 1 - fp).
        const float fp1 = F_ONE_SW - fp;
        o.fac10[lk] = fp1 * ft;
        o.fac00[lk] = fp1 * (F_ONE_SW - ft);
        o.fac11[lk] = fp * ft1;
        o.fac01[lk] = fp * (F_ONE_SW - ft1);

        if (plog > 4.56f) {
            // Lower atmosphere: water vapour foreign and self continuum.
            laytrop_c = k;

            const float tem1f = (332.0f - tav) / 36.0f;
            const int indf = min(2, max(1, (int)tem1f));
            o.indfor[lk] = indf;
            o.forfrac[lk] = tem1f - (float)indf;

            const float tem2s = (tav - 188.0f) / 7.2f;
            const int inds = min(9, max(1, (int)tem2s - 7));
            o.indself[lk] = inds;
            o.selffrac[lk] = tem2s - (float)(inds + 7);
            o.selffac[lk] = h2o * forf;
        } else {
            // Upper atmosphere: no water vapour self continuum.
            const float tem1f = (tav - 188.0f) / 36.0f;
            o.indfor[lk] = 3;
            o.forfrac[lk] = tem1f - F_ONE_SW;

            o.indself[lk] = 0;
            o.selffrac[lk] = F_ZERO_SW;
            o.selffac[lk] = F_ZERO_SW;
        }
    }
    return laytrop_c;
}

// Full domain: one thread per (i0, j0) column.
extern "C" __global__ void rrtm_sw_setcoef(
    const float* __restrict__ pavel,
    const float* __restrict__ tavel,
    const float* __restrict__ h2ovmr,
    const float* __restrict__ preflog,
    const float* __restrict__ tref,
    int nlay, int IM, int JM,
    int* __restrict__ laytrop,
    int* __restrict__ jp,
    int* __restrict__ jt,
    int* __restrict__ jt1,
    float* __restrict__ fac00,
    float* __restrict__ fac01,
    float* __restrict__ fac10,
    float* __restrict__ fac11,
    float* __restrict__ selffac,
    float* __restrict__ selffrac,
    int* __restrict__ indself,
    float* __restrict__ forfac,
    float* __restrict__ forfrac,
    int* __restrict__ indfor)
{
    COLUMN_IJ(IM, JM)
    const SetcoefOut o = {jp, jt, jt1, indself, indfor, fac00, fac01, fac10, fac11,
                          selffac, selffrac, forfac, forfrac};
    const size_t col = (size_t)j0 * IM + i0;
    const int lt = setcoef_layers(pavel, tavel, h2ovmr, preflog, tref, nlay,
                                  col, (size_t)JM * IM, o);
    laytrop[IDX2(i0, j0, IM)] = lt;
}

// Driver chunk: chunk column t is day column cstart + t. Arrays are [nc, nlay] (t, k) and laytrop [nc].
extern "C" __global__ void rrtm_sw_setcoef_chunk(
    const float* __restrict__ pavel,
    const float* __restrict__ tavel,
    const float* __restrict__ h2ovmr,
    const float* __restrict__ preflog,
    const float* __restrict__ tref,
    const int* __restrict__ ndayp, int cstart, int nc, int nlay,
    int* __restrict__ laytrop,
    int* __restrict__ jp,
    int* __restrict__ jt,
    int* __restrict__ jt1,
    float* __restrict__ fac00,
    float* __restrict__ fac01,
    float* __restrict__ fac10,
    float* __restrict__ fac11,
    float* __restrict__ selffac,
    float* __restrict__ selffrac,
    int* __restrict__ indself,
    float* __restrict__ forfac,
    float* __restrict__ forfrac,
    int* __restrict__ indfor)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nc || cstart + t >= *ndayp) return;
    const SetcoefOut o = {jp, jt, jt1, indself, indfor, fac00, fac01, fac10, fac11,
                          selffac, selffrac, forfac, forfrac};
    laytrop[t] = setcoef_layers(pavel, tavel, h2ovmr, preflog, tref, nlay,
                                (size_t)t * nlay, 1, o);
}
