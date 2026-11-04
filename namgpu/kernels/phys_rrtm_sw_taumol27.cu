// phys_rrtm_sw_taumol27.cu -- TAUMOL27: SW band 27 (29000-38000 cm-1, low and high ozone) g-point
// optical depths and Rayleigh optical depths of the NMMB RRTM shortwave (RRTM group). One thread per
// (i, j) column; the layer loop k = 1..nlay runs inside the thread; every term of the Fortran routine
// is present.
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
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:4486-4554 (SUBROUTINE taumol27).
// Tables: module_radsw_kgb27_nmmb (phys/radsw_datatb_nmmb.f:17159-17749), extracted by
// tools/extract-rrtm-sw-kgb27.py into namgpu/data/rrtm_sw/kgb27.npz and passed in by the wrapper:
//   absa [MSA27=65, NG27=8], absb [MSB27=235, NG27=8], rayl [NG27=8] (module rayl, radsw_datatb_nmmb.f:17189).
// Constants written as literals: NG27 = 8, NS27 = 86 (radsw_param_nmmb.f:160-176: NS27 = NS22 + NG22 + NG23
// + NG24 + NG25 + NG26 with NS22 = 54), MSA27 = 65, MSB27 = 235 (radsw_datatb_nmmb.f:17162-17163),
// nspa(27) = 1 and nspb(27) = 1 (radsw_main_nmmb.f:308-309, band-27 slot of the 16..29 list).
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   colmol, fac00, fac01, fac10, fac11                                          [nlay, JM, IM]
//   jp, jt, jt1                                                                 [nlay, JM, IM] int32
//   colamt                                                                      [MAXGAS=6, nlay, JM, IM]
//                                                                               (plane 2 used: o3, Fortran colamt(:,k,3))
//   laytrop                                                                     [JM, IM] int32
//   absa [MSA27, NG27], absb [MSB27, NG27]  (g index fastest; Fortran absa(ind, ig))
//   rayl [NG27]
//   taug, taur                                                                  [NOUT, nlay, JM, IM];
//                                                                               the NG27 planes this routine
//                                                                               writes are plane0 .. plane0+7
//                                                                               (plane0 = 86 for the full
//                                                                               112-g-point array, 0 for an
//                                                                               NG27-plane array).
//
// Translation notes:
//   - The Fortran driver (taumol, radsw_main_nmmb.f:3251-3257) computes id0/id1 with the lower formula
//     where k <= laytrop and the upper formula where k > laytrop, with nspa(27) = nspb(27) = 1. This
//     kernel recomputes them per layer with the same rule.
//   - taumol27 runs two loops over the chunk. The lower loop writes k <= laytrop (WHERE k .LE. laytrop);
//     the upper loop, DO k = MINVAL(laytrop)+1, NLAY with WHERE k .GE. laytrop, overwrites k == laytrop.
//     The kernel therefore uses the absa expression for k < laytrop and the absb expression for k >= laytrop,
//     per column. The one case that differs is a column whose laytrop equals the minimum laytrop of its
//     CHK chunk: the upper loop does not reach k == laytrop there, so the lower value would stay. That
//     depends on chunk composition (a CPU chunking detail), so it is not reproduced.
//   - taur (taur = colmol * rayl(j)) is written for every layer k = 1..nlay, as in the Fortran.
//   - Band 27 has no continuum and no species mixing: taug = colamt(k,3) * (interpolation of absa or absb
//     in ln(p) and T), with ind02 = ind01 + 1 and ind12 = ind11 + 1 (clamped to the table size).
//   - No solar source is computed here; the band-27 solar flux is set by the SW driver (taumol case
//     default), outside this routine.
//   - Fortran MIN/MAX are integer min/max as written; single precision everywhere (kind_taum = 4 in
//     machine.f), fmad on, no host sync, no allocation.

#include "nmmb_common.cuh"

#define NG27_SW     8
#define NS27_SW     86               // first 0-based g-point of band 27 in the 112-point array
#define MSA27_SW    65
#define MSB27_SW    235
#define NSPA27_SW   1
#define NSPB27_SW   1

extern "C" __global__ void rrtm_sw_taumol27(
    const float* __restrict__ colmol,
    const float* __restrict__ colamt,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const int*   __restrict__ laytrop,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ rayl,
    int nlay, int IM, int JM, int plane0,
    float* __restrict__ taug,
    float* __restrict__ taur)
{
    COLUMN_IJ(IM, JM)

    const int lt = laytrop[IDX2(i0, j0, IM)];

    for (int l0 = 0; l0 < nlay; ++l0) {
        const size_t c = IDX3(i0, j0, l0, IM, JM);      // [l, j, i] offset of this layer
        const int k1 = l0 + 1;                           // Fortran k
        const float colo3 = colamt[IDX4(i0, j0, l0, 2, IM, JM, nlay)];   // colamt(:,k,3), o3

        // taur: taur(:, k, NS27+j) = colmol(k) * rayl(j) for every layer (k = 1..nlay).
        const float colm = colmol[c];
        for (int jg = 0; jg < NG27_SW; ++jg)
            taur[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = colm * rayl[jg];

        // Driver id0/id1 (radsw_main_nmmb.f:3251-3257): lower formula where k <= laytrop, upper
        // formula where k > laytrop.
        const int jpv = jp[c], jtv = jt[c], jt1v = jt1[c];
        int id0, id1;
        if (k1 <= lt) {
            id0 = ((jpv - 1) * 5 + (jtv - 1)) * NSPA27_SW;
            id1 = (jpv * 5 + (jt1v - 1)) * NSPA27_SW;
        } else {
            id0 = ((jpv - 13) * 5 + (jtv - 1)) * NSPB27_SW;
            id1 = ((jpv - 12) * 5 + (jt1v - 1)) * NSPB27_SW;
        }

        const float f00 = fac00[c], f01 = fac01[c], f10 = fac10[c], f11 = fac11[c];

        if (k1 < lt) {
            // ---- lower loop (k < laytrop): absa table
            const int ind01 = max(1, min(MSA27_SW, id0 + 1));
            const int ind02 = min(MSA27_SW, ind01 + 1);
            const int ind11 = max(1, min(MSA27_SW, id1 + 1));
            const int ind12 = min(MSA27_SW, ind11 + 1);

            for (int jg = 0; jg < NG27_SW; ++jg) {
                const float a01 = absa[(size_t)(ind01 - 1) * NG27_SW + jg];
                const float a02 = absa[(size_t)(ind02 - 1) * NG27_SW + jg];
                const float a11 = absa[(size_t)(ind11 - 1) * NG27_SW + jg];
                const float a12 = absa[(size_t)(ind12 - 1) * NG27_SW + jg];
                const float tg = colo3 * (f00 * a01 + f10 * a02 + f01 * a11 + f11 * a12);
                taug[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = tg;
            }
        }

        if (k1 >= lt) {
            // ---- upper loop (k >= laytrop): absb table (overwrites k == laytrop, as the Fortran does)
            const int ind01 = max(1, min(MSB27_SW, id0 + 1));
            const int ind02 = min(MSB27_SW, ind01 + 1);
            const int ind11 = max(1, min(MSB27_SW, id1 + 1));
            const int ind12 = min(MSB27_SW, ind11 + 1);

            for (int jg = 0; jg < NG27_SW; ++jg) {
                const float b01 = absb[(size_t)(ind01 - 1) * NG27_SW + jg];
                const float b02 = absb[(size_t)(ind02 - 1) * NG27_SW + jg];
                const float b11 = absb[(size_t)(ind11 - 1) * NG27_SW + jg];
                const float b12 = absb[(size_t)(ind12 - 1) * NG27_SW + jg];
                const float tg = colo3 * (f00 * b01 + f10 * b02 + f01 * b11 + f11 * b12);
                taug[IDX4(i0, j0, l0, plane0 + jg, IM, JM, nlay)] = tg;
            }
        }
    }
}
