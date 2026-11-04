// phys_rrtm_sw_taumol28.cu -- TAUMOL28: SW band 28 (38000-50000 cm-1, O3 and O2) gaseous and Rayleigh
// optical depths for the NMMB RRTM shortwave (RRTM group). One thread per (i, j) column; the layer loop
// k = 1..nlay runs inside the thread, the g-point loop j = 1..NG28 (NG28 = 6) runs inside the layer loop,
// and every term of the Fortran routine is present. Band 28 has no self- or foreign-continuum term.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes NCEP modifications of the rrtm-sw radiation code from AER inc.     !
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4559-4678 (SUBROUTINE taumol28, an internal
// procedure of taumol, which USEs module_radsw_kgb28). The k-tables are module_radsw_kgb28_nmmb
// (phys/radsw_datatb_nmmb.f:17754-20060), extracted by tools/extract_rrtm_sw_kgb28.py into
// namgpu/data/rrtm_sw/kgb28.npz.
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]). Integer index
// values (indices into the k-tables, laytrop, kmin) are the Fortran values (1-based):
//   colamt        [MAXGAS=6, nlay, JM, IM]  (Fortran colamt(k,m); m = 1..6 -> plane m-1; O3 = m 3, O2 = m 6)
//   colmol, fac00, fac01, fac10, fac11                                         [nlay, JM, IM]
//   jp, jt, jt1                                                                 [nlay, JM, IM] int32
//   id0, id1 (band 28) are evaluated per layer by rrtm_sw_taumol_index(28, ...) (rrtm_sw_bands.cuh),
//   the Fortran id0(:,k,28) / id1(:,k,28) of the dispatcher (radsw_main_nmmb.f:3137-3158)
//   laytrop                                   [JM, IM] int32
//   kmin          [1] int32 = MINVAL(laytrop(1:CHK)) over the whole chunk, on the device
//   absa (Fortran absa(585,NG28))   [NG28=6, 585]   absa[g*585 + (ind-1)]
//   absb (Fortran absb(1175,NG28))  [NG28, 1175]    absb[g*1175 + (ind-1)]
//   taug, taur (Fortran taug(CHK,nlay,ngptsw)) [NG28 planes, nlay, JM, IM]: the caller passes the
//   view of planes NS28+1..NS28+NG28 (NS28 = 94, radsw_param_nmmb.f:176), so this kernel writes
//   plane 0..5 of the view.
//
// Translation notes:
//   - The Fortran WHERE blocks are per-column masks. Loop 1 (k = 1..MAXVAL(laytrop)) with mask
//     k <= laytrop is k <= laytrop(column) here. Loop 2 (k = MINVAL(laytrop)+1..nlay) with mask
//     k >= laytrop is k >= kmin+1 and k >= laytrop. Both loops can write taug at k == laytrop(column)
//     when laytrop(column) > MINVAL; the second write (upper-atmosphere form) wins, so the two blocks
//     run in that order per layer.
//   - taur is written for every layer and every g-point (Fortran "do k = 1, nlay" that sets
//     tauray = colmol*rayl).
//   - Fortran MOD(a, f_one) for a >= 0 is fmodf(a, 1.0f). Fortran INT() truncates toward zero.
//   - Fortran MIN(a, oneminus) is fminf(a, oneminus); oneminus = 1.0 - eps, eps = 1.0e-6
//     (radsw_main_nmmb.f:292), i.e. SW_ONEMINUS from rrtm_sw_bands.cuh.
//   - Lower block uses specmult = 8.0 and the ind offsets +1, +9, +10 into absa (MSA28 bound);
//     upper block uses specmult = 4.0 and the offsets +1, +5, +6 into absb (MSB28 bound), as written.
//   - strrat(28) (radsw_datatb_nmmb.f:1108-1110, entry 13 of the 14-entry strrat(16:29) list) is
//     passed as strrat28 from the npz; rayl = 2.02e-05 (module_radsw_kgb28_nmmb parameter) too.
//
// Fast numerics: fmad on (NVRTC option), IEEE division, no host sync, no device allocation.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define TAU28_NG28   6
#define TAU28_MSA28  585
#define TAU28_MSB28  1175

extern "C" __global__ void rrtm_sw_taumol28(
    const float* __restrict__ colamt,
    const float* __restrict__ colmol,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const int*   __restrict__ laytrop,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const int*   __restrict__ kmin_p,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float  rayl,
    const float  strrat28,
    const int nlay, const int im, const int jm,
    float* __restrict__ taug,
    float* __restrict__ taur)
{
    COLUMN_IJ(im, jm)                         // i0, j0; returns early past the last column
    const int ij = IDX2(i0, j0, im);
    const int L = laytrop[ij];
    const int kmin = kmin_p[0];

    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t lij = IDX3(i0, j0, k0, im, jm);
        const int k = k0 + 1;                 // Fortran layer index
        int id0, id1;                         // band-28 table offsets, Fortran id0/id1(k,28)
        rrtm_sw_taumol_index(28, k, L, jp[lij], jt[lij], jt1[lij], &id0, &id1);

        // Rayleigh optical depth, every layer and g-point (taur(1:CHK,k,NS28+j) = colmol*rayl).
        const float tauray = colmol[lij] * rayl;
        for (int g = 0; g < TAU28_NG28; ++g) {
            taur[IDX4(i0, j0, k0, g, im, jm, nlay)] = tauray;
        }

        // Loop 1: lower atmosphere, k <= laytrop (O3 and O2 binary species, specmult = 8).
        if (k <= L) {
            const float c3 = colamt[IDX4(i0, j0, k0, 2, im, jm, nlay)];   // colamt(k,3) O3
            const float c6 = colamt[IDX4(i0, j0, k0, 5, im, jm, nlay)];   // colamt(k,6) O2
            const float speccomb = c3 + strrat28 * c6;
            const float specmult = 8.0f * fminf(SW_ONEMINUS, c3 / speccomb);

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

            const int ind01 = max(1, min(TAU28_MSA28, id0 + js));
            const int ind02 = min(TAU28_MSA28, ind01 + 1);
            const int ind03 = min(TAU28_MSA28, ind01 + 9);
            const int ind04 = min(TAU28_MSA28, ind01 + 10);
            const int ind11 = max(1, min(TAU28_MSA28, id1 + js));
            const int ind12 = min(TAU28_MSA28, ind11 + 1);
            const int ind13 = min(TAU28_MSA28, ind11 + 9);
            const int ind14 = min(TAU28_MSA28, ind11 + 10);

            for (int g = 0; g < TAU28_NG28; ++g) {
                const float* A = absa + g * TAU28_MSA28;
                taug[IDX4(i0, j0, k0, g, im, jm, nlay)] = speccomb
                    * (fac000 * A[ind01 - 1] + fac100 * A[ind02 - 1]
                     + fac010 * A[ind03 - 1] + fac110 * A[ind04 - 1]
                     + fac001 * A[ind11 - 1] + fac101 * A[ind12 - 1]
                     + fac011 * A[ind13 - 1] + fac111 * A[ind14 - 1]);
            }
        }

        // Loop 2: upper atmosphere, k >= MINVAL(laytrop)+1 and k >= laytrop (specmult = 4, absb).
        if (k >= kmin + 1 && k >= L) {
            const float c3 = colamt[IDX4(i0, j0, k0, 2, im, jm, nlay)];   // colamt(k,3) O3
            const float c6 = colamt[IDX4(i0, j0, k0, 5, im, jm, nlay)];   // colamt(k,6) O2
            const float speccomb = c3 + strrat28 * c6;
            const float specmult = 4.0f * fminf(SW_ONEMINUS, c3 / speccomb);

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

            const int ind01 = max(1, min(TAU28_MSB28, id0 + js));
            const int ind02 = min(TAU28_MSB28, ind01 + 1);
            const int ind03 = min(TAU28_MSB28, ind01 + 5);
            const int ind04 = min(TAU28_MSB28, ind01 + 6);
            const int ind11 = max(1, min(TAU28_MSB28, id1 + js));
            const int ind12 = min(TAU28_MSB28, ind11 + 1);
            const int ind13 = min(TAU28_MSB28, ind11 + 5);
            const int ind14 = min(TAU28_MSB28, ind11 + 6);

            for (int g = 0; g < TAU28_NG28; ++g) {
                const float* B = absb + g * TAU28_MSB28;
                taug[IDX4(i0, j0, k0, g, im, jm, nlay)] = speccomb
                    * (fac000 * B[ind01 - 1] + fac100 * B[ind02 - 1]
                     + fac010 * B[ind03 - 1] + fac110 * B[ind04 - 1]
                     + fac001 * B[ind11 - 1] + fac101 * B[ind12 - 1]
                     + fac011 * B[ind13 - 1] + fac111 * B[ind14 - 1]);
            }
        }
    }
}
