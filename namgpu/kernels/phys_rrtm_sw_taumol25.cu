// phys_rrtm_sw_taumol25.cu -- TAUMOL25: SW band 25 (16000-22650 cm-1) gaseous and Rayleigh optical
// depths for the NMMB RRTM shortwave (RRTM group). One thread per (i, j) column; the layer loop
// k = 1..nlay runs inside the thread, the g-point loop j = 1..NG25 (NG25 = 6) runs inside the layer
// loop, and every term of the Fortran routine is present except the chunk-dependent overlap noted below.
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
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:4386-4445 (SUBROUTINE taumol25, an
// internal procedure of taumol, which USEs module_radsw_kgb25_nmmb). The k-tables are
// module_radsw_kgb25_nmmb (phys/radsw_datatb_nmmb.f:16972-17119), extracted by
// tools/extract_rrtm_sw_kgb25.py into namgpu/data/rrtm_sw/kgb25.npz.
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]). Integer index
// values (laytrop, jp, jt, jt1) are the Fortran values (1-based):
//   colamt         [MAXGAS=6, nlay, JM, IM]   (Fortran colamt(k,m); m = 1..6 -> plane m-1)
//   colmol, fac00, fac01, fac10, fac11       [nlay, JM, IM]
//   jp, jt, jt1                              [nlay, JM, IM] int32
//   laytrop                                  [JM, IM] int32
//   kmin           [1] int32 = MINVAL(laytrop) over the whole domain (wrapper), on the device
//   absa   (Fortran absa(65,NG25))  [NG25=6, 65]  absa[g*65 + (ind-1)]
//   abso3a, abso3b, rayl (Fortran (NG25))  [NG25]
//   taug, taur [NG25=6, nlay, JM, IM]: plane p (0-based) holds Fortran g-point NS25+1+p, i.e. the
//   Fortran planes 74..79 (0-based) of the 112-plane taug(CHK,nlay,ngptsw). Only this unit's six
//   planes are allocated and written; the other 106 planes belong to the other band units.
//
// Translation notes:
//   - The Fortran index id0(:,k,25) / id1(:,k,25) are the dispatcher's values (radsw_main_nmmb.f:
//     3251-3257) with nspa(25) = 1 and nspb(25) = 0. They are evaluated here by
//     rrtm_sw_taumol_index() from rrtm_sw_bands.cuh (same formula, per layer, no stored array).
//   - Fortran MAX(1, MIN(MSA25, id0 + 1)) and MIN(MSA25, ind01 + 1) are kept as written.
//   - Loop 1 (Fortran "do k = 1, MAXVAL(laytrop)", mask k <= laytrop) is k <= laytrop(column).
//   - Loop 2 (Fortran "do k = MINVAL(laytrop(1:CHK))+1, NLAY", mask k >= laytrop) is k >= kmin+1 and
//     k >= laytrop(column) here.
//     DECLARED DIVERGENCE (NAM-GPU, 2026-11-04 review): the Fortran MINVAL is over the native
//     CHNK_RRTM = 8 column tile, so the lower bound is per 8-column tile, not per domain. This kernel
//     uses one domain-wide kmin. Where a column's laytrop equals its tile minimum and exceeds the
//     domain minimum, the Fortran keeps the lower H2O form at k == laytrop and this kernel writes the
//     upper O3-only form. Where laytrop exceeds the tile minimum, both use the upper form, as in the
//     Fortran. Tile boundaries depend on the process and thread decomposition, so no fixed rule
//     reproduces them (the same finding as taumol16). Declared, not replicated; taumol16, 17, 20 and 21
//     declare the same divergence. Named decision for the project lead: replicate the tile minima (one segmented
//     min per 8-column group) if a bitwise taumol25 is ever wanted.
//   - taur is written for every layer and every g-point (Fortran "do k = 1, nlay; do j = 1, NG25" that
//     sets taur = colmol*rayl(j)).
//   - Upper-atmosphere taug uses only colamt(k,3) (O3) times abso3b(j); the lower-atmosphere taug is
//     colamt(k,1) (H2O) times the four-point absa interpolation plus colamt(k,3) times abso3a(j).
//     Band 25 has no binary-species (speccomb) term and no self/foreign continuum (nspa = 1,
//     nspb = 0 and no selfref/forref table in module_radsw_kgb25_nmmb).
//
// Fast numerics: fmad on (NVRTC option), IEEE division, no host sync, no device allocation.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define TAU25_NG25    6
#define TAU25_MSA25   65
#define TAU25_BAND    25      // Fortran band index jb; id0/id1 are for this band

extern "C" __global__ void rrtm_sw_taumol25(
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
    const float* __restrict__ abso3a,
    const float* __restrict__ abso3b,
    const float* __restrict__ rayl,
    const int nlay, const int im, const int jm,
    float* __restrict__ taug,      // [NG25, nlay, JM, IM]: output plane g = Fortran g-point NS25+1+g
    float* __restrict__ taur)      // [NG25, nlay, JM, IM]
{
    COLUMN_IJ(im, jm)                         // i0, j0; returns early past the last column
    const int ij = IDX2(i0, j0, im);
    const int L = laytrop[ij];
    const int kmin = kmin_p[0];

    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t lij = IDX3(i0, j0, k0, im, jm);
        const int k = k0 + 1;                 // Fortran layer index

        // Rayleigh optical depth, every layer and g-point (taur(1:CHK,k,NS25+j) = colmol*rayl(j)).
        const float cm = colmol[lij];
        for (int g = 0; g < TAU25_NG25; ++g) {
            taur[IDX4(i0, j0, k0, g, im, jm, nlay)] = cm * rayl[g];
        }

        // Loop 1: lower atmosphere, k <= laytrop (H2O absorption interpolated from absa, plus O3 at abso3a).
        if (k <= L) {
            const float c1 = colamt[IDX4(i0, j0, k0, 0, im, jm, nlay)];   // colamt(k,1) H2O
            const float c3 = colamt[IDX4(i0, j0, k0, 2, im, jm, nlay)];   // colamt(k,3) O3

            const float f00 = fac00[lij];
            const float f01 = fac01[lij];
            const float f10 = fac10[lij];
            const float f11 = fac11[lij];

            // id0(k,25), id1(k,25) of the dispatcher, lower-atmosphere branch (k <= laytrop).
            int id0, id1;
            rrtm_sw_taumol_index(TAU25_BAND, k, L, jp[lij], jt[lij], jt1[lij], &id0, &id1);
            const int ind01 = max(1, min(TAU25_MSA25, id0 + 1));
            const int ind02 = min(TAU25_MSA25, ind01 + 1);
            const int ind11 = max(1, min(TAU25_MSA25, id1 + 1));
            const int ind12 = min(TAU25_MSA25, ind11 + 1);

            for (int g = 0; g < TAU25_NG25; ++g) {
                const float* A = absa + g * TAU25_MSA25;
                taug[IDX4(i0, j0, k0, g, im, jm, nlay)] =
                    c1 * (f00 * A[ind01 - 1] + f10 * A[ind02 - 1]
                        + f01 * A[ind11 - 1] + f11 * A[ind12 - 1])
                  + c3 * abso3a[g];
            }
        }

        // Loop 2: upper atmosphere, k >= kmin+1 and k >= laytrop (O3 times abso3b). kmin is domain-wide
        // here; see the declared divergence in the header.
        if (k >= kmin + 1 && k >= L) {
            const float c3 = colamt[IDX4(i0, j0, k0, 2, im, jm, nlay)];   // colamt(k,3) O3
            for (int g = 0; g < TAU25_NG25; ++g) {
                taug[IDX4(i0, j0, k0, g, im, jm, nlay)] = c3 * abso3b[g];
            }
        }
    }
}
