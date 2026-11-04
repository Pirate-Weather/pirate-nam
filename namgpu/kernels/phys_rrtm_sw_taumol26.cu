// phys_rrtm_sw_taumol26.cu -- taumol26: SW band 26 (22650-29000 cm-1) optical depths and Rayleigh
// optical depths, NMMB RRTM shortwave (physics-rrtm group). One thread per (i, j) column; the layer
// loop k = 1..NLAY and the g-point loop j = 1..NG26 run inside the thread.
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
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4450-4481 (SUBROUTINE taumol26, nam.v4.2.11,
// 83df4e13c1). The routine USEs module_radsw_kgb26, whose only content is rayl(NG26): six Rayleigh
// coefficients, one per g-point (phys/radsw_datatb_nmmb.f:17124-17154), extracted by
// tools/extract-rrtm-sw-kgb26.py into namgpu/data/rrtm_sw/kgb26.npz.
//
// Fortran lines covered:
//   4457       use module_radsw_kgb26 (rayl); 4460 locals j, k
//   4470-4477  do k = 1, nlay; do j = 1, NG26:
//                4473 taug(1:CHK, k, NS26+j) = 0      (no gaseous absorption in band 26)
//                4475 taur(1:CHK, k, NS26+j) = colmol(1:CHK, k) * rayl(j)
//   NS26 = 80 (radsw_param_nmmb.f:175: NS25 + NG25 = 74 + 6), NG26 = 6 (radsw_param_nmmb.f:163).
//   Band 26 has no interpolation, no species mixing, no laytrop split: every layer is the same formula.
//   The Fortran has no further assignments (no tauaer, no ssa, no solar source in this band).
//
// Index conventions (CONTRACT.md sections 1 and 3, Fortran A(i,j,l) -> C [l, j, i]):
//   colmol          [NLAY, JM, IM]        (Fortran colmol(i,k) -> [k-1, j-1, i-1])
//   rayl            [NG26 = 6]            (Fortran rayl(j) -> rayl[j-1])
//   taug26          [NG26, NLAY, JM, IM]  band-local planes: plane g-1 = Fortran taug(:,:,NS26+g)
//   taur26          [NG26, NLAY, JM, IM]  plane g-1 = Fortran taur(:,:,NS26+g)
// The 112-plane taug and taur of the Fortran (21 GB each at 954 x 835 x 60) are not allocated; this band
// writes only its six planes, as the taumol22 unit does for its two. The spectral-sum unit maps g to
// NS26+g.
//
// Every term of the routine is present. No deviations. The taug write is a zero fill (the Fortran sets
// it to f_zero for every layer and g-point, f_zero = 0.0 at radsw_main_nmmb.f:298, module scope of
// radsw_main_nmmb); the taur write is colmol*rayl as written.
//
// Precision (declared divergence, review 2026-11-04). In the Fortran, taug and taur are dummies declared
// real(kind=kind_phys) (radsw_main_nmmb.f:3231 and the taumol26 call sites), rayl(NG26) is
// real(kind=kind_phys) (module_radsw_kgb26_nmmb, radsw_datatb_nmmb.f:17147), and kind_phys is set in
// module machine (gate0/nmmb/src/atmos/phys/machine.f): kind_phys = 8 unless SINGLE_PREC is defined. The
// NAM configure we hold (gate0/nmmb/src/conf/configure.nems.Wcoss_C.intel_ESMF_630r_gsm, CPPFLAGS line 106)
// has -DSINGLE_PREC commented out, so the build as configured runs taug, taur and rayl in 64-bit. The
// port computes and stores float32 (fast-numerics rule; see CONTRACT.md section 1b). The per-value error
// against the 64-bit product is one rounding of rayl and one of the product: relative <= 2^-23 (about
// 1.2e-7). Sign, zero structure and the g-to-plane layout are exact.
//
// Band-local layout (CONTRACT.md section 1b). The Fortran writes planes NS26+j of the 112-plane taug and
// taur (NS26 = 80, radsw_param_nmmb.f:175; NG26 = 6, radsw_param_nmmb.f:163). This unit writes planes
// g0 = 0..5 of its own [NG26, NLAY, JM, IM] arrays, so plane g0 is Fortran plane NS26 + g0 + 1.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation, no printf.

#include "nmmb_common.cuh"

#define NG26_SW   6

extern "C" __global__ void rrtm_sw_taumol26(
    const float* __restrict__ colmol,
    const float* __restrict__ rayl,
    const int nlay, const int im, const int jm,
    float* __restrict__ taug,
    float* __restrict__ taur)
{
    COLUMN_IJ(im, jm)                         // i0, j0; returns early past the last column

    // Fortran: do k = 1, nlay (k0 = k-1)
    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t lij = IDX3(i0, j0, k0, im, jm);
        const float cm = colmol[lij];         // colmol(1:CHK, k) for this column

        // Fortran: do j = 1, NG26 (g0 = j-1). Both writes land in the band's own planes.
        for (int g0 = 0; g0 < NG26_SW; ++g0) {
            taug[IDX4(i0, j0, k0, g0, im, jm, nlay)] = 0.0f;               // taug(...,NS26+j) = f_zero
            taur[IDX4(i0, j0, k0, g0, im, jm, nlay)] = cm * rayl[g0];     // taur(...,NS26+j) = colmol*rayl(j)
        }
    }
}
