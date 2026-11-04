// phys_rrtm_sw_swflux.cu -- swflux: upward and downward shortwave fluxes at layer interfaces.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NAM source file phys/radsw_main_nmmb.f (NCEP/EMC, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file).
// The routine header (module_radsw_main_nmmb, swflux) is kept as the notice:
//
//   subroutine swflux (zrefb,zrefd,ztrab,ztrad,zldbt,ztdbt,
//                      NLAY, NLP1, nday,lmask,
//                      zfu, zfd)
//   purpose:  computes the upward and downward radiation fluxes
//   interface: "swflux" is called by "spcvrc" and "spcvrm"
//   NCEP sw-rrtm package (rrtm-sw from AER, NCEP modifications, NMMB version).
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:2977-3102 (nam.v4.2.11, 83df4e13c1).
//
// Level order: the Fortran index k = 1 is the lowest interface (the surface side) and k = NLP1 the top;
// the arrays are dimensioned (CHK, nlp1) and every k is kept exactly as in the Fortran. Kernel-local
// arrays below use the same 1-based Fortran index (size SWF_MAXL+2).
//
// Column mapping: Fortran i = 1..CHK (the column chunk) becomes one thread per (i, j) column of the
// full grid, one launch (grid_cols). Each thread runs the three k loops of the routine in the Fortran
// order, with every statement and every term kept:
//   zrupb(1), zrupd(1) = zrefb(1), zrefd(1)                 -> rup init
//   k = 1..NLAY:  zrupb(k+1), zrupd(k+1)                      -> bottom-to-top pass
//   ztdn(NLP1)=1, zrdnd(NLP1)=0, ztdn(NLAY)=ztrab(NLP1), zrdnd(NLAY)=zrefd(NLP1)  -> upper BC
//   k = NLAY..2:  ztdn(k-1), zrdnd(k-1)                       -> top-to-bottom pass
//   k = 1..NLP1:  zfu(k), zfd(k)                              -> fluxes at levels
// Float expressions are written left to right exactly as the Fortran, so FMA contraction is the only
// rounding freedom (fast numerics, CONTRACT section 3).
//
// Fortran -> CUDA map, statement by statement:
//   f_one / f_zero (radsw_main_nmmb.f:298-299, kind_phys parameters) -> 1.0f / 0.0f
//   CHK (radsw_main_nmmb.f:351, CHNK_RRTM)              -> the full-grid column loop (CHK chunking dropped)
//   lmask                                               -> not read: the Fortran routine never references it
//   nday                                                -> not read: it appears only under RUNTIME_CHECKING
//
// Precision: float (kfpt = single, CONTRACT). The Fortran kind_phys is 8 unless SINGLE_PREC is defined
// (phys/machine.f); this is a declared divergence from the default NAM build, see the wrapper.
//
// Limits: NLP1 <= SWF_MAXL (64). The NAM 12 km parent has LM = 60, LTP = 1, so NLAY = 61, NLP1 = 62.
// Each thread keeps its four column recurrences in local memory; no shared memory, no atomics, no sync.

#include "nmmb_common.cuh"

#define SWF_MAXL 64

// Fortran 1-based level index k of a [NLP1, JM, IM] array (level first, then j, then i).
#define SWF_AT(arr, k) arr[IDX3(i0, j0, (k) - 1, IM, JM)]

extern "C" __global__ void phys_rrtm_sw_swflux(
    const int IM, const int JM, const int NLAY, const int NLP1,
    const float* __restrict__ zrefb, const float* __restrict__ zrefd,
    const float* __restrict__ ztrab, const float* __restrict__ ztrad,
    const float* __restrict__ zldbt, const float* __restrict__ ztdbt,
    float* __restrict__ zfu, float* __restrict__ zfd)
{
    COLUMN_IJ(IM, JM)

    const float f_zero = 0.0f;
    const float f_one  = 1.0f;

    float zrupb[SWF_MAXL + 2];
    float zrupd[SWF_MAXL + 2];
    float zrdnd[SWF_MAXL + 2];
    float ztdn [SWF_MAXL + 2];
    float zden1;
    int k, kp;

    // --- link lowest layer with surface ---
    zrupb[1] = SWF_AT(zrefb, 1);        // direct beam
    zrupd[1] = SWF_AT(zrefd, 1);        // diffused

    // --- pass from bottom to top ---
    for (k = 1; k <= NLAY; ++k) {
        kp = k + 1;
        zden1 = f_one / (f_one - zrupd[k] * SWF_AT(zrefd, kp));
        zrupb[kp] = SWF_AT(zrefb, kp) + (SWF_AT(ztrad, kp) *
                    ((SWF_AT(ztrab, kp) - SWF_AT(zldbt, kp)) * zrupd[k] +
                     SWF_AT(zldbt, kp) * zrupb[k])) * zden1;
        zrupd[kp] = SWF_AT(zrefd, kp) + SWF_AT(ztrad, kp) * SWF_AT(ztrad, kp) * zrupd[k] * zden1;
    }

    // --- upper boundary conditions ---
    ztdn [NLP1] = f_one;
    zrdnd[NLP1] = f_zero;
    ztdn [NLAY] = SWF_AT(ztrab, NLP1);
    zrdnd[NLAY] = SWF_AT(zrefd, NLP1);

    // --- pass from top to bottom ---
    for (k = NLAY; k >= 2; --k) {
        zden1 = f_one / (f_one - SWF_AT(zrefd, k) * zrdnd[k]);
        ztdn[k - 1] = SWF_AT(ztdbt, k) * SWF_AT(ztrab, k) + (SWF_AT(ztrad, k) *
                      ((ztdn[k] - SWF_AT(ztdbt, k)) + SWF_AT(ztdbt, k) *
                       SWF_AT(zrefb, k) * zrdnd[k])) * zden1;
        zrdnd[k - 1] = SWF_AT(zrefd, k) + SWF_AT(ztrad, k) * SWF_AT(ztrad, k) * zrdnd[k] * zden1;
    }

    // --- up- and down-welling fluxes at levels ---
    for (k = 1; k <= NLP1; ++k) {
        zden1 = f_one / (f_one - zrdnd[k] * zrupd[k]);
        SWF_AT(zfu, k) = (SWF_AT(ztdbt, k) * zrupb[k] +
                          (ztdn[k] - SWF_AT(ztdbt, k)) * zrupd[k]) * zden1;
        SWF_AT(zfd, k) = SWF_AT(ztdbt, k) + (ztdn[k] - SWF_AT(ztdbt, k) +
                         SWF_AT(ztdbt, k) * zrupb[k] * zrdnd[k]) * zden1;
    }
}
