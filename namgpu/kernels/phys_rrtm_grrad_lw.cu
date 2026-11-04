// phys_rrtm_grrad_lw.cu -- LW section of GRRAD_NMMB (RRTM group): the glue around setemis and
// lwrad. The LW radiative transfer itself is phys-rrtm-setemis and phys-rrtm-lw-lwrad; this file
// holds only the statements of grrad_nmmb that surround those calls.
//
// NOAA notice: phys/grrad_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, US government work
// (no licence file; see NOTICE). Header kept as the notice:
// !!!!!  ==========================================================  !!!!!
// !!!!!             module_radiation_driver descriptions             !!!!!
// !!!!!  ==========================================================  !!!!!
// !   this is the radiation driver module.  it prepares atmospheric      !
// !   profiles and invokes main radiation calculations.                  !
//
// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:1959-2075 (LW radiation block, under
// if (lslwr)), 2082-2169 (LW, cloud and cldcov accumulations under if (lssav)) and 2201
// (coszen copy, moved to post_flux). Constant CPATHFAC4LW = 1.5 is module_RA_RRTM.F90:212,
// passed in by the wrapper. Rows j = 1 and j = JM get no LW work (see the row guards).
//
// Kernels (one launch each, one thread per (i, j) column, every level in the column loop):
//   grrad_lw_pre        grrad_nmmb.f:1978-1990  np3d==3: save clouds(.,.,2)/(.,.,4) to dummys,
//                                               scale them by CPATHFAC4LW, before setemis/lwrad.
//   grrad_lw_post_heat  grrad_nmmb.f:2006-2013, 2046-2055, 2064-2071
//                                               htrlwb (if present), semis, tsflw, htrlw, restore
//                                               clouds from dummys.
//   grrad_lw_post_flux  grrad_nmmb.f:2201, 2089-2093, 2130-2165, 2166-2169
//                                               LW fluxr (TOA/sfc up/dn, clear and total sky),
//                                               cloud fraction and top/bottom accumulations, cldcov.
//
// Dependencies not in this file: setemis (sfcemis), lwrad (htlwc, htlwb, topflw, sfcflw).
// Sizes: IM x JM columns, LM model layers, LMK = LM+1 local layers (LTP = 1).
// Layouts (CONTRACT section 1): 2-D [JM, IM]; 3-D [LM|LMK, JM, IM]; 4-D [n, l, JM, IM].
//   clouds  [NF_CLDS=9, LMK, JM, IM]   dummys [2, LMK, JM, IM]: slot 0 holds clouds n=1 (Fortran
//           clouds(.,.,2), liquid path), slot 1 holds clouds n=3 (Fortran clouds(.,.,4), ice path).
//   htlwc   [LMK, JM, IM]    htlwb [NBDLW=16, LMK, JM, IM] (optional, lwrad HLWB)
//   htrlw   [LM, JM, IM]     htrlwb [NBDLW, LM, JM, IM] (optional)
//   plvl    [LM+1, JM, IM]   prsi (interfaces)     T [LM, JM, IM] (tgrs)
//   cldsa   [5, JM, IM]      mtopa, mbota int32 [3, JM, IM]
//   fluxr   [NFXR, JM, IM]   (Fortran fluxr(IX,NFXR); 1-based n maps to 0-based n-1)
//   topflw_*/sfcflw_* [JM, IM]: the proflw_type fields upfxc, dnfxc, upfx0, dnfx0.
// kd, kt, kb are the index offsets set in grrad_nmmb (lines 1165-1193), passed in.
//
// Fast numerics: float throughout, FMA contraction on (NVRTC --fmad=true). No host sync.

#include "nmmb_common.cuh"

#define NBDLW 16

// Local helper: 1-based Fortran fluxr(i, n1) for one column.
__device__ inline float* fluxr_at(float* fluxr, int n1, int i0, int j0, int im, int jm)
{
    return &fluxr[IDX3(i0, j0, n1 - 1, im, jm)];
}

// ---------------------------------------------------------------------------------------------
// grrad_nmmb.f:1978-1990. Only when np3d == 3 (Ferrier stratiform clouds). Saves the liquid and
// ice cloud paths (clouds(i,k,2), clouds(i,k,4)) to dummys and scales them by CPATHFAC4LW for the
// LW call. The loop covers all LMK local levels, including the extra top layer (LTP = 1).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void grrad_lw_pre(float* __restrict__ clouds,
                                        float* __restrict__ dummys,
                                        int lmk, int jm, int im, int np3d,
                                        float cpathfac)
{
    COLUMN_IJ(im, jm)
    if (j0 == 0 || j0 == jm - 1) return;      // rows j=1 and j=JM: RRTM is gated on JDS+1..JDE-1
    if (np3d != 3) return;
    for (int k0 = 0; k0 < lmk; ++k0) {
        const size_t p2 = IDX4(i0, j0, k0, 1, im, jm, lmk);   // clouds(i,k,2): liquid path
        const size_t p4 = IDX4(i0, j0, k0, 3, im, jm, lmk);   // clouds(i,k,4): ice path
        const size_t d2 = IDX4(i0, j0, k0, 0, im, jm, lmk);   // dummys(i,k,2)
        const size_t d4 = IDX4(i0, j0, k0, 1, im, jm, lmk);   // dummys(i,k,4)
        dummys[d2] = clouds[p2];
        dummys[d4] = clouds[p4];
        clouds[p2] = dummys[d2] * cpathfac;
        clouds[p4] = dummys[d4] * cpathfac;
    }
}

// ---------------------------------------------------------------------------------------------
// After setemis/lwrad. Heating-rate and surface outputs (grrad_nmmb.f:2006-2013, 2046-2055,
// 2064-2071). The restore (2064-2071) mirrors grrad_lw_pre.
//   htrlwb (2006-2013): only if present (has_htlwb); htlwb is the lwrad optional HLWB output.
//   semis, tsflw (2046-2048): semis = sfcemis, tsflw = tsfa (the surface air temperature saved for
//          the diurnal adjustment).
//   htrlw (2050-2058): htrlw(i,k) = htlwc(i,k+kd), k = 1..LM.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void grrad_lw_post_heat(const float* __restrict__ htlwc,
                                              const float* __restrict__ htlwb,
                                              const float* __restrict__ sfcemis,
                                              const float* __restrict__ tsfa,
                                              float* __restrict__ semis,
                                              float* __restrict__ tsflw,
                                              float* __restrict__ htrlw,
                                              float* __restrict__ htrlwb,
                                              float* __restrict__ clouds,
                                              const float* __restrict__ dummys,
                                              int lm, int lmk, int kd, int jm, int im,
                                              int np3d, int lslwr, int has_htlwb)
{
    COLUMN_IJ(im, jm)
    if (j0 == 0 || j0 == jm - 1) return;      // rows j = 1 and j = JM: RRTM is gated on JDS+1..JDE-1
    if (!lslwr) return;                       // everything below is inside if (lslwr) (1959)

    // grrad_nmmb.f:2046-2048
    semis[IDX2(i0, j0, im)] = sfcemis[IDX2(i0, j0, im)];
    tsflw[IDX2(i0, j0, im)] = tsfa[IDX2(i0, j0, im)];

    // grrad_nmmb.f:2050-2058, htrlw(i,k) = htlwc(i,k1), k1 = k + kd
    for (int k0 = 0; k0 < lm; ++k0) {
        htrlw[IDX3(i0, j0, k0, im, jm)] = htlwc[IDX3(i0, j0, k0 + kd, im, jm)];
    }

    // grrad_nmmb.f:2006-2013, htrlwb(i,k,j) = htlwb(i,k1,j), j = 1..NBDLW (present branch only)
    if (has_htlwb) {
        for (int n0 = 0; n0 < NBDLW; ++n0) {
            for (int k0 = 0; k0 < lm; ++k0) {
                htrlwb[IDX4(i0, j0, k0, n0, im, jm, lm)] =
                    htlwb[IDX4(i0, j0, k0 + kd, n0, im, jm, lmk)];
            }
        }
    }

    // grrad_nmmb.f:2064-2071, restore the scaled cloud paths (np3d == 3)
    if (np3d == 3) {
        for (int k0 = 0; k0 < lmk; ++k0) {
            clouds[IDX4(i0, j0, k0, 1, im, jm, lmk)] = dummys[IDX4(i0, j0, k0, 0, im, jm, lmk)];
            clouds[IDX4(i0, j0, k0, 3, im, jm, lmk)] = dummys[IDX4(i0, j0, k0, 1, im, jm, lmk)];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LW fluxes and cloud diagnostics, inside if (lssav) (grrad_nmmb.f:2082).
//   2086-2093 (lslwr): fluxr 1, 19, 20, 28, 30, 33 accumulate dtlw * LW fluxes at TOA and surface.
//   2130-2165 (lsswr .or. lslwr): fluxr 17, 18 from cldsa(.,4) and cldsa(.,5) (times raddt);
//          for j = 1..3 the cloud fraction into fluxr(8-j), the top and bottom interface pressure
//          into fluxr(11-j) and fluxr(14-j), and the top temperature into fluxr(17-j). mtopa and
//          mbota are clamped to [1, LM] before the +kt/+kb offsets, as at 2151-2152.
//   2166-2169 (always inside lssav): cldcov(i,k) = clouds(i,k+kd,1).
// The SW accumulations (grrad_nmmb.f:2094-2128 after the LW block) are not in this unit.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void grrad_lw_post_flux(const float* __restrict__ topflw_upfxc,
                                              const float* __restrict__ topflw_upfx0,
                                              const float* __restrict__ sfcflw_dnfxc,
                                              const float* __restrict__ sfcflw_upfxc,
                                              const float* __restrict__ sfcflw_dnfx0,
                                              const float* __restrict__ sfcflw_upfx0,
                                              const float* __restrict__ cldsa,
                                              const int* __restrict__ mtopa,
                                              const int* __restrict__ mbota,
                                              const float* __restrict__ plvl,
                                              const float* __restrict__ T,
                                              const float* __restrict__ clouds,
                                              float* __restrict__ cldcov,
                                              float* __restrict__ fluxr,
                                              const float* __restrict__ coszen_loc,
                                              float* __restrict__ coszen,
                                              int lm, int lmk, int kd, int kt, int kb,
                                              int jm, int im,
                                              float dtlw, float raddt,
                                              int lssav, int lslwr, int lsswr)
{
    COLUMN_IJ(im, jm)
    if (j0 == 0 || j0 == jm - 1) return;      // rows j=1 and j=JM: RRTM is gated on JDS+1..JDE-1
    // grrad_nmmb.f:2201, coszen(i) = coszen_loc(i). Unconditional in Fortran (after the grrad body).
    coszen[IDX2(i0, j0, im)] = coszen_loc[IDX2(i0, j0, im)];

    if (!lssav) return;                                  // grrad_nmmb.f:2082

    if (lslwr) {
        // grrad_nmmb.f:2089-2093 (total sky 1, 19, 20; clear sky 28, 30, 33)
        *fluxr_at(fluxr, 1,  i0, j0, im, jm) += dtlw * topflw_upfxc[IDX2(i0, j0, im)];
        *fluxr_at(fluxr, 19, i0, j0, im, jm) += dtlw * sfcflw_dnfxc[IDX2(i0, j0, im)];
        *fluxr_at(fluxr, 20, i0, j0, im, jm) += dtlw * sfcflw_upfxc[IDX2(i0, j0, im)];
        *fluxr_at(fluxr, 28, i0, j0, im, jm) += dtlw * topflw_upfx0[IDX2(i0, j0, im)];
        *fluxr_at(fluxr, 30, i0, j0, im, jm) += dtlw * sfcflw_dnfx0[IDX2(i0, j0, im)];
        *fluxr_at(fluxr, 33, i0, j0, im, jm) += dtlw * sfcflw_upfx0[IDX2(i0, j0, im)];
    }

    if (lsswr || lslwr) {
        // grrad_nmmb.f:2132-2133. cldsa is Fortran (i,5): n0 = 3 is cldsa(i,4), n0 = 4 is cldsa(i,5).
        *fluxr_at(fluxr, 17, i0, j0, im, jm) += raddt * cldsa[IDX3(i0, j0, 3, im, jm)];
        *fluxr_at(fluxr, 18, i0, j0, im, jm) += raddt * cldsa[IDX3(i0, j0, 4, im, jm)];

        // grrad_nmmb.f:2136-2161, j = 1..3 (Fortran cldsa(i,j), mtopa(i,j), mbota(i,j))
        for (int j = 1; j <= 3; ++j) {
            const float tem0d = raddt * cldsa[IDX3(i0, j0, j - 1, im, jm)];
            int itop = mtopa[IDX3(i0, j0, j - 1, im, jm)] - kd;
            int ibtc = mbota[IDX3(i0, j0, j - 1, im, jm)] - kd;
            itop = min(max(itop, 1), lm);                // grrad_nmmb.f:2151
            ibtc = min(max(ibtc, 1), lm);                // grrad_nmmb.f:2152

            *fluxr_at(fluxr, 8 - j,  i0, j0, im, jm) += tem0d;
            // prsi(i, itop+kt) and prsi(i, ibtc+kb), 1-based -> 0-based (x - 1)
            *fluxr_at(fluxr, 11 - j, i0, j0, im, jm) +=
                plvl[IDX3(i0, j0, itop + kt - 1, im, jm)] * tem0d;
            *fluxr_at(fluxr, 14 - j, i0, j0, im, jm) +=
                plvl[IDX3(i0, j0, ibtc + kb - 1, im, jm)] * tem0d;
            // tgrs(i, itop) with tgrs = T
            *fluxr_at(fluxr, 17 - j, i0, j0, im, jm) +=
                T[IDX3(i0, j0, itop - 1, im, jm)] * tem0d;
        }
    }

    // grrad_nmmb.f:2166-2169, cldcov(i,k) = clouds(i,k1,1), k1 = k + kd; n0 = 0 is clouds(.,.,1)
    for (int k0 = 0; k0 < lm; ++k0) {
        cldcov[IDX3(i0, j0, k0, im, jm)] = clouds[IDX4(i0, j0, k0 + kd, 0, im, jm, lmk)];
    }
}
