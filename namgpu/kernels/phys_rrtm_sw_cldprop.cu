// !!!!!  ==============================================================  !!!!!
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
// !!!!!              sw-rrtm3 radiation package description              !!!!!
// !!!!!  ==============================================================  !!!!!
// !                                                                          !
// !   this package includes ncep's modifications of the rrtm-sw radiation    !
// !   code from aer inc.                                                     !
// !                                                                          !
// !   notice: the cloud coefficient tables (module_radsw_cldprtb, radsw_datatb.f) and the    !
// !   subroutine cldprop are NOAA/NCEP code (nam.v4.2.11). Ported to CUDA for WOOF-NAM.      !
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:1400-1822 (nam.v4.2.11, 83df4e13c1).
//
// CLDPROP: SW cloud optical properties per cloud layer and per SW band (14 bands, Fortran
// NBLOW..NBHGH = 16..29): taucw, ssacw, asycw, the plain cloud fraction cldfrc = cfrac/cf1 when
// isubcsw <= 0, and the McICA input cldf (cfrac with cfrac < ftiny set to 0) when isubcsw > 0.
// The McICA lcloudy -> cldfmc step is the sub-column generator, phys-rrtm-sw-mcica-subcol, which
// reads ctx.tables["rrtm_sw_cldf"] written here. One thread per column (i, j); the layer loop runs
// inside the thread (CONTRACT section 3).
//
// Layout: per-layer inputs [nlay, JM, IM] in radsw INTERNAL order, surface first: k0 = 0 is the
// lowest layer (Fortran cldprop k = 1, radsw_main_nmmb.f:985-1002 and 1531). The seam (swrad) reverses
// the NMMB top-first state arrays before this unit runs; the CONTRACT section 1 rule (l = 0 is the
// model top) applies to state arrays, not to these seam arrays. Consumers (mcica-subcol, spcvrtc,
// taumol) use the same surface-first order. cf1 [JM, IM] is zcf1(i, j); outputs taucw/ssacw/asycw
// [14, nlay, JM, IM] (band-major, Fortran (ib, k) -> [ib, k]); cldfrc, cldf [nlay, JM, IM].
// Tables: extliq1/ssaliq1/asyliq1 [14, 58], extice2/ssaice2/asyice2 [14, 43],
// extice3/ssaice3/asyice3 [14, 46] (npz from tools/extract_rrtm_sw_cldprtb.py).
// Small constants (Ebert-Curry, rain, snow, band tables, idxebc) are written below with their source
// lines (radsw_datatb.f:179-194, 1803-1813; radsw_main_nmmb.f:313).
//
// Fortran-to-CUDA review (every statement of cldprop, radsw_main_nmmb.f:1400-1822):
//   caller gate (radsw_main_nmmb.f:985-1002): cldprop runs only where zcf1 > 0 (NaN takes the clear
//           branch, as in Fortran). A clear column gets taucw = ssacw = asycw = 0 (ssacw 0, not the
//           cloudy default 1), cldfrc = 0 and, for isubcsw > 0, cldf = 0 (cldfmc = 0 in Fortran).
//   taucw = 0, ssacw = 1, asycw = 0 for every (k, ib) first                -> same.
//   iswcliq > 0, loop k with cfrac(k) > ftiny (ftiny = 1e-12)              -> same.
//   iswcice < 4: tauran = cldran*a0r; tausnw = cldsnw*1.09087*(a0s+a1s/dgesnw) when
//                cldsnw > 0 and refsnw > 10 (dgesnw = 1.0315*refsnw); ssaran, ssasnw, asyran,
//                asysnw per band from b0r, b0s, b1s, c0r, c0s                -> same.
//   liquid: cldliq <= 0 -> 0; iswcliq == 1: factor = refliq-1.5, index 1..57,
//           extinction/ssa/asy interpolation, ssa and asy clamps            -> same.
//           (iswcliq other than 1 with cldliq > 0: Fortran leaves tauliq undefined; zero here.)
//   ice: cldice <= 0 -> 0; iswcice 1 (Ebert-Curry, refice 13..130, idxebc band index),
//        2 (streamer, refice 5..131, index 1..42), 3 (fu, dgeice = max(5, min(140, 1.0315*refice)),
//        index 1..45), 4 (fu table, dgeice = max(10.1, min(129.9, refice)), index 1..45)  -> all four.
//           (iswcice outside 1..4 with cldice > 0: Fortran undefined; zero here.)
//   iswcice == 4 snow: cldsnw <= 0 -> 0; else dgesnw = max(25.1, min(129.9, refsnw)),
//           tausnow = min(35, cldsnw*ext), ssasnw = tausnow*ssa, asysnw = ssasnw*asy -> same.
//   iswcice == 4 final: tauliq, tauice = min(35, .); dummy = min(sum, 35);
//           taucw = max(1e-3, dummy); ssacw and asycw = sums of liq, ice, snow -> same.
//   iswcice < 4 final: taucw = tauliq+tauice+tauran+tausnw, ssacw/asycw sums with the band
//           arrays ssaran/asyran/ssasnw/asysnw                              -> same order.
//   iswcliq == 0: taucw = cdat1, ssacw = cdat1*cdat2, asycw = ssacw*cdat3 where cfrac > ftiny -> same.
//   isubcsw > 0: cldf = cfrac, then cldf = 0 where cldf < ftiny            -> written to cldf.
//   isubcsw <= 0: cldfrc = cfrac/cf1 for every layer                        -> same.
//   NaN gating: Fortran tests cldliq <= 0 (1572), cldice <= 0 (1603) and cldsnw <= 0 (1714); a NaN
//           water path is not <= 0, so it falls through and propagates. The kernel uses the same
//           !(x <= 0) form, so NaN inputs are not zeroed.
// Not in this kernel: the isubcsw > 0 cldfmc (1/0 per g-point) is the McICA generator's output and
// is produced by phys-rrtm-sw-mcica-subcol from cldf. ipseed is an input of that unit, not of cldprop.

#include "nmmb_common.cuh"

#define SW_NBDSW     14     // NBDSW = NBHGH - NBLOW + 1 = 29 - 16 + 1 (radsw_param)
#define SW_NTAB_LIQ  58
#define SW_NTAB_ICE2 43
#define SW_NTAB_ICE3 46

// per-(layer, column) access helpers; k0 = layer (0-based), colbase = column offset in a [*, JM, IM] array
#define SW_IN(arr, k0)          arr[(size_t)(k0) * plane + colbase]
#define SW_OUT(arr, ib0, k0)    arr[((size_t)(ib0) * nlay + (k0)) * plane + colbase]

extern "C" __global__ void phys_rrtm_sw_cldprop(
    const float* __restrict__ cfrac,    /* [nlay, JM, IM] */
    const float* __restrict__ cliqp,    /* [nlay, JM, IM] */
    const float* __restrict__ reliq,    /* [nlay, JM, IM] */
    const float* __restrict__ cicep,    /* [nlay, JM, IM] */
    const float* __restrict__ reice,    /* [nlay, JM, IM] */
    const float* __restrict__ cdat1,    /* [nlay, JM, IM] */
    const float* __restrict__ cdat2,    /* [nlay, JM, IM] */
    const float* __restrict__ cdat3,    /* [nlay, JM, IM] */
    const float* __restrict__ cdat4,    /* [nlay, JM, IM] */
    const float* __restrict__ cf1,      /* [JM, IM]  zcf1 */
    const float* __restrict__ extliq1,  /* [14, 58] */
    const float* __restrict__ ssaliq1,  /* [14, 58] */
    const float* __restrict__ asyliq1,  /* [14, 58] */
    const float* __restrict__ extice2,  /* [14, 43] */
    const float* __restrict__ ssaice2,  /* [14, 43] */
    const float* __restrict__ asyice2,  /* [14, 43] */
    const float* __restrict__ extice3,  /* [14, 46] */
    const float* __restrict__ ssaice3,  /* [14, 46] */
    const float* __restrict__ asyice3,  /* [14, 46] */
    float* __restrict__ taucw,          /* [14, nlay, JM, IM] */
    float* __restrict__ ssacw,          /* [14, nlay, JM, IM] */
    float* __restrict__ asycw,          /* [14, nlay, JM, IM] */
    float* __restrict__ cldfrc,         /* [nlay, JM, IM]; written when isubcsw <= 0 */
    float* __restrict__ cldf,           /* [nlay, JM, IM]; written when isubcsw > 0 */
    int nlay, int iswcliq, int iswcice, int isubcsw, int IM, int JM)
{
    COLUMN_IJ(IM, JM)
    const float f_zero = 0.0f;
    const float f_one  = 1.0f;
    const float ftiny  = 1.0e-12f;      /* radsw_main_nmmb.f:295 */
    const float a0r = 3.07e-3f;         /* radsw_datatb.f:1803  a0r */
    const float a0s = 0.0f;             /* radsw_datatb.f:1803  a0s */
    const float a1s = 1.5f;             /* radsw_datatb.f:1803  a1s */

    /* Ebert-Curry ice coefficients, radsw_datatb.f:189-194 (index ia = 1..5) */
    const float abari[5] = {3.448e-03f, 3.448e-03f, 3.448e-03f, 3.448e-03f, 3.448e-03f};
    const float bbari[5] = {2.431e+00f, 2.431e+00f, 2.431e+00f, 2.431e+00f, 2.431e+00f};
    const float cbari[5] = {1.000e-05f, 1.100e-04f, 1.240e-02f, 3.779e-02f, 4.666e-01f};
    const float dbari[5] = {0.000e+00f, 1.405e-05f, 6.867e-04f, 1.284e-03f, 2.050e-05f};
    const float ebari[5] = {7.661e-01f, 7.730e-01f, 7.865e-01f, 8.172e-01f, 9.595e-01f};
    const float fbari[5] = {5.851e-04f, 5.665e-04f, 7.204e-04f, 7.463e-04f, 1.076e-04f};

    /* rain and snow band coefficients, radsw_datatb.f:1807-1813 (band ib = 1..14) */
    const float b0r[SW_NBDSW] = {0.466f, 0.437f, 0.416f, 0.391f, 0.374f, 0.352f, 0.183f,
                                 0.048f, 0.012f, 0.000f, 0.000f, 0.000f, 0.000f, 0.496f};
    const float c0r[SW_NBDSW] = {0.975f, 0.965f, 0.960f, 0.955f, 0.952f, 0.950f, 0.944f,
                                 0.894f, 0.884f, 0.883f, 0.883f, 0.883f, 0.883f, 0.980f};
    const float b0s[SW_NBDSW] = {0.460f, 0.460f, 0.460f, 0.460f, 0.460f, 0.460f, 0.460f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.460f};
    const float b1s[SW_NBDSW] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                 1.62e-5f, 1.62e-5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float c0s[SW_NBDSW] = {0.970f, 0.970f, 0.970f, 0.970f, 0.970f, 0.970f, 0.970f,
                                 0.970f, 0.970f, 0.700f, 0.700f, 0.700f, 0.700f, 0.970f};
    /* idxebc, radsw_main_nmmb.f:313: eb_&_c band index for the cloud-ice coefficients */
    const int idxebc[SW_NBDSW] = {5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5};

    const size_t plane = (size_t)JM * IM;
    const size_t colbase = (size_t)j0 * IM + i0;

    /* caller gate, radsw_main_nmmb.f:985-1002: cldprop runs only for zcf1 > 0. The clear branch sets
       cldfrc = cldfmc = 0 and taucw = ssacw = asycw = 0. NaN cf1 is not > 0, so it is clear as in Fortran. */
    if (!(cf1[colbase] > f_zero)) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                SW_OUT(taucw, ib0, k0) = f_zero;
                SW_OUT(ssacw, ib0, k0) = f_zero;
                SW_OUT(asycw, ib0, k0) = f_zero;
            }
            if (isubcsw > 0) SW_IN(cldf, k0) = f_zero;
            else             SW_IN(cldfrc, k0) = f_zero;
        }
        return;
    }

    /* radsw_main_nmmb.f:1420-1425 defaults */
    for (int k0 = 0; k0 < nlay; ++k0)
        for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
            SW_OUT(taucw, ib0, k0) = f_zero;
            SW_OUT(ssacw, ib0, k0) = f_one;
            SW_OUT(asycw, ib0, k0) = f_zero;
        }

    if (iswcliq > 0) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            const float cf = SW_IN(cfrac, k0);
            if (!(cf > ftiny)) continue;

            float tauran = f_zero, tausnw = f_zero;
            float ssaran[SW_NBDSW], asyran[SW_NBDSW], ssasnw[SW_NBDSW], asysnw[SW_NBDSW];
            float tauliq[SW_NBDSW], ssaliq[SW_NBDSW], asyliq[SW_NBDSW];
            float tauice[SW_NBDSW], ssaice[SW_NBDSW], asyice[SW_NBDSW], tausnow[SW_NBDSW];
            for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                ssaran[ib0] = f_zero; asyran[ib0] = f_zero; ssasnw[ib0] = f_zero; asysnw[ib0] = f_zero;
                tauliq[ib0] = f_zero; ssaliq[ib0] = f_zero; asyliq[ib0] = f_zero;
                tauice[ib0] = f_zero; ssaice[ib0] = f_zero; asyice[ib0] = f_zero; tausnow[ib0] = f_zero;
            }

            /* rain and snow optical properties (fu's snow formula), radsw_main_nmmb.f:1437-1476 */
            const float cldran = SW_IN(cdat1, k0);
            const float cldsnw = SW_IN(cdat3, k0);
            const float refsnw = SW_IN(cdat4, k0);
            const float dgesnw_fu = 1.0315f * refsnw;
            if (iswcice < 4) {
                tauran = cldran * a0r;
                /* fu's formula needs the snow density factor 1.09087 */
                if (cldsnw > f_zero && refsnw > 10.0f) {
                    tausnw = cldsnw * 1.09087f * (a0s + a1s / dgesnw_fu);
                } else {
                    tausnw = f_zero;
                }
                for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                    ssaran[ib0] = tauran * (f_one - b0r[ib0]);
                    ssasnw[ib0] = tausnw * (f_one - (b0s[ib0] + b1s[ib0] * dgesnw_fu));
                    asyran[ib0] = ssaran[ib0] * c0r[ib0];
                    asysnw[ib0] = ssasnw[ib0] * c0s[ib0];
                }
            }

            const float cldliq = SW_IN(cliqp, k0);
            const float refliq = SW_IN(reliq, k0);
            const float cldice = SW_IN(cicep, k0);
            float refice = SW_IN(reice, k0);

            /* liquid water absorption (hu & stamnes), radsw_main_nmmb.f:1576-1598 */
            if (!(cldliq <= f_zero) && iswcliq == 1) {  /* Fortran 1572: cldliq <= 0 zeroes, NaN falls through */
                const float factor = refliq - 1.5f;
                const int index = max(1, min(SW_NTAB_LIQ - 1, (int)factor));
                const float fint = factor - (float)index;
                for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                    const float* e = extliq1 + ib0 * SW_NTAB_LIQ;
                    const float* s = ssaliq1 + ib0 * SW_NTAB_LIQ;
                    const float* a = asyliq1 + ib0 * SW_NTAB_LIQ;
                    const float extcoliq = fmaxf(f_zero, e[index - 1] + fint * (e[index] - e[index - 1]));
                    const float ssacoliq = fmaxf(f_zero, fminf(f_one, s[index - 1] + fint * (s[index] - s[index - 1])));
                    const float asycoliq = fmaxf(f_zero, fminf(f_one, a[index - 1] + fint * (a[index] - a[index - 1])));
                    tauliq[ib0] = cldliq * extcoliq;
                    ssaliq[ib0] = tauliq[ib0] * ssacoliq;
                    asyliq[ib0] = ssaliq[ib0] * asycoliq;
                }
            }

            /* ice absorption, radsw_main_nmmb.f:1607-1697 */
            if (!(cldice <= f_zero)) {  /* Fortran 1603: cldice <= 0 zeroes, NaN falls through */
                if (iswcice == 1) {
                    /* Ebert & Curry, all particle sizes */
                    refice = fminf(130.0f, fmaxf(13.0f, refice));
                    for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                        const int ia = idxebc[ib0] - 1;
                        const float extcoice = fmaxf(f_zero, abari[ia] + bbari[ia] / refice);
                        const float ssacoice = fmaxf(f_zero, fminf(f_one, f_one - cbari[ia] - dbari[ia] * refice));
                        const float asycoice = fmaxf(f_zero, fminf(f_one, ebari[ia] + fbari[ia] * refice));
                        tauice[ib0] = cldice * extcoice;
                        ssaice[ib0] = tauice[ib0] * ssacoice;
                        asyice[ib0] = ssaice[ib0] * asycoice;
                    }
                } else if (iswcice == 2) {
                    /* streamer, effective radius 5..131 micron */
                    refice = fminf(131.0f, fmaxf(5.0f, refice));
                    const float factor = (refice - 2.0f) / 3.0f;
                    const int index = max(1, min(SW_NTAB_ICE2 - 1, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                        const float* e = extice2 + ib0 * SW_NTAB_ICE2;
                        const float* s = ssaice2 + ib0 * SW_NTAB_ICE2;
                        const float* a = asyice2 + ib0 * SW_NTAB_ICE2;
                        const float extcoice = fmaxf(f_zero, e[index - 1] + fint * (e[index] - e[index - 1]));
                        const float ssacoice = fmaxf(f_zero, fminf(f_one, s[index - 1] + fint * (s[index] - s[index - 1])));
                        const float asycoice = fmaxf(f_zero, fminf(f_one, a[index - 1] + fint * (a[index] - a[index - 1])));
                        tauice[ib0] = cldice * extcoice;
                        ssaice[ib0] = tauice[ib0] * ssacoice;
                        asyice[ib0] = ssaice[ib0] * asycoice;
                    }
                } else if (iswcice == 3) {
                    /* fu (1996), generalized size 5..140 micron */
                    const float dgeice = fmaxf(5.0f, fminf(140.0f, 1.0315f * refice));
                    const float factor = (dgeice - 2.0f) / 3.0f;
                    const int index = max(1, min(SW_NTAB_ICE3 - 1, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                        const float* e = extice3 + ib0 * SW_NTAB_ICE3;
                        const float* s = ssaice3 + ib0 * SW_NTAB_ICE3;
                        const float* a = asyice3 + ib0 * SW_NTAB_ICE3;
                        const float extcoice = fmaxf(f_zero, e[index - 1] + fint * (e[index] - e[index - 1]));
                        const float ssacoice = fmaxf(f_zero, fminf(f_one, s[index - 1] + fint * (s[index] - s[index - 1])));
                        const float asycoice = fmaxf(f_zero, fminf(f_one, a[index - 1] + fint * (a[index] - a[index - 1])));
                        tauice[ib0] = cldice * extcoice;
                        ssaice[ib0] = tauice[ib0] * ssacoice;
                        asyice[ib0] = ssaice[ib0] * asycoice;
                    }
                } else if (iswcice == 4) {
                    /* fu table, size 10.1..129.9 micron (MIN 35 applied in the final sum) */
                    const float dgeice = fmaxf(10.1f, fminf(129.9f, refice));
                    const float factor = (dgeice - 2.0f) / 3.0f;
                    const int index = max(1, min(SW_NTAB_ICE3 - 1, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                        const float* e = extice3 + ib0 * SW_NTAB_ICE3;
                        const float* s = ssaice3 + ib0 * SW_NTAB_ICE3;
                        const float* a = asyice3 + ib0 * SW_NTAB_ICE3;
                        const float extcoice = fmaxf(f_zero, e[index - 1] + fint * (e[index] - e[index - 1]));
                        const float ssacoice = fmaxf(f_zero, fminf(f_one, s[index - 1] + fint * (s[index] - s[index - 1])));
                        const float asycoice = fmaxf(f_zero, fminf(f_one, a[index - 1] + fint * (a[index] - a[index - 1])));
                        tauice[ib0] = cldice * extcoice;
                        ssaice[ib0] = tauice[ib0] * ssacoice;
                        asyice[ib0] = ssaice[ib0] * asycoice;
                    }
                }
            }

            /* snow absorption, new when iswcice == 4 (G. Thompson), radsw_main_nmmb.f:1713-1741 */
            if (iswcice == 4) {
                if (!(cldsnw <= f_zero)) {  /* Fortran 1714: cldsnw <= 0 zeroes, NaN falls through */
                    const float dgesnw = fmaxf(25.1f, fminf(129.9f, refsnw));
                    const float factor = (dgesnw - 2.0f) / 3.0f;
                    const int index = max(1, min(SW_NTAB_ICE3 - 1, (int)factor));
                    const float fint = factor - (float)index;
                    for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                        const float* e = extice3 + ib0 * SW_NTAB_ICE3;
                        const float* s = ssaice3 + ib0 * SW_NTAB_ICE3;
                        const float* a = asyice3 + ib0 * SW_NTAB_ICE3;
                        const float extcosno = fmaxf(f_zero, e[index - 1] + fint * (e[index] - e[index - 1]));
                        const float ssacosno = fmaxf(f_zero, fminf(f_one, s[index - 1] + fint * (s[index] - s[index - 1])));
                        const float asycosno = fmaxf(f_zero, fminf(f_one, a[index - 1] + fint * (a[index] - a[index - 1])));
                        tausnow[ib0] = fminf(35.0f, cldsnw * extcosno);
                        ssasnw[ib0] = tausnow[ib0] * ssacosno;
                        asysnw[ib0] = ssasnw[ib0] * asycosno;
                    }
                }
            }

            /* distribute the band properties to the output, radsw_main_nmmb.f:1743-1771 */
            if (iswcice == 4) {
                for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                    tauliq[ib0] = fminf(35.0f, tauliq[ib0]);
                    tauice[ib0] = fminf(35.0f, tauice[ib0]);
                    const float dummy_tau = fminf(tauliq[ib0] + tauice[ib0] + tausnow[ib0], 35.0f);
                    SW_OUT(taucw, ib0, k0) = fmaxf(1.0e-3f, dummy_tau);
                    SW_OUT(ssacw, ib0, k0) = ssaliq[ib0] + ssaice[ib0] + ssasnw[ib0];
                    SW_OUT(asycw, ib0, k0) = asyliq[ib0] + asyice[ib0] + asysnw[ib0];
                }
            } else {
                for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                    SW_OUT(taucw, ib0, k0) = tauliq[ib0] + tauice[ib0] + tauran + tausnw;
                    SW_OUT(ssacw, ib0, k0) = ssaliq[ib0] + ssaice[ib0] + ssaran[ib0] + ssasnw[ib0];
                    SW_OUT(asycw, ib0, k0) = asyliq[ib0] + asyice[ib0] + asyran[ib0] + asysnw[ib0];
                }
            }
        }
    } else {
        /* diagnostic cloud method: cloud optical depth, ssa and asy are inputs, radsw_main_nmmb.f:1776-1787 */
        for (int k0 = 0; k0 < nlay; ++k0) {
            if (SW_IN(cfrac, k0) > ftiny) {
                for (int ib0 = 0; ib0 < SW_NBDSW; ++ib0) {
                    SW_OUT(taucw, ib0, k0) = SW_IN(cdat1, k0);
                    SW_OUT(ssacw, ib0, k0) = SW_IN(cdat1, k0) * SW_IN(cdat2, k0);
                    SW_OUT(asycw, ib0, k0) = SW_OUT(ssacw, ib0, k0) * SW_IN(cdat3, k0);
                }
            }
        }
    }

    /* cloud fraction for the sub-column generator (isubcsw > 0), radsw_main_nmmb.f:1789-1795 */
    if (isubcsw > 0) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            const float cf = SW_IN(cfrac, k0);
            SW_IN(cldf, k0) = (cf < ftiny) ? f_zero : cf;
        }
    } else {
        /* standard cloud scheme: normalise to the cloudy fraction of the column, radsw_main_nmmb.f:1812-1817 */
        const float cf1c = cf1[colbase];
        for (int k0 = 0; k0 < nlay; ++k0) {
            SW_IN(cldfrc, k0) = SW_IN(cfrac, k0) / cf1c;
        }
    }
}
