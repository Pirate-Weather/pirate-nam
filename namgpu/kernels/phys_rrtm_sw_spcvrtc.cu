// phys_rrtm_sw_spcvrtc.cu -- delta-Eddington / PIFM / discrete-ordinates two-stream layer solution
// and column flux accumulation for the NMMB RRTM shortwave (spcvrtc + swflux), one g-point per launch.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:2144-2964 (subroutine spcvrtc) and
// phys/radsw_main_nmmb.f:2977-3102 (subroutine swflux), nam.v4.2.11, 83df4e13c1.
// Notice (header of radsw_main_nmmb.f): sw-rrtm3 radiation package, NCEP modifications of the
// RRTM-SW radiation code from AER Inc.; the AER lineage and the NOAA/NCEP header of that file apply.
//
// Per column (one thread per (i, j)): for the g-point jg of this launch, the clear-sky layer pass,
// the clear-sky swflux, the clear-sky surface components, and (sunlit, cf1 > eps) the total-sky
// layer pass, swflux and surface components. At the last g-point the column end section of the
// Fortran (band sums, cf1 <= eps copy, cf1 weighting) runs too. At the first g-point the column's
// accumulators are zeroed, as the Fortran does at routine entry.
//
// The Fortran loops over ngptsw g-points inside one call. This kernel does one g-point per launch
// and the wrapper loops the 112 g-points on the host (no per-cell host work). The reason is memory:
// the gas optical depths (taug, taur) for all g-points at 954 x 835 x 60 would not fit a card next to
// the state. The accumulation order over g-points and bands is the Fortran order.
//
// Layer order (radiation-internal, CONTRACT.md section 1 SW exception): the arrays below are the RRTM
// internal order, surface first (Fortran k = 1 is the lowest layer, index k-1 here). The swrad seam
// (phys_rrtm_sw_swrad.cu swrad_prep, kk0 = nlay-1-k0) reverses the NMMB top-first state arrays into this
// order; swrad_post converts the fluxes back to NMMB order. Nothing in this kernel flips.
//   sfluxzen [NGPTSW,JM,IM] (phys-rrtm-sw-taumol): zsolar = ssolar * sfluxzen(jg) per column (radsw_main_nmmb.f:2396)
// Mapping of Fortran variables (1-based Fortran k in 1..nlay, kp = k+1 in 2..nlp1; the local arrays
// below are 1-based so the Fortran text maps line by line):
//   taug, taur (jg), tauae, ssaae, asyae, taucw, ssacw, asycw (band ib) -> device [NLAY,JM,IM]
//   cldfrc [NLAY,JM,IM], cf1 cf0 cosz sntz ssolar [JM,IM], albbm albdf [2,JM,IM]
//   fxupc fxdnc fxup0 fxdn0 [NBDSW,NLP1,JM,IM] (band outermost, as Fortran fxup0(i,k,ib))
//   sfbmc sfdfc sfbm0 sfdf0 [2,JM,IM]; ftoadc ftoauc ftoau0 fsfcuc fsfcu0 fsfcdc fsfcd0 suvbfc suvbf0 [JM,IM]
//   exp_tbl [NTBMX+1] (radsw_param.f NTBMX=10000; exp_tbl(0)=1, exp_tbl(NTBMX)=expeps)
//
// Fixed control values (copied, with source lines):
//   iswmode = 2 (PIFM), physparam.f:62; the other two branches are kept (iswmode is an argument).
//   eps = 1e-6, oneminus = 1-eps, bpade = 1/0.278, ftiny = 1e-12 (radsw_main.f:291-295)
//   zcrit = 0.9999995, zsr3 = sqrt(3), od_lo = 0.06, eps1 = 1e-8 (spcvrtc parameters)
//   NBDSW = 14 (radsw_param.f NBANDS = 29-16+1), NBLOW = 16, uv-b band nuvb = 27 -> ib 12.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no device allocation, no printf.
// Local memory: the column scratch arrays (about 16 x 66 floats) live in the thread's local memory.

#include "nmmb_common.cuh"

#define SWC_NBDSW       14
#define SWC_NLAY_MAX    64
#define SWC_LP          (SWC_NLAY_MAX + 2)   // 1-based level arrays, index 0 unused
#define SWC_NTBMX       10000                // radsw_param.f:113
#define SWC_IBUV0       11                   // uv-b band nuvb=27 -> ib 12 -> 0-based 11
#define SWC_EPS         1.0e-6f              // radsw_main.f:291 eps
#define SWC_ONEMINUS    (1.0f - 1.0e-6f)     // radsw_main.f:292 oneminus = 1 - eps
#define SWC_BPADE       (1.0f / 0.278f)      // radsw_main.f:293 bpade = 1/0.278
#define SWC_FTINY       1.0e-12f             // radsw_main.f:295
#define SWC_ZCRIT       0.9999995f           // spcvrtc zcrit
#define SWC_ZSR3        1.7320508f           // spcvrtc zsr3 = sqrt(3)
#define SWC_OD_LO       0.06f                // spcvrtc od_lo
#define SWC_EPS1        1.0e-8f              // spcvrtc eps1
#define SWC_DAY_COSZ    1.0e-4f              // grrad_nmmb.f:640,644 sunlit when coszen >= 0.0001 (NDAY set)

// Table lookup as spcvrtc writes it in the conservative and non-conservative branches:
//   ftind = zb/(bpade+zb); itind = ftind*NTBMX + 0.5 (truncated); exp_tbl(itind)   (no clamp)
__device__ __forceinline__ float swc_lut_raw(float zb, const float* __restrict__ tbl)
{
    const float ftind = zb / (SWC_BPADE + zb);
    const int itind = (int)(ftind * (float)SWC_NTBMX + 0.5f);
    return tbl[itind];
}

// Table lookup as the direct-beam blocks write it:
//   itind = max(0, min(NTBMX, int(0.5 + NTBMX*ftind)))
__device__ __forceinline__ float swc_lut_clamped(float zb, const float* __restrict__ tbl)
{
    const float ftind = zb / (SWC_BPADE + zb);
    int itind = (int)(0.5f + (float)SWC_NTBMX * ftind);
    itind = max(0, min(SWC_NTBMX, itind));
    return tbl[itind];
}

// exp(-zb) as the Fortran evaluates it: low optical depth expansion, else the table.
__device__ __forceinline__ float swc_expm(float zb, const float* __restrict__ tbl, bool clamped)
{
    if (zb <= SWC_OD_LO) {
        return 1.0f - zb + 0.5f * zb * zb;
    }
    return clamped ? swc_lut_clamped(zb, tbl) : swc_lut_raw(zb, tbl);
}

// Two-stream coefficients zgam1..zgam3 for the three transfer schemes (spcvrtc).
__device__ __forceinline__ void swc_gam(int iswmode, float zssa1, float zasy1, float zasy3, float cz,
                                        float& zgam1, float& zgam2, float& zgam3)
{
    if (iswmode == 1) {                                  // delta-Eddington (Joseph et al. 1976)
        zgam1 = 1.75f - zssa1 * (1.0f + zasy3);
        zgam2 = -0.25f - zssa1 * (1.0f - zasy3);
        zgam3 = 0.5f - zasy3 * cz;
    } else if (iswmode == 2) {                           // PIFM (Zdunkowski et al. 1980)
        zgam1 = 2.0f - zssa1 * (1.25f + zasy3);
        zgam2 = 0.75f * zssa1 * (1.0f - zasy1);
        zgam3 = 0.5f - zasy3 * cz;
    } else {                                             // discrete ordinates (Liou 1973)
        zgam1 = SWC_ZSR3 * (2.0f - zssa1 * (1.0f + zasy1)) * 0.5f;
        zgam2 = SWC_ZSR3 * zssa1 * (1.0f - zasy1) * 0.5f;
        zgam3 = (1.0f - SWC_ZSR3 * zasy1 * cz) * 0.5f;
    }
}

// Layer reflectance and transmittance at one level (spcvrtc clear-sky block and the identical
// total-sky block). Writes rb, rd, tb, td (refb, refd, trab, trad).
__device__ __forceinline__ void swc_layer_rt(float ztau1, float zssaw, float zssa1,
                                             float zgam1, float zgam2, float zgam3, float zgam4,
                                             float cz, float snt, const float* __restrict__ tbl,
                                             float& rb, float& rd, float& tb, float& td)
{
    if (zssaw >= SWC_ZCRIT) {                            // conservative scattering
        const float za1 = zgam1 * cz - zgam3;
        const float za2 = zgam1 * ztau1;
        const float zb1 = fminf(ztau1 * snt, 500.0f);
        const float zb2 = swc_expm(zb1, tbl, false);
        rb = fmaxf(0.0f, fminf(1.0f, (za2 - za1 * (1.0f - zb2)) / (1.0f + za2)));
        tb = fmaxf(0.0f, fminf(1.0f, 1.0f - rb));
        rd = fmaxf(0.0f, fminf(1.0f, za2 / (1.0f + za2)));
        td = fmaxf(0.0f, fminf(1.0f, 1.0f - rd));
    } else {                                             // non-conservative scattering
        const float za1 = zgam1 * zgam4 + zgam2 * zgam3;
        const float za2 = zgam1 * zgam3 + zgam2 * zgam4;
        const float zrk = sqrtf((zgam1 - zgam2) * (zgam1 + zgam2));
        const float zrk2 = 2.0f * zrk;
        const float zrp = zrk * cz;
        const float zrp1 = 1.0f + zrp;
        const float zrm1 = 1.0f - zrp;
        float zrpp = 1.0f - zrp * zrp;
        if (fabsf(zrpp) < SWC_EPS1) {
            zrpp = (zrpp < 0.0f) ? -SWC_EPS1 : SWC_EPS1;   // Fortran SIGN(eps1, zrpp)
        }
        const float zrkg1 = zrk + zgam1;
        const float zrkg3 = zrk * zgam3;
        const float zrkg4 = zrk * zgam4;

        const float zr1 = zrm1 * (za2 + zrkg3);
        const float zr2 = zrp1 * (za2 - zrkg3);
        const float zr3 = zrk2 * (zgam3 - za2 * cz);
        const float zr4 = zrpp * zrkg1;
        const float zr5 = zrpp * (zrk - zgam1);

        const float zt1 = zrp1 * (za1 + zrkg4);
        const float zt2 = zrm1 * (za1 - zrkg4);
        const float zt3 = zrk2 * (zgam4 + za1 * cz);

        const float zb1 = fminf(zrk * ztau1, 500.0f);
        const float zexm1 = swc_expm(zb1, tbl, false);
        const float zexp1 = 1.0f / zexm1;

        const float zb2 = fminf(ztau1 * snt, 500.0f);
        const float zexm2 = swc_expm(zb2, tbl, false);
        const float zexp2 = 1.0f / zexm2;
        const float ze1r45 = zr4 * zexp1 + zr5 * zexm1;

        if (ze1r45 >= -SWC_EPS1 && ze1r45 <= SWC_EPS1) {
            rb = SWC_EPS1;
            tb = zexm2;
        } else {
            const float zden1 = zssa1 / ze1r45;
            rb = fmaxf(0.0f, fminf(1.0f, (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1));
            tb = fmaxf(0.0f, fminf(1.0f, zexm2 * (1.0f - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2) * zden1)));
        }
        const float zden1d = zr4 / (ze1r45 * zrkg1);
        rd = fmaxf(0.0f, fminf(1.0f, zgam2 * (zexp1 - zexm1) * zden1d));
        td = fmaxf(0.0f, fminf(1.0f, zrk2 * zden1d));
    }
}

// swflux (radsw_main_nmmb.f:2977-3102): up/down-welling fluxes from the level reflectances and
// transmittances. All arrays are 1-based (index 1 = surface side, nlp1 = top). The lmask argument of
// the Fortran is not read by swflux, so it is not carried.
__device__ void swc_swflux(const float* zrefb, const float* zrefd, const float* ztrab, const float* ztrad,
                           const float* zldbt, const float* ztdbt, int nlay, int nlp1,
                           float* zfu, float* zfd)
{
    float zrupb[SWC_LP], zrupd[SWC_LP], zrdnd[SWC_LP], ztdn[SWC_LP];

    // link lowest layer with surface
    zrupb[1] = zrefb[1];
    zrupd[1] = zrefd[1];

    // pass from bottom to top
    for (int k = 1; k <= nlay; ++k) {
        const int kp = k + 1;
        const float zden1 = 1.0f / (1.0f - zrupd[k] * zrefd[kp]);
        zrupb[kp] = zrefb[kp] + (ztrad[kp] * ((ztrab[kp] - zldbt[kp]) * zrupd[k] + zldbt[kp] * zrupb[k])) * zden1;
        zrupd[kp] = zrefd[kp] + ztrad[kp] * ztrad[kp] * zrupd[k] * zden1;
    }

    // upper boundary conditions
    ztdn[nlp1] = 1.0f;
    zrdnd[nlp1] = 0.0f;
    ztdn[nlay] = ztrab[nlp1];
    zrdnd[nlay] = zrefd[nlp1];

    // pass from top to bottom
    for (int k = nlay; k >= 2; --k) {
        const float zden1 = 1.0f / (1.0f - zrefd[k] * zrdnd[k]);
        ztdn[k - 1] = ztdbt[k] * ztrab[k] + (ztrad[k] * ((ztdn[k] - ztdbt[k]) + ztdbt[k] * zrefb[k] * zrdnd[k])) * zden1;
        zrdnd[k - 1] = zrefd[k] + ztrad[k] * ztrad[k] * zrdnd[k] * zden1;
    }

    // up and down-welling fluxes at levels
    for (int k = 1; k <= nlp1; ++k) {
        const float zden1 = 1.0f / (1.0f - zrdnd[k] * zrupd[k]);
        zfu[k] = (ztdbt[k] * zrupb[k] + (ztdn[k] - ztdbt[k]) * zrupd[k]) * zden1;
        zfd[k] = ztdbt[k] + (ztdn[k] - ztdbt[k] + ztdbt[k] * zrupb[k] * zrdnd[k]) * zden1;
    }
}

extern "C" __global__ void phys_rrtm_sw_spcvrtc(
    const float* __restrict__ ssolar,     // [JM,IM]
    const float* __restrict__ cosz,       // [JM,IM]
    const float* __restrict__ sntz,       // [JM,IM]
    const float* __restrict__ albbm,      // [2,JM,IM]
    const float* __restrict__ albdf,      // [2,JM,IM]
    const float* __restrict__ cldfrc,     // [NLAY,JM,IM]
    const float* __restrict__ cf1,        // [JM,IM]
    const float* __restrict__ cf0,        // [JM,IM]
    const float* __restrict__ taug,       // [NLAY,JM,IM]  this g-point
    const float* __restrict__ taur,       // [NLAY,JM,IM]  this g-point
    const float* __restrict__ tauae,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ ssaae,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ asyae,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ taucw,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ ssacw,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ asycw,      // [NLAY,JM,IM]  band of this g-point
    const float* __restrict__ exp_tbl,    // [NTBMX+1]
    float* __restrict__ fxupc,            // [NBDSW,NLP1,JM,IM]
    float* __restrict__ fxdnc,            // [NBDSW,NLP1,JM,IM]
    float* __restrict__ fxup0,            // [NBDSW,NLP1,JM,IM]
    float* __restrict__ fxdn0,            // [NBDSW,NLP1,JM,IM]
    float* __restrict__ ftoadc,           // [JM,IM]
    float* __restrict__ ftoauc,           // [JM,IM]
    float* __restrict__ ftoau0,           // [JM,IM]
    float* __restrict__ fsfcuc,           // [JM,IM]
    float* __restrict__ fsfcu0,           // [JM,IM]
    float* __restrict__ fsfcdc,           // [JM,IM]
    float* __restrict__ fsfcd0,           // [JM,IM]
    float* __restrict__ sfbmc,            // [2,JM,IM]
    float* __restrict__ sfdfc,            // [2,JM,IM]
    float* __restrict__ sfbm0,            // [2,JM,IM]
    float* __restrict__ sfdf0,            // [2,JM,IM]
    float* __restrict__ suvbfc,           // [JM,IM]
    float* __restrict__ suvbf0,           // [JM,IM]
    const float* __restrict__ sfluxzen,   // [NGPTSW,JM,IM] TOA flux per g-point and column (taumol)
    const int jg,                         // 1-based g-point index 1..NGPTSW
    const int ib_g,                       // 1-based band index ib = jb + 1 - nblow (1..14)
    const int ibd_g,                      // idxsfc(jb): 0 (average), 1, or 2
    const int first_g,                    // 1 on the first g-point launch
    const int last_g,                     // 1 on the last g-point launch
    const int nlay,
    const int nlp1,
    const int JM,
    const int IM,
    const int iswmode)
{
    COLUMN_IJ(IM, JM)
    if (nlay > SWC_NLAY_MAX) return;

    const int c2 = IDX2(i0, j0, IM);
    const int ib0 = ib_g - 1;

    // ---- first g-point: zero the column accumulators (spcvrtc, routine entry)
    if (first_g) {
        for (int ib = 0; ib < SWC_NBDSW; ++ib) {
            for (int k = 0; k < nlp1; ++k) {
                const size_t q = IDX4(i0, j0, k, ib, IM, JM, nlp1);
                fxupc[q] = 0.0f;
                fxdnc[q] = 0.0f;
                fxup0[q] = 0.0f;
                fxdn0[q] = 0.0f;
            }
        }
        for (int n = 0; n < 2; ++n) {
            const size_t q = IDX3(i0, j0, n, IM, JM);
            sfbmc[q] = 0.0f;
            sfdfc[q] = 0.0f;
            sfbm0[q] = 0.0f;
            sfdf0[q] = 0.0f;
        }
    }

    const float cz = cosz[c2];
    const float snt = sntz[c2];
    const float cf1v = cf1[c2];
    const float cf0v = cf0[c2];

    // Fortran NDAY set: coszen >= 0.0001 (grrad_nmmb.f:640-644). Other columns keep zero fluxes.
    if (cz >= SWC_DAY_COSZ) {
        float zrefb[SWC_LP], zrefd[SWC_LP], ztrab[SWC_LP], ztrad[SWC_LP];
        float ztdbt[SWC_LP], zldbt[SWC_LP], zfu[SWC_LP], zfd[SWC_LP];
        float ztaus[SWC_LP], zssas[SWC_LP], zasys[SWC_LP], zldbt0[SWC_LP];

        const float zsolar = ssolar[c2] * sfluxzen[IDX3(i0, j0, jg - 1, IM, JM)];
        float ztdbt0 = 1.0f;

        // ---- set up toa direct beam and surface values (beam and diff)
        ztdbt[nlp1] = 1.0f;
        zldbt[1] = 0.0f;
        if (ibd_g != 0) {
            zrefb[1] = albbm[IDX3(i0, j0, ibd_g - 1, IM, JM)];
            zrefd[1] = albdf[IDX3(i0, j0, ibd_g - 1, IM, JM)];
        } else {
            zrefb[1] = 0.5f * (albbm[IDX3(i0, j0, 0, IM, JM)] + albbm[IDX3(i0, j0, 1, IM, JM)]);
            zrefd[1] = 0.5f * (albdf[IDX3(i0, j0, 0, IM, JM)] + albdf[IDX3(i0, j0, 1, IM, JM)]);
        }
        ztrab[1] = 0.0f;
        ztrad[1] = 0.0f;

        // ---- clear-sky optical parameters, layer reflectance and transmittance
        for (int k = nlay; k >= 1; --k) {
            const int kp = k + 1;
            const size_t q = IDX3(i0, j0, k - 1, IM, JM);
            const float tr = taur[q];
            const float tg = taug[q];
            const float ta = tauae[q];
            const float sa = ssaae[q];
            const float ya = asyae[q];

            const float ztau0 = fmaxf(SWC_FTINY, tr + tg + ta);
            const float zssa0 = tr + ta * sa;
            const float zasy0 = ya * sa * ta;
            const float zssaw = fminf(SWC_ONEMINUS, zssa0 / ztau0);
            const float zasyw = zasy0 / fmaxf(SWC_FTINY, zssa0);

            // saving clear-sky quantities for later total-sky usage
            ztaus[k] = ztau0;
            zssas[k] = zssa0;
            zasys[k] = zasy0;

            // delta scaling for clear-sky condition
            const float za1s = zasyw * zasyw;
            const float za2s = zssaw * za1s;
            const float ztau1 = (1.0f - za2s) * ztau0;
            const float zssa1 = (zssaw - za2s) / (1.0f - za2s);
            const float zasy1 = zasyw / (1.0f + zasyw);
            const float zasy3 = 0.75f * zasy1;

            float zgam1, zgam2, zgam3;
            swc_gam(iswmode, zssa1, zasy1, zasy3, cz, zgam1, zgam2, zgam3);
            const float zgam4 = 1.0f - zgam3;

            swc_layer_rt(ztau1, zssaw, zssa1, zgam1, zgam2, zgam3, zgam4, cz, snt, exp_tbl,
                         zrefb[kp], zrefd[kp], ztrab[kp], ztrad[kp]);

            // direct beam transmittance (scaled optical depth)
            float zr1 = ztau1 * snt;
            const float zexp3 = swc_expm(zr1, exp_tbl, true);
            ztdbt[k] = zexp3 * ztdbt[kp];
            zldbt[kp] = zexp3;

            // pre-delta-scaling clear direct beam transmittance (unscaled optical depth)
            zr1 = ztau0 * snt;
            const float zexp4 = swc_expm(zr1, exp_tbl, true);
            zldbt0[k] = zexp4;
            ztdbt0 = zexp4 * ztdbt0;
        }

        // ---- clear-sky vertical quadrature and fluxes
        swc_swflux(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1, zfu, zfd);
        for (int k = 1; k <= nlp1; ++k) {
            const size_t q = IDX4(i0, j0, k - 1, ib0, IM, JM, nlp1);
            fxup0[q] = fxup0[q] + zsolar * zfu[k];
            fxdn0[q] = fxdn0[q] + zsolar * zfd[k];
        }

        // ---- clear-sky surface downward beam/diffuse components
        {
            const float zb1 = zsolar * ztdbt0;
            const float zb2 = zsolar * (zfd[1] - ztdbt0);
            if (ibd_g != 0) {
                const size_t q = IDX3(i0, j0, ibd_g - 1, IM, JM);
                sfbm0[q] = sfbm0[q] + zb1;
                sfdf0[q] = sfdf0[q] + zb2;
            } else {
                const float zf1 = 0.5f * zb1;
                const float zf2 = 0.5f * zb2;
                sfbm0[IDX3(i0, j0, 0, IM, JM)] = sfbm0[IDX3(i0, j0, 0, IM, JM)] + zf1;
                sfdf0[IDX3(i0, j0, 0, IM, JM)] = sfdf0[IDX3(i0, j0, 0, IM, JM)] + zf2;
                sfbm0[IDX3(i0, j0, 1, IM, JM)] = sfbm0[IDX3(i0, j0, 1, IM, JM)] + zf1;
                sfdf0[IDX3(i0, j0, 1, IM, JM)] = sfdf0[IDX3(i0, j0, 1, IM, JM)] + zf2;
            }
        }

        // ---- total-sky optical parameters, only for cf1 > eps (lmask)
        if (cf1v > SWC_EPS) {
            // set up toa direct beam and surface values again for the total-sky pass
            ztdbt0 = 1.0f;
            zldbt[1] = 0.0f;

            for (int k = nlay; k >= 1; --k) {
                const int kp = k + 1;
                const size_t q = IDX3(i0, j0, k - 1, IM, JM);
                const float zc1 = cldfrc[q];
                const float zc0 = 1.0f - zc1;

                if (zc1 > SWC_FTINY) {
                    // cloudy layer
                    const float ztau0 = ztaus[k] + taucw[q];
                    const float zssa0 = zssas[k] + ssacw[q];
                    const float zasy0 = zasys[k] + asycw[q];
                    const float zssaw = fminf(SWC_ONEMINUS, zssa0 / ztau0);
                    const float zasyw = zasy0 / fmaxf(SWC_FTINY, zssa0);

                    // delta scaling for total-sky condition
                    const float za1s = zasyw * zasyw;
                    const float za2s = zssaw * za1s;
                    const float ztau1 = (1.0f - za2s) * ztau0;
                    const float zssa1 = (zssaw - za2s) / (1.0f - za2s);
                    const float zasy1 = zasyw / (1.0f + zasyw);
                    const float zasy3 = 0.75f * zasy1;

                    float zgam1, zgam2, zgam3;
                    swc_gam(iswmode, zssa1, zasy1, zasy3, cz, zgam1, zgam2, zgam3);
                    const float zgam4 = 1.0f - zgam3;

                    const float zrefb1 = zrefb[kp];
                    const float zrefd1 = zrefd[kp];
                    const float ztrab1 = ztrab[kp];
                    const float ztrad1 = ztrad[kp];

                    swc_layer_rt(ztau1, zssaw, zssa1, zgam1, zgam2, zgam3, zgam4, cz, snt, exp_tbl,
                                 zrefb[kp], zrefd[kp], ztrab[kp], ztrad[kp]);

                    // combine clear and cloudy contributions for total sky
                    zrefb[kp] = zc0 * zrefb1 + zc1 * zrefb[kp];
                    zrefd[kp] = zc0 * zrefd1 + zc1 * zrefd[kp];
                    ztrab[kp] = zc0 * ztrab1 + zc1 * ztrab[kp];
                    ztrad[kp] = zc0 * ztrad1 + zc1 * ztrad[kp];

                    // direct beam transmittance
                    float zr1 = ztau1 * snt;
                    const float zexp3 = swc_expm(zr1, exp_tbl, true);
                    zldbt[kp] = zc0 * zldbt[kp] + zc1 * zexp3;
                    ztdbt[k] = zldbt[kp] * ztdbt[kp];

                    // pre-delta-scaling clear and cloudy direct beam transmittance (unscaled tau)
                    zr1 = ztau0 * snt;
                    const float zexp4 = swc_expm(zr1, exp_tbl, true);
                    ztdbt0 = (zc0 * zldbt0[k] + zc1 * zexp4) * ztdbt0;
                } else {
                    // clear layer
                    ztdbt[k] = zldbt[kp] * ztdbt[kp];
                    ztdbt0 = zldbt0[k] * ztdbt0;
                }
            }

            // perform vertical quadrature
            swc_swflux(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1, zfu, zfd);

            for (int k = 1; k <= nlp1; ++k) {
                const size_t q = IDX4(i0, j0, k - 1, ib0, IM, JM, nlp1);
                fxupc[q] = fxupc[q] + zsolar * zfu[k];
                fxdnc[q] = fxdnc[q] + zsolar * zfd[k];
            }

            // total-sky surface downward beam/diffuse components
            {
                const float zb1 = zsolar * ztdbt0;
                const float zb2 = zsolar * (zfd[1] - ztdbt0);
                if (ibd_g != 0) {
                    const size_t q = IDX3(i0, j0, ibd_g - 1, IM, JM);
                    sfbmc[q] = sfbmc[q] + zb1;
                    sfdfc[q] = sfdfc[q] + zb2;
                } else {
                    const float zf1 = 0.5f * zb1;
                    const float zf2 = 0.5f * zb2;
                    sfbmc[IDX3(i0, j0, 0, IM, JM)] = sfbmc[IDX3(i0, j0, 0, IM, JM)] + zf1;
                    sfdfc[IDX3(i0, j0, 0, IM, JM)] = sfdfc[IDX3(i0, j0, 0, IM, JM)] + zf2;
                    sfbmc[IDX3(i0, j0, 1, IM, JM)] = sfbmc[IDX3(i0, j0, 1, IM, JM)] + zf1;
                    sfdfc[IDX3(i0, j0, 1, IM, JM)] = sfdfc[IDX3(i0, j0, 1, IM, JM)] + zf2;
                }
            }
        }
    }

    // ---- last g-point: column end section of spcvrtc (band sums, cf1 <= eps copy, cf1 weighting)
    if (last_g) {
        float s_adc = 0.0f, s_au0 = 0.0f, s_su0 = 0.0f, s_sd0 = 0.0f;
        for (int ib = 0; ib < SWC_NBDSW; ++ib) {
            s_adc = s_adc + fxdn0[IDX4(i0, j0, nlp1 - 1, ib, IM, JM, nlp1)];
            s_au0 = s_au0 + fxup0[IDX4(i0, j0, nlp1 - 1, ib, IM, JM, nlp1)];
            s_su0 = s_su0 + fxup0[IDX4(i0, j0, 0, ib, IM, JM, nlp1)];
            s_sd0 = s_sd0 + fxdn0[IDX4(i0, j0, 0, ib, IM, JM, nlp1)];
        }
        ftoadc[c2] = s_adc;
        ftoau0[c2] = s_au0;
        fsfcu0[c2] = s_su0;
        fsfcd0[c2] = s_sd0;

        // uv-b surface downward flux
        suvbf0[c2] = fxdn0[IDX4(i0, j0, 0, SWC_IBUV0, IM, JM, nlp1)];
        suvbfc[c2] = suvbf0[c2];

        if (cf1v <= SWC_EPS) {
            // clear column: total-sky fluxes equal clear-sky fluxes
            for (int ib = 0; ib < SWC_NBDSW; ++ib) {
                for (int k = 0; k < nlp1; ++k) {
                    const size_t q = IDX4(i0, j0, k, ib, IM, JM, nlp1);
                    fxupc[q] = fxup0[q];
                    fxdnc[q] = fxdn0[q];
                }
            }
            ftoauc[c2] = ftoau0[c2];
            fsfcuc[c2] = fsfcu0[c2];
            fsfcdc[c2] = fsfcd0[c2];
            sfbmc[IDX3(i0, j0, 0, IM, JM)] = sfbm0[IDX3(i0, j0, 0, IM, JM)];
            sfdfc[IDX3(i0, j0, 0, IM, JM)] = sfdf0[IDX3(i0, j0, 0, IM, JM)];
            sfbmc[IDX3(i0, j0, 1, IM, JM)] = sfbm0[IDX3(i0, j0, 1, IM, JM)];
            sfdfc[IDX3(i0, j0, 1, IM, JM)] = sfdf0[IDX3(i0, j0, 1, IM, JM)];
        } else {
            // cloudy column: total-sky = cf1 * total-sky + cf0 * clear-sky
            for (int ib = 0; ib < SWC_NBDSW; ++ib) {
                for (int k = 0; k < nlp1; ++k) {
                    const size_t q = IDX4(i0, j0, k, ib, IM, JM, nlp1);
                    fxupc[q] = cf1v * fxupc[q] + cf0v * fxup0[q];
                    fxdnc[q] = cf1v * fxdnc[q] + cf0v * fxdn0[q];
                }
            }
            float t_au = 0.0f, t_su = 0.0f, t_sd = 0.0f;
            for (int ib = 0; ib < SWC_NBDSW; ++ib) {
                t_au = t_au + fxupc[IDX4(i0, j0, nlp1 - 1, ib, IM, JM, nlp1)];
                t_su = t_su + fxupc[IDX4(i0, j0, 0, ib, IM, JM, nlp1)];
                t_sd = t_sd + fxdnc[IDX4(i0, j0, 0, ib, IM, JM, nlp1)];
            }
            ftoauc[c2] = t_au;
            fsfcuc[c2] = t_su;
            fsfcdc[c2] = t_sd;
            suvbfc[c2] = fxdnc[IDX4(i0, j0, 0, SWC_IBUV0, IM, JM, nlp1)];
            {
                const size_t q1 = IDX3(i0, j0, 0, IM, JM);
                const size_t q2 = IDX3(i0, j0, 1, IM, JM);
                sfbmc[q1] = cf1v * sfbmc[q1] + cf0v * sfbm0[q1];
                sfbmc[q2] = cf1v * sfbmc[q2] + cf0v * sfbm0[q2];
                sfdfc[q1] = cf1v * sfdfc[q1] + cf0v * sfdf0[q1];
                sfdfc[q2] = cf1v * sfdfc[q2] + cf0v * sfdf0[q2];
            }
        }
    }
}
