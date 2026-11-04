// phys_rrtm_lw_rtrn.cu -- rtrn (RRTM longwave clear/cloudy transfer), NAM NMMB port.
//
// NOAA notice, phys/radlw_main_nmmb.f header (nam.v4.2.11, 83df4e13c1):
// !!!!!  ==============================================================  !!!!!
// !!!!!               lw-rrtm3 radiation package description             !!!!!
// !!!!!  ==============================================================  !!!!!
// !                                                                          !
// !   this package includes ncep's modifications of the rrtm-lw radiation    !
// !   code from aer inc.                                                     !
// !                                                                          !
// !    the lw-rrtm3 package includes these parts:                            !
// !                                                                          !
// !       'radlw_rrtm3_param.f'                                              !
// !       'radlw_rrtm3_datatb.f'                                             !
// !       'radlw_rrtm3_main.f'                                               !
// !                                                                          !
// !    the 'radlw_rrtm3_param.f' contains:                                   !
// !                                                                          !
// !       'module_radlw_parameters'  -- band parameters set up               !
// !                                                                          !
// !    the 'radlw_rrtm3_datatb.f' contains:                                  !
// !                                                                          !
// !       'module_radlw_avplank'     -- plank flux data                      !
// !       'module_radlw_ref'         -- reference temperature and pressure   !
// !       'module_radlw_cldprlw'     -- cloud property coefficients          !
// !       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
// !                                     bands, where nn = 01-16              !
// !                                                                          !
// !    the 'radlw_rrtm3_main.f' contains:                                    !
// !                                                                          !
// !       'module_radlw_main'        -- main lw radiation transfer           !
// !                                                                          !
// !    in the main module 'module_radlw_main' there are only two             !
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:2127-2484 (subroutine rtrn), nam.v4.2.11,
// 83df4e13c1. Notice: NOAA/NCEP NAM, US government work (see NOTICE). Header block of the Fortran
// file kept above as the notice.
//
// What it computes: clear and cloudy longwave fluxes and heating rates for one column, one g-point
// at a time: downward sweep, surface emission and reflection, upward sweep, with the cloud fraction
// weighting of each layer (random overlap, as in the Fortran rtrn).
//
// Layout (CONTRACT.md sections 1 and 4). The Fortran works in RRTM order (layer 1 = surface), which
// lwrad builds from the NAM top-down arrays with k1 = nlp1 - k (radlw_main_nmmb.f, ivflip == 0 branch).
// The same mapping is applied here in one place:
//   Fortran layer k (1..nlay)     -> NAM slot l0 = LM - k        (slot 0 = model top)
//   Fortran interface m (0..nlay) -> NAM slot    LM - m          (interface arrays are LMK = LM+1 deep)
//   so pklev(ib,k) -> slot l0, pklev(ib,k-1) -> slot l0+1, pklay(ib,0) -> slot LM, fracs(ig,1) -> LM-1.
// Arrays (float32 unless noted; IDX3/IDX4 from nmmb_common.cuh):
//   semiss [NBANDS, JM, IM]             delp [LM, JM, IM]  layer pressure thickness in MB (hPa), the
//                                       Fortran rtrn unit. htr = (flux difference) * heatfac / delp, and
//                                       heatfac = g*1e-2/cp only converts mb to Pa for delp in mb.
//   cldfrc [LM, JM, IM]                 layer values; rtrn reads cldfrc(k) for k = 1..nlay only
//   taucld [NBANDS, LM, JM, IM]         tautot [NGPTLW, LM, JM, IM]
//   pklay, pklev [NBANDS, LMK, JM, IM]  fracs [NGPTLW, LM, JM, IM]   secdif [NBANDS, JM, IM]
//   ngb int32 [NGPTLW]                  band index of each g-point (1-based, as Fortran)
//   tau_tbl, exp_tbl, tfn_tbl [NTBL+1]  built by phys-rrtm-lw-rlwinit (NTBL = 10000)
//   outputs: totuflux, totdflux, totuclfl, totdclfl [LMK, JM, IM]; htr, htrcl [LM, JM, IM];
//            htrb [NBANDS, LM, JM, IM] (written only when lhlwb != 0).
//
// Summation order is the Fortran order: per band, its g-points in ig order; band sums are then added
// to the totals in band order (totuflux(k) = sum over ib of toturad(k,ib)), then scaled by
// flxfac = wtdiff * fluxfac. Both lookup-table branches (odepth <= 0.06 or table; odtot < 0.06 or
// table) are kept.
//
// Per-thread scratch: four band accumulators [LMAX+1] (toturad, totdrad, clrurad, clrdrad for the band
// being summed). The per-g layer quantities the Fortran keeps in trngas, gassrcu, totsrcu and efclrfr
// are recomputed for the upward sweep by rtrn_layer from the same inputs: same values, no per-g array.
//
// Kernel phys_rrtm_lw_rtrn: one thread per (i, j) column (kernels.grid_cols); the level and g loops run
// inside the thread. No host sync, no malloc, no print.
// Fast numerics: float throughout, FMA contraction on (NVRTC --fmad=true).

#include "nmmb_common.cuh"

// radlw_param_nmmb.f:140-141 NBANDS = 16, NGPTLW = 140; radlw_param.f:96 NTBL = 10000.
#define RTRN_NBANDS 16
#define RTRN_NGPTLW 140
#define RTRN_NTBL   10000
// Largest LM the per-thread band accumulators hold (NAM 12 km parent: LM = 60, CONTRACT section 5).
#define RTRN_LMAX   60

// radlw_main_nmmb.f:267 eps = 1.0e-6; :270 bpade = 1.0/0.278; :273 tblint = ntbl; :280 wtdiff = 0.5;
// :2258 rec_6 = 0.166667 (rtrn local parameter).
#define RTRN_EPS    1.0e-6f
#define RTRN_BPADE  (1.0f / 0.278f)
#define RTRN_TBLINT 10000.0f
#define RTRN_WTDIFF 0.5f
#define RTRN_REC_6  0.166667f

// Per-layer quantities of Fortran layer k (1 = bottom, RRTM order) for g-point ig in band ib: the gas
// part (radlw_main_nmmb.f:2302-2328) and the cloud part (2334-2352). The downward sweep and the upward
// sweep both take their values from here (the upward one uses trngas, gassrcu, totsrcu, efclrfr of the
// same g, which are the same numbers the downward pass computed).
__device__ __forceinline__ void rtrn_layer(
    int k, int ig, int ib, int i0, int j0, int IM, int JM, int LM,
    const float* __restrict__ tautot, const float* __restrict__ fracs,
    const float* __restrict__ taucld, const float* __restrict__ cldfrc,
    const float* __restrict__ secdif, const float* __restrict__ pklay,
    const float* __restrict__ pklev,
    const float* __restrict__ tau_tbl, const float* __restrict__ exp_tbl,
    const float* __restrict__ tfn_tbl,
    float& trng, float& gassrcd, float& gassrcu, float& efclrfr,
    float& totsrcd, float& totsrcu, bool& cloudy, float& clfr)
{
    const int LMK = LM + 1;
    const int sl  = LM - k;                       // NAM slot of Fortran layer k

    // gas contribution (radlw_main_nmmb.f:2302-2320)
    float odepth = fmaxf(0.0f, secdif[IDX3(i0, j0, ib, IM, JM)] *
                               tautot[IDX4(i0, j0, sl, ig, IM, JM, LM)]);
    float atrgas, gasfac;
    if (odepth <= 0.06f) {
        atrgas = odepth - 0.5f * odepth * odepth;
        trng   = 1.0f - atrgas;
        gasfac = RTRN_REC_6 * odepth;
    } else {
        const float tblind = odepth / (RTRN_BPADE + odepth);
        const int   itgas  = (int)(RTRN_TBLINT * tblind + 0.5f);
        trng   = exp_tbl[itgas];
        atrgas = 1.0f - trng;
        gasfac = tfn_tbl[itgas];
        odepth = tau_tbl[itgas];
    }

    // Planck terms (radlw_main_nmmb.f:2324-2331)
    const float plfrac = fracs[IDX4(i0, j0, sl, ig, IM, JM, LM)];
    const float blay   = pklay[IDX4(i0, j0, sl, ib, IM, JM, LMK)];
    const float dplnku = pklev[IDX4(i0, j0, sl,     ib, IM, JM, LMK)] - blay;   // pklev(ib,k)
    const float dplnkd = pklev[IDX4(i0, j0, sl + 1, ib, IM, JM, LMK)] - blay;   // pklev(ib,k-1)
    const float bbdgas = plfrac * (blay + dplnkd * gasfac);
    const float bbugas = plfrac * (blay + dplnku * gasfac);
    gassrcd = bbdgas * atrgas;
    gassrcu = bbugas * atrgas;

    // cloud contribution (radlw_main_nmmb.f:2334-2352)
    clfr    = cldfrc[IDX3(i0, j0, sl, IM, JM)];
    cloudy  = (clfr >= RTRN_EPS);
    efclrfr = 1.0f;
    totsrcd = 0.0f;
    totsrcu = 0.0f;
    if (cloudy) {
        const float odcld = secdif[IDX3(i0, j0, ib, IM, JM)] *
                            taucld[IDX4(i0, j0, sl, ib, IM, JM, LM)];
        efclrfr = 1.0f - (1.0f - expf(-odcld)) * clfr;
        const float odtot = odepth + odcld;
        float totfac, atrtot;
        if (odtot < 0.06f) {
            totfac = RTRN_REC_6 * odtot;
            atrtot = odtot - 0.5f * odtot * odtot;
        } else {
            const float tblind = odtot / (RTRN_BPADE + odtot);
            const int   ittot  = (int)(RTRN_TBLINT * tblind + 0.5f);
            totfac = tfn_tbl[ittot];
            atrtot = 1.0f - exp_tbl[ittot];
        }
        const float bbdtot = plfrac * (blay + dplnkd * totfac);
        const float bbutot = plfrac * (blay + dplnku * totfac);
        totsrcd = bbdtot * atrtot;
        totsrcu = bbutot * atrtot;
    }
}

extern "C" __global__ void phys_rrtm_lw_rtrn(
    const float* __restrict__ semiss,
    const float* __restrict__ delp,
    const float* __restrict__ cldfrc,
    const float* __restrict__ taucld,
    const float* __restrict__ tautot,
    const float* __restrict__ pklay,
    const float* __restrict__ pklev,
    const float* __restrict__ fracs,
    const float* __restrict__ secdif,
    const int*   __restrict__ ngb,
    const float* __restrict__ tau_tbl,
    const float* __restrict__ exp_tbl,
    const float* __restrict__ tfn_tbl,
    float* __restrict__ totuflux,
    float* __restrict__ totdflux,
    float* __restrict__ totuclfl,
    float* __restrict__ totdclfl,
    float* __restrict__ htr,
    float* __restrict__ htrcl,
    float* __restrict__ htrb,
    const float fluxfac,
    const float heatfac,
    const int   lhlw0,
    const int   lhlwb,
    const int   IM,
    const int   JM,
    const int   LM)
{
    COLUMN_IJ(IM, JM)
    if (LM > RTRN_LMAX) return;                   // the wrapper refuses LM > RTRN_LMAX before launch

    const int   LMK    = LM + 1;
    const float flxfac = RTRN_WTDIFF * fluxfac;   // radlw_main_nmmb.f:2436 flxfac = wtdiff*fluxfac

    // totals start at f_zero (radlw_main_nmmb.f:2146-2154)
    for (int s = 0; s <= LM; ++s) {
        totuflux[IDX3(i0, j0, s, IM, JM)] = 0.0f;
        totdflux[IDX3(i0, j0, s, IM, JM)] = 0.0f;
        totuclfl[IDX3(i0, j0, s, IM, JM)] = 0.0f;
        totdclfl[IDX3(i0, j0, s, IM, JM)] = 0.0f;
    }

    // band accumulators, Fortran index m = 0..nlay (toturad(m,ib), totdrad, clrurad, clrdrad)
    float bt_u[RTRN_LMAX + 1], bt_d[RTRN_LMAX + 1], bc_u[RTRN_LMAX + 1], bc_d[RTRN_LMAX + 1];

    // Fortran: do ig = 1, ngptlw; ib = ngb(ig). The g-points are visited band by band (ngb is
    // nondecreasing), keeping ig order inside each band; band sums are added in band order.
    for (int ib = 0; ib < RTRN_NBANDS; ++ib) {
        for (int m = 0; m <= LM; ++m) {
            bt_u[m] = 0.0f; bt_d[m] = 0.0f; bc_u[m] = 0.0f; bc_d[m] = 0.0f;
        }
        const float semis  = semiss[IDX3(i0, j0, ib, IM, JM)];   // semiss(ib)
        const float reflct = 1.0f - semis;                        // radlw_main_nmmb.f:2376

        for (int ig = 0; ig < RTRN_NGPTLW; ++ig) {
            if (ngb[ig] != ib + 1) continue;

            // downward transfer, k = nlay .. 1 (radlw_main_nmmb.f:2300-2360)
            float radtotd = 0.0f, radclrd = 0.0f;
            for (int k = LM; k >= 1; --k) {
                float trng, gassrcd, gassrcu, efclrfr, totsrcd, totsrcu, clfr;
                bool cloudy;
                rtrn_layer(k, ig, ib, i0, j0, IM, JM, LM, tautot, fracs, taucld, cldfrc,
                           secdif, pklay, pklev, tau_tbl, exp_tbl, tfn_tbl,
                           trng, gassrcd, gassrcu, efclrfr, totsrcd, totsrcu, cloudy, clfr);
                if (cloudy) {
                    radtotd = radtotd * trng * efclrfr + gassrcd + clfr * (totsrcd - gassrcd);
                    bt_d[k - 1] += radtotd;
                    radclrd = radclrd * trng + gassrcd;
                    bc_d[k - 1] += radclrd;
                } else {
                    radtotd = radtotd * trng + gassrcd;
                    bt_d[k - 1] += radtotd;
                    radclrd = radclrd * trng + gassrcd;
                    bc_d[k - 1] += radclrd;
                }
            }

            // surface emission and reflection (radlw_main_nmmb.f:2377-2384)
            // rad0 = semiss(ib) * fracs(ig,1) * pklay(ib,0): layer 1 is slot LM-1, interface 0 is slot LM
            const float rad0 = semis * fracs[IDX4(i0, j0, LM - 1, ig, IM, JM, LM)]
                                     * pklay[IDX4(i0, j0, LM, ib, IM, JM, LMK)];
            float radtotu = rad0 + reflct * radtotd;
            bt_u[0] += radtotu;
            float radclru = rad0 + reflct * radclrd;
            bc_u[0] += radclru;

            // upward transfer, k = 1 .. nlay (radlw_main_nmmb.f:2387-2440)
            for (int k = 1; k <= LM; ++k) {
                float trng, gassrcd, gassrcu, efclrfr, totsrcd, totsrcu, clfr;
                bool cloudy;
                rtrn_layer(k, ig, ib, i0, j0, IM, JM, LM, tautot, fracs, taucld, cldfrc,
                           secdif, pklay, pklev, tau_tbl, exp_tbl, tfn_tbl,
                           trng, gassrcd, gassrcu, efclrfr, totsrcd, totsrcu, cloudy, clfr);
                if (cloudy) {
                    radtotu = radtotu * trng * efclrfr + gassrcu + clfr * (totsrcu - gassrcu);
                    bt_u[k] += radtotu;
                    radclru = radclru * trng + gassrcu;
                    bc_u[k] += radclru;
                } else {
                    radtotu = radtotu * trng + gassrcu;
                    bt_u[k] += radtotu;
                    radclru = radclru * trng + gassrcu;
                    bc_u[k] += radclru;
                }
            }
        } // end ig

        // add this band to the totals (band order per level, radlw_main_nmmb.f:2446-2452);
        // Fortran index m sits in NAM slot LM - m
        for (int m = 0; m <= LM; ++m) {
            const int s = LM - m;
            totuflux[IDX3(i0, j0, s, IM, JM)] += bt_u[m];
            totdflux[IDX3(i0, j0, s, IM, JM)] += bt_d[m];
            totuclfl[IDX3(i0, j0, s, IM, JM)] += bc_u[m];
            totdclfl[IDX3(i0, j0, s, IM, JM)] += bc_d[m];
        }

        // spectral band heating rates, lhlwb block (radlw_main_nmmb.f:2472-2480)
        if (lhlwb) {
            float fnet_prev = (bt_u[0] - bt_d[0]) * flxfac;
            for (int k = 1; k <= LM; ++k) {
                const float fnet_k = (bt_u[k] - bt_d[k]) * flxfac;
                const float rfdelp = heatfac / delp[IDX3(i0, j0, LM - k, IM, JM)];
                htrb[IDX4(i0, j0, LM - k, ib, IM, JM, LM)] = (fnet_prev - fnet_k) * rfdelp;
                fnet_prev = fnet_k;
            }
        }
    } // end ib

    // flux scaling (radlw_main_nmmb.f:2446-2452): totuflux(k) = totuflux(k) * flxfac, all four totals
    for (int s = 0; s <= LM; ++s) {
        totuflux[IDX3(i0, j0, s, IM, JM)] *= flxfac;
        totdflux[IDX3(i0, j0, s, IM, JM)] *= flxfac;
        totuclfl[IDX3(i0, j0, s, IM, JM)] *= flxfac;
        totdclfl[IDX3(i0, j0, s, IM, JM)] *= flxfac;
    }

    // net fluxes and heating rates (radlw_main_nmmb.f:2456-2465): fnet(k) = totuflux(k) - totdflux(k),
    // htr(k) = (fnet(k-1) - fnet(k)) * heatfac / delp(k); Fortran k-1 is the NAM slot above it, LM-k+1
    {
        float fnet_prev = totuflux[IDX3(i0, j0, LM, IM, JM)] - totdflux[IDX3(i0, j0, LM, IM, JM)];
        for (int k = 1; k <= LM; ++k) {
            const int   s      = LM - k;
            const float rfdelp = heatfac / delp[IDX3(i0, j0, s, IM, JM)];
            const float fnet_k = totuflux[IDX3(i0, j0, s, IM, JM)] - totdflux[IDX3(i0, j0, s, IM, JM)];
            htr[IDX3(i0, j0, s, IM, JM)] = (fnet_prev - fnet_k) * rfdelp;
            fnet_prev = fnet_k;
        }
    }

    // clear-sky heating rates (radlw_main_nmmb.f:2462-2470, lhlw0)
    if (lhlw0) {
        float fnet_prev = totuclfl[IDX3(i0, j0, LM, IM, JM)] - totdclfl[IDX3(i0, j0, LM, IM, JM)];
        for (int k = 1; k <= LM; ++k) {
            const int   s      = LM - k;
            const float rfdelp = heatfac / delp[IDX3(i0, j0, s, IM, JM)];
            const float fnet_k = totuclfl[IDX3(i0, j0, s, IM, JM)] - totdclfl[IDX3(i0, j0, s, IM, JM)];
            htrcl[IDX3(i0, j0, s, IM, JM)] = (fnet_prev - fnet_k) * rfdelp;
            fnet_prev = fnet_k;
        }
    }
}
