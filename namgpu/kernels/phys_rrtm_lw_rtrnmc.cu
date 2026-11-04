// phys_rrtm_lw_rtrnmc.cu -- nam-gpu unit phys-rrtm-lw-rtrnmc: subroutine rtrnmc, the McICA
// longwave radiative transfer (upward/downward fluxes, heating rates, clear and total sky).
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). Header of
// phys/radlw_main_nmmb.f (lw-rrtm3 package, NCEP modifications of RRTM-LW from AER; nam.v4.2.11,
// 83df4e13c1), kept as the notice:
//   lw-rrtm3 radiation package description. this package includes ncep's modifications of the
//   rrtm-lw radiation code from aer inc. (radlw_main_nmmb.f, subroutine rtrnmc, original code
//   description: e. j. mlawer et al. rrtm_v3.0; revision for gcms michael j. iacono 2002; f90 2006).
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:3059-3418 (nam.v4.2.11, 83df4e13c1).
//
// Fortran line ranges covered:
//   3059-3088  argument list                          -> kernel signature (column-flattened)
//   3109-3123  band/level accumulators = 0            -> per-band accumulators zeroed per band
//   3125-3140  flux totals = 0                        -> output arrays zeroed (used as totals)
//   3142-3155  ig loop head, ib = ngb(ig), radtotd=0  -> band-major loop, ig ascending in band
//   3157-3228  downward loop, clear-sky gas terms     -> downward gas block
//   3230-3266  downward loop, cloudy-layer branch     -> cloudy branch (clfm >= eps)
//   3267-3283  downward loop, clear-layer branch      -> clear branch
//   3288-3300  reflectance, rad0, surface radiance    -> surface block
//   3302-3346  upward loop, cloudy and clear branches -> upward loop (clfm > eps)
//   3349-3352  end of ig loop                         -> band accumulation into totals
//   3357-3376  flxfac, totals to fluxes (w/m2)        -> flux totals scaled by flxfac
//   3381-3394  net flux, htr                          -> htr block (rfdelp = heatfac/delp)
//   3399-3407  lhlw0 clear-sky heating rates          -> htrcl block (lhlw0 flag)
//   3412-3418  lhlwb spectral band heating rates      -> htrb block (lhlwb flag), per band
//
// Order of summation (matches the Fortran): for each band ib in 1..NBANDS the g-points of that
// band are summed in ascending ig order into the band accumulators; the band accumulator is then
// added into the flux totals in band order. This is the Fortran order for toturad(k,ib) over ig
// and for totuflux(k) over ib. The arithmetic is kept term by term.
//
// Layout (CONTRACT section 1, column index last). Every per-column Fortran array with dims
// (d1, ..., dn) is stored as [dn, ..., d1, NCOL] in C order, i.e. the CONTRACT reversal of the
// Fortran index tuple, with the flattened column axis last:
//   semiss  (nbands)              -> [NBANDS, NCOL]
//   secdif  (nbands)              -> [NBANDS, NCOL]
//   delp    (nlay)                -> [NLAY, NCOL]
//   cldfmc  (ngptlw, nlay)        -> [NLAY, NGPTLW, NCOL]
//   tautot  (ngptlw, nlay)        -> [NLAY, NGPTLW, NCOL]
//   fracs   (ngptlw, nlay)        -> [NLAY, NGPTLW, NCOL]
//   taucld  (nbands, nlay)        -> [NLAY, NBANDS, NCOL]
//   pklay   (nbands, 0:nlay)      -> [NLAY+1, NBANDS, NCOL]  (index = Fortran level 0..nlay)
//   pklev   (nbands, 0:nlay)      -> [NLAY+1, NBANDS, NCOL]
//   totuflux, totdflux, totuclfl, totdclfl (0:nlay) -> [NLAY+1, NCOL]
//   htr, htrcl (nlay)             -> [NLAY, NCOL]
//   htrb    (nlay, nbands)        -> [NBANDS, NLAY, NCOL]
//   ngb     (ngptlw)              -> [NGPTLW] int32 (module table, built by the params unit)
//   tau_tbl, exp_tbl, tfn_tbl (0:ntbl) -> [NTBL+1] float32 (module tables, built at init)
// The column axis is a flat chunk of (i, j) columns, so one launch covers ncol columns and the
// wrapper slices the domain into chunks (see the wrapper for the reason).
//
// Module constants (radlw_main_nmmb.f:267-275, radlw_param.f:94-96), single precision:
//   eps = 1.0e-6, bpade = 1/0.278, wtdiff = 0.5, tblint = ntbl = 10000, f_zero = 0, f_one = 1,
//   rec_6 = 0.166667 (the same literal at 2258, 2619, 3190).
// fluxfac = pi * 2e4 and heatfac (module values set at radlw_main_nmmb.f:1287-1295) are scalars
// passed in. flxfac = wtdiff * fluxfac is formed here, as the Fortran does.
//
// Fortran integer truncation: itgas and ittot are real-to-integer assignments of
// tblint*tblind + 0.5, which truncate toward zero; the kernel uses (int)(x + 0.5f), the same for
// x >= 0 (tblind >= 0 because odepth >= 0).
//
// Fast numerics: float (kind_phys single), NVRTC --fmad=true. No host sync, no printf, no device
// malloc. Per-thread scratch is local: four [NLAY] arrays per g-point and four [NLAY+1] band
// accumulators. nlay must be <= RRTM_LW_MAXLAY (checked by the wrapper).
//
// Halo / MPI / OpenMP: none (single card; the Fortran ENABLE_SMP branch is dropped).
#include "nmmb_common.cuh"

#define RRTM_LW_NBANDS   16
#define RRTM_LW_NGPTLW   140
#define RRTM_LW_NTBL     10000
#define RRTM_LW_MAXLAY   128

#define RRTM_LW_EPS      1.0e-6f
#define RRTM_LW_BPADE    (1.0f / 0.278f)
#define RRTM_LW_WTDIFF   0.5f
#define RRTM_LW_TBLINT   10000.0f
#define RRTM_LW_REC6     0.166667f

extern "C" __global__ void rrtm_lw_rtrnmc(
    const int ncol, const int nlay,
    const int lhlw0, const int lhlwb,
    const float fluxfac, const float heatfac,
    const float* __restrict__ semiss,   // [NBANDS, NCOL]
    const float* __restrict__ delp,     // [NLAY, NCOL]
    const float* __restrict__ cldfmc,   // [NLAY, NGPTLW, NCOL]
    const float* __restrict__ taucld,   // [NLAY, NBANDS, NCOL]
    const float* __restrict__ tautot,   // [NLAY, NGPTLW, NCOL]
    const float* __restrict__ pklay,    // [NLAY+1, NBANDS, NCOL]
    const float* __restrict__ pklev,    // [NLAY+1, NBANDS, NCOL]
    const float* __restrict__ fracs,    // [NLAY, NGPTLW, NCOL]
    const float* __restrict__ secdif,   // [NBANDS, NCOL]
    const int* __restrict__ ngb,        // [NGPTLW]
    const float* __restrict__ tau_tbl,  // [NTBL+1]
    const float* __restrict__ exp_tbl,  // [NTBL+1]
    const float* __restrict__ tfn_tbl,  // [NTBL+1]
    float* __restrict__ totuflux,       // [NLAY+1, NCOL]
    float* __restrict__ totdflux,       // [NLAY+1, NCOL]
    float* __restrict__ htr,            // [NLAY, NCOL]
    float* __restrict__ totuclfl,       // [NLAY+1, NCOL]
    float* __restrict__ totdclfl,       // [NLAY+1, NCOL]
    float* __restrict__ htrcl,          // [NLAY, NCOL]
    float* __restrict__ htrb)           // [NBANDS, NLAY, NCOL]
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const size_t nc = (size_t)ncol;

    // Per-g scratch (Fortran gassrcu, totsrcu, trngas, efclrfr; index k-1 for level k).
    float gassrcu[RRTM_LW_MAXLAY];
    float totsrcu[RRTM_LW_MAXLAY];
    float trngas[RRTM_LW_MAXLAY];
    float efclrfr[RRTM_LW_MAXLAY];
    // Per-band accumulators (Fortran toturad, totdrad, clrurad, clrdrad; index 0..nlay).
    float tur[RRTM_LW_MAXLAY + 1];
    float tdr[RRTM_LW_MAXLAY + 1];
    float cur[RRTM_LW_MAXLAY + 1];
    float cdr[RRTM_LW_MAXLAY + 1];

    const float flxfac = RRTM_LW_WTDIFF * fluxfac;     // Fortran: flxfac = wtdiff * fluxfac

    // Fortran 3125-3140: flux totals start at zero (they are summed over bands below).
    for (int k = 0; k <= nlay; ++k) {
        totuflux[(size_t)k * nc + c] = 0.0f;
        totdflux[(size_t)k * nc + c] = 0.0f;
        totuclfl[(size_t)k * nc + c] = 0.0f;
        totdclfl[(size_t)k * nc + c] = 0.0f;
    }

    // Band-major loop: for each band, the g-points of that band are visited in ascending ig.
    for (int ib = 1; ib <= RRTM_LW_NBANDS; ++ib) {
        const int bi = ib - 1;
        const float semiss_b = semiss[(size_t)bi * nc + c];
        const float secdif_b = secdif[(size_t)bi * nc + c];

        for (int k = 0; k <= nlay; ++k) {
            tur[k] = 0.0f;
            tdr[k] = 0.0f;
            cur[k] = 0.0f;
            cdr[k] = 0.0f;
        }

        for (int ig = 1; ig <= RRTM_LW_NGPTLW; ++ig) {
            if (ngb[ig - 1] != ib) continue;
            const int g = ig - 1;

            float radtotd = 0.0f;
            float radclrd = 0.0f;

            // ---- downward radiative transfer loop (Fortran k = nlay..1) ----
            for (int k = nlay; k >= 1; --k) {
                // clear sky, gases contribution
                float odepth = fmaxf(0.0f, secdif_b * tautot[((size_t)(k - 1) * RRTM_LW_NGPTLW + g) * nc + c]);
                float atrgas, trng, gasfac;
                if (odepth <= 0.06f) {
                    atrgas = odepth - 0.5f * odepth * odepth;
                    trng = 1.0f - atrgas;
                    gasfac = RRTM_LW_REC6 * odepth;
                } else {
                    const float tblind = odepth / (RRTM_LW_BPADE + odepth);
                    const int itgas = (int)(RRTM_LW_TBLINT * tblind + 0.5f);
                    trng = exp_tbl[itgas];
                    atrgas = 1.0f - trng;
                    gasfac = tfn_tbl[itgas];
                    odepth = tau_tbl[itgas];
                }

                const float plfrac = fracs[((size_t)(k - 1) * RRTM_LW_NGPTLW + g) * nc + c];
                const float blay = pklay[((size_t)k * RRTM_LW_NBANDS + bi) * nc + c];
                const float dplnku = pklev[((size_t)k * RRTM_LW_NBANDS + bi) * nc + c] - blay;
                const float dplnkd = pklev[((size_t)(k - 1) * RRTM_LW_NBANDS + bi) * nc + c] - blay;
                const float bbdgas = plfrac * (blay + dplnkd * gasfac);
                const float bbugas = plfrac * (blay + dplnku * gasfac);
                const float gassrcd = bbdgas * atrgas;
                gassrcu[k - 1] = bbugas * atrgas;
                trngas[k - 1] = trng;

                const float clfm = cldfmc[((size_t)(k - 1) * RRTM_LW_NGPTLW + g) * nc + c];
                if (clfm >= RRTM_LW_EPS) {
                    // cloudy layer
                    const float odcld = secdif_b * taucld[((size_t)(k - 1) * RRTM_LW_NBANDS + bi) * nc + c];
                    efclrfr[k - 1] = 1.0f - (1.0f - expf(-odcld)) * clfm;
                    const float odtot = odepth + odcld;
                    float totfac, atrtot;
                    if (odtot < 0.06f) {
                        totfac = RRTM_LW_REC6 * odtot;
                        atrtot = odtot - 0.5f * odtot * odtot;
                    } else {
                        const float tblind = odtot / (RRTM_LW_BPADE + odtot);
                        const int ittot = (int)(RRTM_LW_TBLINT * tblind + 0.5f);
                        totfac = tfn_tbl[ittot];
                        atrtot = 1.0f - exp_tbl[ittot];
                    }
                    const float bbdtot = plfrac * (blay + dplnkd * totfac);
                    const float bbutot = plfrac * (blay + dplnku * totfac);
                    const float totsrcd = bbdtot * atrtot;
                    totsrcu[k - 1] = bbutot * atrtot;

                    // total sky radiance
                    radtotd = radtotd * trng * efclrfr[k - 1] + gassrcd + clfm * (totsrcd - gassrcd);
                    tdr[k - 1] = tdr[k - 1] + radtotd;

                    // clear sky radiance
                    radclrd = radclrd * trng + gassrcd;
                    cdr[k - 1] = cdr[k - 1] + radclrd;
                } else {
                    // clear layer
                    radtotd = radtotd * trng + gassrcd;
                    tdr[k - 1] = tdr[k - 1] + radtotd;

                    radclrd = radclrd * trng + gassrcd;
                    cdr[k - 1] = cdr[k - 1] + radclrd;
                }
            }

            // ---- spectral emissivity and surface reflectance (Fortran 3288-3300) ----
            const float reflct = 1.0f - semiss_b;
            const float rad0 = semiss_b * fracs[((size_t)0 * RRTM_LW_NGPTLW + g) * nc + c]
                                        * pklay[((size_t)0 * RRTM_LW_NBANDS + bi) * nc + c];

            float radtotu = rad0 + reflct * radtotd;
            tur[0] = tur[0] + radtotu;
            float radclru = rad0 + reflct * radclrd;
            cur[0] = cur[0] + radclru;

            // ---- upward radiative transfer loop (Fortran k = 1..nlay) ----
            for (int k = 1; k <= nlay; ++k) {
                const float clfm = cldfmc[((size_t)(k - 1) * RRTM_LW_NGPTLW + g) * nc + c];
                const float trng = trngas[k - 1];
                const float gasu = gassrcu[k - 1];

                if (clfm > RRTM_LW_EPS) {
                    // cloudy layer: total sky radiance
                    radtotu = radtotu * trng * efclrfr[k - 1] + gasu + clfm * (totsrcu[k - 1] - gasu);
                    tur[k] = tur[k] + radtotu;

                    // clear sky radiance
                    radclru = radclru * trng + gasu;
                    cur[k] = cur[k] + radclru;
                } else {
                    // clear layer
                    radtotu = radtotu * trng + gasu;
                    tur[k] = tur[k] + radtotu;

                    radclru = radclru * trng + gasu;
                    cur[k] = cur[k] + radclru;
                }
            }
        }   // end ig loop (members of this band)

        // Band sums are added into the flux totals in band order (Fortran 3357-3364, k-major).
        for (int k = 0; k <= nlay; ++k) {
            totuflux[(size_t)k * nc + c] = totuflux[(size_t)k * nc + c] + tur[k];
            totdflux[(size_t)k * nc + c] = totdflux[(size_t)k * nc + c] + tdr[k];
            totuclfl[(size_t)k * nc + c] = totuclfl[(size_t)k * nc + c] + cur[k];
            totdclfl[(size_t)k * nc + c] = totdclfl[(size_t)k * nc + c] + cdr[k];
        }

        // Fortran 3412-3418: spectral band heating rates, from this band's radiances.
        if (lhlwb) {
            float fnet_prev = (tur[0] - tdr[0]) * flxfac;
            for (int k = 1; k <= nlay; ++k) {
                const float fnet_k = (tur[k] - tdr[k]) * flxfac;
                const float rfdelp_k = heatfac / delp[(size_t)(k - 1) * nc + c];
                htrb[((size_t)bi * nlay + (k - 1)) * nc + c] = (fnet_prev - fnet_k) * rfdelp_k;
                fnet_prev = fnet_k;
            }
        } else {
            // lhlwb off: Fortran leaves htrb untouched; the unit defines it as zero so no
            // uninitialised device memory is ever read back.
            for (int k = 0; k < nlay; ++k) htrb[((size_t)bi * nlay + k) * nc + c] = 0.0f;
        }
    }   // end band loop

    // Fortran 3357-3376: totals become fluxes (w/m2).
    for (int k = 0; k <= nlay; ++k) {
        totuflux[(size_t)k * nc + c] = totuflux[(size_t)k * nc + c] * flxfac;
        totdflux[(size_t)k * nc + c] = totdflux[(size_t)k * nc + c] * flxfac;
        totuclfl[(size_t)k * nc + c] = totuclfl[(size_t)k * nc + c] * flxfac;
        totdclfl[(size_t)k * nc + c] = totdclfl[(size_t)k * nc + c] * flxfac;
    }

    // Fortran 3381-3394: net fluxes and total-sky heating rates.
    float fnet_prev = totuflux[c] - totdflux[c];
    for (int k = 1; k <= nlay; ++k) {
        const float rfdelp_k = heatfac / delp[(size_t)(k - 1) * nc + c];
        const float fnet_k = totuflux[(size_t)k * nc + c] - totdflux[(size_t)k * nc + c];
        htr[(size_t)(k - 1) * nc + c] = (fnet_prev - fnet_k) * rfdelp_k;
        fnet_prev = fnet_k;
    }

    // Fortran 3399-3407: clear-sky heating rates (lhlw0 flag).
    if (lhlw0) {
        float fnetc_prev = totuclfl[c] - totdclfl[c];
        for (int k = 1; k <= nlay; ++k) {
            const float rfdelp_k = heatfac / delp[(size_t)(k - 1) * nc + c];
            const float fnetc_k = totuclfl[(size_t)k * nc + c] - totdclfl[(size_t)k * nc + c];
            htrcl[(size_t)(k - 1) * nc + c] = (fnetc_prev - fnetc_k) * rfdelp_k;
            fnetc_prev = fnetc_k;
        }
    } else {
        // lhlw0 off: htrcl is defined as zero (Fortran leaves it untouched).
        for (int k = 0; k < nlay; ++k) htrcl[(size_t)k * nc + c] = 0.0f;
    }
}
