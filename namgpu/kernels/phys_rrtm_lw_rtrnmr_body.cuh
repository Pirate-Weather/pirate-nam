// phys_rrtm_lw_rtrnmr_body.cuh -- rtrnmr kernel body, included twice by phys_rrtm_lw_rtrnmr.cu (two slot maps).
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// phys-rrtm-lw-rtrnmr kernel body, included twice by phys_rrtm_lw_rtrnmr.cu with two slot maps (see there).
// The including file defines RTRNMR_NAME, LAY, IFC, CFX and PKX; nothing else differs between the two kernels.
extern "C" __global__ void RTRNMR_NAME(
    const int im, const int jm, const int nlay, const int lhlw0, const int lhlwb,
    const float heatfac, const float fluxfac,
    const float* __restrict__ exp_tbl, const float* __restrict__ tfn_tbl,
    const float* __restrict__ tau_tbl,
    const float* __restrict__ semiss,   /* [NBANDS, JM, IM]                                      */
    const float* __restrict__ delp,     /* [NLAY, JM, IM]      Fortran delp(1:nlay)              */
    const float* __restrict__ cldfrc,   /* [NLAY+2, JM, IM]    Fortran cldfrc(0:nlp1)            */
    const float* __restrict__ taucld,   /* [NBANDS, NLAY, JM, IM]                                */
    const float* __restrict__ tautot,   /* [NGPTLW, NLAY, JM, IM]                                */
    const float* __restrict__ pklay,    /* [NBANDS, NLAY+1, JM, IM] Fortran pklay(nbands,0:nlay) */
    const float* __restrict__ pklev,    /* [NBANDS, NLAY+1, JM, IM] Fortran pklev(nbands,0:nlay) */
    const float* __restrict__ fracs,    /* [NGPTLW, NLAY, JM, IM]                                */
    const float* __restrict__ secdif,   /* [NBANDS, JM, IM]                                      */
    float* __restrict__ totuflux,       /* [NLAY+1, JM, IM]    Fortran 0:nlay                    */
    float* __restrict__ totdflux,       /* [NLAY+1, JM, IM]                                      */
    float* __restrict__ htr,            /* [NLAY, JM, IM]      Fortran 1:nlay                    */
    float* __restrict__ totuclfl,       /* [NLAY+1, JM, IM]                                      */
    float* __restrict__ totdclfl,       /* [NLAY+1, JM, IM]                                      */
    float* __restrict__ htrcl,          /* [NLAY, JM, IM]                                        */
    float* __restrict__ htrb)           /* [NBANDS, NLAY, JM, IM] Fortran htrb(nlay,nbands)      */
{
    const size_t N = (size_t)jm * (size_t)im;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    // ---- constants (radlw_main_nmmb.f:267-275, 1220 expeps, 1286 fluxfac) ----
    const float eps    = 1.0e-6f;
    const float bpade  = 1.0f / 0.278f;     /* radlw_main_nmmb.f:270 */
    const float wtdiff = 0.5f;              /* radlw_main_nmmb.f:272 */
    const float tblint = (float)LW_NTBL;    /* radlw_main_nmmb.f:273 */
    const float rec_6  = 0.166667f;         /* rtrnmr parameter rec_6 */
    const float f_zero = 0.0f;
    const float f_one  = 1.0f;
    const float c06    = 0.06f;
    const int   nlp1   = nlay + 1;

    // cldfrc(0:nlp1) for this column, memory index = Fortran index
    float cf[LW_MAXL + 3];
    for (int m = 0; m <= nlp1; ++m) cf[m] = cldfrc[(size_t)CFX(m) * N + col];

    // ---- overlap factors, upward ----
    float faccld1u[LW_MAXL + 3], faccld2u[LW_MAXL + 3], facclr1u[LW_MAXL + 3];
    float facclr2u[LW_MAXL + 3], faccmb1u[LW_MAXL + 3], faccmb2u[LW_MAXL + 3];
    int   lstu[LW_MAXL + 3];
    for (int k = 0; k <= nlp1; ++k) {
        faccld1u[k] = f_zero; faccld2u[k] = f_zero; facclr1u[k] = f_zero;
        facclr2u[k] = f_zero; faccmb1u[k] = f_zero; faccmb2u[k] = f_zero;
    }
    lstu[1] = (cf[1] > eps) ? 1 : 0;
    float rat1 = f_zero;
    float rat2 = f_zero;
    for (int k = 1; k <= nlay - 1; ++k) {
        lstu[k + 1] = (cf[k + 1] > eps && cf[k] <= eps) ? 1 : 0;
        if (cf[k] > eps) {
            if (cf[k + 1] >= cf[k]) {
                if (lstu[k]) {
                    if (cf[k] < f_one) {
                        facclr2u[k + 1] = (cf[k + 1] - cf[k]) / (f_one - cf[k]);
                    }
                    facclr2u[k] = f_zero;
                    faccld2u[k] = f_zero;
                } else {
                    float fmax = fmaxf(cf[k], cf[k - 1]);
                    if (cf[k + 1] > fmax) {
                        facclr1u[k + 1] = rat2;
                        facclr2u[k + 1] = (cf[k + 1] - fmax) / (f_one - fmax);
                    } else if (cf[k + 1] < fmax) {
                        facclr1u[k + 1] = (cf[k + 1] - cf[k]) / (cf[k - 1] - cf[k]);
                    } else {
                        facclr1u[k + 1] = rat2;
                    }
                }
                if (facclr1u[k + 1] > f_zero || facclr2u[k + 1] > f_zero) {
                    rat1 = f_one; rat2 = f_zero;
                } else {
                    rat1 = f_zero; rat2 = f_zero;
                }
            } else {
                if (lstu[k]) {
                    faccld2u[k + 1] = (cf[k] - cf[k + 1]) / cf[k];
                    facclr2u[k] = f_zero;
                    faccld2u[k] = f_zero;
                } else {
                    float fmin = fminf(cf[k], cf[k - 1]);
                    if (cf[k + 1] <= fmin) {
                        faccld1u[k + 1] = rat1;
                        faccld2u[k + 1] = (fmin - cf[k + 1]) / fmin;
                    } else {
                        faccld1u[k + 1] = (cf[k] - cf[k + 1]) / (cf[k] - fmin);
                    }
                }
                if (faccld1u[k + 1] > f_zero || faccld2u[k + 1] > f_zero) {
                    rat1 = f_zero; rat2 = f_one;
                } else {
                    rat1 = f_zero; rat2 = f_zero;
                }
            }
            faccmb1u[k + 1] = facclr1u[k + 1] * faccld2u[k] * cf[k - 1];
            faccmb2u[k + 1] = faccld1u[k + 1] * facclr2u[k] * (f_one - cf[k - 1]);
        }
    }

    // ---- overlap factors, downward ----
    float faccld1d[LW_MAXL + 3], faccld2d[LW_MAXL + 3], facclr1d[LW_MAXL + 3];
    float facclr2d[LW_MAXL + 3], faccmb1d[LW_MAXL + 3], faccmb2d[LW_MAXL + 3];
    int   lstd[LW_MAXL + 3];
    for (int k = 0; k <= nlay; ++k) {
        faccld1d[k] = f_zero; faccld2d[k] = f_zero; facclr1d[k] = f_zero;
        facclr2d[k] = f_zero; faccmb1d[k] = f_zero; faccmb2d[k] = f_zero;
    }
    lstd[nlay] = (cf[nlay] > eps) ? 1 : 0;
    rat1 = f_zero;
    rat2 = f_zero;
    for (int k = nlay; k >= 2; --k) {
        lstd[k - 1] = (cf[k - 1] > eps && cf[k] <= eps) ? 1 : 0;
        if (cf[k] > eps) {
            if (cf[k - 1] >= cf[k]) {
                if (lstd[k]) {
                    if (cf[k] < f_one) {
                        facclr2d[k - 1] = (cf[k - 1] - cf[k]) / (f_one - cf[k]);
                    }
                    facclr2d[k] = f_zero;
                    faccld2d[k] = f_zero;
                } else {
                    float fmax = fmaxf(cf[k], cf[k + 1]);
                    if (cf[k - 1] > fmax) {
                        facclr1d[k - 1] = rat2;
                        facclr2d[k - 1] = (cf[k - 1] - fmax) / (f_one - fmax);
                    } else if (cf[k - 1] < fmax) {
                        facclr1d[k - 1] = (cf[k - 1] - cf[k]) / (cf[k + 1] - cf[k]);
                    } else {
                        facclr1d[k - 1] = rat2;
                    }
                }
                if (facclr1d[k - 1] > f_zero || facclr2d[k - 1] > f_zero) {
                    rat1 = f_one; rat2 = f_zero;
                } else {
                    rat1 = f_zero; rat2 = f_zero;
                }
            } else {
                if (lstd[k]) {
                    faccld2d[k - 1] = (cf[k] - cf[k - 1]) / cf[k];
                    facclr2d[k] = f_zero;
                    faccld2d[k] = f_zero;
                } else {
                    float fmin = fminf(cf[k], cf[k + 1]);
                    if (cf[k - 1] <= fmin) {
                        faccld1d[k - 1] = rat1;
                        faccld2d[k - 1] = (fmin - cf[k - 1]) / fmin;
                    } else {
                        faccld1d[k - 1] = (cf[k] - cf[k - 1]) / (cf[k] - fmin);
                    }
                }
                if (faccld1d[k - 1] > f_zero || faccld2d[k - 1] > f_zero) {
                    rat1 = f_zero; rat2 = f_one;
                } else {
                    rat1 = f_zero; rat2 = f_zero;
                }
            }
            faccmb1d[k - 1] = facclr1d[k - 1] * faccld2d[k] * cf[k + 1];
            faccmb2d[k - 1] = faccld1d[k - 1] * facclr2d[k] * (f_one - cf[k + 1]);
        }
    }

    // ---- per-column flux totals and layer heating-rate prep ----
    float tuf[LW_MAXL + 2], tdf[LW_MAXL + 2], tuc[LW_MAXL + 2], tdc[LW_MAXL + 2];
    float rfdelp[LW_MAXL + 2], fnet[LW_MAXL + 2], fnetc[LW_MAXL + 2];
    for (int k = 0; k <= nlay; ++k) {
        tuf[k] = f_zero; tdf[k] = f_zero; tuc[k] = f_zero; tdc[k] = f_zero;
    }
    for (int k = 1; k <= nlay; ++k) {
        rfdelp[k] = heatfac / delp[(size_t)LAY(k - 1) * N + col];   /* rfdelp(k) = heatfac/delp(k) */
    }
    const float flxfac = wtdiff * fluxfac;

    // per-band accumulators (Fortran toturad/totdrad/clrurad/clrdrad(0:nlay, ib) for one ib)
    float toturad[LW_MAXL + 2], totdrad[LW_MAXL + 2], clrurad[LW_MAXL + 2], clrdrad[LW_MAXL + 2];
    // per-g temporaries indexed by layer k (set in the downward pass, used in the upward pass)
    float totsrcu[LW_MAXL + 2], trntot[LW_MAXL + 2], gassrcu[LW_MAXL + 2], trngas[LW_MAXL + 2];

    // persistent Fortran locals of rtrnmr
    float radtotd = f_zero, radclrd = f_zero, totradd = f_zero, clrradd = f_zero, rad = f_zero;
    float radtotu = f_zero, radclru = f_zero, totradu = f_zero, clrradu = f_zero;

    for (int ib = 0; ib < LW_NBANDS; ++ib) {
        for (int k = 0; k <= nlay; ++k) {
            toturad[k] = f_zero; totdrad[k] = f_zero; clrurad[k] = f_zero; clrdrad[k] = f_zero;
        }
        const float semi = semiss[(size_t)ib * N + col];
        const float sdif = secdif[(size_t)ib * N + col];

        for (int ig = LW_NS[ib]; ig < LW_NS[ib] + LW_NG[ib]; ++ig) {   /* ngb(ig) == ib */
            radtotd = f_zero;
            radclrd = f_zero;

            // ---- downward pass, k = nlay..1 ----
            for (int k = nlay; k >= 1; --k) {
                float odepth = fmaxf(f_zero, sdif * tautot[((size_t)ig * nlay + LAY(k - 1)) * N + col]);
                float atrgas, trng, gasfac;
                if (odepth <= c06) {
                    atrgas = odepth - 0.5f * odepth * odepth;
                    trng = f_one - atrgas;
                    gasfac = rec_6 * odepth;
                } else {
                    float tblind = odepth / (bpade + odepth);
                    int itgas = (int)(tblint * tblind + 0.5f);
                    trng = exp_tbl[itgas];
                    atrgas = f_one - trng;
                    gasfac = tfn_tbl[itgas];
                    odepth = tau_tbl[itgas];
                }
                const float plfrac = fracs[((size_t)ig * nlay + LAY(k - 1)) * N + col];
                const float blay   = pklay[PKX(ib, IFC(k)) * N + col];
                const float dplnku = pklev[PKX(ib, IFC(k)) * N + col] - blay;
                const float dplnkd = pklev[PKX(ib, IFC(k - 1)) * N + col] - blay;
                const float bbdgas = plfrac * (blay + dplnkd * gasfac);
                const float bbugas = plfrac * (blay + dplnku * gasfac);
                const float gassrcd = bbdgas * atrgas;
                gassrcu[k] = bbugas * atrgas;
                trngas[k] = trng;
                const float clfr = cf[k];

                if (lstd[k]) {
                    totradd = clfr * radtotd;
                    clrradd = radtotd - totradd;
                    rad = f_zero;
                }
                if (clfr >= eps) {
                    const float odcld = sdif * taucld[((size_t)ib * nlay + LAY(k - 1)) * N + col];
                    const float odtot = odepth + odcld;
                    float totfac, atrtot, trnt;
                    if (odtot < c06) {
                        totfac = rec_6 * odtot;
                        atrtot = odtot - 0.5f * odtot * odtot;
                        trnt = f_one - atrtot;
                    } else {
                        float tblind = odtot / (bpade + odtot);
                        int ittot = (int)(tblint * tblind + 0.5f);
                        totfac = tfn_tbl[ittot];
                        trnt = exp_tbl[ittot];
                        atrtot = f_one - trnt;
                    }
                    const float bbdtot = plfrac * (blay + dplnkd * totfac);
                    const float bbutot = plfrac * (blay + dplnku * totfac);
                    const float totsrcd = bbdtot * atrtot;
                    totsrcu[k] = bbutot * atrtot;
                    trntot[k] = trnt;
                    totradd = totradd * trnt + clfr * totsrcd;
                    clrradd = clrradd * trng + (f_one - clfr) * gassrcd;
                    radtotd = totradd + clrradd;
                    totdrad[k - 1] = totdrad[k - 1] + radtotd;
                    radclrd = radclrd * trng + gassrcd;
                    clrdrad[k - 1] = clrdrad[k - 1] + radclrd;
                    const float radmod = rad * (facclr1d[k - 1] * trng + faccld1d[k - 1] * trnt)
                                       - faccmb1d[k - 1] * gassrcd + faccmb2d[k - 1] * totsrcd;
                    rad = -radmod + facclr2d[k - 1] * (clrradd + radmod)
                        - faccld2d[k - 1] * (totradd - radmod);
                    totradd = totradd + rad;
                    clrradd = clrradd - rad;
                } else {
                    radtotd = radtotd * trng + gassrcd;
                    totdrad[k - 1] = totdrad[k - 1] + radtotd;
                    radclrd = radclrd * trng + gassrcd;
                    clrdrad[k - 1] = clrdrad[k - 1] + radclrd;
                }
            }

            // ---- surface emission and reflection ----
            const float reflct = f_one - semi;
            const float rad0 = semi * fracs[((size_t)ig * nlay + LAY(0)) * N + col]
                             * pklay[PKX(ib, IFC(0)) * N + col];
            radtotu = rad0 + reflct * radtotd;
            toturad[0] = toturad[0] + radtotu;
            radclru = rad0 + reflct * radclrd;
            clrurad[0] = clrurad[0] + radclru;

            // ---- upward pass, k = 1..nlay ----
            for (int k = 1; k <= nlay; ++k) {
                const float clfr = cf[k];
                const float trng = trngas[k];
                const float gasu = gassrcu[k];
                if (lstu[k]) {
                    totradu = clfr * radtotu;
                    clrradu = radtotu - totradu;
                    rad = f_zero;
                }
                if (clfr >= eps) {
                    const float trnt = trntot[k];
                    const float totu = totsrcu[k];
                    totradu = totradu * trnt + clfr * totu;
                    clrradu = clrradu * trng + (f_one - clfr) * gasu;
                    radtotu = totradu + clrradu;
                    toturad[k] = toturad[k] + radtotu;
                    radclru = radclru * trng + gasu;
                    clrurad[k] = clrurad[k] + radclru;
                    const float radmod = rad * (facclr1u[k + 1] * trng + faccld1u[k + 1] * trnt)
                                       - faccmb1u[k + 1] * gasu + faccmb2u[k + 1] * totu;
                    rad = -radmod + facclr2u[k + 1] * (clrradu + radmod)
                        - faccld2u[k + 1] * (totradu - radmod);
                    totradu = totradu + rad;
                    clrradu = clrradu - rad;
                } else {
                    radtotu = radtotu * trng + gasu;
                    toturad[k] = toturad[k] + radtotu;
                    radclru = radclru * trng + gasu;
                    clrurad[k] = clrurad[k] + radclru;
                }
            }
        }   // end ig loop (band ib)

        // band ib complete: add into the column totals in band order (Fortran totals loop)
        for (int k = 0; k <= nlay; ++k) {
            tuf[k] = tuf[k] + toturad[k];
            tdf[k] = tdf[k] + totdrad[k];
            tuc[k] = tuc[k] + clrurad[k];
            tdc[k] = tdc[k] + clrdrad[k];
        }

        // spectral band heating rates htrb(k, ib) (Fortran lhlwb block)
        if (lhlwb) {
            fnet[0] = (toturad[0] - totdrad[0]) * flxfac;
            for (int k = 1; k <= nlay; ++k) {
                fnet[k] = (toturad[k] - totdrad[k]) * flxfac;
                htrb[((size_t)ib * nlay + LAY(k - 1)) * N + col] = (fnet[k - 1] - fnet[k]) * rfdelp[k];
            }
        }
    }   // end band loop

    // ---- totals times flxfac, written out ----
    for (int k = 0; k <= nlay; ++k) {
        tuf[k] = tuf[k] * flxfac;
        tdf[k] = tdf[k] * flxfac;
        tuc[k] = tuc[k] * flxfac;
        tdc[k] = tdc[k] * flxfac;
        totuflux[(size_t)IFC(k) * N + col] = tuf[k];
        totdflux[(size_t)IFC(k) * N + col] = tdf[k];
        totuclfl[(size_t)IFC(k) * N + col] = tuc[k];
        totdclfl[(size_t)IFC(k) * N + col] = tdc[k];
    }

    // ---- total-sky heating rate ----
    fnet[0] = tuf[0] - tdf[0];
    for (int k = 1; k <= nlay; ++k) {
        fnet[k] = tuf[k] - tdf[k];
        htr[(size_t)LAY(k - 1) * N + col] = (fnet[k - 1] - fnet[k]) * rfdelp[k];
    }

    // ---- clear-sky heating rate (optional output, lhlw0) ----
    if (lhlw0) {
        fnetc[0] = tuc[0] - tdc[0];
        for (int k = 1; k <= nlay; ++k) {
            fnetc[k] = tuc[k] - tdc[k];
            htrcl[(size_t)LAY(k - 1) * N + col] = (fnetc[k - 1] - fnetc[k]) * rfdelp[k];
        }
    }
}
