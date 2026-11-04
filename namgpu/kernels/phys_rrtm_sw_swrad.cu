// phys_rrtm_sw_swrad.cu -- SW driver per column (swrad) of the NMMB RRTM SW package.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The package header of radsw_main_nmmb.f is kept as the notice:
//   "sw-rrtm3 radiation package description ... this package includes ncep's modifications of the
//    rrtm-sw radiation code from aer inc."
// The AER copyright notice that follows that header in the source file (Copyright 2002-2007,
// Atmospheric & Environmental Research, Inc.) is reproduced in NOTICE of the nam-gpu tree.
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:374-1239 (subroutine swrad, module
// module_radsw_main), statements of swrad itself. The routines swrad calls are separate units and
// are NOT in this file:
//   cldprop      radsw_main_nmmb.f:1400-1822   unit phys-rrtm-sw-cldprop    (cloudy columns only)
//   setcoef      radsw_main_nmmb.f:1989-2139   unit phys-rrtm-sw-setcoef
//   taumol       radsw_main_nmmb.f:3107-3383   unit phys-rrtm-sw-taumol     (+ taumol16..29)
//   spcvrtc      radsw_main_nmmb.f:2144-2964   unit phys-rrtm-sw-spcvrtc    (isubcsw <= 0)
//   mcica_subcol radsw_main_nmmb.f:1827-1984   unit phys-rrtm-sw-mcica-subcol (isubcsw > 0 only)
//   spcvrtm      stub in the NAM source (the TODO branch of swrad), unit phys-rrtm-sw-spcvrtm (skip)
// swrad is therefore four launches around those calls, each one thread per DAY column (the day
// list idxday[0..NDAY), NDAY read on the device; no host sync):
//   swrad_zero_cld  the else branch of swrad's cloud test: clear columns (zcf1 <= 0) get zeros for
//                   cldfrc, cldfmc, taucw, ssacw, asycw before the seam runs (cldprop overwrites
//                   the cloudy columns only).
//   swrad_prep      swrad statements before the scheme calls: ipseed, the ivflip branches, layer
//                   pressure/temperature/thickness, absorber amounts, rare gases, aerosol and
//                   cloud unpack, cloud overlap (zcf0, zcf1), cosz1/sntz1/ssolar, albbm/albdf.
//   (seam)          cldprop, setcoef, taumol (+ taumol16..29), spcvrtc: they read the chunk scratch
//                   written by swrad_prep and write fxupc..fxdn0 and the TOA/surface scalars that
//                   swrad_post reads.
//   swrad_post      output assembly after the calls: flux sums over bands (flxuc, flxdc, flxu0,
//                   flxd0), topflx, sfcflx, fdncmp, the heating rates hswc, hsw0, hswb and the
//                   flux profiles flxprf (both ivflip orders).
//
// Index convention. Fortran swrad works on the LOCAL interval arrays of grrad_nmmb: npts = IM*JM,
// nlay = LMK = LM + 1 layers, nlp1 = LMP = nlay + 1 levels. Fortran level k (1-based) maps to
// zero-based k0 = k-1. The full-domain inputs are indexed by global (i0, j0) through the day list:
// chunk column t is grid point g = idxday[cstart + t] (flat g = j0*IM + i0).
// The chunk scratch (index t) keeps the Fortran internal order: ivflip = 1 reads the input
// surface->top, so internal level 0 is the surface; ivflip = 0 reads top->surface and flips.
// ivflip (NAM value 0, from rad_initialize iflip; module_SOLVER_GRID_COMP.F90:10714): both branches are kept.
//
// Fast numerics: float throughout (kind_phys single), FMA contraction on (NVRTC --fmad=true).
// No host sync, no device malloc, no printf. The Fortran lprnt/buggal diagnostic prints are not
// ported (debug output only; they do not change fluxes or heating rates).

#include "nmmb_common.cuh"

#define SW_NBDSW     14        // NBDSW = NBANDS (radsw_param_nmmb.f:149,156)
#define SW_NGPTSW    112       // NGPT   (radsw_param_nmmb.f:150)
#define SW_MAXGAS    6         // MAXGAS (radsw_param_nmmb.f:152)
#define SW_MAXLEV    80        // upper bound for the per-thread flux arrays (nlp1 <= 80)

// physcons (phys/physcons.f): con_g, con_avgd, con_amd, con_amw, con_amo3, con_cp.
#define SW_CON_G     9.80665f
#define SW_CON_AVGD  6.0221415e23f
#define SW_CON_AMD   28.9644f
#define SW_CON_AMW   18.0154f
#define SW_CON_AMO3  47.9982f
#define SW_CON_CP    1.0046e+3f
#define SW_AMDW      (SW_CON_AMD / SW_CON_AMW)     // amdw  (radsw_main_nmmb.f:302)
#define SW_AMDO3     (SW_CON_AMD / SW_CON_AMO3)    // amdo3 (radsw_main_nmmb.f:303)
#define SW_EPS       1.0e-6f                        // eps   (radsw_main_nmmb.f:291)
#define SW_ONEMINUS  (1.0f - SW_EPS)                // oneminus (radsw_main_nmmb.f:292)
#define SW_FTINY     1.0e-12f                       // ftiny (radsw_main_nmmb.f:295)
#define SW_S0        1368.22f                       // internal solar constant (radsw_main_nmmb.f:296)

// Flat index of a field [n, LEV, JM, IM] at zero-based (i0, j0, l0, n0) (CONTRACT section 1).
// For faersw the leading index is n0 = m*NBDSW + ib (m = 0 tau, 1 ssa, 2 asy).
#define SW_I4(i0, j0, l0, n0, LEV, JM, IM) \
    ((((size_t)(n0) * (LEV) + (l0)) * (JM) + (j0)) * (IM) + (i0))

// ---------------------------------------------------------------------------------------------
// swrad_zero_cld: the Fortran else branch of the cloud test (radsw_main_nmmb.f:~1040-1050, the
// cloudy-sky test before the cldprop call): clear columns get zero cloud fraction and zero cloud
// optics. One thread per day column of the chunk. zcf1 comes from swrad_prep.
// Chunk scratch: cldfrc [nc, nlay], cldfmc [nc, nlay, 112], taucw/ssacw/asycw [nc, nlay, 14].
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void swrad_zero_cld(
    const float* __restrict__ zcf1, const int* __restrict__ ndayp, int cstart, int nc, int nlay,
    float* __restrict__ cldfrc, float* __restrict__ cldfmc,
    float* __restrict__ taucw, float* __restrict__ ssacw, float* __restrict__ asycw)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nc || cstart + t >= *ndayp) return;
    if (zcf1[t] > 0.0f) return;                       // cloudy sky column: cldprop runs
    for (int k0 = 0; k0 < nlay; ++k0) {
        cldfrc[(size_t)t * nlay + k0] = 0.0f;
        for (int g = 0; g < SW_NGPTSW; ++g) cldfmc[((size_t)t * nlay + k0) * SW_NGPTSW + g] = 0.0f;
        for (int ib = 0; ib < SW_NBDSW; ++ib) {
            const size_t q = ((size_t)t * nlay + k0) * SW_NBDSW + ib;
            taucw[q] = 0.0f;
            ssacw[q] = 0.0f;
            asycw[q] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// swrad_prep: swrad statements before the scheme calls, one thread per day column.
//
// Inputs (full domain, CONTRACT layout):
//   plyr, tlyr, qlyr, olyr   [nlay, JM, IM]         layer pressure (mb), temperature, q, ozone
//   plvl                     [nlp1, JM, IM]         interface pressure (mb); tlvl is not in use
//                                                   (radsw_main_nmmb.f header "not in use")
//   gasvmr                   [10, nlay, JM, IM]     fields 1..9 used (gasvmr(:,:,1..9))
//   clouds                   [9, nlay, JM, IM]      prognostic (iswcliq>0) or diagnostic layout
//   faersw                   [42, nlay, JM, IM]     flattened [3 (tau, ssa, asy), 14 bands, ...]
//   sfcalb                   [4, JM, IM]            surface albedo, components 1..4
//   cosz                     [JM, IM]               cosine of solar zenith angle (CZEN)
//   icsdsw                   [JM, IM] int32         permutation seed (isubcsw == 2 only)
// Day list: idxday[0..*ndayp) flat grid indices; cstart is the chunk start in that list.
//
// Chunk scratch outputs (index t = day column in the chunk; per-column blocks):
//   pavel, tavel, delp, coldry, colmol, h2ovmr, o3vmr, temcol [nc, nlay]
//   colamt                   [nc, nlay, 6]          absorber amounts (maxgas 1..6 as 0..5)
//   cfrac, cliqp, reliq, cicep, reice, cdat1..cdat4 [nc, nlay]
//   tauae, ssaae, asyae      [nc, nlay, 14]
//   zcf0, zcf1               [nc]                   clear-sky and cloudy-sky fractions
//   cosz1, sntz1, ssolar     [nc]
//   albbm, albdf             [nc, 2]                beam and diffuse albedo (nir, uvv)
//   ipseed                   [nc] int32
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void swrad_prep(
    const float* __restrict__ plyr, const float* __restrict__ plvl, const float* __restrict__ tlyr,
    const float* __restrict__ qlyr, const float* __restrict__ olyr,
    const float* __restrict__ gasvmr, const float* __restrict__ clouds,
    const float* __restrict__ faersw, const float* __restrict__ cosz,
    const float* __restrict__ sfcalb, const int* __restrict__ icsdsw,
    const int* __restrict__ idxday, const int* __restrict__ ndayp,
    int cstart, int nc, int nlay, int nlp1, int im, int jm,
    int ivflip, int iswrgas, int iswcliq, int iovrsw, int isubcsw, int ipsdsw0, float solcon,
    float* __restrict__ pavel, float* __restrict__ tavel, float* __restrict__ delp,
    float* __restrict__ coldry, float* __restrict__ colmol, float* __restrict__ h2ovmr,
    float* __restrict__ o3vmr, float* __restrict__ temcol, float* __restrict__ colamt,
    float* __restrict__ cfrac, float* __restrict__ cliqp, float* __restrict__ reliq,
    float* __restrict__ cicep, float* __restrict__ reice, float* __restrict__ cdat1,
    float* __restrict__ cdat2, float* __restrict__ cdat3, float* __restrict__ cdat4,
    float* __restrict__ tauae, float* __restrict__ ssaae, float* __restrict__ asyae,
    float* __restrict__ zcf0, float* __restrict__ zcf1,
    float* __restrict__ cosz1, float* __restrict__ sntz1, float* __restrict__ ssolar,
    float* __restrict__ albbm, float* __restrict__ albdf, int* __restrict__ ipseed)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int j1 = cstart + t;                        // zero-based day index
    if (t >= nc || j1 >= *ndayp) return;
    const int g  = idxday[j1];
    const int i0 = g % im;
    const int j0 = g / im;
    const size_t JMIM = (size_t)jm * im;

    // --- solar geometry and surface albedo (the DO j1 = 1,NDAY block of swrad) ---
    const float s0fac = solcon / SW_S0;               // s0fac = solcon / s0
    const float cz = cosz[g];
    cosz1[t] = cz;
    sntz1[t] = 1.0f / cz;                             // sntz1 = f_one / cosz(j1)
    ssolar[t] = s0fac * cz;
    // albbm(j1,1)=sfcalb(j1,1); albdf(j1,1)=sfcalb(j1,2); albbm(j1,2)=sfcalb(j1,3); albdf(j1,2)=sfcalb(j1,4)
    albbm[2 * t + 0] = sfcalb[0 * JMIM + g];
    albdf[2 * t + 0] = sfcalb[1 * JMIM + g];
    albbm[2 * t + 1] = sfcalb[2 * JMIM + g];
    albdf[2 * t + 1] = sfcalb[3 * JMIM + g];

    // --- permutation seed (swrad, isubcsw == 1 or 2). Fortran i runs over npts; here it is the
    //     1-based flat grid point number. Only used when isubcsw > 0. ---
    if (isubcsw == 1)      ipseed[t] = ipsdsw0 + (g + 1);
    else if (isubcsw == 2) ipseed[t] = icsdsw[g];
    else                   ipseed[t] = 0;

    const float tem1 = 100.0f * SW_CON_G;
    const float tem2 = 1.0e-20f * 1.0e3f * SW_CON_AVGD;

    // --- profile preparation. ivflip==0 reads input top->surface (kk = nlp1-k), ivflip==1 reads
    //     input surface->top (kk = k). Both are one loop here: kk0 is the zero-based input level. ---
    for (int k0 = 0; k0 < nlay; ++k0) {
        const int kk0 = (ivflip == 0) ? (nlay - 1 - k0) : k0;   // zero-based input level
        const size_t L = (size_t)t * nlay + k0;                  // chunk-scratch index

        float dl;
        if (ivflip == 0) {
            pavel[L] = plyr[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
            tavel[L] = tlyr[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
            dl = plvl[SW_I4(i0, j0, kk0 + 1, 0, nlp1, jm, im)] - plvl[SW_I4(i0, j0, kk0, 0, nlp1, jm, im)];
        } else {
            pavel[L] = plyr[SW_I4(i0, j0, k0, 0, nlay, jm, im)];
            tavel[L] = tlyr[SW_I4(i0, j0, k0, 0, nlay, jm, im)];
            dl = plvl[SW_I4(i0, j0, k0, 0, nlp1, jm, im)] - plvl[SW_I4(i0, j0, k0 + 1, 0, nlp1, jm, im)];
        }
        delp[L] = dl;

        // absorber amounts (NCEP model use: specific humidity and mass mixing ratio of ozone)
        const float qv = qlyr[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
        const float ov = olyr[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
        const float h2o = fmaxf(0.0f, qv * SW_AMDW / (1.0f - qv));
        const float o3  = fmaxf(0.0f, ov * SW_AMDO3);
        h2ovmr[L] = h2o;
        o3vmr[L] = o3;
        const float tem0 = (1.0f - h2o) * SW_CON_AMD + h2o * SW_CON_AMW;
        const float cd = tem2 * dl / (tem1 * tem0 * (1.0f + h2o));
        coldry[L] = cd;
        const float tc = 1.0e-12f * cd;
        temcol[L] = tc;

        const size_t C = L * SW_MAXGAS;
        colamt[C + 0] = fmaxf(0.0f, cd * h2o);                                       // h2o
        colamt[C + 1] = fmaxf(tc, cd * gasvmr[SW_I4(i0, j0, kk0, 0, nlay, jm, im)]); // co2 (gasvmr 1)
        colamt[C + 2] = fmaxf(0.0f, cd * o3);                                        // o3
        colmol[L] = cd + colamt[C + 0];

        // rare gases: n2o (gasvmr 2), ch4 (gasvmr 3), o2 (gasvmr 4)
        if (iswrgas > 0) {
            colamt[C + 3] = fmaxf(tc, cd * gasvmr[SW_I4(i0, j0, kk0, 1, nlay, jm, im)]);
            colamt[C + 4] = fmaxf(tc, cd * gasvmr[SW_I4(i0, j0, kk0, 2, nlay, jm, im)]);
            colamt[C + 5] = fmaxf(tc, cd * gasvmr[SW_I4(i0, j0, kk0, 3, nlay, jm, im)]);
        } else {
            colamt[C + 3] = tc;
            colamt[C + 4] = tc;
            colamt[C + 5] = tc;
        }

        // aerosol optical properties: faersw(m, ib, kk) -> tauae (m=0), ssaae (m=1), asyae (m=2)
        for (int ib = 0; ib < SW_NBDSW; ++ib) {
            const size_t Q = L * SW_NBDSW + ib;
            tauae[Q] = faersw[SW_I4(i0, j0, kk0, 0 * SW_NBDSW + ib, nlay, jm, im)];
            ssaae[Q] = faersw[SW_I4(i0, j0, kk0, 1 * SW_NBDSW + ib, nlay, jm, im)];
            asyae[Q] = faersw[SW_I4(i0, j0, kk0, 2 * SW_NBDSW + ib, nlay, jm, im)];
        }

        // cloud unpack: prognostic (iswcliq > 0) or diagnostic (iswcliq == 0) layout
        if (iswcliq > 0) {
            cfrac[L] = clouds[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
            cliqp[L] = clouds[SW_I4(i0, j0, kk0, 1, nlay, jm, im)];
            reliq[L] = clouds[SW_I4(i0, j0, kk0, 2, nlay, jm, im)];
            cicep[L] = clouds[SW_I4(i0, j0, kk0, 3, nlay, jm, im)];
            reice[L] = clouds[SW_I4(i0, j0, kk0, 4, nlay, jm, im)];
            cdat1[L] = clouds[SW_I4(i0, j0, kk0, 5, nlay, jm, im)];
            cdat2[L] = clouds[SW_I4(i0, j0, kk0, 6, nlay, jm, im)];
            cdat3[L] = clouds[SW_I4(i0, j0, kk0, 7, nlay, jm, im)];
            cdat4[L] = clouds[SW_I4(i0, j0, kk0, 8, nlay, jm, im)];
        } else {
            cfrac[L] = clouds[SW_I4(i0, j0, kk0, 0, nlay, jm, im)];
            cdat1[L] = clouds[SW_I4(i0, j0, kk0, 1, nlay, jm, im)];
            cdat2[L] = clouds[SW_I4(i0, j0, kk0, 2, nlay, jm, im)];
            cdat3[L] = clouds[SW_I4(i0, j0, kk0, 3, nlay, jm, im)];
        }
    }

    // --- clear-sky view fractions (overlap, iovrsw 0 / 1 / 2) ---
    float z0 = 1.0f;
    float z1 = 1.0f;
    if (iovrsw == 0) {                                // random overlapping
        for (int k0 = 0; k0 < nlay; ++k0) {
            z0 = z0 * (1.0f - cfrac[(size_t)t * nlay + k0]);
        }
    } else if (iovrsw == 1) {                         // max/ran overlapping
        for (int k0 = 0; k0 < nlay; ++k0) {
            const float cf = cfrac[(size_t)t * nlay + k0];
            if (cf > SW_FTINY) {                      // cloudy layer
                z1 = fminf(z1, 1.0f - cf);
            } else if (z1 < 1.0f) {                   // clear layer
                z0 = z0 * z1;
                z1 = 1.0f;
            }
        }
        z0 = z0 * z1;
    } else if (iovrsw == 2) {                         // maximum overlapping
        for (int k0 = 0; k0 < nlay; ++k0) {
            z0 = fminf(z0, 1.0f - cfrac[(size_t)t * nlay + k0]);
        }
    }
    // WHERE (zcf0 <= ftiny) zcf0 = 0; WHERE (zcf0 > oneminus) zcf0 = 1; zcf1 = 1 - zcf0
    if (z0 <= SW_FTINY) z0 = 0.0f;
    if (z0 > SW_ONEMINUS) z0 = 1.0f;
    zcf0[t] = z0;
    zcf1[t] = 1.0f - z0;
}

// ---------------------------------------------------------------------------------------------
// Heating rate from a net flux profile (the fnet / rfdelp loops of swrad).
// fnet[k] is the Fortran fnet(k+1) (zero-based, nlp1 entries); delp[k] is the Fortran delp(k+1)
// of the internal (surface-first) layers. For Fortran level k' = k+1 (k = 1..nlp1-1) the rate is
//   rfdelp(k'-1) * (fnet(k') - fnet(k'-1)),  rfdelp = heatfac / delp,
// stored at output level lev = k-1 (ivflip 1, surface-first output) or lev = nlp1-1-k (ivflip 0,
// top-first output, Fortran kk = nlp1-k'+1). out[lev * stride] is the field element.
// ---------------------------------------------------------------------------------------------
__device__ inline void swrad_hr_profile(const float* fnet, const float* delp, float heatfac,
                                        int nlp1, int ivflip, float* out, size_t stride)
{
    for (int k = 1; k < nlp1; ++k) {
        const float rf = heatfac / delp[k - 1];
        const float h = (fnet[k] - fnet[k - 1]) * rf;
        const int lev = (ivflip == 1) ? (k - 1) : (nlp1 - 1 - k);
        out[(size_t)lev * stride] = h;
    }
}

// ---------------------------------------------------------------------------------------------
// swrad_post: swrad statements after the scheme calls, one thread per day column.
//
// Chunk inputs (delp from swrad_prep; the rest written by the seam units):
//   delp                       [nc, nlay]              swrad_prep
//   fxupc, fxdnc, fxup0, fxdn0 [nc, nlp1, 14]          spcvrtc spectral fluxes (total and clear sky)
//   ftoauc, ftoau0, ftoadc     [nc]                    TOA fluxes (spcvrtc)
//   fsfcuc, fsfcu0, fsfcdc, fsfcd0 [nc]                surface fluxes (spcvrtc)
//   suvbfc, suvbf0             [nc]                    UV-B surface fluxes (spcvrtc)
//   sfbmc, sfdfc               [nc, 2]                 surface beam and diffuse (spcvrtc)
// Full-domain outputs (CONTRACT layout; zero-filled by the wrapper before the chunk loop):
//   htswc                      [nlay, JM, IM]          total-sky heating rate
//   topflx                     [3, JM, IM]             upfxc, dnfxc, upfx0 at TOA
//   sfcflx                     [4, JM, IM]             upfxc, dnfxc, upfx0, dnfx0 at the surface
//   hsw0 (opt)                 [nlay, JM, IM]          clear-sky heating rate (lhsw0)
//   htswb (opt)                [14, nlay, JM, IM]      spectral band heating rates (lhswb)
//   flxprf (opt)               [4, nlp1, JM, IM]       upfxc, dnfxc, upfx0, dnfx0 per level (lflxprf)
//   fdncmp (opt)               [6, JM, IM]             uvbf0, uvbfc, nirbm, nirdf, visbm, visdf (lfdncmp)
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void swrad_post(
    const float* __restrict__ delp,
    const float* __restrict__ fxupc, const float* __restrict__ fxdnc,
    const float* __restrict__ fxup0, const float* __restrict__ fxdn0,
    const float* __restrict__ ftoauc, const float* __restrict__ ftoau0, const float* __restrict__ ftoadc,
    const float* __restrict__ fsfcuc, const float* __restrict__ fsfcu0,
    const float* __restrict__ fsfcdc, const float* __restrict__ fsfcd0,
    const float* __restrict__ suvbfc, const float* __restrict__ suvbf0,
    const float* __restrict__ sfbmc, const float* __restrict__ sfdfc,
    const int* __restrict__ idxday, const int* __restrict__ ndayp,
    int cstart, int nc, int nlay, int nlp1, int im, int jm,
    int ivflip, float heatfac, int lhsw0, int lhswb, int lflxprf, int lfdncmp,
    float* __restrict__ htswc, float* __restrict__ topflx, float* __restrict__ sfcflx,
    float* __restrict__ hsw0, float* __restrict__ htswb,
    float* __restrict__ flxprf, float* __restrict__ fdncmp)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int j1 = cstart + t;
    if (t >= nc || j1 >= *ndayp) return;
    const int g = idxday[j1];
    const size_t JMIM = (size_t)jm * im;
    const float* D = delp + (size_t)t * nlay;

    // flxuc, flxdc (total sky) and flxu0, flxd0 (clear sky): sum over the 14 bands in band order
    // starting from zero.
    float flxuc[SW_MAXLEV], flxdc[SW_MAXLEV], flxu0[SW_MAXLEV], flxd0[SW_MAXLEV];
    float fnet[SW_MAXLEV];
    for (int k = 0; k < nlp1; ++k) {
        float su = 0.0f;
        float sd = 0.0f;
        for (int ib = 0; ib < SW_NBDSW; ++ib) {
            su = su + fxupc[((size_t)t * nlp1 + k) * SW_NBDSW + ib];
            sd = sd + fxdnc[((size_t)t * nlp1 + k) * SW_NBDSW + ib];
        }
        flxuc[k] = su;
        flxdc[k] = sd;
        if (lhsw0 || lflxprf) {
            float s0u = 0.0f;
            float s0d = 0.0f;
            for (int ib = 0; ib < SW_NBDSW; ++ib) {
                s0u = s0u + fxup0[((size_t)t * nlp1 + k) * SW_NBDSW + ib];
                s0d = s0d + fxdn0[((size_t)t * nlp1 + k) * SW_NBDSW + ib];
            }
            flxu0[k] = s0u;
            flxd0[k] = s0d;
        }
    }

    // toa and sfc fluxes
    topflx[0 * JMIM + g] = ftoauc[t];                 // upfxc
    topflx[1 * JMIM + g] = ftoadc[t];                 // dnfxc
    topflx[2 * JMIM + g] = ftoau0[t];                 // upfx0
    sfcflx[0 * JMIM + g] = fsfcuc[t];                 // upfxc
    sfcflx[1 * JMIM + g] = fsfcdc[t];                 // dnfxc
    sfcflx[2 * JMIM + g] = fsfcu0[t];                 // upfx0
    sfcflx[3 * JMIM + g] = fsfcd0[t];                 // dnfx0

    // optional component surface fluxes
    if (lfdncmp) {
        fdncmp[0 * JMIM + g] = suvbf0[t];             // uvbf0
        fdncmp[1 * JMIM + g] = suvbfc[t];             // uvbfc
        fdncmp[2 * JMIM + g] = sfbmc[(size_t)t * 2 + 0];   // nirbm
        fdncmp[3 * JMIM + g] = sfdfc[(size_t)t * 2 + 0];   // nirdf
        fdncmp[4 * JMIM + g] = sfbmc[(size_t)t * 2 + 1];   // visbm
        fdncmp[5 * JMIM + g] = sfdfc[(size_t)t * 2 + 1];   // visdf
    }

    // total-sky heating rates: fnet(k) = flxdc(k) - flxuc(k), then rfdelp-scaled differences
    for (int k = 0; k < nlp1; ++k) fnet[k] = flxdc[k] - flxuc[k];
    swrad_hr_profile(fnet, D, heatfac, nlp1, ivflip, htswc + g, JMIM);

    // optional clear-sky heating rates (lhsw0)
    if (lhsw0) {
        for (int k = 0; k < nlp1; ++k) fnet[k] = flxd0[k] - flxu0[k];
        swrad_hr_profile(fnet, D, heatfac, nlp1, ivflip, hsw0 + g, JMIM);
    }

    // optional spectral band heating rates (lhswb): fnet from the band fluxes
    if (lhswb) {
        for (int mb = 0; mb < SW_NBDSW; ++mb) {
            for (int k = 0; k < nlp1; ++k) {
                const size_t q = ((size_t)t * nlp1 + k) * SW_NBDSW + mb;
                fnet[k] = fxdnc[q] - fxupc[q];
            }
            swrad_hr_profile(fnet, D, heatfac, nlp1, ivflip,
                             htswb + (size_t)mb * nlay * JMIM + g, JMIM);
        }
    }

    // optional flux profiles (lflxprf): ivflip 1 keeps the level order, ivflip 0 reverses it
    if (lflxprf) {
        for (int k = 0; k < nlp1; ++k) {
            const int lev = (ivflip == 1) ? k : (nlp1 - 1 - k);
            flxprf[(0 * (size_t)nlp1 + lev) * JMIM + g] = flxuc[k];
            flxprf[(1 * (size_t)nlp1 + lev) * JMIM + g] = flxdc[k];
            flxprf[(2 * (size_t)nlp1 + lev) * JMIM + g] = flxu0[k];
            flxprf[(3 * (size_t)nlp1 + lev) * JMIM + g] = flxd0[k];
        }
    }
}
