// phys_rrtm_lw_lwrad.cu -- LW driver per column (lwrad) of the NMMB RRTM LW package.
//
// NOAA notice: phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, US government work
// (no licence file; see NOTICE). The package header of radlw_main_nmmb.f is kept as the notice:
//   "lw-rrtm3 radiation package description ... this package includes ncep modifications of the
//    rrtm-lw radiation code from aer inc."
// The AER copyright notice (Copyright 2002-2007, Atmospheric & Environmental Research, Inc.) is in the
// source file radlw_main_nmmb.f itself, not in NOTICE. Its terms allow use and redistribution only as
// long as the code is not sold. That is a licence limit: this code stays in the private tree and is
// held out of any public export until the project lead rules on it (CLAUDE.md, licences and defaults).
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:352-1139 (subroutine lwrad, module
// module_radlw_main_nmmb), statements of lwrad itself. The routines lwrad calls are separate units and
// are NOT in this file:
//   cldprop   radlw_main_nmmb.f:1344-1697   unit phys-rrtm-lw-cldprop
//   setcoef   radlw_main_nmmb.f:1863-2121   unit phys-rrtm-lw-setcoef
//   taumol    radlw_main_nmmb.f:3423-3609   unit phys-rrtm-lw-taumol (+ taugb01..16)
//   rtrn      radlw_main_nmmb.f:2127-2484   unit phys-rrtm-lw-rtrn
//   rtrnmr    radlw_main_nmmb.f:2489-3054   unit phys-rrtm-lw-rtrnmr
//   rtrnmc    radlw_main_nmmb.f:3059-3418   unit phys-rrtm-lw-rtrnmc (McICA, isubclw > 0)
// lwrad is therefore three launches around those calls, all one thread per column (i, j):
//   lwrad_prep   lwrad body before the calls: ivflip branches, colamt and wx, tauaer, cloud unpack
//                (cldfrc, clwp ...), cldfrc(0)/(nlp1) padding, pwvcm, colbrd, secdiff (the a0/a1/a2
//                data table), lcf1 test, semiss, stemp, ipseed.
//   (seam)       the dependency units run on the scratch written by lwrad_prep (listed in the wrapper).
//   lwrad_post   output assembly after the calls: topflx, sfcflx, flxprf, hlwc, hlw0, hlwb (both ivflip
//                orders).
//
// Index convention. Fortran lwrad works on the LOCAL interval arrays of grrad_nmmb: npts = IM, nlay =
// LMK = LM + 1 layers, nlp1 = LMP = nlay + 1 levels. Fortran (iplon, k) with 1-based k maps to CuPy
// [k-1, j, i] (CONTRACT section 1). The inputs are FULL-DOMAIN arrays indexed by global (i, j); the
// scratch covers one chunk of JC rows starting at row jstart (the wrapper loops the chunks, so scratch
// stays bounded). Scratch index uses local row il = j - jstart.
// ivflip = 0 on the NAM path: input TOA->surface (Fortran index k1 = nlp1 - k). Both branches are kept.
//
// Fast numerics: float throughout (kind_phys single), FMA contraction on (NVRTC --fmad=true).
// No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

#define LW_NBDLW     16     // NBANDS = NBDLW (radlw_param_nmmb.f:140,148)
#define LW_NGPTLW    140    // NGPTLW (radlw_param_nmmb.f:141)
#define LW_MAXGAS    7      // maxgas (radlw_param_nmmb.f)
#define LW_MAXXSEC   4      // maxxsec: ccl4, cfc11, cfc12, cfc22
#define LW_NF_CLDS   9      // clouds fields (radiation_clouds_nmmb NF_CLDS)
#define LW_NF_AELW   3      // aerosols fields (NF_AELW, grrad_nmmb.f:1041)
#define LW_NF_VGAS   10     // gasvmr fields (NF_VGAS, grrad_nmmb.f:1035); fields 1..9 are read

// NAM runtime values (module_SOLVER_GRID_COMP.F90:10690-10718 sets IFLIP=0, IOVR_LW=1, ISUBCLW=0,
// ICLIQ_LW=1, ICICE_LW=1, and the rare-gas switch is 1 per physparam.f:73). The wrapper passes
// ivflip, ilwrgas, ilwcliq, isubclw, ipsdlw0 explicitly. ivflip = 0 is the NAM path.
// ipsdlw0 = ngptlw; used only when isubclw == 1.

// physcons (phys/physcons.f): con_g, con_avgd, con_amd, con_amw, con_amo3.
#define LW_CON_G     9.80665f
#define LW_CON_AVGD  6.0221415e23f
#define LW_CON_AMD   28.9644f
#define LW_CON_AMW   18.0154f
#define LW_CON_AMO3  47.9982f
#define LW_AMDW      (LW_CON_AMD / LW_CON_AMW)    // amdw  (radlw_main_nmmb.f:278)
#define LW_AMDO3     (LW_CON_AMD / LW_CON_AMO3)   // amdo3 (radlw_main_nmmb.f:279)
#define LW_EPS       1.0e-6f                       // eps   (radlw_main_nmmb.f:~285)

// Diffusivity-angle data (radlw_main_nmmb.f data statements a0, a1, a2 of lwrad, bands 1..16).
__device__ __constant__ float LW_A0[16] = {1.66f, 1.55f, 1.58f, 1.66f, 1.54f, 1.454f, 1.89f, 1.33f,
                                           1.668f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f};
__device__ __constant__ float LW_A1[16] = {0.00f, 0.25f, 0.22f, 0.00f, 0.13f, 0.446f, -0.10f, 0.40f,
                                           -0.006f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f};
__device__ __constant__ float LW_A2[16] = {0.00f, -12.0f, -11.7f, 0.00f, -0.72f, -0.243f, 0.19f, -0.062f,
                                           0.414f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f};

// 5-D aerosols index: aerosols[NF_AELW, NBDLW, nlay, JM, IM] (Fortran faerlw(i, l, b, n)).
#define LW_AERO_IDX(i0, j0, l0, b0, n0, IM, JM, LM) \
    (((((size_t)(n0) * LW_NBDLW + (b0)) * (LM) + (l0)) * (JM) + (j0)) * (IM) + (i0))

// ---------------------------------------------------------------------------------------------
// lwrad_prep: lwrad statements before the scheme calls, for one column.
//
// Scratch outputs (chunk-local, index IDX3(i0, il, l0, im, jc) unless noted):
//   pavel, tavel, delp  [nlay]          tz [nlp1] (tz(0..nlay), Fortran index = l0)
//   coldry, colbrd      [nlay]          colamt [MAXGAS, nlay] (IDX4)   wx [MAXXSEC, nlay] (IDX4)
//   tauaer [NBDLW, nlay] (IDX4 with b)  cldfrc [nlp1+1] (Fortran index 0..nlp1)
//   cldin [8, nlay] (IDX4): clwp, relw, ciwp, reiw, cda1, cda2, cda3, cda4 (ilwcliq>0 fills all eight;
//                           ilwcliq==0 fills cda1 only, as lwrad does)
//   secdiff, semiss [NBDLW] (IDX3 with band as l0)
//   pwvcm, stemp [1] (IDX2), ipseed, lcf1 int [1] (IDX2)
//   taucld [NBDLW, nlay] (IDX4), cldfmc [NGPTLW, nlay] (IDX4, only when isubclw > 0)
//
// Fortran lwrad (radlw_main_nmmb.f:924-944): cldprop is called only when lcf1, otherwise cldfmc = 0
// and taucld = 0. This kernel owns that zeroing (lcf1 == 0 columns). The cldprop unit writes taucld and
// cldfmc on lcf1 == 1 columns and must leave lcf1 == 0 columns alone.
// Transfer dispatch (radlw_main_nmmb.f:1020-1054) is NOT here: it is selected by isubclw and iovrlw in
// the wrapper (rtrn_unit). rtrn and rtrnmr read cldfrc; rtrnmc reads cldfmc.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void lwrad_prep(
    const float* __restrict__ plyr,      // [nlay, JM, IM]
    const float* __restrict__ plvl,      // [nlp1, JM, IM]
    const float* __restrict__ tlyr,      // [nlay, JM, IM]
    const float* __restrict__ tlvl,      // [nlp1, JM, IM]
    const float* __restrict__ qlyr,      // [nlay, JM, IM]
    const float* __restrict__ olyr,      // [nlay, JM, IM]
    const float* __restrict__ gasvmr,    // [NF_VGAS, nlay, JM, IM]
    const float* __restrict__ clouds,    // [NF_CLDS, nlay, JM, IM]
    const float* __restrict__ aerosols,  // [NF_AELW, NBDLW, nlay, JM, IM]
    const float* __restrict__ sfemis,    // [JM, IM]
    const float* __restrict__ sfgtmp,    // [JM, IM]
    const int*   __restrict__ icseed,    // [JM, IM]
    float* __restrict__ pavel, float* __restrict__ tavel, float* __restrict__ delp,
    float* __restrict__ tz,
    float* __restrict__ coldry, float* __restrict__ colbrd,
    float* __restrict__ colamt, float* __restrict__ wx, float* __restrict__ tauaer,
    float* __restrict__ cldfrc, float* __restrict__ cldin,
    float* __restrict__ taucld, float* __restrict__ cldfmc,
    float* __restrict__ secdiff, float* __restrict__ semiss,
    float* __restrict__ pwvcm, float* __restrict__ stemp,
    int* __restrict__ ipseed, int* __restrict__ lcf1,
    int nlay, int nlp1, int jm, int im, int jstart, int jc,
    int ivflip, int ilwrgas, int ilwcliq, int isubclw, int ipsdlw0)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jc) return;
    const int i0 = col % im;          // Fortran i-1
    const int il = col / im;          // local row inside the chunk
    const int j0 = jstart + il;       // global row j-1
    const int nlay_ = nlay;

    // colamt(:,:) = f_zero, wx, tauaer, cldin start from zero (every entry is then set below)
    for (int m = 0; m < LW_MAXGAS; ++m)
        for (int k = 0; k < nlay_; ++k)
            colamt[IDX4(i0, il, k, m, im, jc, nlay_)] = 0.0f;
    for (int m = 0; m < LW_MAXXSEC; ++m)
        for (int k = 0; k < nlay_; ++k)
            wx[IDX4(i0, il, k, m, im, jc, nlay_)] = 0.0f;
    for (int b = 0; b < LW_NBDLW; ++b)
        for (int k = 0; k < nlay_; ++k)
            tauaer[IDX4(i0, il, k, b, im, jc, nlay_)] = 0.0f;
    for (int s = 0; s < 8; ++s)
        for (int k = 0; k < nlay_; ++k)
            cldin[IDX4(i0, il, k, s, im, jc, nlay_)] = 0.0f;

    // Constants (lwrad lines ~667-672): tem1 = 100 g, tem2 = 1e-20 * 1e3 * avgd.
    const float tem1g = 100.0f * LW_CON_G;
    const float tem2c = 1.0e-20f * 1.0e3f * LW_CON_AVGD;

    // tz(0) (lwrad: tz(0)=tlvl(nlp1) for ivflip 0, tlvl(1) for ivflip 1)
    tz[IDX3(i0, il, 0, im, jc)] = (ivflip == 0) ? tlvl[IDX3(i0, j0, nlay_, im, jm)]
                                                : tlvl[IDX3(i0, j0, 0, im, jm)];

    // ---- layer and level loop (ivflip branches) ----
    for (int k0 = 0; k0 < nlay_; ++k0) {
        // input indices: layer kin, levels for delp and tz
        const int kin = (ivflip == 0) ? (nlay_ - 1 - k0) : k0;
        float dlp, tzk;
        if (ivflip == 0) {
            // delp(k) = plvl(k1+1) - plvl(k1); tz(k) = tlvl(k1)   (1-based k1 = nlp1 - k)
            dlp = plvl[IDX3(i0, j0, nlay_ - k0, im, jm)] - plvl[IDX3(i0, j0, nlay_ - k0 - 1, im, jm)];
            tzk = tlvl[IDX3(i0, j0, nlay_ - k0 - 1, im, jm)];
        } else {
            // delp(k) = plvl(k) - plvl(k+1); tz(k) = tlvl(k+1)
            dlp = plvl[IDX3(i0, j0, k0, im, jm)] - plvl[IDX3(i0, j0, k0 + 1, im, jm)];
            tzk = tlvl[IDX3(i0, j0, k0 + 1, im, jm)];
        }
        const float qv = qlyr[IDX3(i0, j0, kin, im, jm)];
        const float ov = olyr[IDX3(i0, j0, kin, im, jm)];

        pavel[IDX3(i0, il, k0, im, jc)] = plyr[IDX3(i0, j0, kin, im, jm)];
        tavel[IDX3(i0, il, k0, im, jc)] = tlyr[IDX3(i0, j0, kin, im, jm)];
        delp[IDX3(i0, il, k0, im, jc)] = dlp;
        tz[IDX3(i0, il, k0 + 1, im, jc)] = tzk;

        // absorber amounts (ncep model use: specific humidity, mass mixing ratio ozone)
        const float h2ovmr = fmaxf(0.0f, qv * LW_AMDW / (1.0f - qv));
        const float o3vmr = fmaxf(0.0f, ov * LW_AMDO3);
        // tem0 is the molecular weight of moist air
        const float tem0 = (1.0f - h2ovmr) * LW_CON_AMD + h2ovmr * LW_CON_AMW;
        const float cdry = tem2c * dlp / (tem1g * tem0 * (1.0f + h2ovmr));
        const float temcol = 1.0e-12f * cdry;
        coldry[IDX3(i0, il, k0, im, jc)] = cdry;

        colamt[IDX4(i0, il, k0, 0, im, jc, nlay_)] = fmaxf(0.0f, cdry * h2ovmr);            // h2o
        colamt[IDX4(i0, il, k0, 1, im, jc, nlay_)] =
            fmaxf(temcol, cdry * gasvmr[IDX4(i0, j0, kin, 0, im, jm, nlay_)]);              // co2
        colamt[IDX4(i0, il, k0, 2, im, jc, nlay_)] = fmaxf(temcol, cdry * o3vmr);            // o3

        if (ilwrgas > 0) {
            // rare gases: gasvmr fields 2..5 (n2o, ch4, o2, co) and 6..9 (cfc11, cfc12, cfc22, ccl4)
            colamt[IDX4(i0, il, k0, 3, im, jc, nlay_)] =
                fmaxf(temcol, cdry * gasvmr[IDX4(i0, j0, kin, 1, im, jm, nlay_)]);          // n2o
            colamt[IDX4(i0, il, k0, 4, im, jc, nlay_)] =
                fmaxf(temcol, cdry * gasvmr[IDX4(i0, j0, kin, 2, im, jm, nlay_)]);          // ch4
            colamt[IDX4(i0, il, k0, 5, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 3, im, jm, nlay_)]);            // o2
            colamt[IDX4(i0, il, k0, 6, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 4, im, jm, nlay_)]);            // co
            wx[IDX4(i0, il, k0, 0, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 8, im, jm, nlay_)]);            // ccl4
            wx[IDX4(i0, il, k0, 1, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 5, im, jm, nlay_)]);            // cf11
            wx[IDX4(i0, il, k0, 2, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 6, im, jm, nlay_)]);            // cf12
            wx[IDX4(i0, il, k0, 3, im, jc, nlay_)] =
                fmaxf(0.0f, cdry * gasvmr[IDX4(i0, j0, kin, 7, im, jm, nlay_)]);            // cf22
        } else {
            colamt[IDX4(i0, il, k0, 3, im, jc, nlay_)] = 0.0f;
            colamt[IDX4(i0, il, k0, 4, im, jc, nlay_)] = 0.0f;
            colamt[IDX4(i0, il, k0, 5, im, jc, nlay_)] = 0.0f;
            colamt[IDX4(i0, il, k0, 6, im, jc, nlay_)] = 0.0f;
            for (int m = 0; m < LW_MAXXSEC; ++m) wx[IDX4(i0, il, k0, m, im, jc, nlay_)] = 0.0f;
        }

        // aerosol optical properties: tauaer = tau * (1 - ssa), band by band
        for (int b = 0; b < LW_NBDLW; ++b) {
            const float tau = aerosols[LW_AERO_IDX(i0, j0, kin, b, 0, im, jm, nlay_)];
            const float ssa = aerosols[LW_AERO_IDX(i0, j0, kin, b, 1, im, jm, nlay_)];
            tauaer[IDX4(i0, il, k0, b, im, jc, nlay_)] = tau * (1.0f - ssa);
        }

        // clouds (Fortran clouds(iplon,k1,n), n = 1..9; NF_CLDS slots by IDX4 with n0 = n-1)
        if (ilwcliq > 0) {
            cldfrc[IDX3(i0, il, k0 + 1, im, jc)] = clouds[IDX4(i0, j0, kin, 0, im, jm, nlay_)];
            cldin[IDX4(i0, il, k0, 0, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 1, im, jm, nlay_)]; // clwp
            cldin[IDX4(i0, il, k0, 1, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 2, im, jm, nlay_)]; // relw
            cldin[IDX4(i0, il, k0, 2, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 3, im, jm, nlay_)]; // ciwp
            cldin[IDX4(i0, il, k0, 3, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 4, im, jm, nlay_)]; // reiw
            cldin[IDX4(i0, il, k0, 4, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 5, im, jm, nlay_)]; // cda1
            cldin[IDX4(i0, il, k0, 5, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 6, im, jm, nlay_)]; // cda2
            cldin[IDX4(i0, il, k0, 6, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 7, im, jm, nlay_)]; // cda3
            cldin[IDX4(i0, il, k0, 7, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 8, im, jm, nlay_)]; // cda4
        } else {
            cldfrc[IDX3(i0, il, k0 + 1, im, jc)] = clouds[IDX4(i0, j0, kin, 0, im, jm, nlay_)];
            cldin[IDX4(i0, il, k0, 4, im, jc, nlay_)] = clouds[IDX4(i0, j0, kin, 1, im, jm, nlay_)]; // cda1
        }
    }

    // padding (lwrad: cldfrc(0) = 1, cldfrc(nlp1) = 0); cldfrc holds Fortran indices 0..nlp1
    cldfrc[IDX3(i0, il, 0, im, jc)] = 1.0f;
    cldfrc[IDX3(i0, il, nlp1, im, jc)] = 0.0f;

    // column colbrd: colbrd(k) = coldry(k) - sum_{i=2..maxgas} colamt(k,i)
    for (int k0 = 0; k0 < nlay_; ++k0) {
        float summol = 0.0f;
        for (int m = 1; m < LW_MAXGAS; ++m) summol = summol + colamt[IDX4(i0, il, k0, m, im, jc, nlay_)];
        colbrd[IDX3(i0, il, k0, im, jc)] = coldry[IDX3(i0, il, k0, im, jc)] - summol;
    }

    // precipitable water vapour for the diffusivity-angle adjustment
    float tem1 = 0.0f, tem2 = 0.0f;
    for (int k0 = 0; k0 < nlay_; ++k0) {
        tem1 = tem1 + coldry[IDX3(i0, il, k0, im, jc)] + colamt[IDX4(i0, il, k0, 0, im, jc, nlay_)];
        tem2 = tem2 + colamt[IDX4(i0, il, k0, 0, im, jc, nlay_)];
    }
    const float tem0p = 10.0f * tem2 / (LW_AMDW * tem1 * LW_CON_G);
    // ivflip 0: pwvcm = tem0 * plvl(nlp1); ivflip 1: pwvcm = tem0 * plvl(1)
    const float plsfc = (ivflip == 0) ? plvl[IDX3(i0, j0, nlay_, im, jm)] : plvl[IDX3(i0, j0, 0, im, jm)];
    const float pwv = tem0p * plsfc;
    pwvcm[IDX2(i0, il, im)] = pwv;

    // diffusivity angle per band (band j = 1, 4, 10 fixed at 1.66; others 1.50..1.80 vs pwv)
    for (int b0 = 0; b0 < LW_NBDLW; ++b0) {
        const int j1 = b0 + 1;
        if (j1 == 1 || j1 == 4 || j1 == 10) {
            secdiff[IDX3(i0, il, b0, im, jc)] = 1.66f;
        } else {
            const float tem1d = 1.80f, tem2d = 1.50f;
            secdiff[IDX3(i0, il, b0, im, jc)] =
                fminf(tem1d, fmaxf(tem2d, LW_A0[b0] + LW_A1[b0] * expf(LW_A2[b0] * pwv)));
        }
    }

    // cloud flag: any layer with cldfrc > eps (the seam calls cldprop only when lcf1)
    int lcf = 0;
    for (int k = 1; k <= nlay_; ++k) {
        if (cldfrc[IDX3(i0, il, k, im, jc)] > LW_EPS) { lcf = 1; break; }
    }
    lcf1[IDX2(i0, il, im)] = lcf;
    if (lcf == 0) {
        // lwrad: cldprop not called, cldfmc = 0 and taucld = 0 (radlw_main_nmmb.f:942-944)
        for (int b = 0; b < LW_NBDLW; ++b)
            for (int k = 0; k < nlay_; ++k)
                taucld[IDX4(i0, il, k, b, im, jc, nlay_)] = 0.0f;
        if (isubclw > 0)
            for (int g = 0; g < LW_NGPTLW; ++g)
                for (int k = 0; k < nlay_; ++k)
                    cldfmc[IDX4(i0, il, k, g, im, jc, nlay_)] = 0.0f;
    }

    // surface emissivity per band and surface temperature
    const float sfe = sfemis[IDX2(i0, j0, im)];
    for (int b0 = 0; b0 < LW_NBDLW; ++b0) {
        // semiss0 = 1.0 for every band (radlw_main_nmmb.f data statement)
        semiss[IDX3(i0, il, b0, im, jc)] = (sfe > LW_EPS && sfe <= 1.0f) ? sfe : 1.0f;
    }
    stemp[IDX2(i0, il, im)] = sfgtmp[IDX2(i0, j0, im)];

    // permutation seeds (isubclw 1: ipsdlw0 + iplon; 2: icseed; 0: not used, set 0)
    int seed = 0;
    if (isubclw == 1) seed = ipsdlw0 + (i0 + 1);
    else if (isubclw == 2) seed = icseed[IDX2(i0, j0, im)];
    ipseed[IDX2(i0, il, im)] = seed;
}

// ---------------------------------------------------------------------------------------------
// lwrad_post: lwrad statements after the scheme calls, for one column.
// Inputs (scratch from the transfer units, chunk-local):
//   totuflux, totdflux, totuclfl, totdclfl [nlp1, JC, IM] (index 0..nlay: 0 surface, nlay TOA)
//   htr, htrcl [nlay, JC, IM]        htrb [NBDLW, nlay, JC, IM]
// Outputs (global [.., JM, IM]):
//   topflw_upfxc, topflw_upfx0          = totuflux(nlay), totuclfl(nlay)
//   sfcflw_upfxc, sfcflw_dnfxc, sfcflw_upfx0, sfcflw_dnfx0 = (0) of the four flux arrays
//   hlwc [nlay, JM, IM], hlw0 [nlay, JM, IM] (lhlw0), hlwb [NBDLW, nlay, JM, IM] (lhlwb),
//   flxprf [4, nlp1, JM, IM] (lflxprf; fields upfxc, dnfxc, upfx0, dnfx0)
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void lwrad_post(
    const float* __restrict__ totuflux, const float* __restrict__ totdflux,
    const float* __restrict__ totuclfl, const float* __restrict__ totdclfl,
    const float* __restrict__ htr, const float* __restrict__ htrcl,
    const float* __restrict__ htrb,
    float* __restrict__ topflw_upfxc, float* __restrict__ topflw_upfx0,
    float* __restrict__ sfcflw_upfxc, float* __restrict__ sfcflw_upfx0,
    float* __restrict__ sfcflw_dnfxc, float* __restrict__ sfcflw_dnfx0,
    float* __restrict__ hlwc, float* __restrict__ hlw0, float* __restrict__ hlwb,
    float* __restrict__ flxprf,
    int nlay, int nlp1, int jm, int im, int jstart, int jc,
    int ivflip, int lhlw0, int lhlwb, int lflxprf)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= im * jc) return;
    const int i0 = col % im;
    const int il = col / im;
    const int j0 = jstart + il;

    // topflx / sfcflx (Fortran: upfxc = totuflux(nlay) at TOA, sfc = index 0)
    topflw_upfxc[IDX2(i0, j0, im)] = totuflux[IDX3(i0, il, nlay, im, jc)];
    topflw_upfx0[IDX2(i0, j0, im)] = totuclfl[IDX3(i0, il, nlay, im, jc)];
    sfcflw_upfxc[IDX2(i0, j0, im)] = totuflux[IDX3(i0, il, 0, im, jc)];
    sfcflw_upfx0[IDX2(i0, j0, im)] = totuclfl[IDX3(i0, il, 0, im, jc)];
    sfcflw_dnfxc[IDX2(i0, j0, im)] = totdflux[IDX3(i0, il, 0, im, jc)];
    sfcflw_dnfx0[IDX2(i0, j0, im)] = totdclfl[IDX3(i0, il, 0, im, jc)];

    // optional level fluxes: ivflip 0 -> flxprf(iplon, nlp1-k) (1-based); ivflip 1 -> flxprf(iplon, k+1)
    if (lflxprf) {
        const size_t stride = (size_t)jm * im * nlp1;      // stride of one field of flxprf
        for (int k = 0; k <= nlay; ++k) {
            const int lo = (ivflip == 0) ? (nlay - k) : k;     // 0-based output level
            const size_t base = IDX3(i0, j0, lo, im, jm);
            flxprf[0 * stride + base] = totuflux[IDX3(i0, il, k, im, jc)];   // upfxc
            flxprf[1 * stride + base] = totdflux[IDX3(i0, il, k, im, jc)];   // dnfxc
            flxprf[2 * stride + base] = totuclfl[IDX3(i0, il, k, im, jc)];   // upfx0
            flxprf[3 * stride + base] = totdclfl[IDX3(i0, il, k, im, jc)];   // dnfx0
        }
    }

    // heating rates: ivflip 0 -> hlwc(iplon, nlp1-k) with k 1-based; ivflip 1 -> hlwc(iplon, k)
    for (int k0 = 0; k0 < nlay; ++k0) {
        const int lo = (ivflip == 0) ? (nlay - 1 - k0) : k0;   // 0-based output level
        hlwc[IDX3(i0, j0, lo, im, jm)] = htr[IDX3(i0, il, k0, im, jc)];
        if (lhlw0) hlw0[IDX3(i0, j0, lo, im, jm)] = htrcl[IDX3(i0, il, k0, im, jc)];
        if (lhlwb) {
            for (int b0 = 0; b0 < LW_NBDLW; ++b0)
                hlwb[IDX4(i0, j0, lo, b0, im, jm, nlay)] = htrb[IDX4(i0, il, k0, b0, im, jc, nlay)];
        }
    }
}
