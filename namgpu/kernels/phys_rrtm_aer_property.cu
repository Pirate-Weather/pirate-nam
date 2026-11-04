// phys_rrtm_aer_property.cu -- aer_property and its contained radclimaer of the NMMB radiation
// aerosol module: maps the 5-degree climatological aerosol data onto each model column and
// computes the sw and lw aerosol optical properties (optical depth, single-scatter albedo,
// asymmetry) in every spectral band, with the vertical smoothing at domain boundaries.
//
// NOTICE: NAM source file phys/radiation_aerosols_nmmb.f (NOAA/NCEP/EMC, US Government work, no
// licence file). The module header of that file (source lines 1-12) is kept as the notice:
//   !!!!!  'module_radiation_aerosols' description
//   !   this module contains climatological atmospheric aerosol schemes for radiation computations.
// Ported to CUDA for WOOF-NAM from phys/radiation_aerosols_nmmb.f:2377-3107 (nam.v4.2.11, 83df4e13c1):
//   aer_property 2377-2805 (lon/lat bracket, kp/h1 profile choice, bilinear denn/cmix, layer domain
//                index idmaer and delz for ivflip = 1 and 0, call radclimaer, sw/lw copy-out)
//   radclimaer   2806-3103 (rh bracket, domain 5/4/3/1/2/other optical properties, smoothing).
//
// Column kernel: one thread per horizontal point (i, j), all NLAY layers and all bands inside the
// thread (CONTRACT section 3, kernels.grid_cols). Single precision (kind_phys = 4), FMA allowed.
// No host sync, no device allocation, no printf.
//
// Inputs (device, CONTRACT layout: Fortran A(i,j,l) -> [l, j, i]; Fortran index k -> level k-1 in
// memory; the orientation is the Fortran ivflip convention, see the note below):
//   prsi  [NLAY+1, JM, IM]  interface pressure, mb (Fortran prsi(i,k))
//   dz    [NLAY,   JM, IM]  layer thickness, m (dz(i,k))
//   hz    [NLAY+1, JM, IM]  level height, m (hz(i,k))
//   rhlay [NLAY,   JM, IM]  layer mean relative humidity (rhlay(i,k))
//   alon, alat, slmsk [JM, IM] longitude (deg, 0..360), latitude (deg, -90..90), sea/land mask
//   Tables (built once by phys-rrtm-aer-init and phys-rrtm-clim-aerinit; read-only here). Fortran
//   index order is reversed in memory, last Fortran index first:
//   haer, prsref, sigref  [NAE=7, NDM=5]          haer(NDM,NAE) etc.
//   rhlev                 [NRHLEV=8]
//   extrhi, scarhi, ssarhi, asyrhi  [NSWLWBD=30, NCM1=6]     (ic, ib)
//   extrhd, scarhd, ssarhd, asyrhd  [NSWLWBD=30, NCM2=4, NRHLEV=8]   (ir, ic1, ib)
//   extstra               [NSWLWBD=30]
//   cmixg                 [JMXAE=37, IMXAE=72, NXC=5]    (m, i, j)
//   denng                 [JMXAE=37, IMXAE=72, 2]        (m, i, j)
//   idxcg, kprfg          int32 [JMXAE, IMXAE, NXC] and [JMXAE, IMXAE]
// Outputs (device float32, written when the flag is set):
//   aerosw [3, NBDSW=14, NLAY, JM, IM]   (f, m, k) = Fortran aerosw(i,k,m,f)
//   aerolw [3, NBDLW=16, NLAY, JM, IM]
//
// Scalars: IM, JM, NLAY (<= 64), laersw, laerlw (logical flags), ivflip (physparam.f:164,
// default 1 = input from surface to top), nswbnd, nlwbnd, nswlwbd (module variables set in
// aer_init: NSWBND = 14 with laswflg, NLWBND = 16 with lalw1bd false, NSWLWBD = NSWBND+NLWBND).
//
// NaN guard: where ext1 (or sca1) is zero the Fortran ssa1/ext1 is 0/0. MIN(NaN, 1) is compiler-dependent in
// Fortran, so the port sets the no-aerosol values (ssa 1, asy 0) explicitly. tau is 0 there, so the radiation
// result does not depend on ssa or asy in those layers.
//
// Not bound (Fortran arguments the body never reads): prsl, prslk, tvly, tracer (aer_property
// header: "not used"). Not ported (error paths): the print/stop calls for unrealistic surface
// pressure (prsi <= 100), alon outside [0,360], alat outside [-90,90] (the caller clamps alat;
// alon is made non-negative by the caller). Out-of-range table reads that the Fortran leaves
// undefined (sigref(6,kp), a lat bracket j2 = JMXAE+1 at alat = -90) are clamped or skipped.
//
// Bracket search: the Fortran keeps i1/j1 from the previous column as the starting point of its
// search. A fresh search from index 1 gives the same bracket except exactly on a 5-degree grid
// line, where the choice of neighbour (and so the set of kp candidates) can differ. That is the
// only order dependence the port does not reproduce.

#include "nmmb_common.cuh"

#define AER_NDM      5     // NDM   atmos aerosol domains
#define AER_NAE      7     // NAE   aerosol profile structures
#define AER_IMXAE    72    // IMXAE lon points of the climatological aerosol set
#define AER_JMXAE    37    // JMXAE lat points
#define AER_NXC      5     // NXC   max components in a profile
#define AER_NRHLEV   8     // NRHLEV rh levels
#define AER_NCM1     6     // NCM1  rh-independent species
#define AER_NCM2     4     // NCM2  rh-dependent species
#define AER_NCM      10    // NCM   NCM1 + NCM2
#define AER_NSWLWBD  30    // table stride of the sw+lw band dimension (NBDSW + NBDLW)
#define AER_NBDSW    14    // radsw_param_nmmb NBDSW (NBANDS = 29 - 16 + 1)
#define AER_NBDLW    16    // radlw_param_nmmb NBDLW
#define AER_NLAY_MAX 64    // local scratch bound (NAM LM = 60)

// Table accessors (Fortran index -> memory offset, see header).
#define HAER(idm, kp)          haer[((kp) - 1) * AER_NDM + ((idm) - 1)]
#define PRSREF(idm, kp)        prsref[((kp) - 1) * AER_NDM + ((idm) - 1)]
#define SIGREF(idm, kp)        sigref[((kp) - 1) * AER_NDM + ((idm) - 1)]
#define RHLEV(ir)              rhlev[(ir) - 1]
#define EXTRHI(ic, ib)         extrhi[((ib) - 1) * AER_NCM1 + ((ic) - 1)]
#define SCARHI(ic, ib)         scarhi[((ib) - 1) * AER_NCM1 + ((ic) - 1)]
#define SSARHI(ic, ib)         ssarhi[((ib) - 1) * AER_NCM1 + ((ic) - 1)]
#define ASYRHI(ic, ib)         asyrhi[((ib) - 1) * AER_NCM1 + ((ic) - 1)]
#define EXTRHD(ir, ic1, ib)    extrhd[(((ib) - 1) * AER_NCM2 + ((ic1) - 1)) * AER_NRHLEV + ((ir) - 1)]
#define SCARHD(ir, ic1, ib)    scarhd[(((ib) - 1) * AER_NCM2 + ((ic1) - 1)) * AER_NRHLEV + ((ir) - 1)]
#define SSARHD(ir, ic1, ib)    ssarhd[(((ib) - 1) * AER_NCM2 + ((ic1) - 1)) * AER_NRHLEV + ((ir) - 1)]
#define ASYRHD(ir, ic1, ib)    asyrhd[(((ib) - 1) * AER_NCM2 + ((ic1) - 1)) * AER_NRHLEV + ((ir) - 1)]
#define EXTSTRA(ib)            extstra[(ib) - 1]
#define CMIXG(m, i, j)         cmixg[(((j) - 1) * AER_IMXAE + ((i) - 1)) * AER_NXC + ((m) - 1)]
#define IDXCG(m, i, j)         idxcg[(((j) - 1) * AER_IMXAE + ((i) - 1)) * AER_NXC + ((m) - 1)]
#define DENNG(m, i, j)         denng[(((j) - 1) * AER_IMXAE + ((i) - 1)) * 2 + ((m) - 1)]
#define KPRFG(i, j)            kprfg[((j) - 1) * AER_IMXAE + ((i) - 1)]

// Level k (Fortran 1-based) of a [NLAY(+1), JM, IM] field at this column.
#define LV(arr, k)             arr[(size_t)((k) - 1) * NCOL + col]

__device__ __forceinline__ size_t aer_out_idx(int f, int m, int nb, int k0, int nlay,
                                              size_t ncol, size_t col)
{
    // [f, m, k, j, i] with i fastest; k0 zero-based layer.
    return ((((size_t)f * (size_t)nb + (size_t)m) * (size_t)nlay + (size_t)k0) * ncol) + col;
}

extern "C" __global__ void phys_rrtm_aer_property_column(
    const float* __restrict__ prsi, const float* __restrict__ dz, const float* __restrict__ hz,
    const float* __restrict__ rhlay, const float* __restrict__ alon, const float* __restrict__ alat,
    const float* __restrict__ slmsk,
    const float* __restrict__ haer, const float* __restrict__ prsref, const float* __restrict__ sigref,
    const float* __restrict__ rhlev,
    const float* __restrict__ extrhi, const float* __restrict__ scarhi,
    const float* __restrict__ ssarhi, const float* __restrict__ asyrhi,
    const float* __restrict__ extrhd, const float* __restrict__ scarhd,
    const float* __restrict__ ssarhd, const float* __restrict__ asyrhd,
    const float* __restrict__ extstra,
    const float* __restrict__ cmixg, const float* __restrict__ denng,
    const int* __restrict__ idxcg, const int* __restrict__ kprfg,
    float* __restrict__ aerosw, float* __restrict__ aerolw,
    int IM, int JM, int NLAY, int laersw, int laerlw, int ivflip,
    int nswbnd, int nlwbnd, int nswlwbd)
{
    COLUMN_IJ(IM, JM)
    const size_t NCOL = (size_t)IM * (size_t)JM;
    const size_t col = (size_t)col_;
    if (NLAY > AER_NLAY_MAX) return;

    const float f_zero = 0.0f;
    const float f_one  = 1.0f;
    const float dltg = 360.0f / (float)AER_IMXAE;       // 5 deg
    const float hdlt = 0.5f * dltg;
    const float rdlt = f_one / dltg;

    // ---- aer_property: map the climatology onto this point (fresh bracket search) ----------
    const float alon_v = alon[col];
    const float alat_v = alat[col];
    const float slm_v  = slmsk[col];

    // Fortran lab_do_IMXAE: starts at i3 = 1, steps up while dtmp > dltg.
    int i3 = 1;
    float dtmp = alon_v;
    // alon < 0 and alon > 360 are refused by the wrapper (Fortran stops here). The clamp below only
    // keeps the device path in bounds for inputs that were never checked.
    if (dtmp < f_zero) dtmp = f_zero;
    while (dtmp > dltg && i3 < AER_IMXAE) {
        ++i3;
        dtmp = alon_v - dltg * (float)(i3 - 1);
    }
    const int i1 = i3;
    const int i2 = i3 % AER_IMXAE + 1;                    // longitude wraps
    const float wi = dtmp * rdlt;
    const int kpi = (dtmp <= hdlt) ? i3 : i2;

    // Fortran lab_do_JMXAE: tmp2 = 90 - dltg*(j3-1), dtmp = tmp2 - alat.
    int j3 = 1;
    float tmp2 = 90.0f;
    float dtl = tmp2 - alat_v;
    while (dtl > dltg && j3 < AER_JMXAE) {
        ++j3;
        tmp2 = 90.0f - dltg * (float)(j3 - 1);
        dtl = tmp2 - alat_v;
    }
    const int j1 = j3;
    // j3 reaches JMXAE only when alat < -90, which the wrapper refuses (Fortran lab_do_JMXAE stops).
    // For alat in [-90, 90] this clamp is never taken: at alat = -90 exactly j3 = 36 and j2 = 37.
    const int j2 = (j3 + 1 > AER_JMXAE) ? AER_JMXAE : j3 + 1;
    const float wj = dtl * rdlt;
    const int kpj = (dtl <= hdlt) ? j3 : j2;

    // nearest typical profile, then the domain-overlay rules
    int kp = KPRFG(kpi, kpj);
    const int kpa = max(max(KPRFG(i1, j1), KPRFG(i1, j2)), max(KPRFG(i2, j1), KPRFG(i2, j2)));
    float h1 = HAER(1, kp);
    int ii = 1;
    if (kp != kpa) {
        if (kpa == 6) {                                  // ocean profile with mineral overlay
            ii = 2;
            if (slm_v > f_zero) {                        // land/sea-ice point
                kp = 7;
                h1 = 0.5f * (HAER(1, 6) + HAER(1, 7));
            } else {
                kp = kpa;
                h1 = HAER(1, 6);
            }
        } else if (kpa == 7) {                           // land profile with mineral overlay
            ii = 2;
            if (slm_v <= f_zero) {                       // ocean point
                kp = 6;
                h1 = 0.5f * (HAER(1, 6) + HAER(1, 7));
            } else {
                kp = kpa;
                h1 = HAER(1, 7);
            }
        } else {                                         // lower atmos without mineral overlay
            h1 = HAER(1, kpa);
            kp = kpa;
        }
    }

    // bilinear weights (w11 = (i1,j1), w12 = (i1,j2), w21 = (i2,j1), w22 = (i2,j2))
    const float w11 = (f_one - wi) * (f_one - wj);
    const float w12 = (f_one - wi) * wj;
    const float w21 = wi * (f_one - wj);
    const float w22 = wi * wj;

    // aerosol number densities (denn(1..2), 1-based), ii = 1 or 2
    float denn[3] = {f_zero, f_zero, f_zero};
    for (int m = 1; m <= ii; ++m) {
        denn[m] = w11 * DENNG(m, i1, j1) + w12 * DENNG(m, i1, j2)
                + w21 * DENNG(m, i2, j1) + w22 * DENNG(m, i2, j2);
    }

    // interpolated component mixing ratios cmix(1..NCM)
    float cmix[AER_NCM + 1];
    for (int n = 0; n <= AER_NCM; ++n) cmix[n] = f_zero;
    for (int m = 1; m <= AER_NXC; ++m) {
        int ic = IDXCG(m, i1, j1);
        if (ic > 0) cmix[ic] = cmix[ic] + w11 * CMIXG(m, i1, j1);
        ic = IDXCG(m, i1, j2);
        if (ic > 0) cmix[ic] = cmix[ic] + w12 * CMIXG(m, i1, j2);
        ic = IDXCG(m, i2, j1);
        if (ic > 0) cmix[ic] = cmix[ic] + w21 * CMIXG(m, i2, j1);
        ic = IDXCG(m, i2, j2);
        if (ic > 0) cmix[ic] = cmix[ic] + w22 * CMIXG(m, i2, j2);
    }

    // layer domain index idmaer(k) and effective thickness delz(k), k 1-based
    float delz[AER_NLAY_MAX + 2];
    int   idmaer[AER_NLAY_MAX + 2];
    int   ih1v[AER_NLAY_MAX + 2];
    int   ih2v[AER_NLAY_MAX + 2];
    float rdrhv[AER_NLAY_MAX + 2];

    if (ivflip == 1) {                                   // input from surface to top
        const float rps = f_one / LV(prsi, 1);           // Fortran: stop if prsi(i,1) <= 100
        ii = 1;
        for (int k = 1; k <= NLAY; ++k) {
            // ii <= NDM guard. The Fortran reads sigref(6,kp), which aliases the next profile. The guard is
            // inert with the shipped table: prsref(5,kp) = 0 gives sigref(5,kp) = 0, so ii never passes 5.
            if (ii <= AER_NDM && LV(prsi, k + 1) * rps < SIGREF(ii, kp)) {
                ii = ii + 1;
                if (ii == 2 && PRSREF(2, kp) == PRSREF(3, kp)) ii = 3;
            }
            idmaer[k] = ii;
            const float tmp1 = (ii > 1) ? ((ii <= AER_NDM) ? HAER(ii, kp) : f_zero) : h1;
            if (tmp1 > f_zero) {
                const float tmp2b = f_one / tmp1;
                delz[k] = tmp1 * (expf(-LV(hz, k) * tmp2b) - expf(-LV(hz, k + 1) * tmp2b));
            } else {
                delz[k] = LV(dz, k);
            }
        }
    } else {                                             // input from top to surface
        const float rps = f_one / LV(prsi, NLAY + 1);   // Fortran: stop if prsi(i,NLP1) <= 100
        ii = 1;
        for (int k = NLAY; k >= 1; --k) {
            if (ii <= AER_NDM && LV(prsi, k) * rps < SIGREF(ii, kp)) {
                ii = ii + 1;
                if (ii == 2 && PRSREF(2, kp) == PRSREF(3, kp)) ii = 3;
            }
            idmaer[k] = ii;
            const float tmp1 = (ii > 1) ? ((ii <= AER_NDM) ? HAER(ii, kp) : f_zero) : h1;
            if (tmp1 > f_zero) {
                const float tmp2b = f_one / tmp1;
                delz[k] = tmp1 * (expf(-LV(hz, k + 1) * tmp2b) - expf(-LV(hz, k) * tmp2b));
            } else {
                delz[k] = LV(dz, k);
            }
        }
    }

    // ---- radclimaer: rh bracket per layer (independent of band) ----------------------------
    for (int kk = 1; kk <= NLAY; ++kk) {
        const float rh1 = LV(rhlay, kk);
        int ih2 = 1;
        while (ih2 <= AER_NRHLEV && rh1 > RHLEV(ih2)) ++ih2;
        int ih1 = max(1, ih2 - 1);
        ih2 = min(AER_NRHLEV, ih2);
        const float drh0 = RHLEV(ih2) - RHLEV(ih1);
        const float drh1 = rh1 - RHLEV(ih1);
        rdrhv[kk] = (ih1 == ih2) ? f_zero : drh1 / drh0;
        ih1v[kk] = ih1;
        ih2v[kk] = ih2;
    }

    // ---- radclimaer: optical properties band by band (layers are independent per band) -----
    // Each band is computed for all layers, smoothed across layers (the Fortran smoothing acts on
    // one band at a time), then copied to the output arrays.
    const float crt1 = 30.0f;
    const float crt2 = 0.03333f;
    float tau[AER_NLAY_MAX + 2];
    float ssa[AER_NLAY_MAX + 2];
    float asy[AER_NLAY_MAX + 2];

    for (int ib = 1; ib <= nswlwbd; ++ib) {
        for (int kk = 1; kk <= NLAY; ++kk) {
            const int idom = idmaer[kk];
            const int ih1 = ih1v[kk];
            const int ih2 = ih2v[kk];
            const float rdrh = rdrhv[kk];
            const float dzk = delz[kk];

            if (idom == 5) {                             // 5th domain: upper stratosphere, no aerosol
                tau[kk] = f_zero;
                if (ib <= nswbnd) { ssa[kk] = 0.99f; asy[kk] = 0.696f; }
                else              { ssa[kk] = 0.5f;  asy[kk] = 0.3f;   }
            } else if (idom == 4) {                      // 4th domain: stratospheric layers
                tau[kk] = EXTSTRA(ib) * dzk;
                if (ib <= nswbnd) { ssa[kk] = 0.99f; asy[kk] = 0.696f; }
                else              { ssa[kk] = 0.5f;  asy[kk] = 0.3f;   }
            } else if (idom == 3) {                      // 3rd domain: free troposphere
                const float ex01 = EXTRHI(1, ib);
                const float sc01 = SCARHI(1, ib);
                const float ss01 = SSARHI(1, ib);
                const float as01 = ASYRHI(1, ib);

                const float ex02 = EXTRHI(2, ib);
                const float sc02 = SCARHI(2, ib);
                const float ss02 = SSARHI(2, ib);
                const float as02 = ASYRHI(2, ib);

                const float ex03 = EXTRHD(ih1, 1, ib) + rdrh * (EXTRHD(ih2, 1, ib) - EXTRHD(ih1, 1, ib));
                const float sc03 = SCARHD(ih1, 1, ib) + rdrh * (SCARHD(ih2, 1, ib) - SCARHD(ih1, 1, ib));
                const float ss03 = SSARHD(ih1, 1, ib) + rdrh * (SSARHD(ih2, 1, ib) - SSARHD(ih1, 1, ib));
                const float as03 = ASYRHD(ih1, 1, ib) + rdrh * (ASYRHD(ih2, 1, ib) - ASYRHD(ih1, 1, ib));

                const float ext1 = 0.17e-3f * ex01 + 0.4f * ex02 + 0.59983f * ex03;
                const float sca1 = 0.17e-3f * sc01 + 0.4f * sc02 + 0.59983f * sc03;
                const float ssa1 = 0.17e-3f * ss01 * ex01 + 0.4f * ss02 * ex02 + 0.59983f * ss03 * ex03;
                const float asy1 = 0.17e-3f * as01 * sc01 + 0.4f * as02 * sc02 + 0.59983f * as03 * sc03;

                tau[kk] = ext1 * 730.0f * dzk;
                ssa[kk] = (ext1 > f_zero) ? fminf(f_one, ssa1 / ext1) : f_one;
                asy[kk] = (sca1 > f_zero) ? fminf(f_one, asy1 / sca1) : f_zero;
            } else if (idom == 1) {                      // 1st domain: mixing layer
                float ext1 = f_zero;
                float sca1 = f_zero;
                float ssa1 = f_zero;
                float asy1 = f_zero;

                for (int icmp = 1; icmp <= AER_NCM; ++icmp) {
                    const float cm = cmix[icmp];
                    if (cm > f_zero) {
                        if (icmp <= AER_NCM1) {          // component without rh dependence
                            const float tt0 = cm * EXTRHI(icmp, ib);
                            ext1 = ext1 + tt0;
                            sca1 = sca1 + cm * SCARHI(icmp, ib);
                            ssa1 = ssa1 + cm * SSARHI(icmp, ib) * EXTRHI(icmp, ib);
                            asy1 = asy1 + cm * ASYRHI(icmp, ib) * SCARHI(icmp, ib);
                        } else {                         // component with rh dependence
                            const int ic1 = icmp - AER_NCM1;
                            const float ex00 = EXTRHD(ih1, ic1, ib)
                                             + rdrh * (EXTRHD(ih2, ic1, ib) - EXTRHD(ih1, ic1, ib));
                            const float sc00 = SCARHD(ih1, ic1, ib)
                                             + rdrh * (SCARHD(ih2, ic1, ib) - SCARHD(ih1, ic1, ib));
                            const float ss00 = SSARHD(ih1, ic1, ib)
                                             + rdrh * (SSARHD(ih2, ic1, ib) - SSARHD(ih1, ic1, ib));
                            const float as00 = ASYRHD(ih1, ic1, ib)
                                             + rdrh * (ASYRHD(ih2, ic1, ib) - ASYRHD(ih1, ic1, ib));
                            const float tt0 = cm * ex00;
                            ext1 = ext1 + tt0;
                            sca1 = sca1 + cm * sc00;
                            ssa1 = ssa1 + cm * ss00 * ex00;
                            asy1 = asy1 + cm * as00 * sc00;
                        }
                    }
                }

                tau[kk] = ext1 * denn[1] * dzk;
                ssa[kk] = (ext1 > f_zero) ? fminf(f_one, ssa1 / ext1) : f_one;
                asy[kk] = (sca1 > f_zero) ? fminf(f_one, asy1 / sca1) : f_zero;
            } else if (idom == 2) {                      // 2nd domain: mineral transport layers
                tau[kk] = EXTRHI(6, ib) * denn[2] * dzk;
                ssa[kk] = SSARHI(6, ib);
                asy[kk] = ASYRHI(6, ib);
            } else {                                     // domain index out of range: no aerosol
                tau[kk] = f_zero;
                ssa[kk] = f_one;
                asy[kk] = f_zero;
            }
        }

        // vertical smoothing at domain boundaries (ivflip == 0: toa to sfc; else sfc to toa)
        if (ivflip == 0) {
            for (int kk = 2; kk <= NLAY; ++kk) {
                const float ratio = (tau[kk] > f_zero) ? tau[kk - 1] / tau[kk] : f_one;
                const float tt0 = tau[kk] + tau[kk - 1];
                const float tt1 = 0.2f * tt0;
                const float tt2 = tt0 - tt1;
                if (ratio > crt1) { tau[kk] = tt1; tau[kk - 1] = tt2; }
                if (ratio < crt2) { tau[kk] = tt2; tau[kk - 1] = tt1; }
            }
        } else {
            for (int kk = NLAY - 1; kk >= 1; --kk) {
                const float ratio = (tau[kk] > f_zero) ? tau[kk + 1] / tau[kk] : f_one;
                const float tt0 = tau[kk] + tau[kk + 1];
                const float tt1 = 0.2f * tt0;
                const float tt2 = tt0 - tt1;
                if (ratio > crt1) { tau[kk] = tt1; tau[kk + 1] = tt2; }
                if (ratio < crt2) { tau[kk] = tt2; tau[kk + 1] = tt1; }
            }
        }

        // copy out: sw bands 1..NBDSW use band m; lw bands use NSWBND + m (or band NSWBND+1 for all m)
        if (laersw && ib <= AER_NBDSW) {
            const int m = ib - 1;
            for (int kk = 1; kk <= NLAY; ++kk) {
                aerosw[aer_out_idx(0, m, AER_NBDSW, kk - 1, NLAY, NCOL, col)] = tau[kk];
                aerosw[aer_out_idx(1, m, AER_NBDSW, kk - 1, NLAY, NCOL, col)] = ssa[kk];
                aerosw[aer_out_idx(2, m, AER_NBDSW, kk - 1, NLAY, NCOL, col)] = asy[kk];
            }
        }
        if (laerlw && nlwbnd > 0) {                      // NLWBND = 0: no lw bands to copy (Fortran would overrun)
            if (nlwbnd == 1) {                           // one lw band: every lw output uses NSWBND+1
                if (ib == nswbnd + 1) {
                    for (int m = 0; m < AER_NBDLW; ++m) {
                        for (int kk = 1; kk <= NLAY; ++kk) {
                            aerolw[aer_out_idx(0, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = tau[kk];
                            aerolw[aer_out_idx(1, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = ssa[kk];
                            aerolw[aer_out_idx(2, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = asy[kk];
                        }
                    }
                }
            } else if (ib > nswbnd) {
                const int m = ib - nswbnd - 1;           // Fortran m1 = NSWBND + m, m = 1..NBDLW
                if (m >= 0 && m < AER_NBDLW) {
                    for (int kk = 1; kk <= NLAY; ++kk) {
                        aerolw[aer_out_idx(0, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = tau[kk];
                        aerolw[aer_out_idx(1, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = ssa[kk];
                        aerolw[aer_out_idx(2, m, AER_NBDLW, kk - 1, NLAY, NCOL, col)] = asy[kk];
                    }
                }
            }
        }
    }
}
