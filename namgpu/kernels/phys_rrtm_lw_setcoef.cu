// phys_rrtm_lw_setcoef.cu -- SETCOEF: per-layer interpolation coefficients for the NMMB RRTM
// longwave (RRTM group). One thread per (i, j) column, the layer loop k = 1..nlay runs inside
// the thread, every term of the Fortran routine is present.
//
// NOAA notice: phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, US government work
// (no licence file; see NOTICE). The lw-rrtm3 package header of that file is kept as the notice:
// !!!!!  lw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep modifications of the rrtm-lw radiation code from aer inc.     !
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:1863-2121 (SUBROUTINE setcoef).
// The lookup tables it reads are the RRTM reference data the NMMB build links: module_radlw_avplank
// totplnk (radlw_datatb.f:58-728), module_radlw_ref preflog/tref/chi_mls (radlw_datatb.f:733-905) and
// the band width delwave (radlw_param.f:133), passed in by the wrapper. radlw_datatb_nmmb.f defines
// other modules (module_radlw_avplank_nmmb) that radlw_main_nmmb.f does not use.
//
// Vertical order: SURFACE FIRST, radlw's INTERNAL order (not the NMMB top-first order of CONTRACT
// section 1). Fortran setcoef indexes its layers from the surface: k = 1 is the lowest layer and
// tz(0) is the surface level (radlw_main_nmmb.f:660, "vertical index of internal array is from
// surface to top"; the ivflip == 0 branch sets k1 = nlp1 - k at lines 673-680). Every per-layer
// array below has index 0 = lowest layer, and tz plane 0 = surface. The kernel does no flip. The
// flip is done once in lwrad_prep (phys_rrtm_lw_lwrad.cu, kin = nlay-1-k0), which writes the scratch
// this kernel reads. laytrop counts layers with ln(p) > 4.56 in that order.
//
// Precision: float throughout (the kfpt = single ruling). The NAM configure that the makefiles include
// (src/conf/configure.nems) does not define SINGLE_PREC, so machine.f selects kind_phys = 8 there. The
// pinned reference build's flags are not recorded in the tree. Open: project lead to rule the reference precision.
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   pavel, tavel, h2ovmr, coldry, colbrd, layer outputs   [nlay, JM, IM]    (Fortran k = 1..nlay -> 0-based k-1,
//                                                                          surface first, see Vertical order)
//   tz                                                     [nlay+1, JM, IM] (Fortran tz(0:nlay) -> plane k, plane 0 = tz(0))
//   stemp                                                  [JM, IM]
//   colamt                                                 [MAXGAS, nlay, JM, IM] (Fortran colamt(k,m))
//   totplnk (Fortran totplnk(NPLNK,NBANDS))                [NBANDS, NPLNK]
//   delwave [NBANDS];  preflog, tref [NREF]; chi_mls (Fortran chi_mls(7,NREF)) [NREF, 7]
//   laytrop                                                [JM, IM] int32
//   pklay, pklev (Fortran pklay(NBANDS,0:nlay))            [nlay+1, NBANDS, JM, IM]
//   jp, jt, jt1, indself, indfor, indminor                 [nlay, JM, IM] int32
//   rfrate (Fortran rfrate(nlay,NRATES,2))                 [2, NRATES, nlay, JM, IM]
//   all other per-layer outputs                            [nlay, JM, IM] float32
//
// Notes on the translation:
//   - Fortran INT() truncates toward zero; the C casts do the same for these ranges.
//   - Single precision literals (f suffix) everywhere, as kfpt = single.
//   - rfrate entries the Fortran does not set on a branch (upper atmosphere sets rates 1 and 6 only,
//     lower sets 1..5) are left as the wrapper's zeroed scratch; the Fortran leaves them undefined and
//     the transfer routines do not read them on that branch.
//   - parameter STPFAC = 296.0/1013.0 (radlw_main_nmmb.f:271), NRATES = 6 (radlw_param.f:99).
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation.

#include "nmmb_common.cuh"

#define NBANDS_LW   16
#define NPLNK_LW    181
#define NREF_LW     59
#define NGAS_REF    7
#define MAXGAS_LW   6
#define NRATES_LW   6
#define STPFAC_LW   (296.0f / 1013.0f)

// Fortran 1-based accessors, mapped onto the row-major tables.
#define TOT(k, b)    totplnk[(size_t)((b) - 1) * NPLNK_LW + ((k) - 1)]     // totplnk(k, b)
#define DELW(b)      delwave[(b) - 1]                                       // delwave(b)
#define PREF(k)      preflog[(k) - 1]                                       // preflog(k)
#define TREF(k)      tref[(k) - 1]                                          // tref(k)
#define CHI(g, lev)  chi_mls[((size_t)((lev) - 1)) * NGAS_REF + ((g) - 1)]  // chi_mls(g, lev)

// pklay / pklev (k = 0..nlay, band b): [k, band, j, i]
#define PKIDX(k, b)  ((((size_t)(k)) * NBANDS_LW + ((b) - 1)) * ((size_t)JM * IM) + IDX2(i0, j0, IM))
// rfrate(k, m, n) Fortran, stored [n, m, k, j, i] (n = side 1|2, m = rate 1..6, k = layer 1..nlay)
#define RFIDX(m, n, k) (((((size_t)((n) - 1) * NRATES_LW + ((m) - 1)) * nlay) + ((k) - 1)) * ((size_t)JM * IM) + IDX2(i0, j0, IM))
// colamt(k, m) Fortran, stored [m, k, j, i]
// colamt(k, m) as a VALUE (the macro once returned the flat index, so selffac, forfac and scaleminorn2 were
// scaled by an array offset: LW optical depths came out ~1e3-1e4 too large and OLR ~140 W m-2).
#define COLAMT(m, k) colamt[IDX4(i0, j0, (k) - 1, (m) - 1, IM, JM, nlay)]

extern "C" __global__ void rrtm_lw_setcoef(
    const float* __restrict__ pavel,
    const float* __restrict__ tavel,
    const float* __restrict__ tz,
    const float* __restrict__ stemp,
    const float* __restrict__ h2ovmr,
    const float* __restrict__ colamt,
    const float* __restrict__ coldry,
    const float* __restrict__ colbrd,
    const float* __restrict__ totplnk,
    const float* __restrict__ delwave,
    const float* __restrict__ preflog,
    const float* __restrict__ tref,
    const float* __restrict__ chi_mls,
    int nlay, int IM, int JM,
    int* __restrict__ laytrop,
    float* __restrict__ pklay,
    float* __restrict__ pklev,
    int* __restrict__ jp,
    int* __restrict__ jt,
    int* __restrict__ jt1,
    float* __restrict__ rfrate,
    float* __restrict__ fac00,
    float* __restrict__ fac01,
    float* __restrict__ fac10,
    float* __restrict__ fac11,
    float* __restrict__ selffac,
    float* __restrict__ selffrac,
    int* __restrict__ indself,
    float* __restrict__ forfac,
    float* __restrict__ forfrac,
    int* __restrict__ indfor,
    float* __restrict__ minorfrac,
    float* __restrict__ scaleminor,
    float* __restrict__ scaleminorn2,
    int* __restrict__ indminor)
{
    COLUMN_IJ(IM, JM)

    // ---- surface (level 0) Planck functions: pklay(i,0) from stemp, pklev(i,0) from tz(0). ----
    const float stemp_v = stemp[IDX2(i0, j0, IM)];
    const float tz0 = tz[IDX3(i0, j0, 0, IM, JM)];
    const int indlay0 = min(180, max(1, (int)(stemp_v - 159.0f)));
    const int indlev0 = min(180, max(1, (int)(tz0 - 159.0f)));
    const float tlyrfr0 = stemp_v - (float)((int)stemp_v);
    const float tlvlfr0 = tz0 - (float)((int)tz0);
    for (int i = 1; i <= NBANDS_LW; ++i) {
        const float tem1 = TOT(indlay0 + 1, i) - TOT(indlay0, i);
        const float tem2 = TOT(indlev0 + 1, i) - TOT(indlev0, i);
        pklay[PKIDX(0, i)] = DELW(i) * (TOT(indlay0, i) + tlyrfr0 * tem1);
        pklev[PKIDX(0, i)] = DELW(i) * (TOT(indlev0, i) + tlvlfr0 * tem2);
    }

    int laytrop_c = 0;

    // ---- layer loop ----
    for (int k = 1; k <= nlay; ++k) {
        const size_t lk = IDX3(i0, j0, k - 1, IM, JM);
        const float pav = pavel[lk];
        const float tav = tavel[lk];
        const float tzk = tz[IDX3(i0, j0, k, IM, JM)];

        // Planck functions at layer and level temperatures.
        const int indlay = min(180, max(1, (int)(tav - 159.0f)));
        const float tlyrfr = tav - (float)((int)tav);
        const int indlev = min(180, max(1, (int)(tzk - 159.0f)));
        const float tlvlfr = tzk - (float)((int)tzk);
        for (int i = 1; i <= NBANDS_LW; ++i) {
            pklay[PKIDX(k, i)] = DELW(i) * (TOT(indlay, i) + tlyrfr * (TOT(indlay + 1, i) - TOT(indlay, i)));
            pklev[PKIDX(k, i)] = DELW(i) * (TOT(indlev, i) + tlvlfr * (TOT(indlev + 1, i) - TOT(indlev, i)));
        }

        // Reference pressures on either side of the layer pressure.
        const float plog = logf(pav);
        const int jpk = max(1, min(58, (int)(36.0f - 5.0f * (plog + 0.04f))));
        const int jp1 = jpk + 1;
        const float fp = fmaxf(0.0f, fminf(1.0f, 5.0f * (PREF(jpk) - plog)));

        // Reference temperatures nearest the layer temperature, and the fractions ft, ft1.
        const float tem1t = (tav - TREF(jpk)) / 15.0f;
        const float tem2t = (tav - TREF(jp1)) / 15.0f;
        const int jtk = max(1, min(4, (int)(3.0f + tem1t)));
        const int jt1k = max(1, min(4, (int)(3.0f + tem2t)));
        const float ft = fmaxf(-0.5f, fminf(1.5f, tem1t - (float)(jtk - 3)));
        const float ft1 = fmaxf(-0.5f, fminf(1.5f, tem2t - (float)(jt1k - 3)));

        jp[lk] = jpk;
        jt[lk] = jtk;
        jt1[lk] = jt1k;

        // Interpolation factors for the four reference points.
        const float tem1 = 1.0f - fp;
        fac10[lk] = tem1 * ft;
        fac00[lk] = tem1 * (1.0f - ft);
        fac11[lk] = fp * ft1;
        fac01[lk] = fp * (1.0f - ft1);

        // Water vapour foreign and self continuum scale factors (scaled by colamt(k,1) below).
        const float h2o = h2ovmr[lk];
        const float forf = pav * STPFAC_LW / (tav * (1.0f + h2o));
        const float selff = h2o * forf;

        // Minor gas factors.
        scaleminor[lk] = pav / tav;
        scaleminorn2[lk] = (pav / tav) * (colbrd[lk] / (coldry[lk] + COLAMT(1, k)));
        const float tem1m = (tav - 180.8f) / 7.2f;
        const int indmin = min(18, max(1, (int)tem1m));
        indminor[lk] = indmin;
        minorfrac[lk] = tem1m - (float)indmin;

        if (plog > 4.56f) {
            // Lower atmosphere: foreign and self continuum, and the h2o/co2, h2o/o3, h2o/n2o,
            // h2o/ch4 and n2o/co2 binary species reference ratios.
            laytrop_c += 1;

            const float tem1f = (332.0f - tav) / 36.0f;
            const int indf = min(2, max(1, (int)tem1f));
            indfor[lk] = indf;
            forfrac[lk] = tem1f - (float)indf;

            const float tem1s = (tav - 188.0f) / 7.2f;
            const int inds = min(9, max(1, (int)tem1s - 7));
            indself[lk] = inds;
            selffrac[lk] = tem1s - (float)(inds + 7);

            rfrate[RFIDX(1, 1, k)] = CHI(1, jpk) / CHI(2, jpk);
            rfrate[RFIDX(1, 2, k)] = CHI(1, jp1) / CHI(2, jp1);

            rfrate[RFIDX(2, 1, k)] = CHI(1, jpk) / CHI(3, jpk);
            rfrate[RFIDX(2, 2, k)] = CHI(1, jp1) / CHI(3, jp1);

            rfrate[RFIDX(3, 1, k)] = CHI(1, jpk) / CHI(4, jpk);
            rfrate[RFIDX(3, 2, k)] = CHI(1, jp1) / CHI(4, jp1);

            rfrate[RFIDX(4, 1, k)] = CHI(1, jpk) / CHI(6, jpk);
            rfrate[RFIDX(4, 2, k)] = CHI(1, jp1) / CHI(6, jp1);

            rfrate[RFIDX(5, 1, k)] = CHI(4, jpk) / CHI(2, jpk);
            rfrate[RFIDX(5, 2, k)] = CHI(4, jp1) / CHI(2, jp1);
        } else {
            // Upper atmosphere: no water vapour self continuum, and the h2o/co2 and o3/co2
            // binary species reference ratios.
            const float tem1f = (tav - 188.0f) / 36.0f;
            indfor[lk] = 3;
            forfrac[lk] = tem1f - 1.0f;

            indself[lk] = 0;
            selffrac[lk] = 0.0f;

            rfrate[RFIDX(1, 1, k)] = CHI(1, jpk) / CHI(2, jpk);
            rfrate[RFIDX(1, 2, k)] = CHI(1, jp1) / CHI(2, jp1);

            rfrate[RFIDX(6, 1, k)] = CHI(3, jpk) / CHI(2, jpk);
            rfrate[RFIDX(6, 2, k)] = CHI(3, jp1) / CHI(2, jp1);
        }

        // Rescale selffac and forfac by the water vapour column amount colamt(k,1).
        selffac[lk] = COLAMT(1, k) * selff;
        forfac[lk] = COLAMT(1, k) * forf;
    }

    laytrop[IDX2(i0, j0, IM)] = laytrop_c;
}
