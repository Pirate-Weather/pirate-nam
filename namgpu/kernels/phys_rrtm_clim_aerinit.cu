// phys_rrtm_clim_aerinit.cu -- CLIM_AERINIT, SET_AERCOEF and OPTAVG (module_radiation_aerosols_nmmb).
//
// NOTICE: NAM source file phys/radiation_aerosols_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). Subroutine header kept here as the notice:
//   subroutine clim_aerinit (solfwv, eirfwv, me): the opac-climatology aerosol initialization program.
//   set_aercoef maps the pre-tabulated aerosol optical spectral data onto the sw and lw radiation bands
//   (solar-flux and ir-flux weights per aerosol data band); optavg computes the band-mean optical properties
//   for each species (follows the GFDL approach for thick-cloud optical properties in sw, 2000).
// Ported to CUDA for nam-gpu from phys/radiation_aerosols_nmmb.f:821-1498 (nam.v4.2.11, 83df4e13c1).
//
// Host side (namgpu/physics/rrtm/phys_rrtm_clim_aerinit.py) reads the data file
// parm/nam_climaeropac_global.txt (the 12 monthly global distributions are skipped, as the Fortran does)
// and uploads the optical tables; the file has no per-cell content. The band mapping, the weights and
// the optical averages are computed here, in two launches.
//
// Launch 1: clim_aerinit_bands  (set_aercoef band mapping, sigref)
//   thread t < NSW  : sw band t  (1..NSWBND): loops iw1..iw2 over solfwv, fills solbnd, solwaer, nv1, nv2
//   thread t < NLW  : lw band t  (1..NLWBND): loops iw1..iw2 over eirfwv, fills eirbnd, eirwaer, nr1, nr2
//                     (NLWBND == 1 uses the fixed 400..2500 range, as the Fortran)
//   thread t >= NSW+NLW: sigref(e) = 0.001 * prsref(e)
// Launch 2: clim_aerinit_optavg  (optavg)
//   one thread per (output band ib = 0..nswbnd+nlwbnd-1, slot k = 0..38). ib is the table row, Fortran
//   ib: sw band ib (ib < nswbnd) or lw band ib-nswbnd (ib = NSWBND + nb). The rows are not fixed at 30:
//   with laswflg false (nswbnd 0) lw band nb lands at row nb, as the Fortran does.
//     k < NCM1                : rh-independent species nc = k          (extrhi, scarhi, ssarhi, asyrhi)
//     NCM1 <= k < NCM1+NCM2*8 : rh-dependent nc, nh (m = k-NCM1, nc = m/8, nh = m%8)
//                               (extrhd, scarhd, ssarhd, asyrhd)
//     k == 38                 : stratospheric background extstra
//   Each thread runs the Fortran ni = nv1..nv2 (sw) or nr1..nr2 (lw) sum in the Fortran order.
//
// Fortran-to-CUDA review (every assignment of the line range):
//   set_aercoef sw band loop (sw block of the file):
//     zero solbnd(:), solwaer(:,:)        -> band-row zeroing in launch 1 (rows of active bands)
//     ii/iw1/iw2 mapping, do while        -> while (ii != NAERBND && iw1 > iendwv(ii)) ii++
//     solbnd, sumsol, solwaer(ib,ii)      -> solbnd, sumsol, solwaer assignment at iw == iendwv(ii)
//     nv1(ib) = ii (start), nv2(ib) = ii (end), final solwaer(ib,ii)=sumsol when iw2 /= iendwv(ii)
//   set_aercoef lw block: same, with NLWBND==1 special case (400..2500) and eirbnd/eirwaer/nr1/nr2
//   sigref(:,:) = 0.001 * prsref(:,:)     -> launch 1 tail threads
//   optavg sw: rsolbd = 1/solbnd; per species sums sumk, sums, sumok, sumokg, sumreft; refb, extrhi,
//              scarhi, asyrhi (sumokg/(sumok+1e-10)), ssarhi (4*refb/((1+refb)^2-asy*(1-refb)^2))
//   optavg sw rh-dependent: same with rhdp* arrays, nh loop inside nc loop
//   optavg sw strat: sumk = sum straext0*solwaer, extstra = sumk*rsolbd
//   optavg lw: identical to sw with eirwaer/nr1/nr2/eirbnd, output row ib = NSWBND + nb (NSWBND = 0 when
//   laswflg is false, so the row is nb). Output tables are sized nswbnd + nlwbnd rows (wrapper _band_outputs).
//   The Fortran laswflg / lalwflg gates are the nswbnd / nlwbnd / flag arguments.
//   Precision: kind_phys = 4 (machine.f SINGLE_PREC), float everywhere; sqrt and division IEEE; fmad on.
//   Not ported: me == 0 diagnostic prints and the commented-out print blocks (no state written).
//   The aerosol data file not-found print and stop (radiation_aerosols_nmmb.f:1008-1013) is raised as
//   FileNotFoundError with the same text in the wrapper (parse_optical).
//   Gates: laswflg/lalwflg/lalw1bd enter as the counts nswbnd (14 or 0) and nlwbnd (16, 1 or 0); a disabled
//   set has no output rows and no threads.
//   The solfwv/eirfwv inputs are computed by aer_init (unit phys-rrtm-aer-init), not by this unit.

#define NAERBND 61        // num of bands for clim aer data (opac)
#define NRHLEV  8         // num of rh levels for rh-dep components
#define NCM1    6         // rh-independent species
#define NCM2    4         // rh-dependent species
#define NDM     5         // atmos aerosol domains
#define NAE     7         // aerosol profile structures
#define NSW     14        // NBDSW: sw bands (radsw_param_nmmb.f NBANDS = NBHGH-NBLOW+1)
#define NLW     16        // NBDLW: lw bands (radlw_param_nmmb.f NBANDS)
#define NSWLWBD (NSW + NLW)
#define NSLOT   (NCM1 + NCM2 * NRHLEV + 1)   // 39 slots per band in optavg

// ---------------------------------------------------------------------------
// Launch 1: band mapping (set_aercoef) and sigref
// ---------------------------------------------------------------------------
extern "C" __global__ void clim_aerinit_bands(
    const float* __restrict__ solfwv,     // [NWVTOT=57600]  Fortran solfwv(1..)
    const float* __restrict__ eirfwv,     // [NWVTIR=4000]   Fortran eirfwv(1..)
    const int*   __restrict__ iendwv,     // [NAERBND]       ending wavenumber per aerosol band
    const float* __restrict__ wvnsw1,     // [NSW]           sw band start wavenumber (wvnum1)
    const float* __restrict__ wvnsw2,     // [NSW]           sw band end wavenumber (wvnum2)
    const float* __restrict__ wvnlw1,     // [NLW]           lw band start wavenumber
    const float* __restrict__ wvnlw2,     // [NLW]           lw band end wavenumber
    const float* __restrict__ prsref,     // [NAE*NDM]       prsref(NDM,NAE), flat (ni fastest)
    float* __restrict__ sigref,           // [NAE*NDM]       output
    float* __restrict__ solbnd,           // [NSW]           output
    float* __restrict__ solwaer,          // [NAERBND*NSW]   solwaer(NSW,NAERBND): index ni*NSW + ib
    int*   __restrict__ nv1,              // [NSW]
    int*   __restrict__ nv2,              // [NSW]
    float* __restrict__ eirbnd,           // [NLW]
    float* __restrict__ eirwaer,          // [NAERBND*NLW]   eirwaer(NLW,NAERBND): index ni*NLW + ib
    int*   __restrict__ nr1,              // [NLW]
    int*   __restrict__ nr2,              // [NLW]
    const int nswbnd,                     // NSWBND (= NBDSW when laswflg, else 0)
    const int nlwbnd,                     // NLWBND (= NBDLW, or 1 when lalw1bd, or 0 when not lalwflg)
    const int laswflg,
    const int lalwflg)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < NSW) {
        // ---- sw band ib = t (Fortran ib = t+1, mb = ib + NSWSTR - 1 = ib, NSWSTR = 1)
        if (!laswflg || t >= nswbnd) return;
        const int ib = t;
        for (int ni = 0; ni < NAERBND; ++ni) solwaer[ni * NSW + ib] = 0.0f;   // solwaer(:,:) = 0
        float sb = 0.0f;                                                         // solbnd(ib) = 0
        const int iw1 = (int)roundf(wvnsw1[ib]);
        const int iw2 = (int)roundf(wvnsw2[ib]);

        int ii = 1;                                                              // Fortran 1-based ii
        while (ii != NAERBND && iw1 > iendwv[ii - 1]) ++ii;                      // do while (iw1 > iendwv(ii))

        float sumsol = 0.0f;
        nv1[ib] = ii;
        for (int iw = iw1; iw <= iw2; ++iw) {
            const float f = solfwv[iw - 1];
            sb = sb + f;                                                         // solbnd(ib) += solfwv(iw)
            sumsol = sumsol + f;
            if (iw == iendwv[ii - 1]) {
                solwaer[(ii - 1) * NSW + ib] = sumsol;                           // solwaer(ib,ii) = sumsol
                if (ii < NAERBND) {
                    sumsol = 0.0f;
                    ii = ii + 1;
                }
            }
        }
        if (iw2 != iendwv[ii - 1]) solwaer[(ii - 1) * NSW + ib] = sumsol;
        nv2[ib] = ii;
        solbnd[ib] = sb;
        return;
    }

    if (t < NSW + NLW) {
        // ---- lw band nb = t - NSW (Fortran nb = 1..NLWBND, ib = NSWBND + nb in optavg)
        const int nb = t - NSW;
        if (!lalwflg || nb >= nlwbnd) return;
        for (int ni = 0; ni < NAERBND; ++ni) eirwaer[ni * NLW + nb] = 0.0f;   // eirwaer(:,:) = 0
        float eb = 0.0f;                                                         // eirbnd(nb) = 0
        int iw1, iw2;
        if (nlwbnd == 1) {                                                       // NLWBND == 1 (lalw1bd)
            iw1 = 400;                                                           // corresponding 25 mu
            iw2 = 2500;                                                          // corresponding 4 mu
        } else {
            iw1 = (int)roundf(wvnlw1[nb]);
            iw2 = (int)roundf(wvnlw2[nb]);
        }

        int ii = 1;
        while (ii != NAERBND && iw1 > iendwv[ii - 1]) ++ii;

        float sumir = 0.0f;
        nr1[nb] = ii;
        for (int iw = iw1; iw <= iw2; ++iw) {
            const float f = eirfwv[iw - 1];
            eb = eb + f;                                                         // eirbnd(nb) += eirfwv(iw)
            sumir = sumir + f;
            if (iw == iendwv[ii - 1]) {
                eirwaer[(ii - 1) * NLW + nb] = sumir;
                if (ii < NAERBND) {
                    sumir = 0.0f;
                    ii = ii + 1;
                }
            }
        }
        if (iw2 != iendwv[ii - 1]) eirwaer[(ii - 1) * NLW + nb] = sumir;
        nr2[nb] = ii;
        eirbnd[nb] = eb;
        return;
    }

    // ---- sigref(:,:) = 0.001 * prsref(:,:)   (NDM*NAE entries)
    const int e = t - NSWLWBD;
    if (e < NDM * NAE) sigref[e] = 0.001f * prsref[e];
}

// ---------------------------------------------------------------------------
// Launch 2: optavg. One thread per (band, slot).
// ---------------------------------------------------------------------------
extern "C" __global__ void clim_aerinit_optavg(
    const float* __restrict__ rhidext0,   // [NCM1, NAERBND]          rhidext0(NAERBND,NCM1), index nc*NAERBND + ni
    const float* __restrict__ rhidsca0,   // [NCM1, NAERBND]
    const float* __restrict__ rhidssa0,   // [NCM1, NAERBND]
    const float* __restrict__ rhidasy0,   // [NCM1, NAERBND]
    const float* __restrict__ rhdpext0,   // [NCM2, NRHLEV, NAERBND]  index (nc*NRHLEV + nh)*NAERBND + ni
    const float* __restrict__ rhdpsca0,
    const float* __restrict__ rhdpssa0,
    const float* __restrict__ rhdpasy0,
    const float* __restrict__ straext0,   // [NAERBND]
    const float* __restrict__ solbnd,     // [NSW]
    const float* __restrict__ solwaer,    // [NAERBND*NSW]
    const int*   __restrict__ nv1,        // [NSW]
    const int*   __restrict__ nv2,        // [NSW]
    const float* __restrict__ eirbnd,     // [NLW]
    const float* __restrict__ eirwaer,    // [NAERBND*NLW]
    const int*   __restrict__ nr1,        // [NLW]
    const int*   __restrict__ nr2,        // [NLW]
    float* __restrict__ extrhi,           // [nswbnd+nlwbnd, NCM1]    extrhi(NCM1,NSWBND+NLWBND): index ib*NCM1 + nc
    float* __restrict__ scarhi,
    float* __restrict__ ssarhi,
    float* __restrict__ asyrhi,
    float* __restrict__ extrhd,           // [nswbnd+nlwbnd, NCM2, NRHLEV]  index (ib*NCM2 + nc)*NRHLEV + nh
    float* __restrict__ scarhd,
    float* __restrict__ ssarhd,
    float* __restrict__ asyrhd,
    float* __restrict__ extstra,          // [nswbnd+nlwbnd]
    const int nswbnd,
    const int nlwbnd,
    const int laswflg,
    const int lalwflg)
{
    const int nbt = nswbnd + nlwbnd;          // output bands (table rows)
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nbt * NSLOT) return;
    const int ib = gid / NSLOT;               // output row: sw 0..nswbnd-1, then lw (Fortran ib = NSWBND + nb)
    const int k  = gid - ib * NSLOT;          // slot
    const int isw = (ib < nswbnd);
    const int nb = isw ? ib : ib - nswbnd;    // Fortran nb (0-based) into solbnd/nv1 or eirbnd/nr1

    // weights and band range for this band (sw: solwaer/nv1/nv2/solbnd; lw: eirwaer/nr1/nr2/eirbnd)
    const float* W = isw ? solwaer : eirwaer;
    const int    S = isw ? NSW : NLW;
    const int lo = isw ? nv1[nb] : nr1[nb];   // 1-based Fortran ni start
    const int hi = isw ? nv2[nb] : nr2[nb];   // 1-based Fortran ni end
    const float rnorm = 1.0f / (isw ? solbnd[nb] : eirbnd[nb]);   // rsolbd / rirbd

    if (k < NCM1) {
        // ---- rh-independent species nc = k
        const int nc = k;
        float sumk = 0.0f, sums = 0.0f, sumok = 0.0f, sumokg = 0.0f, sumreft = 0.0f;
        for (int ni = lo; ni <= hi; ++ni) {
            const float w   = W[(ni - 1) * S + nb];
            const int   idx = nc * NAERBND + (ni - 1);
            const float ssa = rhidssa0[idx];
            const float asy = rhidasy0[idx];
            const float ext = rhidext0[idx];
            const float sca = rhidsca0[idx];
            const float sp   = sqrtf((1.0f - ssa) / (1.0f - ssa * asy));
            const float reft = (1.0f - sp) / (1.0f + sp);
            sumreft = sumreft + reft * w;
            sumk    = sumk + ext * w;
            sums    = sums + sca * w;
            sumok   = sumok + ssa * w * ext;
            sumokg  = sumokg + ssa * w * ext * asy;
        }
        const float refb = sumreft * rnorm;
        const float asyo = sumokg / (sumok + 1.0e-10f);
        const int o = ib * NCM1 + nc;
        extrhi[o] = sumk * rnorm;
        scarhi[o] = sums * rnorm;
        asyrhi[o] = asyo;
        ssarhi[o] = 4.0f * refb / ((1.0f + refb) * (1.0f + refb) - asyo * ((1.0f - refb) * (1.0f - refb)));
        return;
    }

    if (k < NCM1 + NCM2 * NRHLEV) {
        // ---- rh-dependent species nc, rh level nh
        const int m  = k - NCM1;
        const int nc = m / NRHLEV;
        const int nh = m - nc * NRHLEV;
        float sumk = 0.0f, sums = 0.0f, sumok = 0.0f, sumokg = 0.0f, sumreft = 0.0f;
        for (int ni = lo; ni <= hi; ++ni) {
            const float w   = W[(ni - 1) * S + nb];
            const int   idx = (nc * NRHLEV + nh) * NAERBND + (ni - 1);
            const float ssa = rhdpssa0[idx];
            const float asy = rhdpasy0[idx];
            const float ext = rhdpext0[idx];
            const float sca = rhdpsca0[idx];
            const float sp   = sqrtf((1.0f - ssa) / (1.0f - ssa * asy));
            const float reft = (1.0f - sp) / (1.0f + sp);
            sumreft = sumreft + reft * w;
            sumk    = sumk + ext * w;
            sums    = sums + sca * w;
            sumok   = sumok + ssa * w * ext;
            sumokg  = sumokg + ssa * w * ext * asy;
        }
        const float refb = sumreft * rnorm;
        const float asyo = sumokg / (sumok + 1.0e-10f);
        const int o = (ib * NCM2 + nc) * NRHLEV + nh;
        extrhd[o] = sumk * rnorm;
        scarhd[o] = sums * rnorm;
        asyrhd[o] = asyo;
        ssarhd[o] = 4.0f * refb / ((1.0f + refb) * (1.0f + refb) - asyo * ((1.0f - refb) * (1.0f - refb)));
        return;
    }

    // ---- stratospheric background aerosol
    float sumk = 0.0f;
    for (int ni = lo; ni <= hi; ++ni) {
        sumk = sumk + straext0[ni - 1] * W[(ni - 1) * S + nb];
    }
    extstra[ib] = sumk * rnorm;
}
