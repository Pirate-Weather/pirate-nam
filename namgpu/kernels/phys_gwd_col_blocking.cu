// phys_gwd_col_blocking.cu -- GWD_col part 1: setup, Lott-Miller mountain blocking and the
// mountain-induced drag terms (DB) of the NAM gravity wave drag / mountain blocking scheme.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). The module
// header of MODULE_GWD (nmm/module_GWD.F90) is kept as the notice:
//
//   !-----------------------------------------------------------------------
//         MODULE MODULE_GWD
//   !***  Module for Gravity Wave Drag (GWD) and Mountain Blocking (MB)
//   !***  Initially incorporated into the WRF NMM from the GFS by B. Ferrier
//   !***  in April/May 2007.
//   !***  Ratko added in NMM-B (July '08)
//   !***  Mountain blocking made more robust in July 2016 by BSF
//   !-----------------------------------------------------------------------
//   Routine GWD_col: original GFS documentation block and version history as in
//   nmm/module_GWD.F90:368-420 (Lott & Miller mountain blocking, *j*, 4/02).
//
// Ported to CUDA for nam-gpu from nmm/module_GWD.F90:368-897 (nam.v4.2.11, 83df4e13c1).
//
// PRECISION: kind_phys = SELECTED_REAL_KIND(13,60) (module_GWD.F90 line 19 neighbourhood, real*8),
// so every REAL(kind=kind_phys) variable and PARAMETER of GWD_col is double here. The tunables
// CDMB, SIGFAC, DPMIN are default REAL dummy arguments (module_GWD.F90:471). The as-compiled NMM
// build compiles module_GWD.f90 with -fdefault-real-8 -fdefault-double-8 (gate0 logs/build_nmmb_clean.log,
// the mpif90 line that follows the module_GWD.F90 cpp line), so they are REAL*8 there and are passed
// as double here. Under the same flags the PARAMETER literals above (9.806, 1.0E-5, ...) are double
// too, so the double literals in this file are the Fortran values, not single-rounded ones.
// CONTRACT section 3 exception: this unit is double because module_GWD declares
// KIND_PHYS=SELECTED_REAL_KIND(13,60); the FP64 cost on GeForce is not measured (open).
// Fast numerics: fmad left on.
//
// VERTICAL ORDER: GWD_col runs BOTTOM-UP (Fortran K = 1 is the surface; GWD_DRIVER flips the NAM
// top-down layers, module_GWD.F90:142 KFLIP = LM-K+1). Arrays here use the same order: the input
// [LM, JM, IM] fields and PRSI/PHII [LM+1, JM, IM] are indexed by the Fortran K (D3 macro below).
//
// THREADS: one thread per column (COLUMN_IJ). The Fortran compacts the qualifying points into
// ipt(1:npt) and loops over npt; each column is independent in this routine, so the compaction is
// not needed and every qualifying column is computed in its own thread. The Fortran-global KBPS
// (max over points of kref) and KMPS (min) are reduced with atomicMax/atomicMin into device
// scalars (initialised by the wrapper with fill, no host sync). The Fortran loop
// "DO K=1,KBPS; IF (K < kref)" is run per column as K = 1 .. kref-1, the same set of terms.
//
// Launch: one launch, all arguments device pointers or scalars (CONTRACT section 3).
//
// Outputs carried to part 2 (phys-gwd-col-wave) and to phys-gwd-driver: ELEVMX, IDXZB, KREF, OA,
// CLX, UBAR, VBAR, ROLL, BNV2BAR, DELKS, DELKS1, DB, BNV2, RI_N, RO, TAUP (zeroed here, filled by
// part 2), KBPS, KMPS, DUsfc, DVsfc (zeroed here). ZBK and WK are diagnostics kept for completeness.
// Locals (ANG, UDS, VTJ, VTK, IWK, KMPBL) are recomputed where used with the Fortran expressions.

#pragma once
#include "nmmb_common.cuh"

// Fortran module_GWD.F90 PARAMETER block (kind_phys, double), lines 487-520.
#define GWD_PI          3.1415926535897931
#define GWD_G           9.806
#define GWD_CP          1004.6
#define GWD_RD          287.04
#define GWD_RV          461.6
#define GWD_FV          (GWD_RV / GWD_RD - 1.0)
#define GWD_RDI         (1.0 / GWD_RD)
#define GWD_GOCP        (GWD_G / GWD_CP)
#define GWD_ROG         (1.0 / GWD_G)
#define GWD_ROG2        (GWD_ROG * GWD_ROG)
#define GWD_DW2MIN      1.0
#define GWD_RIMIN       (-100.0)
#define GWD_BNV2MIN     1.0e-5
#define GWD_HZERO       0.0
#define GWD_HNCRIT      8000.0      /* hncrit: max ELVMAX (m) for the blocking height */
#define GWD_HMINMT      50.0        /* hminmt: min mountain height (m) */
#define GWD_HPMIN       1.0         /* hpmin: minimum HPRIME (std dev of terrain) */
#define GWD_MINWND      0.1         /* minwnd: min wind component */
#define GWD_RAD_TO_DEG  (180.0 / GWD_PI)
#define GWD_DEG_TO_RAD  (GWD_PI / 180.0)
#define GWD_MDIR        8
#define GWD_FDIR        (8.0 / (GWD_PI + GWD_PI))   /* FDIR = mdir/(PI+PI) */
#define GWD_LMAX        64          /* host check: LM <= GWD_LMAX (bnv2lm column buffer) */

// Fortran F(i,j,k) with 1-based bottom-up level k, for any 3-D field [n, JM, IM] (n = LM, LM+1, 4).
#define D3(p, k)  (p)[IDX3(i0, j0, (k) - 1, IM, JM)]
// Fortran F(i,j) for a 2-D field [JM, IM].
#define D2(p)     (p)[IDX2(i0, j0, IM)]

// Fortran VTJ, VTK, RO at level k (blocking section and general section of GWD_col).
#define GWD_VTJ(k)  (D3(T1, k) * (1.0 + GWD_FV * D3(Q1, k)))
#define GWD_VTK(k)  (GWD_VTJ(k) / D3(PRSLK, k))
#define GWD_RO(k)   (GWD_RDI * D3(PRSL, k) / GWD_VTJ(k))

// NWDIR template (data nwdir /6,7,5,8,2,3,1,4/): direction index 1..8 -> NWD.
__constant__ int GWD_NWDIR[8] = {6, 7, 5, 8, 2, 3, 1, 4};

// ANG(I,K) of the Fortran: angle between the wind and the range, branch-corrected, radians.
__device__ __forceinline__ double gwd_ang(double thet, double u, double v)
{
    const double phiang = atan2(v, u) * GWD_RAD_TO_DEG;
    double ang = thet - phiang;
    if (ang > 90.0)  ang -= 180.0;
    if (ang < -90.0) ang += 180.0;
    return ang * GWD_DEG_TO_RAD;
}

// UDS(I,K) of the Fortran: low-level wind speed, floored at MINWND.
__device__ __forceinline__ double gwd_uds(double u, double v)
{
    return fmax(sqrt(u * u + v * v), GWD_MINWND);
}

__device__ bool gwd_col_blocking_dev(const int i0, const int j0,
    const double* __restrict__ U1,      // U1      zonal wind                 [LM, JM, IM]
    const double* __restrict__ V1,      // V1      meridional wind            [LM, JM, IM]
    const double* __restrict__ T1,      // T1      temperature                [LM, JM, IM]
    const double* __restrict__ Q1,      // Q1      specific humidity          [LM, JM, IM]
    const double* __restrict__ DEL,     // DEL     layer pressure thickness   [LM, JM, IM]
    const double* __restrict__ PRSL,    // PRSL    mid-layer pressure         [LM, JM, IM]
    const double* __restrict__ PRSLK,   // PRSLK   Exner function             [LM, JM, IM]
    const double* __restrict__ PHIL,    // PHIL    mid-layer geopotential     [LM, JM, IM]
    const double* __restrict__ PRSI,    // PRSI    interface pressure         [LM+1, JM, IM]
    const double* __restrict__ PHII,    // PHII    interface geopotential    [LM+1, JM, IM]
    const double* __restrict__ OA4,     // OA4     orographic asymmetry (4)   [4, JM, IM]
    const double* __restrict__ CLX4,    // CLX4    orographic length (4)     [4, JM, IM]
    const double* __restrict__ HPRIME,  // HPRIME  orographic std dev         [JM, IM]
    const double* __restrict__ THETA,   // THETA   range angle (deg)          [JM, IM]
    const double* __restrict__ SIGMA,   // SIGMA   orographic slope           [JM, IM]
    const double* __restrict__ GAMMA,   // GAMMA   anisotropy                 [JM, IM]
    const double* __restrict__ ELVMAX,  // ELVMAX  max height above mean      [JM, IM]
    const int*    __restrict__ KPBL,    // KPBL    PBL top index (bottom-up)  [JM, IM]
    double* DUsfc,                      // DUsfc   surface zonal stress (zeroed)      [JM, IM]
    double* DVsfc,                      // DVsfc   surface meridional stress (zeroed) [JM, IM]
    double* ELEVMX,                     // ELEVMX  blocking height limit (out)        [JM, IM]
    int*    IDXZB,                      // IDXZB   dividing streamline level (out)    [JM, IM]
    int*    KREF,                       // KREF    reference level (out)              [JM, IM]
    double* OA,                         // OA      signed asymmetry for direction (out) [JM, IM]
    double* CLX,                        // CLX     length scale for direction (out)   [JM, IM]
    double* UBAR,                       // UBAR    mean U below kref (out)            [JM, IM]
    double* VBAR,                       // VBAR    mean V below kref (out)            [JM, IM]
    double* ROLL,                       // ROLL    mean density below kref (out)      [JM, IM]
    double* BNV2BAR,                    // BNV2BAR mean N^2 below kref (out)          [JM, IM]
    double* DELKS,                      // DELKS   1/(PRSI(1)-PRSI(kref)) (out)       [JM, IM]
    double* DELKS1,                     // DELKS1  1/(PRSL(1)-PRSL(kref)) (out)       [JM, IM]
    double* ZBK,                        // ZBK     blocking-height diagnostic (out)   [JM, IM]
    double* WK,                         // WK      interpolated mountain diagnostic  [JM, IM]
    double* DB,                         // DB      mountain-blocking drag (out)       [LM, JM, IM]
    double* BNV2,                       // BNV2    N^2 (out)                          [LM, JM, IM]
    double* RI_N,                       // RI_N    Richardson number (out)            [LM, JM, IM]
    double* RO,                         // RO      density (out)                     [LM, JM, IM]
    double* TAUP,                       // TAUP    momentum flux (zeroed; part 2)     [LM+1, JM, IM]
    int*    KBPS,                       // KBPS    max kref over points (device scalar)
    int*    KMPS,                       // KMPS    min kref over points (device scalar)
    int*    GUARD,                      // GUARD   [0] = points that read an unset BNV2LM (must be 0)
    double CDMB,                         // CDMB    mountain blocking coefficient (REAL)
    double SIGFAC,                       // SIGFAC  ELVMAX height factor (REAL)
    double DPMIN,                        // DPMIN   min reference layer thickness (REAL, centibars)
    int IM, int JM, int LM)
{
    const double cdmb   = CDMB;
    const double sigfac = SIGFAC;
    const double dpmin  = DPMIN;
    const int    KMM1   = LM - 1;                 // kmm1
    const int    KMLL   = LM - 1;                 // kmll = kmm1 (blocking section)
    const int    KMPBL  = LM / 2;                 // kmpbl = km/2 (integer division)
    const size_t c2     = IDX2(i0, j0, IM);

    // ---- Fortran first loop: DUsfc, DVsfc = 0; ELEVMX = ELVMAX; zeroing of DB (ANG/UDS are
    // recomputed where used, see gwd_ang/gwd_uds).
    D2(DUsfc) = 0.0;
    D2(DVsfc) = 0.0;
    D2(ELEVMX) = ELVMAX[c2];
    for (int k = 1; k <= LM; ++k) {
        D3(DB, k) = 0.0;
        D3(BNV2, k) = 0.0;
        D3(RI_N, k) = 0.0;
        D3(RO, k) = 0.0;
    }
    for (int k = 1; k <= LM + 1; ++k) D3(TAUP, k) = 0.0;   // Fortran TAUP(I,K)=0.0
    // BNV2(I,LM) and RI_N(I,LM) are never set by the Fortran (its loops stop at K = KM-1,
    // module_GWD.F90:820-836) and nothing in 368-897 or in phys-gwd-col-wave reads level LM. The
    // Fortran value there is undefined, so it is NaN-poisoned here instead of a silent zero. The
    // check tool requires NaN exactly at level LM of these two fields and finite everywhere else.
    D3(BNV2, LM) = nan("");
    D3(RI_N, LM) = nan("");
    D2(IDXZB) = 0;
    D2(KREF) = 0;
    D2(OA) = 0.0;
    D2(CLX) = 0.0;
    D2(UBAR) = 0.0;
    D2(VBAR) = 0.0;
    D2(ROLL) = 0.0;
    D2(BNV2BAR) = 0.0;
    D2(DELKS) = 0.0;
    D2(DELKS1) = 0.0;
    D2(ZBK) = 0.0;
    D2(WK) = 0.0;

    // ---- point selection, NMTVR >= 14 branch. NMTVR = 14 is set by GWD_init (module_GWD.F90:79),
    // so the NMTVR < 14 branch is not reachable in this build and is not ported.
    const double hp      = HPRIME[c2];
    const double elvmax  = ELVMAX[c2];
    if (!(elvmax > GWD_HMINMT && hp > GWD_HPMIN)) return false;   // not a GWD/MB point: outputs stay 0

    const double gam  = GAMMA[c2];
    const double sig  = SIGMA[c2];
    const double thet = THETA[c2];

    // ==== Mountain blocking section (NMTVR .eq. 14) ====
    int    iwklm  = 2;
    int    kreflm = 0;
    int    idxzb  = 0;
    // ELEVMX = min(ELEVMX + sigfac*hprime, hncrit)
    const double elevmx = fmin(elvmax + sigfac * hp, GWD_HNCRIT);
    double wk_b = 0.0;

    // BNV2LM(K) for K = 1 .. KMLL-1. Entries K >= KMLL are never set by the Fortran (undefined
    // there); they are 0 here, and GUARD[0] counts any point whose reads reach them (see below).
    double bnv2lm[GWD_LMAX];
    for (int k = 0; k < GWD_LMAX; ++k) bnv2lm[k] = 0.0;

    // Interpolate the max mountain height to an index: IWKLM and WK.
    for (int k = 1; k <= KMLL; ++k) {
        const double pkp1log = D3(PHIL, k + 1) * GWD_ROG;
        const double pklog   = D3(PHIL, k) * GWD_ROG;
        if (elevmx <= pkp1log && elevmx >= pklog) {
            wk_b = GWD_G * elevmx / (D3(PHIL, k + 1) - D3(PHIL, k));
            iwklm = max(iwklm, k + 1);
        }
    }

    // Runtime witness for the unset BNV2LM entries. The PE loop below reads BNV2LM(K) for K <= iwklm,
    // and kreflm < iwklm; both stay inside the set range K <= KMLL-1 exactly when iwklm <= KMLL-1.
    // iwklm = k+1 with ELEVMX <= phil(k+1)/G, and ELEVMX <= HNCRIT = 8000 m, so for NAM's 60 levels
    // this never fires; the count makes that a measured fact, and the check tool requires zero.
    if (iwklm >= KMLL && GUARD) atomicAdd(&GUARD[0], 1);

    // Brunt-Vaisala frequency BNV2LM from the virtual potential temperature VTK.
    for (int k = 1; k <= KMLL - 1; ++k) {
        const double rdz = GWD_G / (D3(PHIL, k + 1) - D3(PHIL, k));
        const double vk1 = GWD_VTK(k + 1);
        const double vk  = GWD_VTK(k);
        const double b   = (GWD_G + GWD_G) * rdz * (vk1 - vk) / (vk1 + vk);
        bnv2lm[k - 1] = fmax(b, GWD_BNV2MIN);
    }

    // Reference level of the dividing streamline: the largest Ktrial < IWKLM, searched down from KMLL.
    for (int kt = KMLL; kt >= 1; --kt) {
        if (kt < iwklm && kreflm == 0) kreflm = kt;
    }

    const double delks_b  = 1.0 / (D3(PRSI, 1) - D3(PRSI, iwklm));
    const double delks1_b = 1.0 / (D3(PRSL, 1) - D3(PRSL, iwklm));
    double ubar_b = 0.0, vbar_b = 0.0, pe = 0.0, ek = 0.0;
    double bnv2bar_b = (D3(PRSL, 1) - D3(PRSL, 2)) * delks1_b * bnv2lm[0];

    // Trial mean wind and N^2 below the reference level kreflm (ROLL_b is diagnostic only in the
    // Fortran and is not carried; see the tail of this kernel).
    for (int k = 1; k <= kreflm; ++k) {
        double rdelks = D3(DEL, k) * delks_b;
        ubar_b += rdelks * D3(U1, k);
        vbar_b += rdelks * D3(V1, k);
        rdelks = (D3(PRSL, k) - D3(PRSL, k + 1)) * delks1_b;
        bnv2bar_b += bnv2lm[k - 1] * rdelks;
    }

    // Dividing streamline: PE (potential energy) against EK (wind normal to the range), from
    // IWKLM downward. IDXZB is the first level where PE >= EK.
    for (int k = iwklm; k >= 1; --k) {
        const double u = D3(U1, k);
        const double v = D3(V1, k);
        const double ang_k = gwd_ang(thet, u, v);
        const double uds_k = gwd_uds(u, v);
        if (idxzb == 0) {
            pe += bnv2lm[k - 1] * (GWD_G * elevmx - D3(PHIL, k))
                  * (D3(PHII, k + 1) - D3(PHII, k)) * GWD_ROG2;
            const double up = uds_k * cos(ang_k);
            ek = 0.5 * up * up;
            if (pe >= ek) idxzb = k;
        }
    }

    // ZBK diagnostic: ELEVMX - SQRT(UBAR**2+VBAR**2)/BNV2bar (blocking-section values).
    const double zbk_b = elevmx - sqrt(ubar_b * ubar_b + vbar_b * vbar_b) / bnv2bar_b;

    // Mountain-blocked drag DB(K) for K = IDXZB .. 1 (lm eq 14 and modified lm eq 15).
    if (idxzb > 0) {
        for (int k = idxzb; k >= 1; --k) {
            if (D3(PHIL, idxzb) > D3(PHIL, k)) {
                const double zlen = sqrt((D3(PHIL, idxzb) - D3(PHIL, k))
                                         / (D3(PHIL, k) + GWD_G * hp));
                const double ang_k  = gwd_ang(thet, D3(U1, k), D3(V1, k));
                const double uds_k  = gwd_uds(D3(U1, k), D3(V1, k));
                const double cosang = cos(ang_k);
                const double sinang = sin(ang_k);
                const double R = (cosang * cosang + gam * sinang * sinang)
                               / (gam * cosang * cosang + sinang * sinang);
                // (negative of DB, see sign at tendency); CDMB adjusts mountain blocking
                const double dbtmp = 0.25 * cdmb
                                   * fmax(2.0 - 1.0 / R, GWD_HZERO) * sig
                                   * fmax(fmax(GWD_HZERO, cosang), gam * sinang)
                                   * zlen / hp;
                D3(DB, k) = fmax(GWD_HZERO, dbtmp * uds_k);
            }
        }
    }

    D2(ELEVMX) = elevmx;
    D2(IDXZB) = idxzb;
    D2(ZBK) = zbk_b;
    D2(WK) = wk_b;

    // ==== General section: RO on all levels, N^2 and Richardson number ====
    for (int k = 1; k <= LM; ++k) {
        D3(RO, k) = GWD_RO(k);
    }

    for (int k = 1; k <= KMM1; ++k) {
        const double ti   = 2.0 / (D3(T1, k) + D3(T1, k + 1));
        // The Fortran TEM = TI/(PRSL(K)-PRSL(K+1)) is set here and not read in this routine
        // segment (later TEM uses are fresh assignments), so it is not kept.
        const double rdz  = GWD_G / (D3(PHIL, k + 1) - D3(PHIL, k));
        const double tem1 = D3(U1, k) - D3(U1, k + 1);
        const double tem2 = D3(V1, k) - D3(V1, k + 1);
        const double dw2  = tem1 * tem1 + tem2 * tem2;
        const double shr2 = fmax(dw2, GWD_DW2MIN) * rdz * rdz;
        const double bvf2 = GWD_G * (GWD_GOCP + rdz * (GWD_VTJ(k + 1) - GWD_VTJ(k))) * ti;
        D3(RI_N, k) = fmax(bvf2 / shr2, GWD_RIMIN);
        const double b = (GWD_G + GWD_G) * rdz * (GWD_VTK(k + 1) - GWD_VTK(k))
                       / (GWD_VTK(k + 1) + GWD_VTK(k));
        D3(BNV2, k) = fmax(b, GWD_BNV2MIN);
    }

    // Reference level: IWK = largest K in 3..KMPBL with PRSI(1)-PRSI(K) < DPMIN, else 2.
    int iwk = 2;
    for (int k = 3; k <= KMPBL; ++k) {
        const double tem = D3(PRSI, 1) - D3(PRSI, k);
        if (tem < dpmin) iwk = k;
    }
    const int kref = max(iwk, KPBL[c2] + 1);
    D2(KREF) = kref;
    if (KBPS) atomicMax(KBPS, kref);
    if (KMPS) atomicMin(KMPS, kref);

    const double delks  = 1.0 / (D3(PRSI, 1) - D3(PRSI, kref));
    const double delks1 = 1.0 / (D3(PRSL, 1) - D3(PRSL, kref));
    double ubar = 0.0, vbar = 0.0, roll = 0.0;
    double bnv2bar = (D3(PRSL, 1) - D3(PRSL, 2)) * delks1 * D3(BNV2, 1);

    // Mean wind, density and N^2 below kref. The Fortran loops K = 1..KBPS and tests K < kref(I);
    // per column that is K = 1 .. kref-1.
    for (int k = 1; k < kref; ++k) {
        double rdelks = D3(DEL, k) * delks;
        ubar += rdelks * D3(U1, k);
        vbar += rdelks * D3(V1, k);
        roll += rdelks * D3(RO, k);
        rdelks = (D3(PRSL, k) - D3(PRSL, k + 1)) * delks1;
        bnv2bar += D3(BNV2, k) * rdelks;
    }

    // Low-level wind direction and the asymmetry / length scale for that direction.
    //   NWD  1   2   3   4   5   6   7   8  ;  WD  W  S  SW  NW  E  N  NE  SE
    const double wdir = atan2(ubar, vbar) + GWD_PI;
    const int idir = (int)round(GWD_FDIR * wdir) % GWD_MDIR + 1;   // mod(nint(fdir*wdir),mdir)+1
    const int nwd  = GWD_NWDIR[idir - 1];
    const int m4   = (nwd - 1) % 4;                                // MOD(NWD-1,4), 0-based
    const int sgn  = 1 - 2 * ((nwd - 1) / 4);                      // 1 - 2*INT((NWD-1)/4)
    D2(OA)  = (double)sgn * D3(OA4, m4 + 1);
    D2(CLX) = D3(CLX4, m4 + 1);

    D2(UBAR) = ubar;
    D2(VBAR) = vbar;
    D2(ROLL) = roll;
    D2(BNV2BAR) = bnv2bar;
    D2(DELKS) = delks;
    D2(DELKS1) = delks1;

    // Blocking-section pe and ek are internal to the PE/IDXZB search and are not needed after it.
    (void)pe;
    (void)ek;
    return true;
}

// Launchable form (one thread per column over the full domain); the column body above is also
// called column-locally by gwd_col() in phys_gwd_col.cuh with IM = JM = 1 and local arrays.
extern "C" __global__ void phys_gwd_col_blocking(
    const double* __restrict__ U1,      // U1      zonal wind                 [LM, JM, IM]
    const double* __restrict__ V1,      // V1      meridional wind            [LM, JM, IM]
    const double* __restrict__ T1,      // T1      temperature                [LM, JM, IM]
    const double* __restrict__ Q1,      // Q1      specific humidity          [LM, JM, IM]
    const double* __restrict__ DEL,     // DEL     layer pressure thickness   [LM, JM, IM]
    const double* __restrict__ PRSL,    // PRSL    mid-layer pressure         [LM, JM, IM]
    const double* __restrict__ PRSLK,   // PRSLK   Exner function             [LM, JM, IM]
    const double* __restrict__ PHIL,    // PHIL    mid-layer geopotential     [LM, JM, IM]
    const double* __restrict__ PRSI,    // PRSI    interface pressure         [LM+1, JM, IM]
    const double* __restrict__ PHII,    // PHII    interface geopotential    [LM+1, JM, IM]
    const double* __restrict__ OA4,     // OA4     orographic asymmetry (4)   [4, JM, IM]
    const double* __restrict__ CLX4,    // CLX4    orographic length (4)     [4, JM, IM]
    const double* __restrict__ HPRIME,  // HPRIME  orographic std dev         [JM, IM]
    const double* __restrict__ THETA,   // THETA   range angle (deg)          [JM, IM]
    const double* __restrict__ SIGMA,   // SIGMA   orographic slope           [JM, IM]
    const double* __restrict__ GAMMA,   // GAMMA   anisotropy                 [JM, IM]
    const double* __restrict__ ELVMAX,  // ELVMAX  max height above mean      [JM, IM]
    const int*    __restrict__ KPBL,    // KPBL    PBL top index (bottom-up)  [JM, IM]
    double* DUsfc,                      // DUsfc   surface zonal stress (zeroed)      [JM, IM]
    double* DVsfc,                      // DVsfc   surface meridional stress (zeroed) [JM, IM]
    double* ELEVMX,                     // ELEVMX  blocking height limit (out)        [JM, IM]
    int*    IDXZB,                      // IDXZB   dividing streamline level (out)    [JM, IM]
    int*    KREF,                       // KREF    reference level (out)              [JM, IM]
    double* OA,                         // OA      signed asymmetry for direction (out) [JM, IM]
    double* CLX,                        // CLX     length scale for direction (out)   [JM, IM]
    double* UBAR,                       // UBAR    mean U below kref (out)            [JM, IM]
    double* VBAR,                       // VBAR    mean V below kref (out)            [JM, IM]
    double* ROLL,                       // ROLL    mean density below kref (out)      [JM, IM]
    double* BNV2BAR,                    // BNV2BAR mean N^2 below kref (out)          [JM, IM]
    double* DELKS,                      // DELKS   1/(PRSI(1)-PRSI(kref)) (out)       [JM, IM]
    double* DELKS1,                     // DELKS1  1/(PRSL(1)-PRSL(kref)) (out)       [JM, IM]
    double* ZBK,                        // ZBK     blocking-height diagnostic (out)   [JM, IM]
    double* WK,                         // WK      interpolated mountain diagnostic  [JM, IM]
    double* DB,                         // DB      mountain-blocking drag (out)       [LM, JM, IM]
    double* BNV2,                       // BNV2    N^2 (out)                          [LM, JM, IM]
    double* RI_N,                       // RI_N    Richardson number (out)            [LM, JM, IM]
    double* RO,                         // RO      density (out)                     [LM, JM, IM]
    double* TAUP,                       // TAUP    momentum flux (zeroed; part 2)     [LM+1, JM, IM]
    int*    KBPS,                       // KBPS    max kref over points (device scalar)
    int*    KMPS,                       // KMPS    min kref over points (device scalar)
    int*    GUARD,                      // GUARD   [0] = points that read an unset BNV2LM (must be 0)
    double CDMB,                         // CDMB    mountain blocking coefficient (REAL)
    double SIGFAC,                       // SIGFAC  ELVMAX height factor (REAL)
    double DPMIN,                        // DPMIN   min reference layer thickness (REAL, centibars)
    int IM, int JM, int LM)
{
    COLUMN_IJ(IM, JM)
    gwd_col_blocking_dev(i0, j0, U1, V1, T1, Q1, DEL, PRSL, PRSLK, PHIL, PRSI, PHII, OA4, CLX4, HPRIME, THETA, SIGMA, GAMMA, ELVMAX, KPBL, DUsfc, DVsfc, ELEVMX, IDXZB, KREF, OA, CLX, UBAR, VBAR, ROLL, BNV2BAR, DELKS, DELKS1, ZBK, WK, DB, BNV2, RI_N, RO, TAUP, KBPS, KMPS, GUARD, CDMB, SIGFAC, DPMIN, IM, JM, LM);
}
