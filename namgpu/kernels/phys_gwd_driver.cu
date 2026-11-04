// phys_gwd_driver.cu -- GWD_DRIVER for WOOF-NAM: column prep (rotate winds to geodetic, pressure
// and exner, geopotential), the GWD_col call, and the write-back of wind tendencies rotated back
// to model coordinates, plus the surface stress diagnostics.
//
// Notice: module_GWD.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1, NMMB physics,
// NOAA/NCEP/EMC) keeps its author headers. There is no licence file in the NAM tree; the NOAA
// source notice in NOTICE applies. NAM is US Government work.
//   "MODULE MODULE_GWD: Module for Gravity Wave Drag (GWD) and Mountain Blocking (MB).
//    Initially incorporated into the WRF NMM from the GFS by B. Ferrier in April/May 2007.
//    Ratko added in NMM-B (July 2008). Mountain blocking made more robust in July 2016 by BSF."
//
// Ported to CUDA for WOOF-NAM from nmm/module_GWD.F90:125-362 (nam.v4.2.11, 83df4e13c1), the
// GWD_driver subroutine body. The GWD_col call (module_GWD.F90:368-1132) is the phys-gwd-col-blocking
// and phys-gwd-col-wave units, called here as gwd_col(...) from phys_gwd_col.cuh (prototype
// expected from those units; argument order = the Fortran CALL GWD_col list). This kernel contains
// every other statement of GWD_driver.
//
// Configuration (parm/nam_configfile): gwdflg true, GLOBAL false (so IMX = IDE-1), IM (the Fortran
// column dimension) = 1, cdmb 0.1, cleffamp 1.0, sigfac 3.0, factop 0.5, rlolev 50, dpmin 0.
//
// Precision: in module_GWD.F90 KIND_PHYS = SELECTED_REAL_KIND(13,60) (64-bit). The column
// arrays (Ucol, Tcol, DUDTcol, PINTcol, ...), the terrain inputs passed to GWD_col, DELTIM,
// RDELTIM, the constants G, GHALF, THRESH and the DUsfc/DVsfc diagnostics are KIND_PHYS, so they
// are double here. The state arrays and the scalars the caller passes as REAL are float. The
// REAL locals Ugeo, Vgeo, Umod, Vmod, TERRtest, TERRmin are default REAL, so they are float.
//
// Column storage: one thread per (i, j). The per-column arrays are thread-local with LM = 60
// (GWD_LM_MAX; the wrapper refuses other LM). Their VRAM footprint is the local-memory cost of
// this port, reported in the unit note.
//
// Index convention (CONTRACT.md 1): Fortran (i,j,l) with 1-based l -> [l0, j0, i0]. Column
// level k (1-based, bottom-up in the GWD column: k = 1 is the lowest layer) maps to the Fortran
// level KFLIP = LM-K+1, zero-based fl0 = LM-1-k0 with k0 = K-1. Interface and Z levels for
// KFLIP+1 are zero-based fl0+1; Z(LM+1) is zero-based LM; Z(1) and PINT(1) are zero-based 0.
//
// Values not ported, with the reason:
//   - UGWDsfc/VGWDsfc are local arrays in the Fortran (lines 165-166), not dummy arguments; they
//     are written here into ctx.st scratch (UGWDsfc, VGWDsfc) so the surface-stress term is kept.
//   - DELTIM module variable: assigned from DTPHS on every call (lines 212-213), kept as the local.
//   - dtlarge (parameter, unused in the Fortran body) is not ported.

#include "nmmb_common.cuh"
#include "phys_gwd_col.cuh"   // gwd_col: GWD_col, phys-gwd-col-blocking + phys-gwd-col-wave

#ifndef GWD_LM_MAX
#define GWD_LM_MAX 60
#endif

// Fortran (module_GWD.F90:211-212): G=9.806, GHALF=.5*G, THRESH=1.E-6 are PARAMETERs of type
// REAL(KIND=KIND_PHYS), so the default-REAL literals are rounded to single before widening to
// double. The kernel reproduces that: float literal, then widen. GHALF = .5*G is exact in float.
#define GWDD_G       ((double)9.806f)
#define GWDD_GHALF   ((double)(0.5f * 9.806f))
#define GWDD_THRESH  ((double)1.e-6f)

extern "C" __global__ void phys_gwd_driver(
    float* __restrict__ DUDT, float* __restrict__ DVDT,
    const float* __restrict__ U, const float* __restrict__ V,
    const float* __restrict__ T, const float* __restrict__ Q,
    const float* __restrict__ DP, const float* __restrict__ PINT,
    const float* __restrict__ PMID, const float* __restrict__ EXNR,
    const float* __restrict__ Z, const int* __restrict__ KPBL,
    const float* __restrict__ HSTDV, const float* __restrict__ HCNVX,
    const float* __restrict__ HASYW, const float* __restrict__ HASYS,
    const float* __restrict__ HASYSW, const float* __restrict__ HASYNW,
    const float* __restrict__ HLENW, const float* __restrict__ HLENS,
    const float* __restrict__ HLENSW, const float* __restrict__ HLENNW,
    const float* __restrict__ HANGL, const float* __restrict__ HANIS,
    const float* __restrict__ HSLOP, const float* __restrict__ HZMAX,
    const float* __restrict__ CROT, const float* __restrict__ SROT,
    float* __restrict__ UGWDsfc, float* __restrict__ VGWDsfc,
    float dtphs, float cdmb, const float* __restrict__ cleff_dev, float sigfac, float factop,
    float rlolev, float dpmin,
    int imx, int im, int jm, int lm, int its, int ite, int jts, int jte)
{
    // Fortran DO J=JTS,JTE / DO I=ITS,ITE: the tile, not the memory block.
    const int nti = ite - its + 1;
    const int ntj = jte - jts + 1;
    const int col_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_ >= nti * ntj) return;
    const int i0 = its + col_ % nti;
    const int j0 = jts + col_ / nti;
    const size_t ij = IDX2(i0, j0, im);

    // CLEFF is the device scalar written once by phys-gwd-init (GWD_init, module_GWD.F90:116).
    const float cleff = cleff_dev[0];

    // Fortran: DUDT, DVDT and UGWDsfc, VGWDsfc are zeroed over the whole memory block first.
    for (int k0 = 0; k0 < lm; ++k0) {
        const size_t idx = IDX3(i0, j0, k0, im, jm);
        DUDT[idx] = 0.0f;
        DVDT[idx] = 0.0f;
    }
    UGWDsfc[ij] = 0.0f;
    VGWDsfc[ij] = 0.0f;

    // Fortran: DELTIM=DTPHS; RDELTIM=1./DTPHS (REAL divide, then stored as KIND_PHYS).
    const double DELTIM  = (double)dtphs;
    const float  rdtphs  = 1.0f / dtphs;
    const double RDELTIM = (double)rdtphs;

    // Fortran: skip column when KPBL <= 1 or KPBL > LM (the column keeps zero tendencies).
    const int kpbl = KPBL[ij];
    if (kpbl <= 1 || kpbl > lm) return;

    // Initial test: TERRtest = HZMAX + SIGFAC*HSTDV against TERRmin = Z(LM) - Z(LM+1).
    const float TERRtest = HZMAX[ij] + sigfac * HSTDV[ij];
    const float TERRmin  = Z[IDX3(i0, j0, lm - 1, im, jm)] - Z[IDX3(i0, j0, lm, im, jm)];
    if (TERRtest < TERRmin) return;

    const double G      = GWDD_G;
    const double GHALF  = GWDD_GHALF;
    const double THRESH = GWDD_THRESH;

    double DUDTcol[GWD_LM_MAX], DVDTcol[GWD_LM_MAX];
    double Ucol[GWD_LM_MAX], Vcol[GWD_LM_MAX], Tcol[GWD_LM_MAX], Qcol[GWD_LM_MAX];
    double DPcol[GWD_LM_MAX], Pcol[GWD_LM_MAX], EXNcol[GWD_LM_MAX], PHIcol[GWD_LM_MAX];
    double PINTcol[GWD_LM_MAX + 1], PHILIcol[GWD_LM_MAX + 1];
    double DUsfc[1], DVsfc[1];
    double HPRIME[1], OC[1], OA4[4], CLX4[4], THETA[1], SIGMA[1], GAMMA[1], ELVMAX[1];
    int LPBL[1];

    const float crot = CROT[ij];
    const float srot = SROT[ij];

    // Rotate winds to geodetic (Earth) coordinates; pressure, exner, geopotential prep.
    for (int k0 = 0; k0 < lm; ++k0) {
        const int fl0 = lm - 1 - k0;             // zero-based KFLIP-1 (mass level)
        DUDTcol[k0] = 0.0;
        DVDTcol[k0] = 0.0;
        const float uf = U[IDX3(i0, j0, fl0, im, jm)];
        const float vf = V[IDX3(i0, j0, fl0, im, jm)];
        Ucol[k0] = uf * crot + vf * srot;
        Vcol[k0] = vf * crot - uf * srot;

        Tcol[k0] = T[IDX3(i0, j0, fl0, im, jm)];
        Qcol[k0] = Q[IDX3(i0, j0, fl0, im, jm)];

        // Pa -> centibar (the Fortran literal .001 is REAL).
        DPcol[k0] = 0.001f * DP[IDX3(i0, j0, fl0, im, jm)];
        PINTcol[k0] = 0.001f * PINT[IDX3(i0, j0, fl0 + 1, im, jm)];
        Pcol[k0] = 0.001f * PMID[IDX3(i0, j0, fl0, im, jm)];
        EXNcol[k0] = EXNR[IDX3(i0, j0, fl0, im, jm)];

        // Geopotential above the surface at the lower interface and at mid-layer.
        PHILIcol[k0] = G * (double)(Z[IDX3(i0, j0, fl0 + 1, im, jm)]
                                    - Z[IDX3(i0, j0, lm, im, jm)]);
        PHIcol[k0] = GHALF * (double)(Z[IDX3(i0, j0, fl0 + 1, im, jm)]
                                      + Z[IDX3(i0, j0, fl0, im, jm)])
                     - G * (double)Z[IDX3(i0, j0, lm, im, jm)];
    }
    PINTcol[lm] = 0.001f * PINT[IDX3(i0, j0, 0, im, jm)];
    PHILIcol[lm] = G * (double)(Z[IDX3(i0, j0, 0, im, jm)] - Z[IDX3(i0, j0, lm, im, jm)]);

    // Terrain-specific inputs.
    HPRIME[0] = HSTDV[ij];
    OC[0] = HCNVX[ij];
    OA4[0] = HASYW[ij];
    OA4[1] = HASYS[ij];
    OA4[2] = HASYSW[ij];
    OA4[3] = HASYNW[ij];
    CLX4[0] = HLENW[ij];
    CLX4[1] = HLENS[ij];
    CLX4[2] = HLENSW[ij];
    CLX4[3] = HLENNW[ij];
    THETA[0] = HANGL[ij];
    SIGMA[0] = HSLOP[ij];
    GAMMA[0] = HANIS[ij];
    ELVMAX[0] = HZMAX[ij];
    LPBL[0] = lm + 1 - kpbl;

    DUsfc[0] = 0.0;
    DVsfc[0] = 0.0;

    // The GWD_col call (Fortran argument order). IM = 1 is the Fortran column dimension.
    gwd_col(DVDTcol, DUDTcol, DUsfc, DVsfc,
            Ucol, Vcol, Tcol, Qcol, PINTcol, DPcol, Pcol, EXNcol, PHILIcol, PHIcol,
            HPRIME, OC, OA4, CLX4, THETA, SIGMA, GAMMA, ELVMAX,
            cdmb, cleff, sigfac, factop, rlolev, dpmin,
            LPBL, imx, 1, lm, DELTIM);   // DELTIM: module variable read by GWD_col (wave part)

    // Update winds in geodetic coordinates, rotate back to model coordinates, form tendencies.
    for (int k0 = 0; k0 < lm; ++k0) {
        const int fl0 = lm - 1 - k0;
        const double TEST = fabs(DUDTcol[k0]) + fabs(DVDTcol[k0]);
        if (TEST > THRESH) {
            const float Ugeo = (float)(Ucol[k0] + DUDTcol[k0] * DELTIM);
            const float Vgeo = (float)(Vcol[k0] + DVDTcol[k0] * DELTIM);
            const float Umod = Ugeo * crot - Vgeo * srot;
            const float Vmod = Ugeo * srot + Vgeo * crot;
            const size_t idx = IDX3(i0, j0, fl0, im, jm);
            DUDT[idx] = (float)(RDELTIM * (double)(Umod - U[idx]));
            DVDT[idx] = (float)(RDELTIM * (double)(Vmod - V[idx]));
        }
    }

    // Surface stresses, geodetic -> model coordinates (diagnostics).
    UGWDsfc[ij] = (float)(DUsfc[0] * (double)crot - DVsfc[0] * (double)srot);
    VGWDsfc[ij] = (float)(DUsfc[0] * (double)srot + DVsfc[0] * (double)crot);
}
