// phys_gwd_col.cuh -- GWD_col for one column, as called by GWD_driver (phys-gwd-driver).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), MODULE_GWD
// (nmm/module_GWD.F90) header as in phys_gwd_col_blocking.cu.
// Ported to CUDA for WOOF-NAM from nmm/module_GWD.F90:368-1132 (nam.v4.2.11, 83df4e13c1): the column
// form of SUBROUTINE GWD_col. Integration glue (Opus integrator, 2026-11-04): it runs the two reviewed
// part bodies, gwd_col_blocking_dev (368-897, unit phys-gwd-col-blocking) and gwd_col_wave_dev
// (898-1132, unit phys-gwd-col-wave), on thread-local column arrays. The part bodies index their
// arrays as [K, JM, IM] / [K, NP]; with IM = JM = NP = NPT = 1 and i0 = j0 = c = p = 0 that is plain
// column indexing, so the same code serves the full-domain launches and this column call.
//
// Fortran argument map (CALL GWD_col at module_GWD.F90:320 -> SUBROUTINE header at 368):
//   A = DVDTcol, B = DUDTcol, PRSI = PINTcol, DEL = DPcol, PRSL = Pcol, PRSLK = EXNcol,
//   PHII = PHILIcol, PHIL = PHIcol, KPBL = LPBL (bottom-up). IM = 1 (one column per call).
// The Fortran compacts qualifying points (ELVMAX > HMINMT and HPRIME > HPMIN) into IPT and runs the
// wave part only on them; here a column that does not qualify returns after the blocking part with
// A, B, DUsfc, DVsfc unchanged (zero), as the Fortran leaves non-IPT columns.
// KBPS/KMPS/GUARD are row reductions of the full-domain launch; they only bound loops (see the part
// headers) and are not needed per column (null pointers skip the atomics).
// DELTIM is the MODULE_GWD variable that GWD_driver sets from DTPHS (module_GWD.F90:212); the wave part
// reads it, so the driver passes it as the last argument.
#pragma once
#include "nmmb_common.cuh"
#include "phys_gwd_col_blocking.cu"
#include "phys_gwd_col_wave.cu"

#ifndef GWD_COL_LMAX
#define GWD_COL_LMAX 64
#endif

__device__ void gwd_col(double* A, double* B, double* DUsfc, double* DVsfc,
                        const double* U1, const double* V1, const double* T1, const double* Q1,
                        const double* PRSI, const double* DEL, const double* PRSL,
                        const double* PRSLK, const double* PHII, const double* PHIL,
                        const double* HPRIME, const double* OC, const double* OA4,
                        const double* CLX4, const double* THETA, const double* SIGMA,
                        const double* GAMMA, const double* ELVMAX,
                        float cdmb, float cleff, float sigfac, float factop, float rlolev,
                        float dpmin, const int* KPBL, int imx, int im, int lm, double deltim)
{
    (void)imx; (void)im;
    double ELEVMX, UBAR, VBAR, ROLL, BNV2BAR, DELKS, DELKS1, ZBK, WK, OA, CLX;
    int IDXZB, KREF;
    double DB[GWD_COL_LMAX], BNV2[GWD_COL_LMAX], RI_N[GWD_COL_LMAX], RO[GWD_COL_LMAX];
    double TAUP[GWD_COL_LMAX + 1], TAUPW[GWD_COL_LMAX + 1], TAUD[GWD_COL_LMAX + 1];
    double VELCO[GWD_COL_LMAX + 1];

    // Tunables are default REAL in the call; the as-compiled build widens them (see the part headers).
    const bool qualifies = gwd_col_blocking_dev(0, 0,
        U1, V1, T1, Q1, DEL, PRSL, PRSLK, PHIL, PRSI, PHII, OA4, CLX4, HPRIME, THETA, SIGMA, GAMMA,
        ELVMAX, KPBL, DUsfc, DVsfc, &ELEVMX, &IDXZB, &KREF, &OA, &CLX, &UBAR, &VBAR, &ROLL,
        &BNV2BAR, &DELKS, &DELKS1, &ZBK, &WK, DB, BNV2, RI_N, RO, TAUP,
        (int*)nullptr, (int*)nullptr, (int*)nullptr,
        (double)cdmb, (double)sigfac, (double)dpmin, 1, 1, lm);
    if (!qualifies) return;

    gwd_col_wave_dev(0, 0, 1, 1, lm, deltim, (double)cleff, (double)factop, (double)rlolev,
        DUsfc, DVsfc, HPRIME, OC, U1, V1, DEL, PRSI, A, B, &UBAR, &VBAR, &ROLL, &BNV2BAR, &OA, &CLX,
        &KREF, &IDXZB, RO, BNV2, RI_N, DB, TAUPW, TAUD, VELCO);
}
