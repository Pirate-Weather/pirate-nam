// phys_fer_fer_hires.cu -- FER_HIRES column driver for the Ferrier-Aligo microphysics.
//
// NOTICE: NAM source file phys/module_MP_FER_HIRES.F90 (NOAA/NCEP/EMC, US Government work, no
// licence file). Subroutine name kept as the notice: SUBROUTINE FER_HIRES (NOAA/NCEP/EMC,
// Ferrier-Aligo high-resolution microphysics, nam.v4.2.11).
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:208-604 (nam.v4.2.11, 83df4e13c1).
//
// Mapping (Fortran -> CUDA). One thread per (i,j) column. Every statement of 208-604 is either a
// term below or listed in the "dropped" notes.
//   T_phy = TH*PI (all levels)                  -> t_phy[] per column (same values)
//   SR init 0; APREC                            -> sr, aprec (per column)
//   ACPREC, PREC                                -> dead accumulators (see notes)
//   TLATGS_PHY, TRAIN_PHY                       -> dead stores (see notes)
//   L=1 block (qt floor, F_ICE/F_RAIN/F_RIMEF)  -> l = 1 block
//   DO L=1,LSFC column init                     -> col.* arrays, same branches
//   EGCP01COLUMN_hr call (445-453)              -> fer_egcp01column_hr (device, one per column)
//   DO L=1,LSFC update (T, Q, QT, mprates, F_*, REFL_10CM) -> same
//   APREC, SR, RAINNC/RAINNCV                   -> same
//   DO K=KTS,KTE: TH=T/PI, QS/QC/QR split       -> same, all levels
// Index: Fortran (i,j,l) -> IDX3(i0,j0,l0,IM,JM), l0 = l-1; loop levels stay 1-based so every
// bound and the LSFC expression read as in the Fortran. mprates is [d_ss, LM, JM, IM] (IDX4).
// LOWLYR is int32 [JM, IM].
//
// Interface to the column unit: fer_egcp01column_hr is the Fortran EGCP01COLUMN_hr
// (module_MP_FER_HIRES.F90:650-2382), ported by units phys-fer-col-*, phys-fer-condense,
// phys-fer-deposit and phys-fer-get-indexr. It is NOT defined in this file. It reads and writes
// the column arrays in struct FerCol and takes the restart state and saturation tables as
// read-only inputs.
//
// Dropped (no output effect):
//   * TLATGS_PHY, TRAIN_PHY: local arrays, written in the column loop, never read and not passed
//     out of FER_HIRES (dead stores).
//   * PREC and ACPREC: local accumulators, never read and not passed out.
//   * WARN4 / WARN5 WRITE(0,...) diagnostics and their module flags: a kernel does not print
//     (CONTRACT section 3). The warn branches do not change the state.
//   * MP_RESTART_STATE and TBPVS copy-in/copy-out at the top and bottom of FER_HIRES: the Fortran
//     writes back the same values, so the state is read-only here.
//   * OpenMP directives are dropped; one thread per column.
//
// Launch: namgpu.kernels.grid_cols(IM, JM), one kernel, no host sync, no device allocation.
// Requires LM <= FER_MAXLEV (the wrapper checks this before launch).

#include "nmmb_common.cuh"

#define FER_MAXLEV 64

// Module constants used by FER_HIRES: fer_hires.cuh (module_MP_FER_HIRES.F90 and module_CONSTANTS;
// EPSQ 1.0e-12, RHOL 1000, T0C 273.15, T_ICE -40, G 9.8060226 as NMMB_G).
#include "fer_hires.cuh"
#define FER_GRAV     FER_G

// Per-column scratch: the Fortran local arrays dimensioned kts:kte, one entry per level.
// Index 0 is level 1 (model top). All members are REAL in the Fortran.
struct FerCol {
    float P_col[FER_MAXLEV], QI_col[FER_MAXLEV], QR_col[FER_MAXLEV], Q_col[FER_MAXLEV];
    float QW_col[FER_MAXLEV], RimeF_col[FER_MAXLEV], T_col[FER_MAXLEV], THICK_col[FER_MAXLEV];
    float WC_col[FER_MAXLEV], DPCOL[FER_MAXLEV];
    float pcond1d[FER_MAXLEV], pidep1d[FER_MAXLEV], piacw1d[FER_MAXLEV], piacwi1d[FER_MAXLEV];
    float piacwr1d[FER_MAXLEV], piacr1d[FER_MAXLEV], picnd1d[FER_MAXLEV], pievp1d[FER_MAXLEV];
    float pimlt1d[FER_MAXLEV], praut1d[FER_MAXLEV], pracw1d[FER_MAXLEV], prevp1d[FER_MAXLEV];
    float pisub1d[FER_MAXLEV], pevap1d[FER_MAXLEV], DBZ_col[FER_MAXLEV], NR_col[FER_MAXLEV];
    float NS_col[FER_MAXLEV], vsnow1d[FER_MAXLEV], vrain11d[FER_MAXLEV], vrain21d[FER_MAXLEV];
    float vci1d[FER_MAXLEV], NSmICE1d[FER_MAXLEV], INDEXS1d[FER_MAXLEV], INDEXR1d[FER_MAXLEV];
    float RFlag1d[FER_MAXLEV];
};

// EGCP01COLUMN_hr, one column: fer_egcp01column_hr in phys_fer_column.cuh (2026-11-04), with the
// FERRIER_INIT_hr tables and SAVE scalars in FerTbl.
#include "phys_fer_column.cuh"

extern "C" __global__ void fer_hires(
    const float* __restrict__ dz8w, const float* __restrict__ rho_phy,
    const float* __restrict__ p_phy, const float* __restrict__ pi_phy,
    float* th_phy, float* q, float* qt,
    float* qc, float* qr, float* qs,
    float* f_ice_phy, float* f_rain_phy, float* f_rimef_phy,
    float* refl_10cm,
    float* rainnc, float* rainncv, float* sr,
    const int* __restrict__ lowlyr,
    float* mprates, int d_ss,
    const float* mp_state, const float* tbpvs, const float* tbpvs0,
    float dt, float rhgrd, int im, int jm, int lm,
    int its_b1, int ite_b1, int jts_b1, int jte_b1,
    // FERRIER_INIT_hr tables (phys-fer-init, ctx.tables["fer"]) and DEPOSIT counters
    const float* massi, const float* vsnowi, const float* accri, const float* venti1,
    const float* venti2, const float* massr, const float* vrain, const float* accrr,
    const float* ventr1, const float* ventr2, const float* vel_rf, const float* growth,
    const float* sc, int* nonconv, int* warn2)
{
    (void)mp_state;   // MP_RESTART_STATE copy-in is the same values as the tables below (see header)
    FerTbl tb;
    tb.massi = massi; tb.vsnowi = vsnowi; tb.accri = accri; tb.venti1 = venti1; tb.venti2 = venti2;
    tb.massr = massr; tb.vrain = vrain; tb.accrr = accrr; tb.ventr1 = ventr1; tb.ventr2 = ventr2;
    tb.vel_rf = vel_rf; tb.growth = growth; tb.tbpvs = tbpvs; tb.tbpvs0 = tbpvs0; tb.sc = sc;
    tb.nonconv = nonconv; tb.warn2 = warn2;
    COLUMN_IJ(im, jm)             // i0, j0 zero-based; returns for out-of-range threads
    const int kts = 1, kte = lm;  // 1-based level bounds, as in the Fortran
    if (kte > FER_MAXLEV) return; // the wrapper refuses LM > FER_MAXLEV before launch

    const int i = i0 + 1, j = j0 + 1;  // Fortran 1-based I, J
    // Fortran tile (module_MICROPHYSICS.F90:335-338): ITS_B1..ITE_B1, JTS_B1..JTE_B1 (1-based,
    // inclusive). FER_HIRES loops only over this tile; boundary rows and columns are never written.
    if (i < its_b1 || i > ite_b1 || j < jts_b1 || j > jte_b1) return;
#define IDXL(l) IDX3(i0, j0, (l) - 1, im, jm)

    FerCol col;
    float t_phy[FER_MAXLEV];           // T_phy for levels 1..KTE of this column

    // T_phy = TH_PHY * PI_PHY for all levels (first loop of FER_HIRES).
    for (int k = kts; k <= kte; ++k) {
        t_phy[k - 1] = th_phy[IDXL(k)] * pi_phy[IDXL(k)];
    }

    // SR is zeroed here; APREC is per column.
    sr[IDX2(i0, j0, im)] = 0.0f;
    float aprec = 0.0f;

    // Driver for EGCP01COLUMN_hr (module_MP_FER_HIRES.F90:340-539).
    const int lsfc = kte - lowlyr[IDX2(i0, j0, im)] + 1;  // "L" of surface, 1-based

    for (int k = kts; k <= kte; ++k) {
        col.DPCOL[k - 1] = rho_phy[IDXL(k)] * FER_GRAV * dz8w[IDXL(k)];
    }

    // Initialise column data: the L = 1 block.
    {
        const int l = 1;
        if (qt[IDXL(l)] <= FER_EPSQ) qt[IDXL(l)] = FER_EPSQ;
        f_ice_phy[IDXL(l)] = 1.0f;
        f_rain_phy[IDXL(l)] = 0.0f;
        f_rimef_phy[IDXL(l)] = 1.0f;
    }

    for (int l = 1; l <= lsfc; ++l) {
        const int o = IDXL(l);
        col.P_col[l - 1] = p_phy[o];
        // Layer thickness = RHO*DZ = -DP/G.
        col.THICK_col[l - 1] = col.DPCOL[l - 1] * FER_RGRAV;
        col.T_col[l - 1] = t_phy[l - 1];
        const float TC = col.T_col[l - 1] - FER_T0C;
        col.Q_col[l - 1] = fmaxf(FER_EPSQ, q[o]);

        if (qt[o] <= FER_EPSQ1) {
            col.WC_col[l - 1] = 0.0f;
            f_ice_phy[o] = (TC < FER_T_ICE) ? 1.0f : 0.0f;
            f_rain_phy[o] = 0.0f;
            f_rimef_phy[o] = 1.0f;
        } else {
            col.WC_col[l - 1] = qt[o];
            // WARN4 / WARN5 diagnostics dropped (see header).
        }
        if (FER_T_ICE <= -100.0f) f_ice_phy[o] = 0.0f;

        // Composition of condensate in terms of cloud water, ice and rain.
        const float wc = col.WC_col[l - 1];
        float qi = 0.0f, qrdum = 0.0f, qw = 0.0f;
        const float fice = f_ice_phy[o];
        const float frain = f_rain_phy[o];

        if (fice >= 1.0f) {
            qi = wc;
        } else if (fice <= 0.0f) {
            qw = wc;
        } else {
            qi = fice * wc;
            qw = wc - qi;
        }

        if (qw > 0.0f && frain > 0.0f) {
            if (frain >= 1.0f) {
                qrdum = qw;
                qw = 0.0f;
            } else {
                qrdum = frain * qw;
                qw = qw - qrdum;
            }
        }
        if (qi <= 0.0f) f_rimef_phy[o] = 1.0f;
        col.RimeF_col[l - 1] = f_rimef_phy[o];
        col.QI_col[l - 1] = qi;
        col.QR_col[l - 1] = qrdum;
        col.QW_col[l - 1] = qw;
    }

    // Microphysical calculations in this column.
    float arain = 0.0f, asnow = 0.0f;
    fer_egcp01column_hr(&arain, &asnow, dt, rhgrd, i, j, lsfc, kts, kte, &col, tb);

    // Update storage arrays for levels 1..LSFC.
    for (int l = 1; l <= lsfc; ++l) {
        const int o = IDXL(l);
        // TRAIN_PHY and TLATGS_PHY: dead stores (see header).
        t_phy[l - 1] = col.T_col[l - 1];
        q[o] = col.Q_col[l - 1];
        qt[o] = col.WC_col[l - 1];

        // 1-D source/sink terms into the 4-D mprates array [d_ss, LM, JM, IM].
        if (d_ss == 1) {
            mprates[IDX4(i0, j0, l - 1, 0, im, jm, lm)] = 0.0f;
        } else {
#define MPR(n) mprates[IDX4(i0, j0, l - 1, (n) - 1, im, jm, lm)]
            MPR(1)  += col.pcond1d[l - 1];
            MPR(2)  += col.pidep1d[l - 1];
            MPR(3)  += col.piacw1d[l - 1];
            MPR(4)  += col.piacwi1d[l - 1];
            MPR(5)  += col.piacwr1d[l - 1];
            MPR(6)  += col.piacr1d[l - 1];
            MPR(7)  += col.picnd1d[l - 1];
            MPR(8)  += col.pievp1d[l - 1];
            MPR(9)  += col.pimlt1d[l - 1];
            MPR(10) += col.praut1d[l - 1];
            MPR(11) += col.pracw1d[l - 1];
            MPR(12) += col.prevp1d[l - 1];
            MPR(13) += col.pisub1d[l - 1];
            MPR(14) += col.pevap1d[l - 1];
            MPR(15) = col.vsnow1d[l - 1];
            MPR(16) = col.vrain11d[l - 1];
            MPR(17) = col.vrain21d[l - 1];
            MPR(18) = col.vci1d[l - 1];
            MPR(19) = col.NSmICE1d[l - 1];
            MPR(20) = col.NS_col[l - 1];   // conc snow
            MPR(21) = col.NR_col[l - 1];   // conc rain
            MPR(22) = col.INDEXS1d[l - 1];
            MPR(23) = col.INDEXR1d[l - 1];
            MPR(24) = col.RFlag1d[l - 1];
#undef MPR
        }

        // REAL*4 array storage.
        const float qi_c = col.QI_col[l - 1];
        const float wc_c = col.WC_col[l - 1];
        if (qi_c <= FER_EPSQ) {
            f_ice_phy[o] = 0.0f;
            if (col.T_col[l - 1] < FER_T_ICEK) f_ice_phy[o] = 1.0f;
            f_rimef_phy[o] = 1.0f;
        } else {
            f_ice_phy[o] = fmaxf(0.0f, fminf(1.0f, qi_c / wc_c));
            f_rimef_phy[o] = fmaxf(1.0f, col.RimeF_col[l - 1]);
        }
        const float qr_c = col.QR_col[l - 1];
        float dum;
        if (qr_c <= FER_EPSQ) {
            dum = 0.0f;
        } else {
            dum = qr_c / (qr_c + col.QW_col[l - 1]);
        }
        f_rain_phy[o] = dum;
        refl_10cm[o] = col.DBZ_col[l - 1];
    }

    // Accumulated surface precipitation (depth in m); SR = snow fraction.
    aprec = (arain + asnow) * FER_RRHOL;
    if (aprec < 1.0e-8f) {
        sr[IDX2(i0, j0, im)] = 0.0f;
    } else {
        sr[IDX2(i0, j0, im)] = FER_RRHOL * asnow / aprec;
    }

    // Rain, after the column loop: RAINNC += APREC*1000; RAINNCV = APREC*1000.
    rainnc[IDX2(i0, j0, im)] = aprec * 1000.0f + rainnc[IDX2(i0, j0, im)];
    rainncv[IDX2(i0, j0, im)] = aprec * 1000.0f;

    // Final pass over all levels: TH = T/PI and the QS/QC/QR split of total condensate.
    for (int k = kts; k <= kte; ++k) {
        const int o = IDXL(k);
        th_phy[o] = t_phy[k - 1] / pi_phy[o];
        const float wc = qt[o];
        qs[o] = 0.0f;
        qr[o] = 0.0f;
        qc[o] = 0.0f;

        if (f_ice_phy[o] >= 1.0f) {
            qs[o] = wc;
        } else if (f_ice_phy[o] <= 0.0f) {
            qc[o] = wc;
        } else {
            qs[o] = f_ice_phy[o] * wc;
            qc[o] = wc - qs[o];
        }

        if (qc[o] > 0.0f && f_rain_phy[o] > 0.0f) {
            if (f_rain_phy[o] >= 1.0f) {
                qr[o] = qc[o];
                qc[o] = 0.0f;
            } else {
                qr[o] = f_rain_phy[o] * qc[o];
                qc[o] = qc[o] - qr[o];
            }
        }
    }
#undef IDXL
}
