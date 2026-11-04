// phys_fer_col_rain_sedim.cu -- nam-gpu unit phys-fer-col-rain-sedim (one thread per column).
//
// NOTICE: NOAA/NCEP NMMB source, US government work (no licence file in the NAM clone). The header
// of phys/module_MP_FER_HIRES.F90 (lines 1-82, "Modified fer_hires microphysics" changelog) is the
// notice; see fer_hires.cuh.
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:1588-2159 (nam.v4.2.11, 83df4e13c1),
// the tail of the level loop of SUBROUTINE EGCP01COLUMN_hr (lines 650-2382).
//
// Scope: the Fortran range, with the per-level values that the upstream units write (see the
// wrapper for the list). Sections:
//   1588-1625  rain evaporation (PREVP), accretion (PRACW), Biggs/collisional rain freezing (PIACR)
//   1627-1660  INDEXR1 = INDEXR; cloud-water sink limits (PIACW, PIACWI, PRAUT, PRACW, PCOND)
//   1661-1700  DELT, Tnew, DELV, WVnew, QWnew (cloud water budget)
//   1700-1800  ice budget: TOT_ICEnew, RimeF, QInew, ASNOWnew, RQSnew, Nsnow, Zsnow
//   1801-1935  rain budget: TIMLT, TOT_RAINnew, QRnew, 3-pass DSD iteration, ARAINnew, Nrain, Zrain
//   1936-2159  array updates; 1-D process outputs; carry update
//
// The DWVr test of Fortran line 1586-1587 (IF (DWVr.LT.0. .AND. DUM.LE.EPSQ), DUM = QW+PCOND, computed
// by the upstream unit on the same level) is re-evaluated here from DWVr, QW and PCOND. PCOND is not
// changed between the two statements, so it is the same test on the same values.
//
// Omitted, with reason:
//   - PRINT_diag block and DBG_logical (Fortran WRITE(0,...) diagnostics, lines ~1960-2060): host
//     output only; no printf in kernels (CONTRACT section 3); no stored value depends on it.
//   - Ztot = MAX(Ztot, Zrain+Zsnow) is overwritten by Ztot = Zrain+Zsnow on the next line, so only the
//     second assignment is kept.
//   - FPVS, CONDENSE, DEPOSIT, GET_INDEXR and the table setup: used or built in other ranges
//     (other units).
//
// Column carry between levels, in registers, read at column start and written back at the end. The
// rows of `carry` are the FER_COL_CARRY layout of phys-fer-col-setup (rows 0..8, [9, JM, IM]); this unit
// reads and writes rows 0,1,2,3,4,6,7 (ARAIN, ASNOW, VRAIN1, VSNOW1, TIMLT, STRAT, DRZL), leaving rows 5
// (INDEXS0C) and 8 (RHOX0C) untouched. `carry2` [6, JM, IM] (owned by this unit):
//   0 Nrain, 1 Zrain, 2 Nsnow, 3 Zsnow, 4 QInew, 5 ASNOWnew.
// Fortran leaves these locals as they were when the branch that sets them is skipped, so they are
// carried across the chunks of a step. Their level-1 value is 0 (the Fortran value before the first
// assignment is not defined by the source): the first chunk of each column (lev0 == 0) starts them at 0.
// CLEAR levels (in_CLEAR, set by phys-fer-col-satcheck) run only the resets of Fortran 951-956 and write
// no output; ARAIN and ASNOW on entry to every level are written to ARAIN_ENTRY / ASNOW_ENTRY for the ice unit.
//
// Layout: per-level arrays are float32 [LM, JM, IM], level l0 at IDX3(i0, j0, l0, IM, JM); l0 = 0 is
// the model top (CONTRACT section 1). Levels 0 .. lsfc-1 of each column are processed.
#include "nmmb_common.cuh"
#include "fer_hires.cuh"

#define RS_TOLER   5.0E-7f
#define RS_C2      (1.0f / 6.0f)
#define RS_ZMIN    0.01f
#define RS_DBZMIN  (-20.0f)
#define RS_NRMIN   FER_MDR_MIN     /* MDRmin = MDImin = 50 */
#define RS_NRMAX   FER_MDR_MAX     /* MDRmax = MDImax = 1000 */

extern "C" __global__ void phys_fer_col_rain_sedim(
    const int IM, const int JM, const int LM,
    const float DTPH, const float CIACR, const float CRACW, const float ABFR, const float CBFR,
    const float RFMX1, const float CRAIN, const float CN0R0, const float RQR_DRMIN,
    const float RQR_DRMAX,
    const int* __restrict__ lsfc,
    const int lev0, const int nchunk,   // this launch covers global levels lev0 .. lev0+nchunk-1
    // ---- per-level inputs from upstream units, [nchunk, JM, IM] (chunk-local level index) ----
    const float* __restrict__ in_THICK, const float* __restrict__ in_RHO,
    const float* __restrict__ in_RRHO, const float* __restrict__ in_TK,
    const float* __restrict__ in_TK2, const float* __restrict__ in_TC,
    const float* __restrict__ in_TCC, const float* __restrict__ in_WV,
    const float* __restrict__ in_QSW, const float* __restrict__ in_QW,
    const float* __restrict__ in_QI, const float* __restrict__ in_QR,
    const float* __restrict__ in_PCOND, const float* __restrict__ in_PIDEP,
    const float* __restrict__ in_PIEVP, const float* __restrict__ in_PICND,
    const float* __restrict__ in_PIMLT, const float* __restrict__ in_PIACW,
    const float* __restrict__ in_PIACWI, const float* __restrict__ in_PRAUT,
    const float* __restrict__ in_PRACW, const float* __restrict__ in_PREVP,
    const float* __restrict__ in_PIACR, const float* __restrict__ in_PINIT,
    const float* __restrict__ in_PRLOSS, const float* __restrict__ in_QTRAIN,
    const float* __restrict__ in_TOT_RAIN, const float* __restrict__ in_VRAIN2,
    const float* __restrict__ in_INDEXR, const float* __restrict__ in_INDEXRMIN,
    const float* __restrict__ in_N0R, const float* __restrict__ in_GAMMAR,
    const float* __restrict__ in_DIFFUS, const float* __restrict__ in_DYNVIS,
    const float* __restrict__ in_THERM_COND, const float* __restrict__ in_QLICE,
    const float* __restrict__ in_NLICE, const float* __restrict__ in_NSMICE,
    const float* __restrict__ in_VSNOW, const float* __restrict__ in_VCI,
    const float* __restrict__ in_FLIMASS, const float* __restrict__ in_INDEXS,
    const float* __restrict__ in_TOT_ICE, const float* __restrict__ in_RIMEF1,
    const float* __restrict__ in_RCW, const float* __restrict__ in_BLDTRH,
    const float* __restrict__ in_DWVR, const float* __restrict__ in_ICE_LOGICAL,
    const float* __restrict__ in_RAIN_LOGICAL,
    const int* __restrict__ in_CLEAR,
    // ---- lookup tables, index = m - 50 (951 entries each) ----
    const float* __restrict__ massr, const float* __restrict__ vrain,
    const float* __restrict__ accrr, const float* __restrict__ ventr1,
    const float* __restrict__ ventr2, const float* __restrict__ massi,
    // ---- column carry, in and out ----
    float* carry, float* carry2,
    // ---- outputs, [LM, JM, IM] ----
    float* out_T, float* out_Q, float* out_WC, float* out_QI, float* out_QR, float* out_QW,
    float* out_RIMEF, float* out_NR, float* out_NS, float* out_DBZ,
    float* pcond1d, float* pevap1d, float* pidep1d, float* pisub1d, float* piacw1d,
    float* piacwi1d, float* piacwr1d, float* piacr1d, float* picnd1d, float* pievp1d,
    float* pimlt1d, float* praut1d, float* pracw1d, float* prevp1d, float* vsnow1d,
    float* vrain11d, float* vrain21d, float* vci1d, float* NSmICE1d, float* INDEXS1d,
    float* INDEXR1d, float* RFlag1d,
    float* ARAIN_ENTRY, float* ASNOW_ENTRY)
{
    COLUMN_IJ(IM, JM)   // col_, i0, j0; threads outside the domain return

    const size_t S2 = (size_t)JM * (size_t)IM;   // stride of one carry row
    const size_t off = IDX2(i0, j0, IM);
    const int nlev = lsfc[off];

    // ---- carry in ----
    float ARAIN    = carry[0 * S2 + off];
    float ASNOW    = carry[1 * S2 + off];
    float VRAIN1   = carry[2 * S2 + off];
    float VSNOW1   = carry[3 * S2 + off];
    float TIMLT    = carry[4 * S2 + off];
    bool  STRAT    = carry[6 * S2 + off] != 0.0f;
    bool  DRZL     = carry[7 * S2 + off] != 0.0f;
    // carry2 belongs to one column of one step: the first chunk (lev0 == 0) starts it at 0 (the level-1
    // value of the header). Later chunks read what the previous chunk wrote.
    const bool first = (lev0 == 0);
    float Nrain    = first ? 0.0f : carry2[0 * S2 + off];
    float Zrain    = first ? 0.0f : carry2[1 * S2 + off];
    float Nsnow    = first ? 0.0f : carry2[2 * S2 + off];
    float Zsnow    = first ? 0.0f : carry2[3 * S2 + off];
    float QInew    = first ? 0.0f : carry2[4 * S2 + off];
    float ASNOWnew = first ? 0.0f : carry2[5 * S2 + off];

    // Levels are processed top-down (l0 = 0 is the model top). A chunk of levels is one launch; the
    // carry makes the result identical to one launch over all levels.
    for (int lc = 0; lc < nchunk; ++lc) {
        const int l0 = lev0 + lc;
        if (l0 >= nlev) break;
        const size_t k = IDX3(i0, j0, lc, IM, JM);

        // ARAIN and ASNOW on entry to this level (Fortran 895-907 zeroing is done by the setup unit into the
        // carry); the ice unit reads them per level. Written on every level, CLEAR or not.
        ARAIN_ENTRY[k] = ARAIN;
        ASNOW_ENTRY[k] = ASNOW;
        // Fortran 951-956: a CLEAR level resets STRAT, DRZL and TIMLT (INDEXRmin is reset by the ice unit,
        // which owns it), then CYCLE big_loop. No array or 1-D output of this level is written.
        if (in_CLEAR[k] != 0) {
            STRAT = false;
            DRZL = false;
            TIMLT = 0.0f;
            continue;
        }

        // ---- per-level inputs ----
        const float THICK = in_THICK[k], RHO = in_RHO[k], RRHO = in_RRHO[k];
        const float TK = in_TK[k], TK2 = in_TK2[k], TC = in_TC[k], TCC = in_TCC[k];
        const float WV = in_WV[k], QSW = in_QSW[k], QW = in_QW[k], QI = in_QI[k], QR = in_QR[k];
        float PCOND = in_PCOND[k], PIDEP = in_PIDEP[k], PIEVP = in_PIEVP[k], PICND = in_PICND[k];
        float PIMLT = in_PIMLT[k], PIACW = in_PIACW[k], PIACWI = in_PIACWI[k];
        float PRAUT = in_PRAUT[k], PRACW = in_PRACW[k], PREVP = in_PREVP[k], PIACR = in_PIACR[k];
        const float PINIT = in_PINIT[k], PRLOSS = in_PRLOSS[k], QTRAIN = in_QTRAIN[k];
        const float TOT_RAIN = in_TOT_RAIN[k], VRAIN2_in = in_VRAIN2[k];
        const int INDEXR_in = (int)in_INDEXR[k], INDEXRmin = (int)in_INDEXRMIN[k];
        const float N0r_in = in_N0R[k];
        const float GAMMAR = in_GAMMAR[k], DIFFUS = in_DIFFUS[k], DYNVIS = in_DYNVIS[k];
        const float THERM_COND = in_THERM_COND[k], QLICE = in_QLICE[k], NLICE = in_NLICE[k];
        const float NSmICE = in_NSMICE[k], VSNOW = in_VSNOW[k], VCI = in_VCI[k];
        const float FLIMASS = in_FLIMASS[k];
        const int INDEXS = (int)in_INDEXS[k];
        const float TOT_ICE = in_TOT_ICE[k], RimeF1 = in_RIMEF1[k], Rcw = in_RCW[k];
        const float BLDTRH = in_BLDTRH[k], DWVr = in_DWVR[k];
        const bool ICE_logical = in_ICE_LOGICAL[k] != 0.0f;
        const bool RAIN_logical = in_RAIN_LOGICAL[k] != 0.0f;

        float N0r = N0r_in;
        float VRAIN2 = VRAIN2_in;
        int INDEXR = INDEXR_in;
        int INDEXR2 = 0;
        float DUM, DUM1, DUM2, DUM3;
        float RimeF = 1.0f;

        // ===== Rain evaporation, accretion and freezing (1588-1625) =====
        // do_rain block, ELSE branch of IF (TC .LT. T_ICE) (the TC < T_ICE branch keeps PIACR = -PRLOSS
        // from the upstream unit and changes nothing here).
        if (RAIN_logical && !(TC < FER_T_ICE)) {
            if (DWVr < 0.0f && (QW + PCOND) <= FER_EPSQ) {
                // Rain evaporation
                const float RFACTOR = sqrtf(GAMMAR) * powf(RHO / (DIFFUS * DIFFUS * DYNVIS), RS_C2);
                const float ABW = 1.0f / (RHO * FER_XLV2 * QSW * TK2 / THERM_COND + 1.0f / DIFFUS);
                const float VENTR = N0r * (ventr1[INDEXR - RS_NRMIN] + RFACTOR * ventr2[INDEXR - RS_NRMIN]);
                const float CREVP = ABW * VENTR * DTPH;
                PREVP = fmaxf(DWVr * CREVP, PRLOSS);
            } else if (QW > FER_EPSQ) {
                // Accretion of cloud water by rain
                const float FWR = CRACW * GAMMAR * N0r * accrr[INDEXR - RS_NRMIN];
                PRACW = fminf(1.0f, FWR) * QW;
            }
            if (ICE_logical && TC < 0.0f && TCC < 0.0f) {
                // Biggs (1953) heterogeneous freezing of rain
                DUM = 0.001f * (float)INDEXR;
                DUM = (expf(ABFR * TC) - 1.0f) * DUM * DUM * DUM * DUM * DUM * DUM * DUM;
                PIACR = fminf(CBFR * N0r * RRHO * DUM, QTRAIN);
                // Freezing of rain by collisions with large ice (no collisional freezing of small ice)
                float FIR = 0.0f;
                if (QLICE > FER_EPSQ) {
                    DUM = GAMMAR * vrain[INDEXR - RS_NRMIN];
                    DUM1 = DUM - VSNOW;
                    DUM2 = sqrtf(DUM1 * DUM1 + 0.04f * DUM * VSNOW);
                    DUM1 = 5.0E-12f * (float)INDEXR * (float)INDEXR
                         + 2.0E-12f * (float)INDEXR * (float)INDEXS
                         + 0.5E-12f * (float)INDEXS * (float)INDEXS;
                    FIR = fminf(1.0f, CIACR * NLICE * DUM1 * DUM2);
                    PIACR = fminf(PIACR + FIR * QTRAIN, QTRAIN);
                }
                DUM = PREVP - PIACR;
                if (DUM < PRLOSS) {
                    DUM1 = PRLOSS / DUM;
                    PREVP = DUM1 * PREVP;
                    PIACR = DUM1 * PIACR;
                }
            }
        }
        const int INDEXR1 = INDEXR;     // INDEXR1 = INDEXR (after do_rain)

        // ===== Cloud-water sink limits (1628-1660) =====
        DUM1 = PIACW + PRAUT + PRACW - fminf(0.0f, PCOND);
        if (DUM1 > QW) {
            DUM = QW / DUM1;
            PIACW = DUM * PIACW;
            PIACWI = DUM * PIACWI;
            PRAUT = DUM * PRAUT;
            PRACW = DUM * PRACW;
            if (PCOND < 0.0f) PCOND = DUM * PCOND;
        }
        const float PIACWR = PIACW - PIACWI;      // TC >= 0C
        const float DELW = PCOND - PIACW - PRAUT - PRACW;
        float QWnew = QW + DELW;
        if (QWnew <= FER_EPSQ) QWnew = 0.0f;
        if (QW > 0.0f && QWnew != 0.0f) {
            DUM = QWnew / QW;
            if (DUM < RS_TOLER) QWnew = 0.0f;
        }

        // ===== Temperature and vapour (1661-1700) =====
        const float DELT = FER_XLV1 * (PCOND + PIEVP + PICND + PREVP)
                         + FER_XLS1 * PIDEP + FER_XLF1 * (PIACWI + PIACR - PIMLT);
        const float Tnew = TK + DELT;
        const float DELV = -PCOND - PIDEP - PIEVP - PICND - PREVP;
        const float WVnew = WV + DELV;

        // ===== Ice budget (1701-1800) =====
        RimeF = 1.0f;
        if (ICE_logical) {
            const float DELI = PIDEP + PIEVP + PIACWI + PIACR - PIMLT;
            float TOT_ICEnew = TOT_ICE + THICK * DELI;
            if (TOT_ICE > 0.0f && TOT_ICEnew != 0.0f) {
                DUM = TOT_ICEnew / TOT_ICE;
                if (DUM < RS_TOLER) TOT_ICEnew = 0.0f;
            }
            if (TOT_ICEnew <= FER_CLIMIT) {
                TOT_ICEnew = 0.0f;
                RimeF = 1.0f;
                QInew = 0.0f;
                ASNOWnew = 0.0f;
            } else {
                float PSDEP;
                if (PINIT <= FER_EPSQ) PSDEP = fmaxf(0.0f, PIDEP);
                else PSDEP = 0.0f;
                const float DELS = PIACWI + PIACR + PSDEP;
                if (DELS <= FER_EPSQ || Tnew >= FER_T0C) {
                    RimeF = RimeF1;
                } else {
                    // Rime factor: pure-ice density (RFmx) and rimed-ice density (RFrime),
                    // Heymsfield and Pflaum (1985) for the rime density.
                    DUM = 1.0E-6f * (float)INDEXS;
                    DUM1 = DUM * DUM * DUM / massi[INDEXS - RS_NRMIN];
                    const float RFmx = RFMX1 * DUM1;
                    float RFrime;
                    if (PIACWI > 0.0f && Rcw > 0.0f) {
                        DUM = -Rcw * VSNOW / fminf(-0.5f, TC);
                        DUM = fminf(12.14f, fmaxf(0.275f, DUM));
                        DUM = 300.0f * powf(DUM, 0.44f);
                        DUM = fminf(900.0f, fmaxf(170.0f, DUM));
                        RFrime = FER_PI * DUM * DUM1;
                    } else {
                        RFrime = 1.0f;
                    }
                    const float RFnew = (PSDEP + RFrime * PIACWI + RFmx * PIACR) / DELS;
                    DUM2 = TOT_ICE + THICK * DELS;
                    DUM3 = RimeF1 * TOT_ICE + RFnew * THICK * DELS;
                    if (DUM2 <= FER_CLIMIT) RimeF = RimeF1;
                    else RimeF = fminf(RFmx, fmaxf(1.0f, DUM3 / DUM2));
                }
                DUM1 = BLDTRH * (FLIMASS * VSNOW + (1.0f - FLIMASS) * VCI);
                QInew = TOT_ICEnew / (THICK + DUM1);
                if (QInew <= FER_EPSQ) QInew = 0.0f;
                if (QI > 0.0f && QInew != 0.0f) {
                    DUM = QInew / QI;
                    if (DUM < RS_TOLER) QInew = 0.0f;
                }
                ASNOWnew = QInew * DUM1;
                if (ASNOW > 0.0f && ASNOWnew != 0.0f) {
                    DUM = ASNOWnew / ASNOW;
                    if (DUM < RS_TOLER) ASNOWnew = 0.0f;
                }
                const float RQSnew = FLIMASS * RHO * QInew;
                if (RQSnew > 0.0f) {
                    Nsnow = RQSnew / (RimeF * massi[INDEXS - RS_NRMIN]);
                    if (RQSnew > 0.0025f && RimeF > 5.0f) {
                        if (RQSnew <= 0.005f) {
                            DUM = fminf(1.0f, 400.0f * RQSnew - 1.0f);
                            Nsnow = 1.0E3f * DUM + Nsnow * (1.0f - DUM);
                        } else {
                            DUM = 180.0f * (RQSnew - 0.005f);
                            Nsnow = 1.0E3f * (1.0f - fminf(DUM, 0.8f));
                        }
                    }
                    Zsnow = FER_CDRY * RQSnew * RQSnew / Nsnow;
                }
            }
        }

        // ===== Rain budget and DSD iteration (1801-1935) =====
        TIMLT = TIMLT + PIMLT * THICK;
        const float DELR = PRAUT + PRACW + PIACWR - PIACR + PIMLT + PREVP + PICND;
        float TOT_RAINnew = TOT_RAIN + THICK * DELR;
        if (TOT_RAIN > 0.0f && TOT_RAINnew != 0.0f) {
            DUM = TOT_RAINnew / TOT_RAIN;
            if (DUM < RS_TOLER) TOT_RAINnew = 0.0f;
        }
        float QRnew, ARAINnew;
        if (TOT_RAINnew <= FER_CLIMIT) {
            TOT_RAINnew = 0.0f;
            VRAIN2 = 0.0f;
            QRnew = 0.0f;
            ARAINnew = 0.0f;
        } else {
            QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
            float RQRnew = RHO * QRnew;
            if (STRAT && RQRnew > 1.0E-3f) STRAT = false;
            if (DRZL && TIMLT > FER_EPSQ) DRZL = false;
            INDEXR2 = INDEXR1;
            for (int ipass = 1; ipass <= 3; ++ipass) {
                if (RQRnew <= RQR_DRMIN) {
                    // Extremely light rain
                    INDEXR = RS_NRMIN;
                    N0r = RQRnew / massr[INDEXR - RS_NRMIN];
                } else if (DRZL) {
                    // Drizzle: N0r grows for rain contents below 0.5 g m^-3
                    DUM = fmaxf(1.0f, 0.5E-3f / RQRnew);
                    N0r = fminf(1.0E9f, FER_N0R0 * DUM * sqrtf(DUM));
                    DUM1 = RQRnew / (FER_PI * FER_RHOL * N0r);
                    DUM = 1.0E6f * sqrtf(sqrtf(DUM1));
                    INDEXR = (int)fmaxf(FER_XMR_MIN, fminf(DUM, FER_XMR_MAX));
                    N0r = RQRnew / massr[INDEXR - RS_NRMIN];
                } else if (RQRnew >= RQR_DRMAX) {
                    // Extremely heavy rain
                    INDEXR = RS_NRMAX;
                    N0r = RQRnew / massr[INDEXR - RS_NRMIN];
                } else {
                    // Light to heavy rain
                    N0r = FER_N0R0;
                    DUM = CN0R0 * sqrtf(sqrtf(RQRnew));
                    INDEXR = (int)fmaxf(FER_XMR_MIN, fminf(DUM, FER_XMR_MAX));
                    if (STRAT && INDEXR < INDEXRmin) {
                        INDEXR = INDEXRmin;
                        N0r = RQRnew / massr[INDEXR - RS_NRMIN];
                    }
                }
                VRAIN2 = GAMMAR * vrain[INDEXR - RS_NRMIN];
                QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
                RQRnew = RHO * QRnew;
                const int IDR = abs(INDEXR - INDEXR2);
                INDEXR2 = INDEXR;
                if (IDR <= 5) break;             // agreement within 5 microns
            }
            if (QRnew <= FER_EPSQ) QRnew = 0.0f;
            if (QR > 0.0f && QRnew != 0.0f) {
                DUM = QRnew / QR;
                if (DUM < RS_TOLER) QRnew = 0.0f;
            }
            ARAINnew = BLDTRH * VRAIN2 * QRnew;
            if (ARAIN > 0.0f && ARAINnew != 0.0f) {
                DUM = ARAINnew / ARAIN;
                if (DUM < RS_TOLER) ARAINnew = 0.0f;
            }
            RQRnew = RHO * QRnew;
            if (RQRnew > FER_EPSQ) {
                Nrain = N0r * 1.0E-6f * (float)INDEXR2;
                Zrain = CRAIN * RQRnew * RQRnew / Nrain;
            }
        }
        const float WCnew = QWnew + QRnew + QInew;

        // ===== Array updates (1936-1958; print block omitted, see header) =====
        // Ztot = Zrain + Zsnow (the MAX assignment before it is overwritten in the Fortran).
        const float Ztot = Zrain + Zsnow;
        out_T[k] = Tnew;
        out_Q[k] = fmaxf(FER_EPSQ, WVnew / (1.0f + WVnew));
        out_WC[k] = fmaxf(FER_EPSQ, WCnew);
        out_QI[k] = fmaxf(FER_EPSQ, QInew);
        out_QR[k] = fmaxf(FER_EPSQ, QRnew);
        out_QW[k] = fmaxf(FER_EPSQ, QWnew);
        out_RIMEF[k] = RimeF;
        out_NR[k] = Nrain;
        out_NS[k] = Nsnow;
        out_DBZ[k] = (Ztot > RS_ZMIN) ? 10.0f * log10f(Ztot) : RS_DBZMIN;

        ASNOW = ASNOWnew;
        VSNOW1 = VSNOW;
        ARAIN = ARAINnew;
        VRAIN1 = VRAIN2;                         // VRAIN1 = VRAIN2 (the new fall speed)

        // ===== 1-D process outputs (Fortran 1-D arrays of KTS:KTE; set here, zero otherwise) =====
        pcond1d[k] = (PCOND > 0.0f) ? PCOND : 0.0f;
        pevap1d[k] = (PCOND < 0.0f) ? PCOND : 0.0f;
        pidep1d[k] = (PIDEP > 0.0f) ? PIDEP : 0.0f;
        pisub1d[k] = (PIDEP < 0.0f) ? PIDEP : 0.0f;
        piacw1d[k] = PIACW;
        piacwi1d[k] = PIACWI;
        piacwr1d[k] = PIACWR;
        piacr1d[k] = PIACR;
        picnd1d[k] = PICND;
        pievp1d[k] = PIEVP;
        pimlt1d[k] = PIMLT;
        praut1d[k] = PRAUT;
        pracw1d[k] = PRACW;
        prevp1d[k] = PREVP;
        vsnow1d[k] = 0.0f;
        vci1d[k] = 0.0f;
        NSmICE1d[k] = 0.0f;
        INDEXS1d[k] = 0.0f;
        if (QInew > FER_EPSQ) {
            vsnow1d[k] = VSNOW;
            if (FLIMASS < 1.0f) {
                vci1d[k] = VCI;
                NSmICE1d[k] = NSmICE;
            }
            INDEXS1d[k] = (float)INDEXS;
        }
        vrain11d[k] = 0.0f;
        vrain21d[k] = 0.0f;
        INDEXR1d[k] = 0.0f;
        RFlag1d[k] = 0.0f;
        if (QRnew > FER_EPSQ) {
            vrain11d[k] = VRAIN1;
            vrain21d[k] = VRAIN2;
            INDEXR1d[k] = (float)INDEXR2;
            int IDRflag = 0;
            if (STRAT) IDRflag = 1;
            if (DRZL) IDRflag = IDRflag + 2;
            RFlag1d[k] = (float)IDRflag;
        }
        // Carry the level's ice/rain locals that a later level may read without reassignment.
        // (QInew, ASNOWnew, Nsnow, Zsnow, Nrain, Zrain are carried in registers above.)
    }

    // ---- carry out ----
    carry[0 * S2 + off] = ARAIN;
    carry[1 * S2 + off] = ASNOW;
    carry[2 * S2 + off] = VRAIN1;
    carry[3 * S2 + off] = VSNOW1;
    carry[4 * S2 + off] = TIMLT;
    carry[6 * S2 + off] = STRAT ? 1.0f : 0.0f;
    carry[7 * S2 + off] = DRZL ? 1.0f : 0.0f;
    carry2[0 * S2 + off] = Nrain;
    carry2[1 * S2 + off] = Zrain;
    carry2[2 * S2 + off] = Nsnow;
    carry2[3 * S2 + off] = Zsnow;
    carry2[4 * S2 + off] = QInew;
    carry2[5 * S2 + off] = ASNOWnew;
}
