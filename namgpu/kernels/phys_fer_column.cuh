// phys_fer_column.cuh -- EGCP01COLUMN_hr for one column (Ferrier-Aligo microphysics, NAM fer_hires).
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US Government work, no licence file in the
// NAM clone). SUBPROGRAM EGCP01COLUMN_hr header kept as authorship: "Grid-scale microphysical
// processes - condensation & precipitation", PRGRMMR: Ferrier (W/NP22, 08-2001), Jin (WRF structure).
// Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:650-2159 (nam.v4.2.11, 83df4e13c1).
// Integration (Opus integrator, 2026-11-04): one device function, statement-for-statement from the
// Fortran level loop (big_loop) with every section in Fortran order; the section units
// phys-fer-col-setup / -satcheck / -ice-props / -processes / -melt-evap / -rain-sedim were reviewed
// against the same line ranges and keep their launchable forms. CONDENSE, DEPOSIT, FPVS and FPVS0 are
// the reviewed device functions of units phys-fer-condense, phys-fer-deposit, phys-fer-fpvs.
//
// Tables and SAVE scalars come from FERRIER_INIT_hr (phys-fer-init, ctx.tables["fer"]): MASSI, VSNOWI,
// ACCRI, VENTI1, VENTI2 (MDImin:MDImax, index m-50), MASSR, VRAIN, ACCRR, VENTR1, VENTR2 (index m-50),
// VEL_RF(2:9,0:Nrime) ([IXRF][IXS-2]), MY_GROWTH_NMM(1:35), TBPVS/TBPVS0 and the scalar slots sc[39]
// (SCALAR_SLOTS order of phys_fer_init.py).
//
// Dropped (no state effect): the WARN1 and PRINT_diag/DBG WRITE(0,...) diagnostics and the BUDGET
// check that only feeds them; kernels do not print. Ztot = MAX(Ztot, Zrain+Zsnow) is overwritten on the
// next line (Ztot = Zrain+Zsnow), so only the second assignment is kept.
// Undefined in the Fortran before first set: INDEXRmin (set on the first CLEAR level or TC <= 0 level,
// or by the STRAT test) starts at MDRmin here.
#pragma once
#include "nmmb_common.cuh"
#include "phys_fer_condense.cu"   // nam_condense (+ phys_fer_fpvs.cu: nam_fpvs, nam_fpvs0)
#include "phys_fer_deposit.cu"    // nam_deposit

struct FerTbl {
    const float *massi, *vsnowi, *accri, *venti1, *venti2;
    const float *massr, *vrain, *accrr, *ventr1, *ventr2;
    const float *vel_rf, *growth, *tbpvs, *tbpvs0;
    const float* sc;          // SCALAR_SLOTS (39)
    int *nonconv, *warn2;     // DEPOSIT non-convergence counters (phys-fer-deposit)
};

#define FC_TOLER   5.E-7f
#define FC_C2      (1.f / 6.f)
#define FC_RHO0    1.194f
#define FC_XRATIO  .025f
#define FC_ZMIN    0.01f
#define FC_DBZMIN  (-20.f)
#define FC_BLEND   1.f
#define FC_XMIMAX  (1.0e6f * 1.0e-3f)
#define FC_XMIEXP  .0536f
#define FC_XMRMIN  (1.0e6f * 0.05e-3f)
#define FC_XMRMAX  (1.0e6f * 1.0e-3f)

__device__ void fer_egcp01column_hr(float* ARAIN_io, float* ASNOW_io, float DTPH, float RHgrd,
                                    int I_index, int J_index, int LSFC, int KTS, int KTE,
                                    FerCol* col, const FerTbl& tb)
{
    (void)KTS; (void)KTE;
    // FERRIER_INIT_hr SAVE scalars (sc slots)
    const float C1XPVS = tb.sc[0], C2XPVS = tb.sc[1], C1XPVS0 = tb.sc[2], C2XPVS0 = tb.sc[3];
    const float PI = tb.sc[4], ABFR = tb.sc[5], CBFR = tb.sc[6], CIACW = tb.sc[7], CIACR = tb.sc[8];
    const float CRACW = tb.sc[9], BRAUT = tb.sc[10], RQR_DRmin = tb.sc[11], RQR_DRmax = tb.sc[12];
    const float Crain = tb.sc[21], CN0r0 = tb.sc[23], ESW0 = tb.sc[26], RFmax = tb.sc[27];
    const float RFmx1 = tb.sc[28], ARAUT = tb.sc[30], QAUT0 = tb.sc[31], ARcw = tb.sc[32];
    const float RH_NgC = tb.sc[33], RQhail = tb.sc[34], BVhail = tb.sc[35], AVhail = tb.sc[36];
    const float RH_NgT = tb.sc[37];
    const float EPSQ = FER_EPSQ, CLIMIT = FER_CLIMIT, T0C = FER_T0C, EPS = FER_EP2, EPS1 = FER_EP1;
    const float RD = FER_RD, C1 = FER_C1, T_ICE = FER_T_ICE, T_ICE_init = FER_T_ICE_INIT;
    const float XLV1 = FER_XLV1, XLF1 = FER_XLF1, XLS1 = FER_XLS1, XLS2 = FER_XLS2, XLS3 = FER_XLS3;
    const float XLV2 = FER_XLV2, XLF = FER_XLF, N0r0 = FER_N0R0, RHOL = FER_RHOL, Cdry = FER_CDRY;
    const float NSI_max = FER_NSI_MAX, NLImin = FER_NLIMIN;
    const int MDImin = FER_MDI_MIN, MDImax = FER_MDI_MAX, MDRmin = FER_MDR_MIN, MDRmax = FER_MDR_MAX;
    const int Nrime = FER_NRIME, MY_T1 = FER_MY_T1, MY_T2 = FER_MY_T2;
    const int INDEXRstrmax = FER_INDEXRSTRMAX;
#define MASSI(m)  tb.massi[(m) - 50]
#define VSNOWI(m) tb.vsnowi[(m) - 50]
#define ACCRI(m)  tb.accri[(m) - 50]
#define VENTI1(m) tb.venti1[(m) - 50]
#define VENTI2(m) tb.venti2[(m) - 50]
#define MASSR(m)  tb.massr[(m) - 50]
#define VRAIN(m)  tb.vrain[(m) - 50]
#define ACCRR(m)  tb.accrr[(m) - 50]
#define VENTR1(m) tb.ventr1[(m) - 50]
#define VENTR2(m) tb.ventr2[(m) - 50]
#define VEL_RF(ixs, ixrf) tb.vel_rf[FER_TBL_VEL_RF_INDEX(ixs, ixrf)]
#define FPVS(t)   nam_fpvs((t), tb.tbpvs, C1XPVS, C2XPVS)
#define FPVS0(t)  nam_fpvs0((t), tb.tbpvs0, C1XPVS0, C2XPVS0)

    float EMAIRI, N0r, NLICE, NSmICE, NInuclei, Nrain, Nsnow;
    bool CLEAR, ICE_logical, RAIN_logical, STRAT, DRZL;
    int INDEX_MY, INDEXR, INDEXR1, INDEXR2, INDEXS, IPASS, IXRF, IXS, LBEF, L, INDEXRmin, INDEXS0C, IDR;
    float ABI, ABW, AIEVP, ARAINnew, ASNOWnew, BLDTRH, CREVP, DELI, DELR, DELT, DELV, DELW, DIDEP,
          DIEVP, DIFFUS, DLI, DTRHO, DUM, DUM1, DUM2, DUM3, DWV0, DWVi, DWVr, DYNVIS, ESI, ESW, FIR,
          FLIMASS, FWR, FWS, GAMMAR, GAMMAS, PCOND, PIACR, PIACW, PIACWI, PIACWR, PICND, PIDEP,
          PIDEP_max, PIEVP, PILOSS, PIMLT, PINIT, PP, PRACW, PRAUT, PREVP, PRLOSS, QI, QInew, QLICE, QR,
          QRnew, QSI, QSIgrd, QSW, QSW0, QSWgrd, QTICE, QTRAIN, Q, QW, QWnew, Rcw, RFACTOR, RFmx,
          RFrime, RHO, RIMEF, RimeF1, RQR, RRHO, SFACTOR, TC, TCC, TFACTOR, THERM_COND, THICK, TK,
          TK2, TNEW, TOT_ICE, TOT_ICEnew, TOT_RAIN, TOT_RAINnew, VEL_INC, VENTR, VENTIL, VENTIS,
          VRAIN1, VRAIN2, VrimeF, VSNOW, VSNOW1, WC, WCnew, WSgrd, WS, WV, WVnew, XLI, XLIMASS, XRF,
          NSImax, QSmICE, QLgICE, RQLICE, VCI, TIMLT, RQSnew, RQRnew, Zrain, Zsnow, Ztot, RHOX0C,
          RFnew, PSDEP, DELS, ARAIN, ASNOW;
    (void)DELV; (void)DELW; (void)ESI;

    ARAIN = 0.f;
    VRAIN1 = 0.f;
    VSNOW1 = 0.f;
    ASNOW = 0.f;
    INDEXS0C = MDImin;
    RHOX0C = 22.5f;
    TIMLT = 0.f;
    STRAT = false;
    DRZL = false;
    INDEXRmin = MDRmin;
    // locals that keep their value across levels when a branch does not set them (Fortran locals)
    QLICE = 0.f; QTICE = 0.f; NLICE = 0.f; NSmICE = 0.f; VEL_INC = 1.f; PILOSS = 0.f; NInuclei = 0.f;
    FLIMASS = 1.f; VCI = 0.f; VSNOW = 0.f; TOT_ICE = 0.f; QSmICE = 0.f; Rcw = 0.f; DWVi = 0.f;
    RQLICE = 0.f; INDEXS = MDImin; RimeF1 = 1.f;

    for (L = 1; L <= LSFC; ++L) {          // big_loop
        const int l0 = L - 1;
        col->pcond1d[l0] = 0.f;  col->pidep1d[l0] = 0.f;  col->piacw1d[l0] = 0.f;
        col->piacwi1d[l0] = 0.f; col->piacwr1d[l0] = 0.f; col->piacr1d[l0] = 0.f;
        col->picnd1d[l0] = 0.f;  col->pievp1d[l0] = 0.f;  col->pimlt1d[l0] = 0.f;
        col->praut1d[l0] = 0.f;  col->pracw1d[l0] = 0.f;  col->prevp1d[l0] = 0.f;
        col->pisub1d[l0] = 0.f;  col->pevap1d[l0] = 0.f;  col->vsnow1d[l0] = 0.f;
        col->vrain11d[l0] = 0.f; col->vrain21d[l0] = 0.f; col->vci1d[l0] = 0.f;
        col->NSmICE1d[l0] = 0.f; col->DBZ_col[l0] = FC_DBZMIN; col->NR_col[l0] = 0.f;
        col->NS_col[l0] = 0.f;   col->INDEXR1d[l0] = 0.f; col->INDEXS1d[l0] = 0.f;
        col->RFlag1d[l0] = 0.f;

        if (ARAIN > CLIMIT) { CLEAR = false; VRAIN1 = 0.f; }
        else { CLEAR = true; ARAIN = 0.f; }
        if (ASNOW > CLIMIT) { CLEAR = false; VSNOW1 = 0.f; }
        else { ASNOW = 0.f; }

        TK = col->T_col[l0];
        TC = TK - T0C;
        PP = col->P_col[l0];
        Q = col->Q_col[l0];
        WV = Q / (1.f - Q);
        WC = col->WC_col[l0];

        ESW = fminf(1000.f * FPVS0(TK), 0.99f * PP);
        QSW = EPS * ESW / (PP - ESW);
        WS = QSW;
        QSI = QSW;
        if (TC < 0.f) {
            ESI = fminf(1000.f * FPVS(TK), 0.99f * PP);
            QSI = EPS * ESI / (PP - ESI);
            WS = QSI;
        }
        QSWgrd = RHgrd * QSW;
        QSIgrd = RHgrd * QSI;
        WSgrd = RHgrd * WS;
        if (WV > WSgrd || WC > EPSQ) CLEAR = false;
        if (CLEAR) {
            STRAT = false;
            DRZL = false;
            INDEXRmin = MDRmin;
            TIMLT = 0.f;
            continue;
        }

        RHO = PP / (RD * TK * (1.f + EPS1 * Q));
        RRHO = 1.f / RHO;
        DTRHO = DTPH * RHO;
        BLDTRH = FC_BLEND * DTRHO;
        THICK = col->THICK_col[l0];
        ARAINnew = 0.f;
        ASNOWnew = 0.f;
        QI = col->QI_col[l0];
        QInew = 0.f;
        QR = col->QR_col[l0];
        QRnew = 0.f;
        QW = col->QW_col[l0];
        QWnew = 0.f;
        PCOND = 0.f; PIDEP = 0.f; PINIT = 0.f; PIACW = 0.f; PIACWI = 0.f; PIACWR = 0.f; PIACR = 0.f;
        PICND = 0.f; PIEVP = 0.f; PIMLT = 0.f; PRAUT = 0.f; PRACW = 0.f; PREVP = 0.f;
        NSmICE = 0.f; Nrain = 0.f; Nsnow = 0.f; RQRnew = 0.f; RQSnew = 0.f;
        Zrain = 0.f; Zsnow = 0.f; Ztot = 0.f;
        INDEXR = MDRmin;
        INDEXR1 = INDEXR;
        INDEXR2 = INDEXR;
        N0r = 0.f;
        DUM1 = fminf(0.f, TC);
        DUM = FC_XMIMAX * expf(FC_XMIEXP * DUM1);
        INDEXS = min(MDImax, max(MDImin, (int)DUM));
        VCI = 0.f;
        VSNOW = 0.f;
        VRAIN2 = 0.f;
        RimeF1 = 1.f;

        TK2 = 1.f / (TK * TK);
        TFACTOR = sqrtf(TK * TK * TK) / (TK + 120.f);
        DYNVIS = 1.496E-6f * TFACTOR;
        THERM_COND = 2.116E-3f * TFACTOR;
        DIFFUS = 8.794E-5f * powf(TK, 1.81f) / PP;
        GAMMAS = fminf(1.5f, powf(1.E5f / PP, C1));
        GAMMAR = powf(FC_RHO0 / RHO, .4f);

        if (TC < 0.f || QI > EPSQ || ASNOW > CLIMIT) {
            ICE_logical = true;
        } else {
            ICE_logical = false;
            QLICE = 0.f;
            QTICE = 0.f;
        }
        if (T_ICE <= -100.f) { ICE_logical = false; QLICE = 0.f; QTICE = 0.f; }
        RAIN_logical = false;
        if (ARAIN > CLIMIT || QR > EPSQ) RAIN_logical = true;

        if (ICE_logical) {                                           // ice_test
            NInuclei = 0.f;
            NSmICE = 0.f;
            QSmICE = 0.f;
            Rcw = 0.f;
            if (TC < 0.f) {
                NSImax = fmaxf(NSI_max, 0.1f * RHO * QI / MASSI(MDImin));
                NInuclei = fminf(0.01f * expf(-0.6f * TC), NSImax);
                NInuclei = fminf(5.f * expf(-0.304f * TC), NInuclei);
                if (QI > EPSQ) {
                    DUM = RRHO * MASSI(MDImin);
                    NSmICE = fminf(NInuclei, QI / DUM);
                    QSmICE = NSmICE * DUM;
                }
            }
            if (QI <= EPSQ && ASNOW <= CLIMIT) {                     // init_ice
                TOT_ICE = 0.f;
                PILOSS = 0.f;
                RimeF1 = 1.f;
                VrimeF = 1.f;
                VEL_INC = GAMMAS;
                VSNOW = 0.f;
                VSNOW1 = 0.f;
                VCI = 0.f;
                EMAIRI = THICK;
                XLIMASS = RimeF1 * MASSI(INDEXS);
                FLIMASS = 1.f;
                QLICE = 0.f;
                RQLICE = 0.f;
                QTICE = 0.f;
                NLICE = 0.f;
            } else {
                TOT_ICE = THICK * QI + FC_BLEND * ASNOW;
                PILOSS = -TOT_ICE / THICK;
                QLgICE = fmaxf(0.f, QI - QSmICE);
                VCI = GAMMAS * VSNOWI(MDImin);
                LBEF = max(1, L - 1);
                RimeF1 = (col->RimeF_col[l0] * THICK * QLgICE
                          + col->RimeF_col[LBEF - 1] * FC_BLEND * ASNOW) / TOT_ICE;
                if (RimeF1 > 2.f) {
                    DUM3 = RH_NgC * powf(RHO * QLgICE, C1);
                    DUM2 = RH_NgT * powf(RHO * QLgICE, C1);
                    if (RimeF1 >= 10.f) {
                        DUM = DUM3;
                    } else if (RimeF1 >= 5.f) {
                        DUM = 0.2f * (RimeF1 - 5.f);
                        DUM = DUM3 * DUM + DUM2 * (1.f - DUM);
                    } else {
                        DUM1 = (float)INDEXS;
                        DUM = 0.33333f * (RimeF1 - 2.f);
                        DUM = DUM2 * DUM + DUM1 * (1.f - DUM);
                    }
                    INDEXS = min(MDImax, max(MDImin, (int)DUM));
                }
                EMAIRI = THICK + BLDTRH * VSNOW1;
                QLICE = (THICK * QLgICE + FC_BLEND * ASNOW) / EMAIRI;
                RQLICE = RHO * QLICE;
                QTICE = QLICE + QSmICE;
                FLIMASS = QLICE / QTICE;
                for (IPASS = 1; IPASS <= 2; ++IPASS) {               // two_pass
                    DUM = 1.E-6f * (float)INDEXS;
                    RFmx = RFmx1 * DUM * DUM * DUM / MASSI(INDEXS);
                    RimeF1 = fminf(RimeF1, RFmx);
                    if (RimeF1 <= 1.f) {                             // vel_rime
                        RimeF1 = 1.f;
                        VrimeF = 1.f;
                    } else {
                        RimeF1 = fminf(RimeF1, RFmax);
                        IXS = max(2, min(INDEXS / 100, 9));
                        XRF = 10.492f * logf(RimeF1);
                        IXRF = max(0, min((int)XRF, Nrime));
                        if (IXRF >= Nrime) {
                            VrimeF = VEL_RF(IXS, Nrime);
                        } else {
                            VrimeF = VEL_RF(IXS, IXRF) + (XRF - (float)IXRF) *
                                     (VEL_RF(IXS, IXRF + 1) - VEL_RF(IXS, IXRF));
                        }
                        VrimeF = fmaxf(1.f, VrimeF);
                    }
                    VEL_INC = GAMMAS * VrimeF;
                    VSNOW = VEL_INC * VSNOWI(INDEXS);
                    if (RimeF1 >= 5.f && INDEXS == MDImax && RQLICE > RQhail) {
                        DUM = GAMMAS * AVhail * powf(RQLICE, BVhail);
                        if (DUM > VSNOW) {
                            VSNOW = DUM;
                            VEL_INC = VSNOW / VSNOWI(INDEXS);
                        }
                    }
                    XLIMASS = RimeF1 * MASSI(INDEXS);
                    NLICE = RQLICE / XLIMASS;
                    if (IPASS >= 2 || (NLICE >= NLImin && NLICE <= NSI_max)) break;
                    NLICE = fmaxf(NLImin, fminf(NSI_max, NLICE));
                    XLI = RQLICE / (NLICE * RimeF1);
                    if (XLI <= MASSI(MDImin)) {                      // new_size
                        INDEXS = MDImin;
                    } else if (XLI <= MASSI(450)) {
                        DLI = 9.5885E5f * powf(XLI, .42066f);
                        INDEXS = min(MDImax, max(MDImin, (int)DLI));
                    } else if (XLI < MASSI(MDImax)) {
                        DLI = 3.9751E6f * powf(XLI, .49870f);
                        INDEXS = min(MDImax, max(MDImin, (int)DLI));
                    } else {
                        INDEXS = MDImax;
                    }
                }
            }
        }

        if (TC <= 0.f) {
            STRAT = false;
            INDEXRmin = MDRmin;
            TIMLT = 0.f;
            INDEXS0C = INDEXS;
            RHOX0C = 22.5f * fmaxf(1.f, fminf(RimeF1, 40.f));
        } else {
            if (!RAIN_logical) {
                STRAT = false;
                INDEXRmin = MDRmin;
                if (!ICE_logical) TIMLT = 0.f;
            } else {
                if (TIMLT > EPSQ && RHOX0C <= 225.f) {
                    STRAT = true;
                } else {
                    STRAT = false;
                    INDEXRmin = MDRmin;
                }
                if (STRAT && INDEXRmin <= MDRmin) {
                    INDEXRmin = (int)((float)INDEXS0C * powf(0.001f * RHOX0C, C1));
                    INDEXRmin = max(MDRmin, min(INDEXRmin, INDEXRstrmax));
                }
            }
        }
        if (STRAT || TIMLT > EPSQ) DRZL = false;
        else DRZL = true;

        // ---- individual processes (1321-1478)
        if (QW > EPSQ && TC >= T_ICE) {
            DUM2 = RHO * QW;
            if (DUM2 > QAUT0) {
                DUM2 = DUM2 * DUM2;
                DUM = BRAUT * DUM2 * QW;
                DUM1 = ARAUT * DUM2;
                PRAUT = fminf(QW, DUM * (1.f - expf(-DUM1 * DUM1)));
            }
            if (QLICE > EPSQ) {
                FWS = fminf(1.f, CIACW * VEL_INC * NLICE * ACCRI(INDEXS));
                PIACW = FWS * QW;
                if (TC < 0.f) {
                    PIACWI = PIACW;
                    Rcw = ARcw * powf(DUM2, C1);
                }
            }
        }
        if (TC < T_ICE && (WV > QSWgrd || QW > EPSQ)) {              // ice_only
            PIACW = QW;
            PIACWI = PIACW;
            Rcw = 0.f;
            DUM1 = TK + XLF1 * PIACW;
            DUM2 = WV;
            DUM = fminf(1000.f * FPVS(DUM1), 0.99f * PP);
            DUM = RHgrd * EPS * DUM / (PP - DUM);
            if (DUM2 > DUM)
                PIDEP = nam_deposit(PP, DUM1, DUM2, RHgrd, tb.tbpvs, C1XPVS, C2XPVS, tb.nonconv,
                                    tb.warn2, I_index, J_index, L);
            DWVi = 0.f;
        } else if (TC < 0.f) {
            const float DENOMI = 1.f + XLS3 * QSI * TK2;
            DWVi = WV - QSIgrd;
            PIDEP_max = fmaxf(PILOSS, DWVi / DENOMI);
            DIDEP = 0.f;
            if (QTICE > 0.f) {
                ABI = 1.f / (RHO * XLS2 * QSI * TK2 / THERM_COND + 1.f / DIFFUS);
                DUM = powf(RHO / (DIFFUS * DIFFUS * DYNVIS), FC_C2);
                SFACTOR = sqrtf(GAMMAS) * DUM;
                VENTIS = (VENTI1(MDImin) + SFACTOR * VENTI2(MDImin)) * NSmICE;
                SFACTOR = sqrtf(VEL_INC) * DUM;
                VENTIL = (VENTI1(INDEXS) + SFACTOR * VENTI2(INDEXS)) * NLICE;
                DIDEP = ABI * (VENTIL + VENTIS) * DTPH;
            }
            if (WV > QSWgrd && TC < T_ICE_init && NSmICE < NInuclei) {
                INDEX_MY = max(MY_T1, min((int)(.5f - TC), MY_T2));
                DUM = fmaxf(0.f, NInuclei - NSmICE);
                PINIT = fmaxf(0.f, DUM * tb.growth[INDEX_MY - 1] * RRHO);
            }
            if (DWVi > 0.f) PIDEP = fminf(DWVi * DIDEP + PINIT, PIDEP_max);
            else if (DWVi < 0.f) PIDEP = fmaxf(DWVi * DIDEP, PIDEP_max);
        }
        if (TC >= T_ICE && (QW > EPSQ || WV > QSWgrd)) {
            DUM1 = QW - PIACWI;
            DUM2 = TK + XLS1 * PIDEP + XLF1 * PIACWI;
            DUM3 = WV - PIDEP;
            bool conv = true;
            PCOND = nam_condense(PP, DUM1, DUM2, DUM3, RHgrd, tb.tbpvs0, C1XPVS0, C2XPVS0, &conv);
        }
        TCC = TC + XLV1 * PCOND + XLS1 * PIDEP + XLF1 * PIACWI;
        if (TCC > 0.f) {
            PIACWI = 0.f;
            Rcw = 0.f;
            PIDEP = 0.f;
            TCC = TC + XLV1 * PCOND;
        }
        QSW0 = 0.f;
        DWV0 = 0.f;

        // ---- melting and evaporation of ice (1478-1520)
        if (TC > 0.f && TCC > 0.f && ICE_logical) {
            SFACTOR = sqrtf(VEL_INC) * powf(RHO / (DIFFUS * DIFFUS * DYNVIS), FC_C2);
            VENTIL = NLICE * (VENTI1(INDEXS) + SFACTOR * VENTI2(INDEXS));
            AIEVP = VENTIL * DIFFUS * DTPH;
            if (AIEVP < FC_XRATIO) DIEVP = AIEVP;
            else DIEVP = 1.f - expf(-AIEVP);
            DUM = QW + PCOND;
            if (WV < QSW && DUM <= EPSQ) {
                DUM = fminf(ESW0, 0.99f * PP);
                QSW0 = fmaxf(EPSQ, EPS * DUM / (PP - DUM));
                DWV0 = fminf(WV, QSW) - QSW0;
                DUM = DWV0 * DIEVP;
                PIEVP = fmaxf(fminf(0.f, DUM), PILOSS);
                PICND = fmaxf(0.f, DUM);
            }
            PIMLT = THERM_COND * TCC * VENTIL * RRHO * DTPH / XLF;
            DUM1 = fmaxf(0.f, (TCC + XLV1 * PIEVP) / XLF1);
            PIMLT = fminf(PIMLT, DUM1);
            DUM = PIEVP - PIMLT;
            if (DUM < PILOSS) {
                DUM1 = PILOSS / DUM;
                PIMLT = PIMLT * DUM1;
                PIEVP = PIEVP * DUM1;
            }
        }

        // ---- rain (1521-1625)
        TOT_RAIN = 0.f;
        QTRAIN = 0.f;
        PRLOSS = 0.f;
        RQR = 0.f;
        if (RAIN_logical) {                                          // do_rain
            TOT_RAIN = THICK * QR + FC_BLEND * ARAIN;
            PRLOSS = -TOT_RAIN / THICK;
            QTRAIN = TOT_RAIN / (THICK + BLDTRH * VRAIN1);
            RQR = RHO * QTRAIN;
            if (STRAT && RQR > 1.E-3f) STRAT = false;
            if (DRZL && PIMLT > EPSQ) DRZL = false;
            if (RQR <= RQR_DRmin) {                                  // DSD1
                INDEXR = MDRmin;
                N0r = RQR / MASSR(INDEXR);
            } else if (DRZL) {
                DUM = fmaxf(1.0f, 0.5E-3f / RQR);
                N0r = fminf(1.E9f, N0r0 * DUM * sqrtf(DUM));
                DUM1 = RQR / (PI * RHOL * N0r);
                DUM = 1.E6f * sqrtf(sqrtf(DUM1));
                INDEXR = (int)fmaxf(FC_XMRMIN, fminf(DUM, FC_XMRMAX));
                N0r = RQR / MASSR(INDEXR);
            } else if (RQR >= RQR_DRmax) {
                INDEXR = MDRmax;
                N0r = RQR / MASSR(INDEXR);
            } else {
                N0r = N0r0;
                DUM = CN0r0 * sqrtf(sqrtf(RQR));
                INDEXR = (int)fmaxf(FC_XMRMIN, fminf(DUM, FC_XMRMAX));
                if (STRAT && INDEXR < INDEXRmin) {
                    INDEXR = INDEXRmin;
                    N0r = RQR / MASSR(INDEXR);
                }
            }
            VRAIN2 = GAMMAR * VRAIN(INDEXR);
            if (TC < T_ICE) {
                PIACR = -PRLOSS;
            } else {
                DWVr = WV - PCOND - QSW;
                DUM = QW + PCOND;
                if (DWVr < 0.f && DUM <= EPSQ) {
                    RFACTOR = sqrtf(GAMMAR) * powf(RHO / (DIFFUS * DIFFUS * DYNVIS), FC_C2);
                    ABW = 1.f / (RHO * XLV2 * QSW * TK2 / THERM_COND + 1.f / DIFFUS);
                    VENTR = N0r * (VENTR1(INDEXR) + RFACTOR * VENTR2(INDEXR));
                    CREVP = ABW * VENTR * DTPH;
                    PREVP = fmaxf(DWVr * CREVP, PRLOSS);
                } else if (QW > EPSQ) {
                    FWR = CRACW * GAMMAR * N0r * ACCRR(INDEXR);
                    PRACW = fminf(1.f, FWR) * QW;
                }
                if (ICE_logical && TC < 0.f && TCC < 0.f) {
                    DUM = .001f * (float)INDEXR;
                    DUM = (expf(ABFR * TC) - 1.f) * DUM * DUM * DUM * DUM * DUM * DUM * DUM;
                    PIACR = fminf(CBFR * N0r * RRHO * DUM, QTRAIN);
                    FIR = 0.f;
                    if (QLICE > EPSQ) {
                        DUM = GAMMAR * VRAIN(INDEXR);
                        DUM1 = DUM - VSNOW;
                        DUM2 = sqrtf(DUM1 * DUM1 + .04f * DUM * VSNOW);
                        DUM1 = 5.E-12f * INDEXR * INDEXR + 2.E-12f * INDEXR * INDEXS
                               + .5E-12f * INDEXS * INDEXS;
                        FIR = fminf(1.f, CIACR * NLICE * DUM1 * DUM2);
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
        }
        INDEXR1 = INDEXR;

        // ---- cloud water budget (1627-1700)
        DUM1 = PIACW + PRAUT + PRACW - fminf(0.f, PCOND);
        if (DUM1 > QW) {
            DUM = QW / DUM1;
            PIACW = DUM * PIACW;
            PIACWI = DUM * PIACWI;
            PRAUT = DUM * PRAUT;
            PRACW = DUM * PRACW;
            if (PCOND < 0.f) PCOND = DUM * PCOND;
        }
        PIACWR = PIACW - PIACWI;
        DELW = PCOND - PIACW - PRAUT - PRACW;
        QWnew = QW + DELW;
        if (QWnew <= EPSQ) QWnew = 0.f;
        if (QW > 0.f && QWnew != 0.f) {
            DUM = QWnew / QW;
            if (DUM < FC_TOLER) QWnew = 0.f;
        }
        DELT = XLV1 * (PCOND + PIEVP + PICND + PREVP) + XLS1 * PIDEP + XLF1 * (PIACWI + PIACR - PIMLT);
        TNEW = TK + DELT;
        DELV = -PCOND - PIDEP - PIEVP - PICND - PREVP;
        WVnew = WV + DELV;

        // ---- ice budget (1700-1843)
        DELI = 0.f;
        RIMEF = 1.f;
        if (ICE_logical) {
            DELI = PIDEP + PIEVP + PIACWI + PIACR - PIMLT;
            TOT_ICEnew = TOT_ICE + THICK * DELI;
            if (TOT_ICE > 0.f && TOT_ICEnew != 0.f) {
                DUM = TOT_ICEnew / TOT_ICE;
                if (DUM < FC_TOLER) TOT_ICEnew = 0.f;
            }
            if (TOT_ICEnew <= CLIMIT) {
                TOT_ICEnew = 0.f;
                RIMEF = 1.f;
                QInew = 0.f;
                ASNOWnew = 0.f;
            } else {
                if (PINIT <= EPSQ) PSDEP = fmaxf(0.f, PIDEP);
                else PSDEP = 0.f;
                DELS = PIACWI + PIACR + PSDEP;
                if (DELS <= EPSQ || TNEW >= T0C) {
                    RIMEF = RimeF1;
                } else {
                    DUM = 1.E-6f * (float)INDEXS;
                    DUM1 = DUM * DUM * DUM / MASSI(INDEXS);
                    RFmx = RFmx1 * DUM1;
                    if (PIACWI > 0.f && Rcw > 0.f) {
                        DUM = -Rcw * VSNOW / fminf(-0.5f, TC);
                        DUM = fminf(12.14f, fmaxf(0.275f, DUM));
                        DUM = 300.f * powf(DUM, 0.44f);
                        DUM = fminf(900.f, fmaxf(170.f, DUM));
                        RFrime = PI * DUM * DUM1;
                    } else {
                        RFrime = 1.f;
                    }
                    RFnew = (PSDEP + RFrime * PIACWI + RFmx * PIACR) / DELS;
                    DUM2 = TOT_ICE + THICK * DELS;
                    DUM3 = RimeF1 * TOT_ICE + RFnew * THICK * DELS;
                    if (DUM2 <= CLIMIT) RIMEF = RimeF1;
                    else RIMEF = fminf(RFmx, fmaxf(1.f, DUM3 / DUM2));
                }
                DUM1 = BLDTRH * (FLIMASS * VSNOW + (1.f - FLIMASS) * VCI);
                QInew = TOT_ICEnew / (THICK + DUM1);
                if (QInew <= EPSQ) QInew = 0.f;
                if (QI > 0.f && QInew != 0.f) {
                    DUM = QInew / QI;
                    if (DUM < FC_TOLER) QInew = 0.f;
                }
                ASNOWnew = QInew * DUM1;
                if (ASNOW > 0.f && ASNOWnew != 0.f) {
                    DUM = ASNOWnew / ASNOW;
                    if (DUM < FC_TOLER) ASNOWnew = 0.f;
                }
                RQSnew = FLIMASS * RHO * QInew;
                if (RQSnew > 0.f) {
                    Nsnow = RQSnew / (RIMEF * MASSI(INDEXS));
                    if (RQSnew > 0.0025f && RIMEF > 5.f) {
                        if (RQSnew <= 0.005f) {
                            DUM = fminf(1.f, 400.f * RQSnew - 1.f);
                            Nsnow = 1.E3f * DUM + Nsnow * (1.f - DUM);
                        } else {
                            DUM = 180.f * (RQSnew - 0.005f);
                            Nsnow = 1.E3f * (1.f - fminf(DUM, 0.8f));
                        }
                    }
                    Zsnow = Cdry * RQSnew * RQSnew / Nsnow;
                }
            }
        }

        // ---- rain budget (1844-1935)
        TIMLT = TIMLT + PIMLT * THICK;
        DELR = PRAUT + PRACW + PIACWR - PIACR + PIMLT + PREVP + PICND;
        TOT_RAINnew = TOT_RAIN + THICK * DELR;
        if (TOT_RAIN > 0.f && TOT_RAINnew != 0.f) {
            DUM = TOT_RAINnew / TOT_RAIN;
            if (DUM < FC_TOLER) TOT_RAINnew = 0.f;
        }
        if (TOT_RAINnew <= CLIMIT) {                                 // update_rn
            TOT_RAINnew = 0.f;
            VRAIN2 = 0.f;
            QRnew = 0.f;
            ARAINnew = 0.f;
        } else {
            QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
            RQRnew = RHO * QRnew;
            if (STRAT && RQRnew > 1.E-3f) STRAT = false;
            if (DRZL && TIMLT > EPSQ) DRZL = false;
            INDEXR2 = INDEXR1;
            for (IPASS = 1; IPASS <= 3; ++IPASS) {                   // rain_pass
                if (RQRnew <= RQR_DRmin) {                           // DSD2
                    INDEXR = MDRmin;
                    N0r = RQRnew / MASSR(INDEXR);
                } else if (DRZL) {
                    DUM = fmaxf(1.0f, 0.5E-3f / RQRnew);
                    N0r = fminf(1.E9f, N0r0 * DUM * sqrtf(DUM));
                    DUM1 = RQRnew / (PI * RHOL * N0r);
                    DUM = 1.E6f * sqrtf(sqrtf(DUM1));
                    INDEXR = (int)fmaxf(FC_XMRMIN, fminf(DUM, FC_XMRMAX));
                    N0r = RQRnew / MASSR(INDEXR);
                } else if (RQRnew >= RQR_DRmax) {
                    INDEXR = MDRmax;
                    N0r = RQRnew / MASSR(INDEXR);
                } else {
                    N0r = N0r0;
                    DUM = CN0r0 * sqrtf(sqrtf(RQRnew));
                    INDEXR = (int)fmaxf(FC_XMRMIN, fminf(DUM, FC_XMRMAX));
                    if (STRAT && INDEXR < INDEXRmin) {
                        INDEXR = INDEXRmin;
                        N0r = RQRnew / MASSR(INDEXR);
                    }
                }
                VRAIN2 = GAMMAR * VRAIN(INDEXR);
                QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
                RQRnew = RHO * QRnew;
                IDR = abs(INDEXR - INDEXR2);
                INDEXR2 = INDEXR;
                if (IDR <= 5) break;
            }
            if (QRnew <= EPSQ) QRnew = 0.f;
            if (QR > 0.f && QRnew != 0.f) {
                DUM = QRnew / QR;
                if (DUM < FC_TOLER) QRnew = 0.f;
            }
            ARAINnew = BLDTRH * VRAIN2 * QRnew;
            if (ARAIN > 0.f && ARAINnew != 0.f) {
                DUM = ARAINnew / ARAIN;
                if (DUM < FC_TOLER) ARAINnew = 0.f;
            }
            RQRnew = RHO * QRnew;
            if (RQRnew > EPSQ) {
                Nrain = N0r * 1.E-6f * (float)INDEXR2;
                Zrain = Crain * RQRnew * RQRnew / Nrain;
            }
        }

        // ---- array updates (1936-2159)
        WCnew = QWnew + QRnew + QInew;
        col->T_col[l0] = TNEW;
        col->Q_col[l0] = fmaxf(EPSQ, WVnew / (1.f + WVnew));
        col->WC_col[l0] = fmaxf(EPSQ, WCnew);
        col->QI_col[l0] = fmaxf(EPSQ, QInew);
        col->QR_col[l0] = fmaxf(EPSQ, QRnew);
        col->QW_col[l0] = fmaxf(EPSQ, QWnew);
        col->RimeF_col[l0] = RIMEF;
        col->NR_col[l0] = Nrain;
        col->NS_col[l0] = Nsnow;
        Ztot = Zrain + Zsnow;
        if (Ztot > FC_ZMIN) col->DBZ_col[l0] = 10.f * log10f(Ztot);
        ASNOW = ASNOWnew;
        VSNOW1 = VSNOW;
        ARAIN = ARAINnew;
        VRAIN1 = VRAIN2;
        if (PCOND > 0.f) col->pcond1d[l0] = PCOND;
        else if (PCOND < 0.f) col->pevap1d[l0] = PCOND;
        if (PIDEP > 0.f) col->pidep1d[l0] = PIDEP;
        else if (PIDEP < 0.f) col->pisub1d[l0] = PIDEP;
        col->piacw1d[l0] = PIACW;
        col->piacwi1d[l0] = PIACWI;
        col->piacwr1d[l0] = PIACWR;
        col->piacr1d[l0] = PIACR;
        col->picnd1d[l0] = PICND;
        col->pievp1d[l0] = PIEVP;
        col->pimlt1d[l0] = PIMLT;
        col->praut1d[l0] = PRAUT;
        col->pracw1d[l0] = PRACW;
        col->prevp1d[l0] = PREVP;
        if (QInew > EPSQ) {
            col->vsnow1d[l0] = VSNOW;
            if (FLIMASS < 1.f) {
                col->vci1d[l0] = VCI;
                col->NSmICE1d[l0] = NSmICE;
            }
            col->INDEXS1d[l0] = (float)INDEXS;
        }
        if (QRnew > EPSQ) {
            col->vrain11d[l0] = VRAIN1;
            col->vrain21d[l0] = VRAIN2;
            col->INDEXR1d[l0] = (float)INDEXR2;
            IDR = 0;
            if (STRAT) IDR = 1;
            if (DRZL) IDR = IDR + 2;
            col->RFlag1d[l0] = (float)IDR;
        }
    }
    *ARAIN_io = ARAIN;
    *ASNOW_io = ASNOW;
#undef MASSI
#undef VSNOWI
#undef ACCRI
#undef VENTI1
#undef VENTI2
#undef MASSR
#undef VRAIN
#undef ACCRR
#undef VENTR1
#undef VENTR2
#undef VEL_RF
#undef FPVS
#undef FPVS0
}
