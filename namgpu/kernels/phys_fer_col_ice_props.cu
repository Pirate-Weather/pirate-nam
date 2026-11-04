// phys_fer_col_ice_props.cu -- WOOF-NAM unit phys-fer-col-ice-props.
//
// NOTICE: NAM source module_MP_FER_HIRES.F90 (NCEP/EMC, US Government work). The file has no
// copyright or licence block; its "Modified fer_hires microphysics" changelog (lines 1-82) is the
// notice, kept as in fer_hires.cuh.
//
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:1029-1320 (nam.v4.2.11, 83df4e13c1).
//
// What the range computes, per level, in Fortran order:
//   1029-1058  TK2, TFACTOR, DYNVIS, THERM_COND, DIFFUS (air properties), GAMMAS (ice fall-speed
//              air-resistance factor, limited to 1.5), GAMMAR (rain air-resistance factor).
//   1064-1080  ICE_logical / T_ICE gate / RAIN_logical decision tree.
//   1083-1140  small ice (50 micron): NSImax, Fletcher and Cooper nuclei, NSmICE, QSmICE.
//   1141-1204  init_ice (no large ice) or time-averaged large-ice properties: TOT_ICE, PILOSS, QLgICE,
//              VCI, RimeF1 from RimeF_col(L) and RimeF_col(LBEF), mean diameter INDEXS (Houze blend),
//              EMAIRI, QLICE, RQLICE, QTICE, FLIMASS.
//   1205-1280  two_pass DO loop (IPASS = 1, 2): rime-factor ceiling RFmx and RFmax, velocity increase
//              VrimeF from VEL_RF, rimed fall speed VSNOW (AVhail/BVhail hail branch), XLIMASS, NLICE,
//              NLImin/NSI_max bounds, and the new INDEXS from XLI (EXIT on IPASS>=2 or NLICE in range).
//   1296-1316  STRAT / INDEXRmin / TIMLT / DRZL decision (TC<=0 freezing-level memory, stratiform test).
//
// Design. The Fortran level loop runs top to bottom (L = 1 .. LSFC), and several locals persist across
// levels: TIMLT, RHOX0C, INDEXS0C, INDEXRmin, STRAT, DRZL, VSNOW1, and the ice-branch values that only
// some levels assign (they keep their last value, as Fortran locals do). They live in FerIceState, one
// per column, held by the column thread (CONTRACT section 3). The per-level body is the __device__
// function fer_col_ice_level(). fer_col_ice_props_eval is the column kernel (one thread per column, the
// level loop inside). Production column kernels include this file and call fer_col_ice_level() in their
// own level loop, after the sections that the same Fortran loop runs before this one.
//
// Inputs per level (Fortran names): TK (T_col), PP (P_col, mid-level pressure), RHO (air density, line
// 967), THICK (THICK_col), QI (QI_col), QR (QR_col), ARAIN and ASNOW (accumulated fall from above, carried
// by the column driver), RimeF_col(L) and RimeF_col(LBEF) with LBEF = MAX(1, L-1). DTPH is the physics
// step. Tables: MASSI and VSNOWI (MDImin:MDImax, index m - 50), VEL_RF(2:9, 0:Nrime). Scalars: RFmx1,
// RFmax, RH_NgC, RH_NgT, RQhail, AVhail, BVhail (runtime SAVE values set by FERRIER_INIT_HR).
//
// Constants come from fer_hires.cuh (MODULE_CONSTANTS and MP_FER_HIRES PARAMETERs).
#include "nmmb_common.cuh"
#include "fer_hires.cuh"

#define FER_BLEND        1.0f                  // BLEND parameter (line 802)
#define FER_RHO0         1.194f                // RHO0 parameter (line 789)
#define FER_XMIMAX       (1.0e6f * 1.0e-3f)    // XMImax = 1.e6*DMImax (line 132), REAL
#define FER_XMIEXP       0.0536f               // XMIexp (line 132)

// Runtime scalars and tables (uploaded once by the init unit, read-only here).
struct FerIceTables {
    const float* massi;     // MASSI(MDImin:MDImax), index m - FER_MDI_MIN
    const float* vsnowi;    // VSNOWI(MDImin:MDImax), index m - FER_MDI_MIN
    const float* vel_rf;    // VEL_RF(2:9, 0:Nrime), index FER_TBL_VEL_RF_INDEX(IXS, IXRF) = IXRF*8 + (IXS-2)
    float RFmx1;
    float RFmax;
    float RH_NgC;
    float RH_NgT;
    float RQhail;
    float AVhail;
    float BVhail;
};

// Column state. Persistent values and per-level results of the section, in one struct.
struct FerIceState {
    // persistent across levels
    float TIMLT;
    float RHOX0C;
    float VSNOW1;
    int   INDEXS0C;
    int   INDEXRmin;
    int   STRAT;
    int   DRZL;
    // assigned only on some levels, so they keep their last value (Fortran locals)
    float TOT_ICE, PILOSS, QLgICE, QLICE, RQLICE, QTICE, FLIMASS, EMAIRI, NLICE;
    float VSNOW, VEL_INC, VrimeF, XLIMASS, VCI, QSmICE, NSmICE, Rcw, NInuclei, NSImax, RimeF1;
    // per-level results
    int   INDEXS;
    int   ICE_logical;
    int   RAIN_logical;
    float TK2, TFACTOR, DYNVIS, THERM_COND, DIFFUS, GAMMAS, GAMMAR;
};

// Column initialisation: the values the Fortran column setup gives before the level loop (lines 844-852).
__device__ inline void fer_col_ice_init(FerIceState& s)
{
    s.TIMLT = 0.0f;
    s.RHOX0C = 22.5f;
    s.VSNOW1 = 0.0f;
    s.INDEXS0C = FER_MDI_MIN;
    s.INDEXRmin = FER_MDR_MIN;     // not set before the loop in Fortran; MDRmin is the first value any path reads
    s.STRAT = 0;
    s.DRZL = 0;
    s.TOT_ICE = 0.0f; s.PILOSS = 0.0f; s.QLgICE = 0.0f; s.QLICE = 0.0f; s.RQLICE = 0.0f;
    s.QTICE = 0.0f; s.FLIMASS = 0.0f; s.EMAIRI = 0.0f; s.NLICE = 0.0f;
    s.VSNOW = 0.0f; s.VEL_INC = 0.0f; s.VrimeF = 0.0f; s.XLIMASS = 0.0f; s.VCI = 0.0f;
    s.QSmICE = 0.0f; s.NSmICE = 0.0f; s.Rcw = 0.0f; s.NInuclei = 0.0f; s.NSImax = 0.0f;
    s.RimeF1 = 1.0f;
    s.INDEXS = FER_MDI_MIN;
    s.ICE_logical = 0;
    s.RAIN_logical = 0;
    s.TK2 = 0.0f; s.TFACTOR = 0.0f; s.DYNVIS = 0.0f; s.THERM_COND = 0.0f; s.DIFFUS = 0.0f;
    s.GAMMAS = 0.0f; s.GAMMAR = 0.0f;
}

// One level of the FER_HIRES column loop, lines 1029-1320. The arguments are this level's inputs;
// s carries the column state in and out. RimeF_lbef is RimeF_col(LBEF).
__device__ inline void fer_col_ice_level(const FerIceTables& tb,
                                         float TK, float PP, float RHO, float THICK,
                                         float QI, float QR, float ARAIN, float ASNOW,
                                         float RimeF_l, float RimeF_lbef, float DTPH, int CLEAR,
                                         FerIceState& s)
{
    const float TC = TK - FER_T0C;                              // line 915

    // Lines 895-907: ARAIN is zeroed when it is at or below CLIMIT (CLEAR ELSE, line 895). ASNOW is zeroed
    // when it is at or below CLIMIT (line 907), and VSNOW1 is reset when snow is falling in (line 905).
    // These run before the CLEAR gate, as in Fortran.
    ARAIN = (ARAIN > FER_CLIMIT) ? ARAIN : 0.0f;
    if (ASNOW > FER_CLIMIT) {
        s.VSNOW1 = 0.0f;
    } else {
        ASNOW = 0.0f;
    }

    // Lines 951-956: a CLEAR level (clear, subsaturated, condensate-free, set by phys-fer-col-satcheck) resets
    // STRAT, DRZL, INDEXRmin and TIMLT, then CYCLE big_loop. Lines 958-1320 and the end-of-level carry
    // (lines 2096-2098) do not run, so every other state value keeps its last value (Fortran locals).
    if (CLEAR) {
        s.STRAT = 0;
        s.DRZL = 0;
        s.INDEXRmin = FER_MDR_MIN;
        s.TIMLT = 0.0f;
        return;
    }

    // Lines 967-971 (RRHO, DTRHO, BLDTRH) and the per-level resets of lines 1000-1027 that this
    // section reads: INDEXS (1st estimate, lines 1007-1009), RimeF1, VCI, VSNOW, NSmICE.
    const float RRHO = 1.0f / RHO;
    const float DTRHO = DTPH * RHO;
    const float BLDTRH = FER_BLEND * DTRHO;
    {
        float DUM1 = fminf(0.0f, TC);
        float DUM = FER_XMIMAX * expf(FER_XMIEXP * DUM1);
        s.INDEXS = min(FER_MDI_MAX, max(FER_MDI_MIN, (int)DUM));
    }
    s.RimeF1 = 1.0f;
    s.VCI = 0.0f;
    s.VSNOW = 0.0f;
    s.NSmICE = 0.0f;

    // ---- 1029-1058: air properties and fall-speed factors ----
    s.TK2 = 1.0f / (TK * TK);
    s.TFACTOR = sqrtf(TK * TK * TK) / (TK + 120.0f);
    s.DYNVIS = 1.496e-6f * s.TFACTOR;
    s.THERM_COND = 2.116e-3f * s.TFACTOR;
    s.DIFFUS = 8.794e-5f * powf(TK, 1.81f) / PP;
    s.GAMMAS = fminf(1.5f, powf(1.0e5f / PP, FER_C1));
    s.GAMMAR = powf(FER_RHO0 / RHO, 0.4f);

    // ---- 1064-1080: decision tree ----
    int ICE = (TC < 0.0f || QI > FER_EPSQ || ASNOW > FER_CLIMIT) ? 1 : 0;
    if (!ICE) {
        s.QLICE = 0.0f;
        s.QTICE = 0.0f;
    }
    if (FER_T_ICE <= -100.0f) {                                 // never true (T_ICE = -40)
        ICE = 0;
        s.QLICE = 0.0f;
        s.QTICE = 0.0f;
    }
    s.ICE_logical = ICE;
    s.RAIN_logical = (ARAIN > FER_CLIMIT || QR > FER_EPSQ) ? 1 : 0;

    // ---- 1083-1280: ice_test ----
    if (ICE) {
        s.NInuclei = 0.0f;
        s.NSmICE = 0.0f;
        s.QSmICE = 0.0f;
        s.Rcw = 0.0f;
        if (TC < 0.0f) {
            // Max number of small ice crystals from 10% of total ice or NSI_max (line 1122)
            s.NSImax = fmaxf(FER_NSI_MAX, 0.1f * RHO * QI / tb.massi[0]);
            // Fletcher (1962) then Cooper (1984) limited by Fletcher (lines 1131-1132)
            s.NInuclei = fminf(0.01f * expf(-0.6f * TC), s.NSImax);
            s.NInuclei = fminf(5.0f * expf(-0.304f * TC), s.NInuclei);
            if (QI > FER_EPSQ) {
                float DUM = RRHO * tb.massi[0];                 // line 1136
                s.NSmICE = fminf(s.NInuclei, QI / DUM);
                s.QSmICE = s.NSmICE * DUM;
            }
        }

        if (QI <= FER_EPSQ && ASNOW <= FER_CLIMIT) {
            // init_ice (lines 1141-1154): no large ice in this level
            s.TOT_ICE = 0.0f;
            s.PILOSS = 0.0f;
            s.RimeF1 = 1.0f;
            s.VrimeF = 1.0f;
            s.VEL_INC = s.GAMMAS;
            s.VSNOW = 0.0f;
            s.VSNOW1 = 0.0f;
            s.VCI = 0.0f;
            s.EMAIRI = THICK;
            s.XLIMASS = s.RimeF1 * tb.massi[s.INDEXS - FER_MDI_MIN];
            s.FLIMASS = 1.0f;
            s.QLICE = 0.0f;
            s.RQLICE = 0.0f;
            s.QTICE = 0.0f;
            s.NLICE = 0.0f;
        } else {
            // lines 1158-1204: time-averaged large-ice properties
            s.TOT_ICE = THICK * QI + FER_BLEND * ASNOW;
            s.PILOSS = -s.TOT_ICE / THICK;
            s.QLgICE = fmaxf(0.0f, QI - s.QSmICE);          // 1st-guess large ice
            s.VCI = s.GAMMAS * tb.vsnowi[0];                // VSNOWI(MDImin)
            s.RimeF1 = (RimeF_l * THICK * s.QLgICE + RimeF_lbef * FER_BLEND * ASNOW) / s.TOT_ICE;

            // Mean diameter blend for RimeF1 > 2 (Houze), lines 1177-1191
            if (s.RimeF1 > 2.0f) {
                float DUM3 = tb.RH_NgC * powf(RHO * s.QLgICE, FER_C1);      // convective
                float DUM2 = tb.RH_NgT * powf(RHO * s.QLgICE, FER_C1);      // transition
                float DUM;
                if (s.RimeF1 >= 10.0f) {
                    DUM = DUM3;
                } else if (s.RimeF1 >= 5.0f) {
                    DUM = 0.2f * (s.RimeF1 - 5.0f);
                    DUM = DUM3 * DUM + DUM2 * (1.0f - DUM);
                } else {
                    float DUM1 = (float)s.INDEXS;                           // stratiform
                    DUM = 0.33333f * (s.RimeF1 - 2.0f);
                    DUM = DUM2 * DUM + DUM1 * (1.0f - DUM);
                }
                s.INDEXS = min(FER_MDI_MAX, max(FER_MDI_MIN, (int)DUM));
            }
            s.EMAIRI = THICK + BLDTRH * s.VSNOW1;
            s.QLICE = (THICK * s.QLgICE + FER_BLEND * ASNOW) / s.EMAIRI;   // large ice
            s.RQLICE = RHO * s.QLICE;
            s.QTICE = s.QLICE + s.QSmICE;
            s.FLIMASS = s.QLICE / s.QTICE;

            // ---- two_pass, IPASS = 1, 2 (lines 1205-1280) ----
            for (int ipass = 1; ipass <= 2; ++ipass) {
                // Rime factor ceiling from the mean diameter (lines 1213-1216)
                float DUM = 1.0e-6f * (float)s.INDEXS;
                float RFmx = tb.RFmx1 * DUM * DUM * DUM / tb.massi[s.INDEXS - FER_MDI_MIN];
                s.RimeF1 = fminf(s.RimeF1, RFmx);

                if (s.RimeF1 <= 1.0f) {                     // vel_rime
                    s.RimeF1 = 1.0f;
                    s.VrimeF = 1.0f;
                } else {
                    s.RimeF1 = fminf(s.RimeF1, tb.RFmax);
                    int IXS = max(2, min(s.INDEXS / 100, 9));
                    float XRF = 10.492f * logf(s.RimeF1);
                    int IXRF = max(0, min((int)XRF, FER_NRIME));
                    if (IXRF >= FER_NRIME) {
                        s.VrimeF = tb.vel_rf[FER_TBL_VEL_RF_INDEX(IXS, FER_NRIME)];
                    } else {
                        float v0 = tb.vel_rf[FER_TBL_VEL_RF_INDEX(IXS, IXRF)];
                        float v1 = tb.vel_rf[FER_TBL_VEL_RF_INDEX(IXS, IXRF + 1)];
                        s.VrimeF = v0 + (XRF - (float)IXRF) * (v1 - v0);
                    }
                    s.VrimeF = fmaxf(1.0f, s.VrimeF);
                }

                s.VEL_INC = s.GAMMAS * s.VrimeF;                                // may17
                s.VSNOW = s.VEL_INC * tb.vsnowi[s.INDEXS - FER_MDI_MIN];
                if (s.RimeF1 >= 5.0f && s.INDEXS == FER_MDI_MAX && s.RQLICE > tb.RQhail) {
                    // Thompson graupel/hail fall speed (AVhail, BVhail), lines 1236-1241
                    float DUMh = s.GAMMAS * tb.AVhail * powf(s.RQLICE, tb.BVhail);
                    if (DUMh > s.VSNOW) {
                        s.VSNOW = DUMh;
                        s.VEL_INC = s.VSNOW / tb.vsnowi[s.INDEXS - FER_MDI_MIN];
                    }
                }
                s.XLIMASS = s.RimeF1 * tb.massi[s.INDEXS - FER_MDI_MIN];        // line 1244
                s.NLICE = s.RQLICE / s.XLIMASS;

                // EXIT two_pass on IPASS >= 2 or NLICE within [NLImin, NSI_max] (lines 1250-1251)
                if (ipass >= 2 || (s.NLICE >= FER_NLIMIN && s.NLICE <= FER_NSI_MAX)) break;

                // Force NLICE into [NLImin, NSI_max] for IPASS = 1 (line 1258), then the new size
                s.NLICE = fmaxf(FER_NLIMIN, fminf(FER_NSI_MAX, s.NLICE));
                float XLI = s.RQLICE / (s.NLICE * s.RimeF1);                    // mean mass of unrimed ice
                if (XLI <= tb.massi[0]) {                                       // MASSI(MDImin)
                    s.INDEXS = FER_MDI_MIN;
                } else if (XLI <= tb.massi[450 - FER_MDI_MIN]) {                // MASSI(450)
                    float DLI = 9.5885e5f * powf(XLI, 0.42066f);               // microns
                    s.INDEXS = min(FER_MDI_MAX, max(FER_MDI_MIN, (int)DLI));
                } else if (XLI < tb.massi[FER_MDI_MAX - FER_MDI_MIN]) {         // MASSI(MDImax)
                    float DLI = 3.9751e6f * powf(XLI, 0.49870f);
                    s.INDEXS = min(FER_MDI_MAX, max(FER_MDI_MIN, (int)DLI));
                } else {
                    s.INDEXS = FER_MDI_MAX;
                }
            }   // two_pass
        }       // init_ice
    }           // ice_test

    // ---- lines 1296-1316: STRAT, INDEXRmin, TIMLT, DRZL ----
    if (TC <= 0.0f) {
        s.STRAT = 0;
        s.INDEXRmin = FER_MDR_MIN;
        s.TIMLT = 0.0f;
        s.INDEXS0C = s.INDEXS;
        s.RHOX0C = 22.5f * fmaxf(1.0f, fminf(s.RimeF1, 40.0f));    // ice density at 0C
    } else {
        if (!s.RAIN_logical) {
            s.STRAT = 0;
            s.INDEXRmin = FER_MDR_MIN;
            if (!s.ICE_logical) s.TIMLT = 0.0f;
        } else {
            if (s.TIMLT > FER_EPSQ && s.RHOX0C <= 225.0f) {
                s.STRAT = 1;
            } else {
                s.STRAT = 0;
                s.INDEXRmin = FER_MDR_MIN;
            }
            if (s.STRAT && s.INDEXRmin <= FER_MDR_MIN) {
                s.INDEXRmin = (int)((float)s.INDEXS0C * powf(0.001f * s.RHOX0C, FER_C1));
                s.INDEXRmin = max(FER_MDR_MIN, min(s.INDEXRmin, 500));      // INDEXRstrmax = 500
            }
        }
    }

    // DRZL: drizzle unless stratiform or melted ice falls from above (lines 1311-1316)
    s.DRZL = (s.STRAT || s.TIMLT > FER_EPSQ) ? 0 : 1;

    // Line 2097: VSNOW1 = VSNOW carries this level's ice fall speed into the next level (used by EMAIRI, line 1194).
    s.VSNOW1 = s.VSNOW;
}

// Output slots of fer_col_ice_props_eval, written as [slot, LM, JM, IM] float32.
// Logicals and integers are stored as float (exact for these magnitudes).
#define FER_OUT_ICE_LOGICAL   0
#define FER_OUT_RAIN_LOGICAL  1
#define FER_OUT_INDEXS        2
#define FER_OUT_RIMEF1        3
#define FER_OUT_VRIMEF        4
#define FER_OUT_VEL_INC       5
#define FER_OUT_VSNOW         6
#define FER_OUT_VCI           7
#define FER_OUT_QLICE         8
#define FER_OUT_QTICE         9
#define FER_OUT_RQLICE        10
#define FER_OUT_NLICE         11
#define FER_OUT_QSMICE        12
#define FER_OUT_NSMICE        13
#define FER_OUT_TOT_ICE       14
#define FER_OUT_PILOSS        15
#define FER_OUT_EMAIRI        16
#define FER_OUT_XLIMASS       17
#define FER_OUT_FLIMASS       18
#define FER_OUT_QLGICE        19
#define FER_OUT_STRAT         20
#define FER_OUT_DRZL          21
#define FER_OUT_TIMLT         22
#define FER_OUT_RHOX0C        23
#define FER_OUT_INDEXS0C      24
#define FER_OUT_INDEXRMIN     25
#define FER_OUT_TK2           26
#define FER_OUT_TFACTOR       27
#define FER_OUT_DYNVIS        28
#define FER_OUT_THERM_COND    29
#define FER_OUT_DIFFUS        30
#define FER_OUT_GAMMAS        31
#define FER_OUT_GAMMAR        32
#define FER_OUT_VSNOW1        33
#define FER_OUT_RCW           34
#define FER_OUT_NINUCLEI      35
#define FER_OUT_NSIMAX        36
#define FER_OUT_NSLOT         37

// Column kernel: one thread per column, the level loop inside (CONTRACT section 3).
// ARAIN and ASNOW are per-level inputs (the value on entry to each level, as the column driver carries them).
extern "C" __global__ void fer_col_ice_props_eval(
    const float* TK,
    const float* PP,
    const float* RHO,
    const float* THICK,
    const float* QI,
    const float* QR,
    const float* ARAIN,
    const float* ASNOW,
    const float* RIMEF_COL,
    const int* CLEAR,
    const float* MASSI,
    const float* VSNOWI,
    const float* VEL_RF,
    float RFmx1, float RFmax, float RH_NgC, float RH_NgT,
    float RQhail, float AVhail, float BVhail, float DTPH,
    float* OUT,
    int IM, int JM, int LM)
{
    COLUMN_IJ(IM, JM)

    FerIceTables tb;
    tb.massi = MASSI;
    tb.vsnowi = VSNOWI;
    tb.vel_rf = VEL_RF;
    tb.RFmx1 = RFmx1;
    tb.RFmax = RFmax;
    tb.RH_NgC = RH_NgC;
    tb.RH_NgT = RH_NgT;
    tb.RQhail = RQhail;
    tb.AVhail = AVhail;
    tb.BVhail = BVhail;

    const size_t NCELL = (size_t)LM * JM * IM;
    FerIceState s;
    fer_col_ice_init(s);

    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t c = IDX3(i0, j0, l0, IM, JM);
        const int lbef0 = max(0, l0 - 1);                       // LBEF = MAX(1, L-1)
        fer_col_ice_level(tb, TK[c], PP[c], RHO[c], THICK[c], QI[c], QR[c], ARAIN[c], ASNOW[c],
                          RIMEF_COL[c], RIMEF_COL[IDX3(i0, j0, lbef0, IM, JM)], DTPH, CLEAR[c], s);

#define FER_PUT(slot, val) OUT[(size_t)(slot) * NCELL + c] = (float)(val)
        FER_PUT(FER_OUT_ICE_LOGICAL,  s.ICE_logical);
        FER_PUT(FER_OUT_RAIN_LOGICAL, s.RAIN_logical);
        FER_PUT(FER_OUT_INDEXS,       s.INDEXS);
        FER_PUT(FER_OUT_RIMEF1,       s.RimeF1);
        FER_PUT(FER_OUT_VRIMEF,       s.VrimeF);
        FER_PUT(FER_OUT_VEL_INC,      s.VEL_INC);
        FER_PUT(FER_OUT_VSNOW,        s.VSNOW);
        FER_PUT(FER_OUT_VCI,          s.VCI);
        FER_PUT(FER_OUT_QLICE,        s.QLICE);
        FER_PUT(FER_OUT_QTICE,        s.QTICE);
        FER_PUT(FER_OUT_RQLICE,       s.RQLICE);
        FER_PUT(FER_OUT_NLICE,        s.NLICE);
        FER_PUT(FER_OUT_QSMICE,       s.QSmICE);
        FER_PUT(FER_OUT_NSMICE,       s.NSmICE);
        FER_PUT(FER_OUT_TOT_ICE,      s.TOT_ICE);
        FER_PUT(FER_OUT_PILOSS,       s.PILOSS);
        FER_PUT(FER_OUT_EMAIRI,       s.EMAIRI);
        FER_PUT(FER_OUT_XLIMASS,      s.XLIMASS);
        FER_PUT(FER_OUT_FLIMASS,      s.FLIMASS);
        FER_PUT(FER_OUT_QLGICE,       s.QLgICE);
        FER_PUT(FER_OUT_STRAT,        s.STRAT);
        FER_PUT(FER_OUT_DRZL,         s.DRZL);
        FER_PUT(FER_OUT_TIMLT,        s.TIMLT);
        FER_PUT(FER_OUT_RHOX0C,       s.RHOX0C);
        FER_PUT(FER_OUT_INDEXS0C,     s.INDEXS0C);
        FER_PUT(FER_OUT_INDEXRMIN,    s.INDEXRmin);
        FER_PUT(FER_OUT_TK2,          s.TK2);
        FER_PUT(FER_OUT_TFACTOR,      s.TFACTOR);
        FER_PUT(FER_OUT_DYNVIS,       s.DYNVIS);
        FER_PUT(FER_OUT_THERM_COND,   s.THERM_COND);
        FER_PUT(FER_OUT_DIFFUS,       s.DIFFUS);
        FER_PUT(FER_OUT_GAMMAS,       s.GAMMAS);
        FER_PUT(FER_OUT_GAMMAR,       s.GAMMAR);
        FER_PUT(FER_OUT_VSNOW1,       s.VSNOW1);
        FER_PUT(FER_OUT_RCW,          s.Rcw);
        FER_PUT(FER_OUT_NINUCLEI,     s.NInuclei);
        FER_PUT(FER_OUT_NSIMAX,       s.NSImax);
#undef FER_PUT
    }
}
