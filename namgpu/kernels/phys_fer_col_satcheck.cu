// phys_fer_col_satcheck.cu -- WOOF-NAM unit phys-fer-col-satcheck: grid-scale saturation checks,
// effective saturation mixing ratios, RHO, THICK and process initialisation (Fortran lines 911-1028).
//
// NOTICE: NOAA/NCEP NAM source module_MP_FER_HIRES.F90 (US government work, no licence file in the
// NAM clone). Module header kept as the notice (comment-prefixed):
//   MODULE MODULE_MP_FER_HIRES (Ferrier-Aligo high-resolution microphysics, nam.v4.2.11)
//   "Modified" fer_hires microphysics - 11 July 2016 version
//   SUBPROGRAM EGCP01COLUMN_hr (Ferrier, Jin; WRF structure modification).
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:911-1028 (nam.v4.2.11, 83df4e13c1).
//
// Scope: the body of the top-down loop DO L=1,LSFC after the setup part (phys-fer-col-setup,
// lines 856-910): the saturation tests and, for levels that are not CLEAR, RHO, RRHO, DTRHO,
// BLDTRH, THICK and the process initialisation. The CLEAR branch (lines 951-956) and the
// CYCLE big_loop it causes are returned as FerSatLevel.clear = true.
//
// Layout: every per-level column array is [LM, JM, IM] float32 (CONTRACT section 1); level l0 of
// column (i0, j0) is at l0 * (JM*IM) + IDX2(i0, j0, IM). One thread per column, DO L loop inside
// (CONTRACT section 3, grid_cols).
//
// Fortran-to-CUDA, every statement of 911-1028 (line numbers of module_MP_FER_HIRES.F90):
//   914-919  TK=T_col(L); TC=TK-T0C; PP=P_col(L); Q=Q_col(L); WV=Q/(1.-Q); WC=WC_col(L)
//                                                     -> fer_satcheck_level: tk, tc, pp, q, wv, wc
//   927-929  ESW=MIN(1000.*FPVS0(TK),0.99*PP); QSW=EPS*ESW/(PP-ESW); WS=QSW; QSI=QSW
//                                                     -> esw, qsw, ws, qsi
//   931-934  IF (TC .LT. 0.) ESI=MIN(1000.*FPVS(TK),0.99*PP); QSI=EPS*ESI/(PP-ESI); WS=QSI
//                                                     -> esi, qsi, ws. ESI is not assigned by the
//                                                        Fortran when TC >= 0; it is set to 0 here
//                                                        and nothing reads it on that branch.
//   939-941  QSWgrd=RHgrd*QSW; QSIgrd=RHgrd*QSI; WSgrd=RHgrd*WS -> qswgrd, qsigrd, wsgrd
//   945      IF (WV.GT.WSgrd .OR. WC.GT.EPSQ) CLEAR=.FALSE.    -> clear (CLEAR enters from setup)
//   951-956  IF (CLEAR): STRAT=.F.; DRZL=.F.; INDEXRmin=MDRmin; TIMLT=0; CYCLE big_loop
//                                                     -> carry reset; return with o.clear = true
//   967      RHO=PP/(RD*TK*(1.+EPS1*Q))               -> rho
//   968      RRHO=1./RHO                              -> rrho
//   969      DTRHO=DTPH*RHO                           -> dtrho
//   970      BLDTRH=BLEND*DTRHO (BLEND = 1.)          -> bldtrh
//   971      THICK=THICK_col(L)                       -> thick
//   973-1002 ARAINnew, ASNOWnew, QInew, QRnew, QWnew, PCOND, PIDEP, PINIT, PIACW, PIACWI, PIACWR,
//            PIACR, PICND, PIEVP, PIMLT, PRAUT, PRACW, PREVP, NSmICE, Nrain, Nsnow, RQRnew, RQSnew,
//            Zrain, Zsnow, Ztot = 0. These are the per-level accumulators of the process units;
//            they are first read there, which initialises them in the fused column loop. Nothing
//            is stored here, so this unit has no counterpart for them beyond that statement.
//   975      QI=QI_col(L)                             -> qi
//   977      QR=QR_col(L)                             -> qr
//   979      QW=QW_col(L)                             -> qw
//   1003     INDEXR=MDRmin; INDEXR1=INDEXR; INDEXR2=INDEXR -> indexr, indexr1, indexr2
//   1006     N0r=0.                                   -> n0r
//   1007     DUM1=MIN(0.,TC)                          -> dum1
//   1008     DUM=XMImax*EXP(XMIexp*DUM1)              -> dum
//   1009     INDEXS=MIN(MDImax, MAX(MDImin, INT(DUM))) -> indexs
//   1010-1012 VCI=0.; VSNOW=0.; VRAIN2=0.             -> vci, vsnow, vrain2
//   1013     RimeF1=1.                                -> rimef1
//   1015-1021 commented-out double check of the hydrometeor budget: no code, no counterpart.
// Constants: EPS = EP_2 (FER_EP2), EPS1 = EP_1 (FER_EP1), RD = R_D, T0C = TIW, EPSQ = 1e-12,
//   MDRmin = MDImin = 50 (INTEGER parameters, XMImin = 1.e6*DMImin truncated), MDImax = 1000,
//   XMImax = 1.e6*DMImax, XMIexp = .0536, BLEND = 1. (values in fer_hires.cuh).
// Single precision throughout (kfpt = single). FPVS and FPVS0 are the table functions of
// phys_fer_fpvs.cu (nam_fpvs, nam_fpvs0), called with the GPVS_hr tables (phys-fer-gpvs).
//
// Interface to the rest of the chain: fer_satcheck_level is a device function. The fused column
// loop calls it once per level after fer_col_setup_level (phys-fer-col-setup) has set CLEAR, and
// reads FerSatLevel in the ice-property and process units. phys_fer_col_satcheck below is the
// standalone entry kernel used to prove the unit on a full-size state: it writes CLEAR and RHO per
// level and the final INDEXRmin per column.
//
// Open points (named, not guessed):
//   - INDEXRmin is an INTEGER that the Fortran does not assign before the first CLEAR level
//     (line 954). The standalone kernel starts it at MDRmin. The fused loop must start it at the
//     value the Fortran holds at entry, which this source does not define.
//   - ESI for TC >= 0 is set to 0 here (see above).

#include "nmmb_common.cuh"
#include "fer_hires.cuh"
#include "phys_fer_fpvs.cu"

// Column carry for the satcheck section. The setup unit owns ARAIN, ASNOW, VRAIN1, VSNOW1, TIMLT,
// INDEXS0C and RHOX0C; this struct holds only what this section resets or reads.
struct FerSatCarry {
    bool strat;        // STRAT      stratiform rain DSD below the melting level
    bool drzl;         // DRZL       drizzle DSD below the melting level
    int  indexrmin;    // INDEXRmin  minimum mean rain diameter (microns)
    float timlt;       // TIMLT      total ice melting in a layer (drizzle detection)
};

// Column views for one column: level l0 is at p[l0 * stride] (the caller adds the column offset).
struct FerSatColumn {
    const float* T;       // T_col
    const float* P;       // P_col (pressure, Pa)
    const float* Q;       // Q_col (specific humidity)
    const float* WC;      // WC_col (total condensate mixing ratio)
    const float* QI;      // QI_col
    const float* QR;      // QR_col
    const float* QW;      // QW_col
    const float* THICK;   // THICK_col (RHO*DZ)
    const float* CLEAR;   // CLEAR_col after the setup unit (1.0 clear, 0.0 not)
    const float* tbpvs;   // GPVS_hr TBPVS
    const float* tbpvs0;  // GPVS_hr TBPVS0
    float c1xpvs, c2xpvs, c1xpvs0, c2xpvs0;   // GPVS_hr coefficients
    float RHgrd;          // threshold relative humidity (FER_HIRES argument)
    float DTPH;           // physics time step (EGCP01COLUMN_hr argument DTPH)
    size_t stride;        // JM*IM
};

// Per-level results of the satcheck section, consumed by the later units of the fused loop.
struct FerSatLevel {
    bool  clear;                        // CLEAR after the checks; true = CYCLE big_loop
    float tk, tc, pp, q, wv, wc;
    float esw, qsw, esi, qsi, ws;
    float qswgrd, qsigrd, wsgrd;
    float rho, rrho, dtrho, bldtrh, thick;
    float qi, qr, qw;
    int   indexr, indexr1, indexr2, indexs;
    float dum1, dum, n0r;
    float vci, vsnow, vrain2, rimef1;
};

// One level L (l0 = L-1) of the satcheck section, lines 911-1028.
__device__ __forceinline__ void fer_satcheck_level(int l0, const FerSatColumn& in,
                                                   FerSatCarry& c, FerSatLevel& o)
{
    const size_t k = (size_t)l0 * in.stride;

    // Lines 914-919.
    o.tk = in.T[k];
    o.tc = o.tk - FER_T0C;
    o.pp = in.P[k];
    o.q  = in.Q[k];
    o.wv = o.q / (1.0f - o.q);
    o.wc = in.WC[k];

    // Lines 927-929: saturation over water (FPVS0), capped at 0.99 P.
    o.esw = fminf(1000.0f * nam_fpvs0(o.tk, in.tbpvs0, in.c1xpvs0, in.c2xpvs0), 0.99f * o.pp);
    o.qsw = FER_EP2 * o.esw / (o.pp - o.esw);
    o.ws  = o.qsw;
    o.qsi = o.qsw;

    // Lines 931-934: below freezing, the general saturation is the one over ice (FPVS).
    o.esi = 0.0f;
    if (o.tc < 0.0f) {
        o.esi = fminf(1000.0f * nam_fpvs(o.tk, in.tbpvs, in.c1xpvs, in.c2xpvs), 0.99f * o.pp);
        o.qsi = FER_EP2 * o.esi / (o.pp - o.esi);
        o.ws  = o.qsi;
    }

    // Lines 939-941: effective grid-scale saturation mixing ratios.
    o.qswgrd = in.RHgrd * o.qsw;
    o.qsigrd = in.RHgrd * o.qsi;
    o.wsgrd  = in.RHgrd * o.ws;

    // Line 945: CLEAR (from setup) is cleared when the air is supersaturated or has condensate.
    o.clear = in.CLEAR[k] != 0.0f;
    if (o.wv > o.wsgrd || o.wc > FER_EPSQ) o.clear = false;

    // Lines 951-956: clear, subsaturated, condensate-free level: reset and skip (CYCLE big_loop).
    if (o.clear) {
        c.strat     = false;
        c.drzl      = false;
        c.indexrmin = FER_MDR_MIN;
        c.timlt     = 0.0f;
        return;
    }

    // Lines 967-971: air density and its derived products.
    o.rho    = o.pp / (FER_RD * o.tk * (1.0f + FER_EP1 * o.q));
    o.rrho   = 1.0f / o.rho;
    o.dtrho  = in.DTPH * o.rho;
    o.bldtrh = 1.0f * o.dtrho;          // BLEND = 1. (module parameter)
    o.thick  = in.THICK[k];

    // Lines 975-979: hydrometeor mixing ratios of the layer.
    o.qi = in.QI[k];
    o.qr = in.QR[k];
    o.qw = in.QW[k];

    // Lines 1003-1013: rain and snow diameter and fall-speed initialisation.
    o.indexr  = FER_MDR_MIN;
    o.indexr1 = o.indexr;
    o.indexr2 = o.indexr;
    o.n0r     = 0.0f;
    o.dum1    = fminf(0.0f, o.tc);                                    // DUM1=MIN(0.,TC)
    o.dum     = FER_XMI_MAX * expf(FER_XMI_EXP * o.dum1);             // DUM=XMImax*EXP(XMIexp*DUM1)
    o.indexs  = (int)o.dum;                                           // INT(DUM)
    o.indexs  = min(FER_MDI_MAX, max(FER_MDI_MIN, o.indexs));        // MIN(MDImax, MAX(MDImin, ...))
    o.vci    = 0.0f;
    o.vsnow  = 0.0f;
    o.vrain2 = 0.0f;
    o.rimef1 = 1.0f;
}

// Standalone entry kernel: one thread per column, DO L=1,LSFC with LSFC = FER_LSFC of the column
// (Fortran LSFC = KTE-LOWLYR+1; Fortran big_loop runs only L=1..LSFC). Writes the CLEAR flag and RHO
// per level for levels 1..LSFC; levels LSFC+1..LM are written as 0 (never run by the Fortran, the
// same defined-default convention as phys-fer-col-setup). Writes INDEXRmin after the satcheck
// section (see the open points above; it is not the Fortran end-of-column value).
extern "C" __global__ void phys_fer_col_satcheck(
    const float* __restrict__ T, const float* __restrict__ P, const float* __restrict__ Q,
    const float* __restrict__ WC, const float* __restrict__ QI, const float* __restrict__ QR,
    const float* __restrict__ QW, const float* __restrict__ THICK, const float* __restrict__ CLEAR,
    const int* __restrict__ lsfc,     // FER_LSFC per column (phys-fer-fer-hires column prep)
    const float* __restrict__ tbpvs, const float* __restrict__ tbpvs0,
    const float* __restrict__ coef,   // GPVS_hr (c1xpvs, c2xpvs, c1xpvs0, c2xpvs0), device array
    const float RHgrd, const float DTPH,
    int* __restrict__ clear_out, float* __restrict__ rho_out, int* __restrict__ indexrmin_out,
    const int IM, const int JM, const int LM)
{
    COLUMN_IJ(IM, JM)   // defines col_, i0, j0; returns for out-of-range threads

    const size_t off = IDX2(i0, j0, IM);
    const size_t stride = (size_t)JM * (size_t)IM;

    FerSatColumn in;
    in.T = T + off;       in.P = P + off;       in.Q = Q + off;       in.WC = WC + off;
    in.QI = QI + off;     in.QR = QR + off;     in.QW = QW + off;     in.THICK = THICK + off;
    in.CLEAR = CLEAR + off;
    in.tbpvs = tbpvs;     in.tbpvs0 = tbpvs0;
    in.c1xpvs = coef[0];  in.c2xpvs = coef[1];  in.c1xpvs0 = coef[2];   in.c2xpvs0 = coef[3];
    in.RHgrd = RHgrd;     in.DTPH = DTPH;
    in.stride = stride;

    // LSFC of the column. Out-of-range values are counted and refused by phys-fer-col-setup
    // (FER_LSFC_BAD must be 0 on the step path); here they are only clamped so the kernel stays in bounds.
    int n = lsfc[off];
    n = (n < 0) ? 0 : ((n > LM) ? LM : n);

    FerSatCarry c;
    c.strat     = false;
    c.drzl      = false;
    c.indexrmin = FER_MDR_MIN;
    c.timlt     = 0.0f;

    for (int l0 = 0; l0 < n; ++l0) {      // DO L=1,LSFC
        FerSatLevel o;
        fer_satcheck_level(l0, in, c, o);
        const size_t k = (size_t)l0 * stride + off;
        clear_out[k] = o.clear ? 1 : 0;
        rho_out[k]   = o.clear ? 0.0f : o.rho;
    }
    for (int l0 = n; l0 < LM; ++l0) {     // levels above LSFC: defined defaults, never stale
        const size_t k = (size_t)l0 * stride + off;
        clear_out[k] = 0;
        rho_out[k]   = 0.0f;
    }
    indexrmin_out[off] = c.indexrmin;
}
