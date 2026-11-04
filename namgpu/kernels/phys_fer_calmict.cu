// phys_fer_calmict.cu -- nam-gpu unit phys-fer-calmict: CALMICT, Ferrier radar reflectivity
// (REFL_10CM composite) for one point. Device function nam_calmict() plus the eval kernel
// fer_calmict_eval (one thread per (i, j, l) point; every point is independent).
//
// NOTICE: NOAA/NCEP NAM source nmm/module_DIAGNOSE.F90 (US government work, no licence file in the
// NAM clone; NOAA author headers kept as notices). Ferrier-Aligo ice and rain logic follows
// phys/module_MP_FER_HIRES.F90 (NOAA "Modified" fer_hires) and phys/module_MP_ETANEW.F90 for the
// MASSI/MASSR tables, FPVS and RQR_DR*/CN0r* scalars.
//
// Ported to CUDA for nam-gpu from nmm/module_DIAGNOSE.F90:1589-1798 (nam.v4.2.11, 83df4e13c1).
// Fortran-to-CUDA review (every statement of CALMICT):
//   1589-1600  arguments: P1D,T1D,Q1D,C1D,FI1D,FR1D,FS1D,CUREFL -> inputs; DBZ1 -> output;
//              I, J, Ilook, Jlook, MY_DOMAIN_ID are debug-print indices only (the commented
//              WRITE at 1733-1736 is dropped).
//   1602-1613  REAL/INTEGER parameters: MDImin=50, MDImax=1000, XMImax=1000 (1e6*1e-3),
//              MDRmin=50, MDRmax=1000, XMRmin=50, XMRmax=1000.
//   1615-1626  RHgrd=1, T_ICE=-40, NLImax=5e3, NLImin=1e3, N0r0=8e6, N0rmin=1e4.
//   1629-1632  DBZ1=DBZmin; C1D<=EPSQ -> RETURN with DBZmin; else WC=C1D.
//   1637-1663  condensate partition QI1 (ice), QW1 (cloud water), QR1 (rain) from FI1D/FR1D/TC.
//   1664-1666  WV, RHO, RRHO (RHO uses P608 and T1D).
//   1669-1690  rain reflectivity: RQR=RHO*QR1; N0r and INDEXR by the RQR_DRmin/RQR_DRmax
//              branches; DRmm=1e-3*INDEXR; Zrain=0.72*N0r*DRmm**6 (seven factors, Fortran order).
//   1693-1706  QICE, RHO, RRHO; ESAT=1000*FPVS(T1D); QSIgrd=RHgrd*EPSILON*ESAT/(P1D-ESAT);
//              WVQW=WV+QW1.
//   1709-1729  FLARGE (large-ice fraction), FSMALL, XSIMASS = RRHO*MASSI(MDImin)*FSMALL.
//   1731-1737  DUM=XMImax*EXP(.0536*TC); INDEXS = MIN(MDImax, MAX(MDImin, INT(DUM))).
//   1738-1741  RimeF = AMAX1(1, FS1D); XLIMASS=RRHO*RimeF*MASSI(INDEXS); FLIMASS; QLICE; NLICE.
//   1744-1791  new_nlice: NLICE outside [NLImin, NLImax] -> clamp, XLI, INDEXS from the three
//              MASSI breakpoints (DLI power laws, MASSI(450) break), the RimeF density
//              adjustment at the NLImax/MDImax corner, then XLIMASS, FLIMASS, QLICE, NLICE again.
//   1792-1794  QS1 = AMIN1(QI1, QLICE); QI1 = AMAX1(0, QI1-QS1).
//   1795-1797  NLICE>0 and QLICE>0: Zice = Cice*RHO*RHO*QLICE*QLICE/NLICE (Cice = 1.634e13).
//   1800-1804  Ztot = Zrain+Zice+Zconv; Ztot > Zmin -> DBZ1 = 10*log10(Ztot).
//   Constants: DBZmin = -20, EPSQ = 1e-12, TIW = 273.15 (module_CONSTANTS); R_D, R_V, P608
//   (module_CONSTANTS via nmmb_common.cuh); EPSILON = R_D/R_V and Zmin = 0.01 (module_DIAGNOSE 15-16).
//
// Host-supplied tables (phys-fer-init, ctx.tables["fer_*"]):
//   massi  float[951], Fortran MASSI(50:1000), index m-50.
//   tbpvs  float[7501], Fortran TBPVS(1:NX) for FPVS (module_MP_ETANEW), 1-based in Fortran.
//   scalars: C1XPVS, C2XPVS (FPVS), RQR_DRmin, RQR_DRmax, CN0R0, CN0r_DMRmin, CN0r_DMRmax.
// Fortran "INDEXR=MAX(XMRmin, MIN(CN0r0*RQR**.25, XMRmax))" stores a REAL in an INTEGER, which
// truncates toward zero; the argument is >= 50 so this is floor. Same for INT(DUM) and INT(DLI).
// Fortran FPVS (module_MP_ETANEW.F90:2475-2526) is reproduced in calmict_fpvs below.
#include "nmmb_common.cuh"

#define CM_NX        7501
#define CM_XMIN      180.0f
#define CM_XMAX      330.0f

#define CM_DBZMIN    (-20.0f)
#define CM_ZMIN      0.01f
#define CM_TFRZ      NMMB_TIW
#define CM_CICE_MASS 1.634e13f          /* Cice, module_DIAGNOSE line 11 */
#define CM_RHGRD     1.0f
#define CM_T_ICE     (-40.0f)
#define CM_NLIMAX    5.0e3f
#define CM_NLIMIN    1.0e3f
#define CM_N0R0      8.0e6f
#define CM_N0RMIN    1.0e4f
#define CM_MDIMIN    50
#define CM_MDIMAX    1000
#define CM_XMIMAX    1000.0f             /* XMImax = 1e6*DMImax, DMImax = 1e-3 */
#define CM_MDRMIN    50
#define CM_MDRMAX    1000
#define CM_XMRMIN    50.0f
#define CM_XMRMAX    1000.0f

// FPVS(T) of module_MP_ETANEW: table interpolation on [XMIN, XMAX], Magnus-Tetens outside.
// Same expression as phys_fer_fpvs.cu nam_fpvs; kept local so this unit is self-contained.
__device__ inline float calmict_fpvs(float T, const float* __restrict__ tbpvs, float c1, float c2)
{
    if (T >= CM_XMIN && T <= CM_XMAX) {
        float xj = fminf(fmaxf(c1 + c2 * T, 1.0f), (float)CM_NX);
        int jx = (int)fminf(xj, (float)(CM_NX - 1));
        return tbpvs[jx - 1] + (xj - (float)jx) * (tbpvs[jx] - tbpvs[jx - 1]);
    } else if (T > CM_XMAX) {
        return 0.61078f * expf(17.2694f * (T - 273.16f) / (T - 35.86f));
    } else {
        return 0.61078f * expf(21.8746f * (T - 273.16f) / (T - 7.66f));
    }
}

// One CALMICT call. Returns DBZ1. massi is MASSI with index m-50 (m = 50..1000 microns).
__device__ inline float nam_calmict(float P1D, float T1D, float Q1D, float C1D,
                                    float FI1D, float FR1D, float FS1D, float CUREFL,
                                    const float* __restrict__ massi,
                                    const float* __restrict__ tbpvs,
                                    float c1xpvs, float c2xpvs,
                                    float RQR_DRmin, float RQR_DRmax,
                                    float CN0r0, float CN0r_DMRmin, float CN0r_DMRmax)
{
    const float EPSILON = NMMB_R_D / NMMB_R_V;
    float DBZ1 = CM_DBZMIN;
    if (C1D <= NMMB_EPSQ) return DBZ1;                  /* 1629-1632: no condensate */
    const float WC = C1D;

    float Zrain = 0.0f, Zice = 0.0f;
    const float Zconv = CUREFL;
    float QW1 = 0.0f, QI1 = 0.0f, QR1 = 0.0f, QS1 = 0.0f;
    const float TC = T1D - CM_TFRZ;
    const float Fice = FI1D;
    const float Frain = FR1D;
    if (TC <= CM_T_ICE || Fice >= 1.0f) {
        QI1 = WC;
    } else if (Fice <= 0.0f) {
        QW1 = WC;
    } else {
        QI1 = Fice * WC;
        QW1 = WC - QI1;
    }
    if (QW1 > 0.0f && Frain > 0.0f) {
        if (Frain >= 1.0f) {
            QR1 = QW1;
            QW1 = 0.0f;
        } else {
            QR1 = Frain * QW1;
            QW1 = QW1 - QR1;
        }
    }
    const float WV = Q1D / (1.0f - Q1D);
    float RHO = P1D / (NMMB_R_D * T1D * (1.0f + NMMB_P608 * Q1D));
    float RRHO = 1.0f / RHO;

    // rain reflectivity (1669-1690)
    if (QR1 > NMMB_EPSQ) {
        const float RQR = RHO * QR1;
        float N0r;
        int INDEXR;
        if (RQR <= RQR_DRmin) {
            N0r = fmaxf(CM_N0RMIN, CN0r_DMRmin * RQR);
            INDEXR = CM_MDRMIN;
        } else if (RQR >= RQR_DRmax) {
            N0r = CN0r_DMRmax * RQR;
            INDEXR = CM_MDRMAX;
        } else {
            N0r = CM_N0R0;
            INDEXR = (int)fmaxf(CM_XMRMIN, fminf(CN0r0 * powf(RQR, 0.25f), CM_XMRMAX));
        }
        const float DRmm = 1.0e-3f * (float)INDEXR;
        Zrain = 0.72f * N0r * DRmm * DRmm * DRmm * DRmm * DRmm * DRmm * DRmm;
    }

    // ice reflectivity (1693-1797)
    if (QI1 > NMMB_EPSQ) {
        const float QICE = QI1;
        RHO = P1D / (NMMB_R_D * T1D * (1.0f + NMMB_P608 * Q1D));
        RRHO = 1.0f / RHO;
        const float ESAT = 1000.0f * calmict_fpvs(T1D, tbpvs, c1xpvs, c2xpvs);
        const float QSIgrd = CM_RHGRD * EPSILON * ESAT / (P1D - ESAT);
        const float WVQW = WV + QW1;

        float FLARGE, FSMALL;
        if (TC >= 0.0f || WVQW < QSIgrd) {
            FLARGE = 1.0f;
        } else {
            FLARGE = 0.2f;
            if (TC >= -8.0f && TC <= -3.0f) FLARGE = 0.5f * FLARGE;
        }
        FSMALL = (1.0f - FLARGE) / FLARGE;
        const float XSIMASS = RRHO * massi[CM_MDIMIN - 50] * FSMALL;

        float DUM = CM_XMIMAX * expf(0.0536f * TC);
        int INDEXS = min(CM_MDIMAX, max(CM_MDIMIN, (int)DUM));
        float RimeF = fmaxf(1.0f, FS1D);
        float XLIMASS = RRHO * RimeF * massi[INDEXS - 50];
        float FLIMASS = XLIMASS / (XLIMASS + XSIMASS);
        float QLICE = FLIMASS * QICE;
        float NLICE = QLICE / XLIMASS;

        if (NLICE < CM_NLIMIN || NLICE > CM_NLIMAX) {       /* new_nlice, 1744-1791 */
            DUM = fmaxf(CM_NLIMIN, fminf(CM_NLIMAX, NLICE));
            const float XLI = RHO * (QICE / DUM - XSIMASS) / RimeF;
            if (XLI <= massi[CM_MDIMIN - 50]) {
                INDEXS = CM_MDIMIN;
            } else if (XLI <= massi[450 - 50]) {
                const float DLI = 9.5885e5f * powf(XLI, 0.42066f);
                INDEXS = min(CM_MDIMAX, max(CM_MDIMIN, (int)DLI));
            } else if (XLI <= massi[CM_MDIMAX - 50]) {
                const float DLI = 3.9751e6f * powf(XLI, 0.49870f);
                INDEXS = min(CM_MDIMAX, max(CM_MDIMIN, (int)DLI));
            } else {
                INDEXS = CM_MDIMAX;
                if (DUM >= CM_NLIMAX) {
                    RimeF = RHO * (QICE / CM_NLIMAX - XSIMASS) / massi[INDEXS - 50];
                }
            }
            XLIMASS = RRHO * RimeF * massi[INDEXS - 50];
            FLIMASS = XLIMASS / (XLIMASS + XSIMASS);
            QLICE = FLIMASS * QICE;
            NLICE = QLICE / XLIMASS;
        }
        QS1 = fminf(QI1, QLICE);
        QI1 = fmaxf(0.0f, QI1 - QS1);
        if (NLICE > 0.0f && QLICE > 0.0f) {
            Zice = CM_CICE_MASS * RHO * RHO * QLICE * QLICE / NLICE;
        }
    }

    // total reflectivity (1800-1804)
    const float Ztot = Zrain + Zice + Zconv;
    if (Ztot > CM_ZMIN) DBZ1 = 10.0f * log10f(Ztot);
    return DBZ1;
}

// One thread per (i, j, l). Fields are [LM, JM, IM] float32 (CONTRACT layout); CUREFL is [JM, IM].
extern "C" __global__ void fer_calmict_eval(
    const float* __restrict__ P, const float* __restrict__ T, const float* __restrict__ Q,
    const float* __restrict__ C, const float* __restrict__ FI, const float* __restrict__ FR,
    const float* __restrict__ FS, const float* __restrict__ CUREFL,
    const float* __restrict__ massi, const float* __restrict__ tbpvs,
    float c1xpvs, float c2xpvs,
    float RQR_DRmin, float RQR_DRmax, float CN0r0, float CN0r_DMRmin, float CN0r_DMRmax,
    float* __restrict__ DBZ, int IM, int JM, int LM)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t p3 = IDX3(i0, j0, l0, IM, JM);
    const size_t p2 = IDX2(i0, j0, IM);
    DBZ[p3] = nam_calmict(P[p3], T[p3], Q[p3], C[p3], FI[p3], FR[p3], FS[p3], CUREFL[p2],
                          massi, tbpvs, c1xpvs, c2xpvs,
                          RQR_DRmin, RQR_DRmax, CN0r0, CN0r_DMRmin, CN0r_DMRmax);
}
