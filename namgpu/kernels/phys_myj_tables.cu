// nam-gpu unit phys-myj-tables: HERFTBL, HERF, TABLEPT, TABLETT and SPLINE, the MYJ PBL look-up
// tables built once at MYJPBL_INIT (module_BL_MYJPBL.F90:1661-2029, nam.v4.2.11, 83df4e13c1).
// Consumers: MYJPBL (module_BL_MYJPBL.F90:180-753) reads HERF, PTBL and TTBL on the step path.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The module header is kept here as the notice:
//   MODULE MODULE_BL_MYJPBL (Mellor-Yamada-Janjic PBL, NMMB, nam.v4.2.11). The look-up table
//   routines are by Z. Janjic (SPLINE is written for a small scalar machine).
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:1661-2029 (nam.v4.2.11, 83df4e13c1).
//
// Kernels (all extern "C"):
//   phys_myj_herftbl  HERFTBL: half erf table herff[KERFM] (single thread: sequential prefix sum,
//                     301 values; init only).
//   phys_myj_tablept  TABLEPT: ptbl[JTBL, ITBL] saturation pressure table, one thread per kth row
//                     (JTBL = 1201 threads): TH accumulation, P accumulation, eps fix, one SPLINE.
//                     Also qs0[kth], sqs[kth].
//   phys_myj_tablett  TABLETT: ttbl[ITBL, JTBL] temperature table, one thread per kp column
//                     (ITBL = 401 threads): TH accumulation, eps fix, one SPLINE. Also the0[kp], sthe[kp].
// Device functions: myj_spline (SPLINE), myj_herf (HERF).
//
// Fortran -> CUDA map (every statement of the five routines):
//   HERFTBL: DXE, DXH, XGMIN, RDXE; GAUSS(K)=EXP(-X*X*0.5), X=(K-1)*DXE+XGMIN;
//            HERFF(1)=0; HERFF(K)=((GAUSS(K)+GAUSS(K+1))*DXH)*RDEN+HERFF(K-1); HERFF=0.5-HERFF.
//   HERF:    AK=(X-XEMIN)*RDXE; K=INT(AK) clamped to [0, KERFM2];
//            HERF=(HERFF(K+2)-HERFF(K+1))*(AK-REAL(K))+HERFF(K+1)  (1-based, mapped to 0-based).
//   TABLEPT: DTH, DP, RDTH=1/DTH; TH=THL-DTH then TH=TH+DTH per kth; P=PL-DP then P=P+DP per kp;
//            RXNER=(100000/P)**CAPPA; QSOLD=PQ0/P*EXP(A2*(TH-A3*RXNER)/(TH-A4*RXNER));
//            QS0K, SQSK, QSOLD(1)=0, QSOLD(ITBL)=1, eps fix on QSOLD(2..ITBL-1);
//            QS0(kth)=QS0K, SQS(kth)=SQSK; DQS=1/(ITBL-1), RDQ=1/DQS, QSNEW accumulated;
//            Y2P(1)=Y2P(ITBL)=0; CALL SPLINE(JTBL,ITBL,QSOLD,POLD,Y2P,ITBL,QSNEW,PNEW,APP,AQP);
//            PTBL(KP,KTH)=PNEW(KP).
//   TABLETT: DTH, DP, DTHE, RDP=1/DP, RDTHE=1/DTHE; TH=THL-DTH per kp, TH accumulated over kth;
//            P accumulated over kp; RXNER; QS=PQ0/P*EXP(...); TOLD=TH/RXNER;
//            THEOLD=TH*EXP(ELIWV*QS/(CP*TOLD)); THE0K, STHEK, end values, eps fix on THEOLD;
//            THE0(KP)=THE0K, STHE(KP)=STHEK; THENEW accumulated with DTHE;
//            CALL SPLINE(JTBL,JTBL,THEOLD,TOLD,Y2T,JTBL,THENEW,TNEW,APT,AQT); TTBL(KTH,KP)=TNEW(KTH).
//   SPLINE:  natural cubic spline: forward sweep P/Q for k = 3..NOLD-1, back substitution for
//            Y2(k), k = NOLD-1..2; per new point the interval search (first XOLD(K2) > XK gives
//            KOLD=K2-1, otherwise YNEW=YOLD(NOLD)) and the cubic AK*XSQ*X+BK*XSQ+CK*X+YOLD(K).
//            The Fortran caches AK/BK/CK across consecutive points with the same K (label 550);
//            that cache holds the same values, so each point recomputes them here.
//
// Scalars written to scal[] (module variables of the Fortran, read by consumers):
//   scal[0] RDXE (HERFTBL), scal[1] RDTH, scal[2] RDQ (TABLEPT), scal[3] RDP, scal[4] RDTHE (TABLETT).
//
// Layout: PTBL(ITBL,JTBL) -> ptbl[kth, kp], shape [JTBL, ITBL], flat IDX2(kp-1, kth-1, ITBL).
//         TTBL(JTBL,ITBL) -> ttbl[kp, kth], shape [ITBL, JTBL], flat IDX2(kth-1, kp-1, JTBL).
//         QS0, SQS, HERFF, THE0, STHE are 1-D. Local arrays are 1-based (element 0 unused).
//
// Precision: kfpt = single; float arithmetic throughout. Fast numerics (fmad contraction on).
// Parallelism: init only. herftbl is one thread by design; tablept and tablett are one thread
// per table row/column.

#include "nmmb_common.cuh"

#define MYJ_ITBL   401
#define MYJ_JTBL   1201
#define MYJ_KERFM  301
#define MYJ_KERFM2 299            // KERFM-2
#define MYJ_PL     2500.0f
#define MYJ_PH     105000.0f
#define MYJ_THL    210.0f
#define MYJ_THH    365.0f
#define MYJ_XEMIN  0.0f
#define MYJ_XEMAX  3.0f
#define MYJ_PI     3.141592653589793f
#define MYJ_EPS    1.0e-10f

// Natural cubic SPLINE. Arrays are 1-based: xold[1..nold], yold[1..nold], y2[1..nold] (y2[1] and
// y2[nold] set to 0 by the caller), xnew[1..nnew], ynew[1..nnew], scratch p, q [1..nold-2].
__device__ void myj_spline(int nold, int nnew,
                           const float* xold, const float* yold, float* y2,
                           const float* xnew, float* ynew, float* p, float* q)
{
    const int noldm1 = nold - 1;

    float dxl = xold[2] - xold[1];
    float dxr = xold[3] - xold[2];
    float dydxl = (yold[2] - yold[1]) / dxl;
    float dydxr = (yold[3] - yold[2]) / dxr;
    float rtdxc = 0.5f / (dxl + dxr);

    p[1] = rtdxc * (6.0f * (dydxr - dydxl) - dxl * y2[1]);
    q[1] = -rtdxc * dxr;

    // Forward sweep (label 100): K = 3 .. NOLD-1 (empty for NOLD = 3, label 700).
    for (int k = 3; k < nold; ++k) {
        dxl = dxr;
        dydxl = dydxr;
        dxr = xold[k + 1] - xold[k];
        dydxr = (yold[k + 1] - yold[k]) / dxr;
        const float dxc = dxl + dxr;
        const float den = 1.0f / (dxl * q[k - 2] + dxc + dxc);
        p[k - 1] = den * (6.0f * (dydxr - dydxl) - dxl * p[k - 2]);
        q[k - 1] = -den * dxr;
    }

    // Back substitution (label 200): K = NOLD-1 down to 2.
    for (int k = noldm1; k > 1; --k) {
        y2[k] = p[k - 1] + q[k - 1] * y2[k + 1];
    }

    // Evaluation (label 300).
    for (int k1 = 1; k1 <= nnew; ++k1) {
        const float xk = xnew[k1];
        int kold = 0;
        for (int k2 = 2; k2 <= nold; ++k2) {
            if (!(xold[k2] <= xk)) { kold = k2 - 1; break; }
        }
        if (kold == 0) {                         // label 600 path: beyond the last knot
            ynew[k1] = yold[nold];
            continue;
        }
        const int k = kold;
        const float y2k   = y2[k];
        const float y2kp1 = y2[k + 1];
        const float dx    = xold[k + 1] - xold[k];
        const float rdx   = 1.0f / dx;
        const float ak = 0.1666667f * rdx * (y2kp1 - y2k);
        const float bk = 0.5f * y2k;
        const float ck = rdx * (yold[k + 1] - yold[k]) - 0.1666667f * dx * (y2kp1 + y2k + y2k);

        const float x   = xk - xold[k];
        const float xsq = x * x;
        ynew[k1] = ak * xsq * x + bk * xsq + ck * x + yold[k];
    }
}

// HERF (Fortran FUNCTION HERF): half-table erf, linear interpolation. herff is 0-based
// (herff[i] = HERFF(i+1)); rdxe = RDXE from HERFTBL.
__device__ __forceinline__ float myj_herf(float x, const float* herff, float rdxe)
{
    const float ak = (x - MYJ_XEMIN) * rdxe;
    int k = (int)ak;                       // Fortran INT(): truncation toward zero
    k = max(k, 0);
    k = min(k, MYJ_KERFM2);
    // HERF=(HERFF(K+2)-HERFF(K+1))*(AK-REAL(K))+HERFF(K+1), 1-based mapped to 0-based.
    return (herff[k + 1] - herff[k]) * (ak - (float)k) + herff[k];
}

// HERFTBL: herff[KERFM], scal[0] = RDXE. One thread (sequential prefix sum; init only).
extern "C" __global__ void phys_myj_herftbl(float* __restrict__ herff, float* __restrict__ scal)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float dxe   = (MYJ_XEMAX - MYJ_XEMIN) / (float)(MYJ_KERFM - 1);
    const float dxh   = dxe * 0.5f;
    const float xgmin = MYJ_XEMIN - dxh;
    const float rdxe  = 1.0f / dxe;

    float gauss[MYJ_KERFM + 2];            // 1-based: gauss[K] = GAUSS(K), K = 1..KERFM+1
    for (int k = 1; k <= MYJ_KERFM + 1; ++k) {
        const float x = (float)(k - 1) * dxe + xgmin;
        gauss[k] = expf(-x * x * 0.5f);
    }

    const float rden = 1.0f / sqrtf(MYJ_PI + MYJ_PI);
    herff[0] = 0.0f;                       // HERFF(1)
    for (int k = 2; k <= MYJ_KERFM; ++k) {
        herff[k - 1] = ((gauss[k] + gauss[k + 1]) * dxh) * rden + herff[k - 2];
    }
    for (int k = 1; k <= MYJ_KERFM; ++k) {
        herff[k - 1] = 0.5f - herff[k - 1];
    }
    scal[0] = rdxe;
}

// TABLEPT: one thread per kth row (kth = 1..JTBL). ptbl[kth, kp] at IDX2(kp-1, kth-1, ITBL),
// qs0[kth-1], sqs[kth-1]. Thread kth = 1 also writes scal[1] = RDTH and scal[2] = RDQ.
extern "C" __global__ void phys_myj_tablept(float* __restrict__ ptbl,
                                            float* __restrict__ qs0,
                                            float* __restrict__ sqs,
                                            float* __restrict__ scal)
{
    const int kth = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (kth > MYJ_JTBL) return;

    const float dth = (MYJ_THH - MYJ_THL) / (float)(MYJ_JTBL - 1);
    const float dp  = (MYJ_PH - MYJ_PL) / (float)(MYJ_ITBL - 1);
    const float dqs = 1.0f / (float)(MYJ_ITBL - 1);

    // TH: TH=THL-DTH, then TH=TH+DTH once per KTH, so kth additions for this row.
    float th = MYJ_THL - dth;
    for (int i = 1; i <= kth; ++i) th = th + dth;

    float qsold[MYJ_ITBL + 1], pold[MYJ_ITBL + 1], qsnew[MYJ_ITBL + 1];
    float y2p[MYJ_ITBL + 1], app[MYJ_ITBL + 1], aqp[MYJ_ITBL + 1], pnew[MYJ_ITBL + 1];

    // P=PL-DP, then P=P+DP per KP.
    float p = MYJ_PL - dp;
    for (int kp = 1; kp <= MYJ_ITBL; ++kp) {
        p = p + dp;
        const float rxner = powf(100000.0f / p, NMMB_CAPPA);
        qsold[kp] = NMMB_PQ0 / p * expf(NMMB_A2 * (th - NMMB_A3 * rxner) / (th - NMMB_A4 * rxner));
        pold[kp]  = p;
    }

    const float qs0k = qsold[1];
    const float sqsk = qsold[MYJ_ITBL] - qsold[1];
    qsold[1] = 0.0f;
    qsold[MYJ_ITBL] = 1.0f;
    for (int kp = 2; kp <= MYJ_ITBL - 1; ++kp) {
        qsold[kp] = (qsold[kp] - qs0k) / sqsk;
        if ((qsold[kp] - qsold[kp - 1]) < MYJ_EPS) {       // 32-bit precision fix
            qsold[kp] = qsold[kp - 1] + MYJ_EPS;
        }
    }

    qs0[kth - 1] = qs0k;
    sqs[kth - 1] = sqsk;

    qsnew[1] = 0.0f;
    qsnew[MYJ_ITBL] = 1.0f;
    for (int kp = 2; kp <= MYJ_ITBL - 1; ++kp) {
        qsnew[kp] = qsnew[kp - 1] + dqs;
    }

    y2p[1] = 0.0f;
    y2p[MYJ_ITBL] = 0.0f;
    myj_spline(MYJ_ITBL, MYJ_ITBL, qsold, pold, y2p, qsnew, pnew, app, aqp);

    for (int kp = 1; kp <= MYJ_ITBL; ++kp) {
        ptbl[IDX2(kp - 1, kth - 1, MYJ_ITBL)] = pnew[kp];
    }

    if (kth == 1) {
        scal[1] = 1.0f / dth;              // RDTH
        scal[2] = 1.0f / dqs;              // RDQ
    }
}

// TABLETT: one thread per kp column (kp = 1..ITBL). ttbl[kp, kth] at IDX2(kth-1, kp-1, JTBL),
// the0[kp-1], sthe[kp-1]. Thread kp = 1 also writes scal[3] = RDP and scal[4] = RDTHE.
extern "C" __global__ void phys_myj_tablett(float* __restrict__ ttbl,
                                            float* __restrict__ the0,
                                            float* __restrict__ sthe,
                                            float* __restrict__ scal)
{
    const int kp = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (kp > MYJ_ITBL) return;

    const float dth  = (MYJ_THH - MYJ_THL) / (float)(MYJ_JTBL - 1);
    const float dp   = (MYJ_PH - MYJ_PL) / (float)(MYJ_ITBL - 1);
    const float dthe = 1.0f / (float)(MYJ_JTBL - 1);

    // P=PL-DP, then P=P+DP per KP: kp additions for this column.
    float p = MYJ_PL - dp;
    for (int i = 1; i <= kp; ++i) p = p + dp;

    float told[MYJ_JTBL + 1], theold[MYJ_JTBL + 1], thenew[MYJ_JTBL + 1];
    float tnew[MYJ_JTBL + 1], y2t[MYJ_JTBL + 1], apt[MYJ_JTBL + 1], aqt[MYJ_JTBL + 1];

    // TH=THL-DTH once per KP, then TH=TH+DTH per KTH.
    float th = MYJ_THL - dth;
    for (int kth = 1; kth <= MYJ_JTBL; ++kth) {
        th = th + dth;
        const float rxner = powf(100000.0f / p, NMMB_CAPPA);
        const float qs = NMMB_PQ0 / p * expf(NMMB_A2 * (th - NMMB_A3 * rxner) / (th - NMMB_A4 * rxner));
        told[kth]   = th / rxner;
        theold[kth] = th * expf(NMMB_ELIWV * qs / (NMMB_CP * told[kth]));
    }

    const float the0k = theold[1];
    const float sthek = theold[MYJ_JTBL] - theold[1];
    theold[1] = 0.0f;
    theold[MYJ_JTBL] = 1.0f;
    for (int kth = 2; kth <= MYJ_JTBL - 1; ++kth) {
        theold[kth] = (theold[kth] - the0k) / sthek;
        if ((theold[kth] - theold[kth - 1]) < MYJ_EPS) {   // 32-bit precision fix
            theold[kth] = theold[kth - 1] + MYJ_EPS;
        }
    }

    the0[kp - 1] = the0k;
    sthe[kp - 1] = sthek;

    thenew[1] = 0.0f;
    thenew[MYJ_JTBL] = 1.0f;
    for (int kth = 2; kth <= MYJ_JTBL - 1; ++kth) {
        thenew[kth] = thenew[kth - 1] + dthe;
    }

    y2t[1] = 0.0f;
    y2t[MYJ_JTBL] = 0.0f;
    myj_spline(MYJ_JTBL, MYJ_JTBL, theold, told, y2t, thenew, tnew, apt, aqt);

    for (int kth = 1; kth <= MYJ_JTBL; ++kth) {
        ttbl[IDX2(kth - 1, kp - 1, MYJ_JTBL)] = tnew[kth];
    }

    if (kp == 1) {
        scal[3] = 1.0f / dp;               // RDP
        scal[4] = 1.0f / dthe;             // RDTHE
    }
}
