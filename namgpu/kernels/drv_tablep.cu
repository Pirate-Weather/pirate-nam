// nam-gpu unit drv-tablep: TABLEP, the saturation-pressure table ptbl(itb,jtb) used by the moist
// processes (convection, microphysics) to find pressure from saturation specific humidity and
// potential temperature.  Also fills the row constants qs0(jtb), sqs(jtb).  The module scalars
// rdth and rdq are set by the wrapper with the same float32 operations as the Fortran.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_CONTROL.F90:1868-1964 (nam.v4.2.11, 83df4e13c1).
// The Fortran CALL SPLINE (module_CONTROL.F90:2074-2183, unit drv-spline) is inlined as the
// __device__ function tablep_spline below, specialised to nold = nnew = ITB (the only call in
// TABLEP), so the kernel has no dependency on the drv-spline kernel and no host call.
//
// Parallelism: one thread per kth row (JTB = 601 threads).  Each row is a sequential chain
// (the running pressure p and theta th accumulate in float exactly as in Fortran, the eps fix
// reads the value just written, the spline is a tridiagonal forward/back sweep), so a row is
// the natural unit.  Init-only; the table is built once and never touched per step.
//
// Index mapping: Fortran ptbl(kp,kth) (itb x jtb) -> CuPy ptbl[kth, kp] (shape [JTB, ITB]),
// i.e. IDX2(kp-1, kth-1, ITB).  qs0(kth), sqs(kth) -> [JTB].
// Arrays inside the device code keep the Fortran 1-based indices (element 0 unused).
//
// Scalars from the Fortran (kfpt = single):
//   ph = 105000, thh = 350, thl = 200 (module parameters); pl = max(pt, 1000) (passed in);
//   dth = (thh-thl)/real(jtb-1), dp = (ph-pl)/real(itb-1); eps = 1e-10 (tablep local).
//   pq0, a2, a3, a4, cappa: NMMB_PQ0, NMMB_A2, NMMB_A3, NMMB_A4, NMMB_CAPPA from nmmb_common.cuh.

#include "nmmb_common.cuh"

#define TB_ITB 201    // module_CONTROL itb (convection tables, dimension 1)
#define TB_JTB 601    // module_CONTROL jtb (convection tables, dimension 2)
#define TB_PH   105000.0f
#define TB_THH  350.0f
#define TB_THL  200.0f
#define TB_EPS  1.0e-10f

// Direct port of module_CONTROL.F90 subroutine spline (lines 2074-2183) for nold = nnew = ITB.
// Arrays are 1-based: xold[1..nold], yold[1..nold], y2[1..nold] (y2[1], y2[nold] are the natural
// spline end conditions, set to 0 by the caller), xnew[1..nnew], ynew[1..nnew], work p[1..nold-2],
// q[1..nold-2].  Every statement of the Fortran is mirrored; no term is dropped.
__device__ void tablep_spline(const float* xold, const float* yold, float* y2,
                              const float* xnew, float* ynew, float* p, float* q)
{
    const int nold = TB_ITB;
    const int nnew = TB_ITB;
    const int noldm1 = nold - 1;

    float dxl = xold[2] - xold[1];
    float dxr = xold[3] - xold[2];
    float dydxl = (yold[2] - yold[1]) / dxl;
    float dydxr = (yold[3] - yold[2]) / dxr;
    float rtdxc = 0.5f / (dxl + dxr);

    p[1] = rtdxc * (6.0f * (dydxr - dydxl) - dxl * y2[1]);
    q[1] = -rtdxc * dxr;

    // Fortran: if(nold.eq.3) go to 700 skips the forward sweep.  Mirrored as a guard; nold is the
    // compile-time constant ITB = 201, so the branch folds away, but the statement is kept.
    if (nold != 3) {
        // Fortran label 100 loop: k = 3 .. nold-1 (k incremented, continue while k < nold).
        for (int k = 3; k < nold; ++k) {
            dxl = dxr;
            dydxl = dydxr;
            dxr = xold[k + 1] - xold[k];
            dydxr = (yold[k + 1] - yold[k]) / dxr;
            float dxc = dxl + dxr;
            float den = 1.0f / (dxl * q[k - 2] + dxc + dxc);
            p[k - 1] = den * (6.0f * (dydxr - dydxl) - dxl * p[k - 2]);
            q[k - 1] = -den * dxr;
        }
    }

    // Fortran label 200: back substitution, k = noldm1 .. 2.
    for (int k = noldm1; k > 1; --k) {
        y2[k] = p[k - 1] + q[k - 1] * y2[k + 1];
    }

    // Fortran label 300 loop over the new abscissae.  k carries over between k1 (as in Fortran):
    // the coefficients ak, bk, ck are recomputed only when the interval changes.
    int k = 1;
    int kold = 0;
    float ak = 0.0f, bk = 0.0f, ck = 0.0f;
    for (int k1 = 1; k1 <= nnew; ++k1) {
        float xk = xnew[k1];
        bool found = false;
        for (int k2 = 2; k2 <= nold; ++k2) {
            if (xold[k2] <= xk) continue;
            kold = k2 - 1;
            found = true;
            break;
        }
        if (!found) {
            ynew[k1] = yold[nold];
            continue;
        }
        // Fortran 450: if (k1 == 1) goto 500; if (k == kold) goto 550; 500: k = kold; coefficients.
        if (k1 == 1 || k != kold) {
            k = kold;
            float y2k = y2[k];
            float y2kp1 = y2[k + 1];
            float dx = xold[k + 1] - xold[k];
            float rdx = 1.0f / dx;
            ak = 0.1666667f * rdx * (y2kp1 - y2k);
            bk = 0.5f * y2k;
            ck = rdx * (yold[k + 1] - yold[k]) - 0.1666667f * dx * (y2kp1 + y2k + y2k);
        }
        // Fortran 550:
        float x = xk - xold[k];
        float xsq = x * x;
        ynew[k1] = ak * xsq * x + bk * xsq + ck * x + yold[k];
    }
}

extern "C" __global__ void drv_tablep(float* __restrict__ ptbl,   // [JTB, ITB]  (Fortran ptbl(itb,jtb))
                                      float* __restrict__ qs0,    // [JTB]
                                      float* __restrict__ sqs,    // [JTB]
                                      const float pl)             // max(pt, 1000)
{
    const int kth = blockIdx.x * blockDim.x + threadIdx.x + 1;   // 1-based Fortran kth
    if (kth > TB_JTB) return;

    const int itb = TB_ITB;
    const int jtb = TB_JTB;

    // Single-precision locals, 1-based.
    float qsold[TB_ITB + 1], pold[TB_ITB + 1], y2p[TB_ITB + 1];
    float qsnew[TB_ITB + 1], pnew[TB_ITB + 1], app[TB_ITB + 1], aqp[TB_ITB + 1];

    const float dth = (TB_THH - TB_THL) / (float)(jtb - 1);
    const float dp = (TB_PH - pl) / (float)(itb - 1);

    // Fortran: th = thl - dth; do kth: th = th + dth.  Accumulated the same way (float).
    float th = TB_THL - dth;
    for (int kk = 1; kk <= kth; ++kk) th = th + dth;

    // Fortran: p = pl - dp; do kp: p = p + dp; ape = (100000/p)**cappa; qsold, pold.
    float p = pl - dp;
    for (int kp = 1; kp <= itb; ++kp) {
        p = p + dp;
        float ape = powf(100000.0f / p, NMMB_CAPPA);
        qsold[kp] = NMMB_PQ0 / p * expf(NMMB_A2 * (th - NMMB_A3 * ape) / (th - NMMB_A4 * ape));
        pold[kp] = p;
    }

    float qs0k = qsold[1];
    float sqsk = qsold[itb] - qsold[1];
    qsold[1] = 0.0f;
    qsold[itb] = 1.0f;

    // Fortran: do kp = 2, itb-1: qsold = (qsold - qs0k)/sqsk; the 32-bit eps fix uses the
    // already-adjusted qsold(kp-1).
    for (int kp = 2; kp <= itb - 1; ++kp) {
        qsold[kp] = (qsold[kp] - qs0k) / sqsk;
        if ((qsold[kp] - qsold[kp - 1]) < TB_EPS) {
            qsold[kp] = qsold[kp - 1] + TB_EPS;
        }
    }

    qs0[kth - 1] = qs0k;
    sqs[kth - 1] = sqsk;

    // New (regular) saturation-humidity abscissae.  dqs = 1/real(itb-1); rdq is a module scalar
    // set by the wrapper (same value).
    qsnew[1] = 0.0f;
    qsnew[itb] = 1.0f;
    const float dqs = 1.0f / (float)(itb - 1);
    for (int kp = 2; kp <= itb - 1; ++kp) {
        qsnew[kp] = qsnew[kp - 1] + dqs;
    }

    // Natural spline end conditions (y2(1) = y2(itb) = 0); interior is written by the spline
    // before it is read.
    for (int kp = 1; kp <= itb; ++kp) y2p[kp] = 0.0f;
    y2p[1] = 0.0f;
    y2p[itb] = 0.0f;

    tablep_spline(qsold, pold, y2p, qsnew, pnew, app, aqp);

    // ptbl(kp, kth) = pnew(kp): CuPy [kth, kp].
    for (int kp = 1; kp <= itb; ++kp) {
        ptbl[IDX2(kp - 1, kth - 1, itb)] = pnew[kp];
    }
}
