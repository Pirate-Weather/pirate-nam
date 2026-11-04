// WOOF-NAM unit phys-bmj-spline: one-dimensional cubic spline for the BMJ convection tables (init only).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_cu_bmj  (NMMB, nam.v4.2.11)
//   subroutine spline(jtbx,nold,xold,yold,y2,nnew,xnew,ynew,p,q)
//   programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for WOOF-NAM from phys/module_CU_BMJ.F90:2429-2547 (nam.v4.2.11, 83df4e13c1).
//
// Fortran -> CUDA, one thread per independent spline (batch index s). The Fortran goto structure:
//   [start]    dxl, dxr, dydxl, dydxr, rtdxc, p(1), q(1)                    (2445-2455)
//   [100 loop] k = 3 .. nold-1 forward sweep of p(k-1), q(k-1)              (2457-2468; skipped if nold==3, 2452)
//   [200 loop] back substitution y2(k), k = nold-1 down to 2 (do-while)     (2473-2479)
//   [300 loop] k1 = 1..nnew: search kold (2..nold), coefficients ak/bk/ck, evaluate ynew (2483-2508)
//              The Fortran reuses ak/bk/ck when k==kold (label 550). They are a pure function of
//              (kold, y2, xold, yold), so the kernel recomputes them; same float operations, same result.
// Fortran index k is stored at [k-1]. p(k-1) is P[k-2], q(k-1) is Q[k-2], p(k-2) is P[k-3], q(k-2) is Q[k-3].
// Natural spline: the caller sets y2(1) = y2(nold) = 0 (inputs of the routine); the kernel never writes them.
// Ranges kept: k2 search 2..nold, back substitution nold-1..2, sweep 3..nold-1, evaluation k1 = 1..nnew.
// Out of range xnew(k1) (no xold(k2) > xk): ynew(k1) = yold(nold), as in the routine.
// Scratch p, q are [nspl, nold]; the routine reads and writes only indices 1..nold-2 of them.
//
// Precision: kfpt = single, so float everywhere; constant .1666667 is kept as written in the Fortran.
// Fast numerics (fmad on): NVRTC default contraction, no fast-math division.
//
// Layout: [nspl, n] row-major (spline s, knot k at s*n + (k-1)). One launch, on ctx.stream, no host
// sync, no device malloc, no printf.

#include "nmmb_common.cuh"

extern "C" __global__ void phys_bmj_spline_batch(
    const int nspl, const int nold, const int nnew,
    const float* __restrict__ xold,   // [nspl, nold]   ascending knots (Fortran xold)
    const float* __restrict__ yold,   // [nspl, nold]   values at the knots (Fortran yold)
    float* __restrict__ y2,           // [nspl, nold]   in: y2(1), y2(nold) (natural: 0); out: all knots
    const float* __restrict__ xnew,   // [nspl, nnew]   evaluation points (Fortran xnew)
    float* __restrict__ ynew,         // [nspl, nnew]   out (Fortran ynew)
    float* __restrict__ p,            // [nspl, nold]   scratch (Fortran p, uses 1..nold-2)
    float* __restrict__ q)            // [nspl, nold]   scratch (Fortran q, uses 1..nold-2)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nspl) return;

    const float* X  = xold + (size_t)s * nold;
    const float* Y  = yold + (size_t)s * nold;
    float* Y2       = y2   + (size_t)s * nold;
    float* P        = p    + (size_t)s * nold;
    float* Q        = q    + (size_t)s * nold;
    const float* XN = xnew + (size_t)s * nnew;
    float* YN       = ynew + (size_t)s * nnew;

    // ---- setup (Fortran 2445-2455) ----
    const int noldm1 = nold - 1;
    float dxl = X[1] - X[0];                              // xold(2) - xold(1)
    float dxr = X[2] - X[1];                              // xold(3) - xold(2)
    float dydxl = (Y[1] - Y[0]) / dxl;
    float dydxr = (Y[2] - Y[1]) / dxr;
    float rtdxc = 0.5f / (dxl + dxr);

    P[0] = rtdxc * (6.0f * (dydxr - dydxl) - dxl * Y2[0]);   // p(1)
    Q[0] = -rtdxc * dxr;                                      // q(1)

    // ---- forward sweep, label 100: k = 3 .. nold-1 (skipped for nold == 3) ----
    if (nold != 3) {
        for (int k = 3; k < nold; ++k) {
            dxl = dxr;
            dydxl = dydxr;
            dxr = X[k] - X[k - 1];                                  // xold(k+1) - xold(k)
            dydxr = (Y[k] - Y[k - 1]) / dxr;
            const float dxc = dxl + dxr;
            const float den = 1.0f / (dxl * Q[k - 3] + dxc + dxc);  // q(k-2)
            P[k - 2] = den * (6.0f * (dydxr - dydxl) - dxl * P[k - 3]);  // p(k-1) = den*(...-dxl*p(k-2))
            Q[k - 2] = -den * dxr;                                        // q(k-1)
        }
    }

    // ---- back substitution, label 200: k = nold-1 down to 2 (do-while: first pass always runs) ----
    {
        int k = noldm1;
        do {
            Y2[k - 1] = P[k - 2] + Q[k - 2] * Y2[k];         // y2(k) = p(k-1) + q(k-1)*y2(k+1)
            --k;
        } while (k > 1);
    }

    // ---- evaluation, label 300: k1 = 1 .. nnew ----
    for (int k1 = 1; k1 <= nnew; ++k1) {
        const float xk = XN[k1 - 1];
        int kold = -1;
        for (int k2 = 2; k2 <= nold; ++k2) {                 // first xold(k2) > xk
            if (X[k2 - 1] > xk) { kold = k2 - 1; break; }
        }
        if (kold < 0) {                                      // label 600 path: xk >= xold(nold)
            YN[k1 - 1] = Y[nold - 1];                        // ynew(k1) = yold(nold)
            continue;
        }
        // label 500 coefficients for the interval k = kold (recomputed; see header)
        const int k = kold;
        const float y2k   = Y2[k - 1];                       // y2(k)
        const float y2kp1 = Y2[k];                           // y2(k+1)
        const float dx    = X[k] - X[k - 1];                 // xold(k+1) - xold(k)
        const float rdx   = 1.0f / dx;
        const float ak = 0.1666667f * rdx * (y2kp1 - y2k);
        const float bk = 0.5f * y2k;
        const float ck = rdx * (Y[k] - Y[k - 1]) - 0.1666667f * dx * (y2kp1 + y2k + y2k);
        // label 550
        const float x   = xk - X[k - 1];                     // xk - xold(k)
        const float xsq = x * x;
        YN[k1 - 1] = ak * xsq * x + bk * xsq + ck * x + Y[k - 1];
    }
}
