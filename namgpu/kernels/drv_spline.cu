// nam-gpu unit drv-spline: one-dimensional cubic spline (table build, host-side init only).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module and routine header kept as the notice:
//   module module_control  (NMMB, nam.v4.2.11)
//   subroutine spline(jtb,nold,xold,yold,y2,nnew,xnew,ynew,p,q)
//   programer: z. janjic, yugoslav fed. hydromet. inst., beograd
// Ported to CUDA for nam-gpu from nmm/module_CONTROL.F90:2074-2188 (nam.v4.2.11, 83df4e13c1).
//
// Fortran -> CUDA, one thread per independent spline (batch index s). Every loop and every term of
// the routine is kept; the Fortran goto structure is mapped as:
//   [start]   dxl,dxr,dydxl,dydxr,rtdxc, p(1), q(1)                          (lines 2124-2134)
//   [100 loop] k = 3 .. nold-1 forward sweep of p(k-1), q(k-1)               (2136-2146; skipped if nold==3, 2130)
//   [200 loop] back substitution y2(k), k = nold-1 .. 2                     (2151-2157; do-while, k=noldm1 first)
//   [300 loop] k1 = 1..nnew: search kold, coefficients ak/bk/ck, evaluate ynew (2161-2186)
//              Fortran reuses ak/bk/ck when k==kold (label 550). Recomputing them from the same k gives
//              the same float operations, so the kernel recomputes; the reuse path is not needed.
// Fortran 1-based index k is stored at [k-1]; xold(k+1) is X[k], p(k-1) is P[k-2], q(k-2) is Q[k-3].
// Natural spline: the caller sets y2(1) = y2(nold) = 0 (inputs of the routine).
// Ranges kept: k2 search 2..nold, back substitution nold-1..2, sweep 3..nold-1.
// Out of range xnew(k1) (xk >= xold(nold)): ynew = yold(nold), as in the routine.
// NaN xnew(k1): the interval test is the exact negation of the Fortran .le. branch (line 80,
// "if(xold(k2).le.xk) go to 400"), i.e. !(xold(k2) <= xk). For NaN xk the Fortran falls through
// with kold = 1 and returns NaN from the interval-1 polynomial; the kernel does the same.
// nnew == 0: the Fortran k1 loop runs its body once and writes ynew(1) (out of bounds); the kernel
// writes nothing. The wrapper refuses nnew < 1, so the difference cannot be reached from Python.
//
// Scratch p, q are [nspl, nold]; Fortran only touches p(1:nold-2) and q(1:nold-2), the kernel
// writes indices up to nold-2 and never reads beyond them.
//
// Layout: [nspl, n] row-major (spline s, knot k at s*n + (k-1)). No host sync, no device malloc.

#include "nmmb_common.cuh"

extern "C" __global__ void drv_spline_batch(
    const int nspl, const int nold, const int nnew,
    const float* __restrict__ xold,   // [nspl, nold]   ascending knots (Fortran xold)
    const float* __restrict__ yold,   // [nspl, nold]   values at the knots (Fortran yold)
    float* __restrict__ y2,           // [nspl, nold]   in: y2(1), y2(nold) (natural: 0); out: y2(2)..y2(nold-1)
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

    // ---- setup (lines 2124-2134) ----
    const int noldm1 = nold - 1;
    float dxl = X[1] - X[0];                       // xold(2)-xold(1)
    float dxr = X[2] - X[1];                       // xold(3)-xold(2)
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
            dxr = X[k] - X[k - 1];                             // xold(k+1)-xold(k)
            dydxr = (Y[k] - Y[k - 1]) / dxr;
            const float dxc = dxl + dxr;
            const float den = 1.0f / (dxl * Q[k - 3] + dxc + dxc);   // q(k-2)
            P[k - 2] = den * (6.0f * (dydxr - dydxl) - dxl * P[k - 3]); // p(k-1) = ... p(k-2)
            Q[k - 2] = -den * dxr;                             // q(k-1)
        }
    }

    // ---- back substitution, label 200: k = nold-1 down to 2 (do-while: first pass always runs) ----
    {
        int k = noldm1;
        do {
            Y2[k - 1] = P[k - 2] + Q[k - 2] * Y2[k];          // y2(k) = p(k-1) + q(k-1)*y2(k+1)
            --k;
        } while (k > 1);
    }

    // ---- evaluation, label 300: k1 = 1 .. nnew ----
    for (int k1 = 1; k1 <= nnew; ++k1) {
        const float xk = XN[k1 - 1];
        int kold = -1;
        for (int k2 = 2; k2 <= nold; ++k2) {               // first xold(k2) > xk
            if (!(X[k2 - 1] <= xk)) { kold = k2 - 1; break; }  // exact negation of Fortran .le. (NaN-safe)
        }
        if (kold < 0) {                                    // xk >= xold(nold): label 600 path
            YN[k1 - 1] = Y[nold - 1];                      // ynew(k1) = yold(nold)
            continue;
        }
        // label 500 coefficients for interval k = kold (recomputed; identical to label 550 reuse)
        const int k = kold;
        const float y2k   = Y2[k - 1];                     // y2(k)
        const float y2kp1 = Y2[k];                         // y2(k+1)
        const float dx    = X[k] - X[k - 1];               // xold(k+1)-xold(k)
        const float rdx   = 1.0f / dx;
        const float ak = 0.1666667f * rdx * (y2kp1 - y2k);
        const float bk = 0.5f * y2k;
        const float ck = rdx * (Y[k] - Y[k - 1]) - 0.1666667f * dx * (y2kp1 + y2k + y2k);
        // label 550
        const float x  = xk - X[k - 1];                    // xk - xold(k)
        const float xsq = x * x;
        YN[k1 - 1] = ak * xsq * x + bk * xsq + ck * x + Y[k - 1];
    }
}
