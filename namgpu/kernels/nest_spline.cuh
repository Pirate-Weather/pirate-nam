// WOOF-NAM: the NMMB column spline and the underground hyperbola, shared by the concurrent-nest kernels
// (nest_bc.cu, nest_2way.cu).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Ported to CUDA for WOOF-NAM from nmm/module_NESTING.F90 (nam.v4.2.11, 83df4e13c1):
//   12905-13050 SUBROUTINE SPLINE(NOLD,XOLD,YOLD,Y2,Y2_K,NNEW,XNEW,YNEW)
//   13051-13110 SUBROUTINE HYPERBOLA(A): A = X1*Y1/(1-Y1), X1 = 10000, Y1 = 0.05 (KDBL)
// Indices below are the Fortran 1-based ones through the F1() macro, so the code reads line for line
// against the Fortran. Single precision as the Fortran REAL. The Fortran reuses the interval coefficients
// when two consecutive XNEW fall in the same interval (GO TO 550); recomputing them gives the same values.
#ifndef NEST_SPLINE_CUH
#define NEST_SPLINE_CUH

#ifndef NEST_LMAX
#define NEST_LMAX 64            // largest LM the nest kernels hold in registers/local memory (NAM: 60)
#endif

// HYPERBOLA (module_NESTING.F90:13051-13110), evaluated in double as the KDBL Fortran, used as REAL.
#define NEST_HYPER_A ((float)((10000.0 * 0.05) / (1.0 - 0.05)))

#define F1(a, k) (a)[(k) - 1]

// y2 holds NOLD values on entry (the caller zeroes it: natural end conditions, SEC_DERIV = 0).
__device__ __forceinline__ void nmmb_spline(int nold, const float* xold, const float* yold, float* y2,
                                            int nnew, const float* xnew, float* ynew)
{
    float p[NEST_LMAX + 1], q[NEST_LMAX + 1];
    const int noldm1 = nold - 1;
    float dxl = F1(xold, 2) - F1(xold, 1);
    float dxr = F1(xold, 3) - F1(xold, 2);
    float dydxl = (F1(yold, 2) - F1(yold, 1)) / dxl;
    float dydxr = (F1(yold, 3) - F1(yold, 2)) / dxr;
    float rtdxc = 0.5f / (dxl + dxr);
    F1(p, 1) = rtdxc * (6.0f * (dydxr - dydxl) - dxl * F1(y2, 1));
    F1(q, 1) = -rtdxc * dxr;
    if (nold != 3) {
        for (int k = 3; k < nold; ++k) {
            dxl = dxr;
            dydxl = dydxr;
            dxr = F1(xold, k + 1) - F1(xold, k);
            dydxr = (F1(yold, k + 1) - F1(yold, k)) / dxr;
            float dxc = dxl + dxr;
            float den = 1.0f / (dxl * F1(q, k - 2) + dxc + dxc);
            F1(p, k - 1) = den * (6.0f * (dydxr - dydxl) - dxl * F1(p, k - 2));
            F1(q, k - 1) = -den * dxr;
        }
    }
    for (int k = noldm1; k > 1; --k)
        F1(y2, k) = F1(p, k - 1) + F1(q, k - 1) * F1(y2, k + 1);
    for (int k1 = 1; k1 <= nnew; ++k1) {
        float xk = F1(xnew, k1);
        int kold = 0;
        for (int k2 = 2; k2 <= nold; ++k2) {
            if (F1(xold, k2) <= xk) continue;
            kold = k2 - 1;
            break;
        }
        if (kold == 0) { F1(ynew, k1) = F1(yold, nold); continue; }
        const int k = kold;
        float y2k = F1(y2, k), y2kp1 = F1(y2, k + 1);
        float dx = F1(xold, k + 1) - F1(xold, k);
        float rdx = 1.0f / dx;
        float ak = 0.1666667f * rdx * (y2kp1 - y2k);
        float bk = 0.5f * y2k;
        float ck = rdx * (F1(yold, k + 1) - F1(yold, k)) - 0.1666667f * dx * (y2kp1 + y2k + y2k);
        float x = xk - F1(xold, k);
        float xsq = x * x;
        F1(ynew, k1) = ak * xsq * x + bk * xsq + ck * x + F1(yold, k);
    }
}

#endif
