// drv_dfi_dolph.cu -- WOOF-NAM unit drv-dfi-dolph: DOLPH, the Dolph-Chebyshev window of the
// digital (TDFI) filter weights, with its helper CHEBY.
//
// NOTICE: NOAA/NCEP NAM source (nam.v4.2.11, MatthewPyle-NOAA/NAM @ 83df4e13c1), a US government
// work. The Fortran routine carries no author block; its references are kept verbatim here:
//   calculation of dolph-chebyshev window, Antoniou, Andreas, 1993: Digital Filters: Analysis,
//   Design and Applications, McGraw-Hill. CHEBY: Numerical Recipes, p. 184, Chebyshev recurrence.
// No licence file accompanies the NAM source.
//
// Ported to CUDA for WOOF-NAM from nmm/module_DIGITAL_FILTER_NMM.F90:929-1039 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE DOLPH(DELTAT, TAUS, M, WINDOW): WINDOW(0:2M) (float32). N = 2M+1.
// Launch: one thread per NT = 0..M (the Fortran DO NT=0,M loop). Each thread runs the whole
// I=1..M sum of its NT in Fortran statement order, then writes its weight to both symmetric
// positions W2(M+NT) and W2(M-NT) (the Fortran fill loop, folded into the same thread: the two
// stores are the only cross-NT data flow and they never collide except at NT=0, where both
// stores carry the same value). The Fortran FILL/SUMW loops are the same stores or a plain
// copy (WINDOW(NT) = W2(NT)), so no separate launch is needed.
//
// SUMW (Fortran lines 1004-1009, the sum of the window) feeds only the commented-out WRITE and
// is not computed on the device. DB and R/RR diagnostics (WRITE(0,...) lines) are host prints
// and are dropped. TIME(NT) is only used by the dropped prints; not ported.
//
// CHEBY(T, N, X) (Fortran lines 1013-1039) computes T(0:N); DOLPH only reads T(NM1), so
// dolph_cheby_last() returns that last value by the same recurrence without storing the whole
// array. Operation order is the Fortran order: T(NN) = 2*X*T(NN-1) - T(NN-2), evaluated as
// ((2*X)*T(NN-1)) - T(NN-2).
#include "nmmb_common.cuh"

// Fortran PI = 4*ATAN(1.D0) is evaluated in double and stored as REAL: the float nearest pi.
#define DOLPH_PI 3.14159265358979323846f

// CHEBY recurrence returning T(n) for argument x (T(0)=1, T(1)=x, T(nn)=2x T(nn-1) - T(nn-2)).
__device__ static float dolph_cheby_last(int n, float x)
{
    if (n == 0) return 1.0f;
    if (n == 1) return x;
    float tm2 = 1.0f;   // T(nn-2)
    float tm1 = x;      // T(nn-1)
    float t = x;
    for (int nn = 2; nn <= n; ++nn) {
        t = 2.0f * x * tm1 - tm2;
        tm2 = tm1;
        tm1 = t;
    }
    return t;
}

// window: device pointer to 2m+1 floats (Fortran WINDOW(0:2*M)); deltat, taus, m as the Fortran
// arguments (m is NSTEP = NDFISTEP in the caller).
extern "C" __global__ void drv_dfi_dolph(float* __restrict__ window,
                                         const float deltat, const float taus, const int m)
{
    const int nt = blockIdx.x * blockDim.x + threadIdx.x;
    if (nt > m) return;

    const float pi = DOLPH_PI;
    const int n = 2 * m + 1;          // N = 2*M+1
    const int nm1 = n - 1;            // NM1 = N-1

    const float thetas = 2.0f * pi * fabsf(deltat / taus);   // THETAS = 2*PI*ABS(DELTAT/TAUS)
    const float x0 = 1.0f / cosf(thetas / 2.0f);             // X0 = 1/COS(THETAS/2)
    const float term1 = powf(x0 + sqrtf(x0 * x0 - 1.0f), (float)(n - 1));
    const float term2 = powf(x0 - sqrtf(x0 * x0 - 1.0f), (float)(n - 1));
    const float rr = 0.5f * (term1 + term2);                  // RR = 0.5*(TERM1+TERM2)
    const float r = 1.0f / rr;                                // R = 1/RR
    // DB = 20*LOG10(R) is a diagnostic only (dropped).

    float sum = 1.0f;                                         // SUM = 1
    for (int i = 1; i <= m; ++i) {
        const float arg = x0 * cosf((float)i * pi / (float)n); // ARG = X0*COS(I*PI/N)
        const float tnm1 = dolph_cheby_last(nm1, arg);         // CALL CHEBY(T,NM1,ARG); TERM1=T(NM1)
        const float term2c = cosf((float)(2 * nt) * pi * (float)i / (float)n); // COS(2*NT*PI*I/N)
        sum = sum + r * 2.0f * tnm1 * term2c;                  // SUM = SUM + R*2*TERM1*TERM2
    }
    const float w = sum / (float)n;                           // W(NT) = SUM/N

    window[m + nt] = w;                                       // W2(M+NT) = W(NT)
    window[m - nt] = w;                                       // W2(M-NT) = W(NT)
}
