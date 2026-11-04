// drv_tablet.cu -- TABLET: temperature look-up table from pressure and equivalent potential
// temperature (NMMB look-up tables, built once at init).
//
// NOAA/NCEP NAM (nam.v4.2.11, MatthewPyle-NOAA/NAM @ 83df4e13c1), a US government work; the
// NOAA source is kept as the notice. No licence file accompanies the source.
//
// Notice (Fortran header block of the routine, nmm/module_CONTROL.F90:1970-2068, kept verbatim):
//   module module_control  (NMMB control, nam.v4.2.11)
//   subroutine tablet
//   generates the table for finding temperature from pressure and equivalent potential temperature
//
// Ported to CUDA for WOOF-NAM from nmm/module_CONTROL.F90:1970-2068 (nam.v4.2.11, 83df4e13c1).
//
// Two stages, one unit:
//   1. tablet_build (this file): one thread per pressure row kp (ITB = 201 rows). Fills the spline
//      knots of every row: xold = theold, yold = told, xnew = thenew, y2 = 0 (natural ends), and
//      the0, sthe, and the module scalars rdp, rdthe.
//   2. The spline (Fortran CALL SPLINE, module_CONTROL.F90:2074-2183) is NOT repeated here. The
//      wrapper runs it with drv-spline's batched kernel (namgpu/kernels/drv_spline.cu), one spline
//      per row, nold = nnew = JTB, which is the Fortran call TABLET makes for every kp.
//
// Outputs (wrapper, namgpu/steps/drv_tablet.py):
//   work [4, ITB, JTB]  knots: [0] theold (xold), [1] told (yold), [2] y2, [3] thenew (xnew)
//   the0 [ITB], sthe [ITB]
//   scal [2] = { rdp = 1/dp, rdthe = 1/dthe } (Fortran module scalars)
// pl is passed in: pl = max(PT, 1000.) as the consts routine sets it.
#include "nmmb_common.cuh"

#define TB_ITB 201
#define TB_JTB 601
#define TB_PH   105000.0f   /* upper bound of pressure range (module_CONTROL ph) */
#define TB_THH  350.0f      /* upper bound of potential temperature range (thh) */
#define TB_THL  200.0f      /* lower bound of potential temperature range (thl) */
#define TB_EPS  1.0e-10f    /* eps of TABLET: the 32-bit monotonic guard */

// TABLET stage 1: one thread per kp row (Fortran do kp=1,itb).
extern "C" __global__ void tablet_build(float* __restrict__ work,
                                        float* __restrict__ the0,
                                        float* __restrict__ sthe,
                                        float* __restrict__ scal,
                                        const float pl)
{
    // Module scalars rdp (= 1/dp) and rdthe (= 1/dthe), written once by thread 0.
    const float dp_s   = (TB_PH - pl) / (float)(TB_ITB - 1);
    const float dthe_s = 1.0f / (float)(TB_JTB - 1);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scal[0] = 1.0f / dp_s;
        scal[1] = 1.0f / dthe_s;
    }

    const int kp0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (kp0 >= TB_ITB) return;

    float* theold = work + ((size_t)0 * TB_ITB + kp0) * TB_JTB;   // spline xold
    float* told   = work + ((size_t)1 * TB_ITB + kp0) * TB_JTB;   // spline yold
    float* y2     = work + ((size_t)2 * TB_ITB + kp0) * TB_JTB;   // spline y2
    float* thenew = work + ((size_t)3 * TB_ITB + kp0) * TB_JTB;   // spline xnew

    const float dth = (TB_THH - TB_THL) / (float)(TB_JTB - 1);
    const float dp  = (TB_PH - pl) / (float)(TB_ITB - 1);

    // Fortran: p = pl - dp, then p = p + dp at the top of each kp pass, so row kp0 has kp0+1 steps.
    float p = pl - dp;
    for (int s = 0; s <= kp0; s++) p = p + dp;

    // Fortran: th = thl - dth, then th = th + dth for each kth.
    float th = TB_THL - dth;
    for (int kth = 0; kth < TB_JTB; kth++) {
        th = th + dth;
        const float ape = powf(100000.0f / p, NMMB_CAPPA);
        const float qs  = NMMB_PQ0 / p * expf(NMMB_A2 * (th - NMMB_A3 * ape) / (th - NMMB_A4 * ape));
        told[kth]   = th / ape;
        theold[kth] = th * expf(NMMB_ELIWV * qs / (NMMB_CP * told[kth]));
    }

    const float the0k = theold[0];
    const float sthek = theold[TB_JTB - 1] - theold[0];
    theold[0] = 0.0f;
    theold[TB_JTB - 1] = 1.0f;
    for (int kth = 1; kth < TB_JTB - 1; kth++) {       // Fortran kth = 2 .. jtb-1
        theold[kth] = (theold[kth] - the0k) / sthek;
        // 32-bit guard: keep the abscissae strictly increasing (uses the updated theold[kth-1]).
        if (theold[kth] - theold[kth - 1] < TB_EPS) {
            theold[kth] = theold[kth - 1] + TB_EPS;
        }
    }

    the0[kp0] = the0k;
    sthe[kp0] = sthek;

    thenew[0] = 0.0f;
    thenew[TB_JTB - 1] = 1.0f;
    const float dthe = 1.0f / (float)(TB_JTB - 1);
    for (int kth = 1; kth < TB_JTB - 1; kth++) {
        thenew[kth] = thenew[kth - 1] + dthe;
    }

    // Natural-spline end values y2(1) = y2(jtb) = 0; the spline back substitution fills the rest.
    for (int kth = 0; kth < TB_JTB; kth++) y2[kth] = 0.0f;
}
