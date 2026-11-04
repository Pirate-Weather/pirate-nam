// phys_noah_wdfcnd.cu -- WDFCND: soil water diffusivity WDF and conductivity WCND (Noah LSM),
// one launch over every soil layer.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:5312-5369 (nam.v4.2.11, 83df4e13c1).
// Executed statements: see phys_noah_wdfcnd.cuh (every term is there).
//
// Layout: 3-D point kernel. One thread per (i, j, soil layer), soil layer on grid.z (grid_2d with
// lm = nsoil). SMC, WDF, WCND are [NSOIL, JM, IM] (layer 1 at the top, CONTRACT section 1). The
// soil-type parameters SMCMAX, BEXP, DKSAT, DWSAT and SICEMAX are per column [JM, IM]: the NAM
// caller passes one value per column (REDPRM soil type; SICEMAX is the column maximum of frozen
// soil water, SFLX SICE(KS) = SMC(KS) - SH2O(KS)). The layer count is an argument, not a constant:
// the wrapper refuses any nsoil other than the model's NSOIL. Float32 throughout (kfpt = single).

#include "nmmb_common.cuh"
#include "phys_noah_wdfcnd.cuh"

extern "C" __global__ void phys_noah_wdfcnd(
    const float* __restrict__ smc,      // SMC     total soil moisture          [NSOIL, JM, IM]
    const float* __restrict__ smcmax,   // SMCMAX  saturation moisture          [JM, IM]
    const float* __restrict__ bexp,     // BEXP    soil B parameter             [JM, IM]
    const float* __restrict__ dksat,    // DKSAT   saturated hydraulic cond.    [JM, IM]
    const float* __restrict__ dwsat,    // DWSAT   saturated diffusivity        [JM, IM]
    const float* __restrict__ sicemax,  // SICEMAX column max frozen soil water  [JM, IM]
    float* __restrict__ wdf,            // WDF     soil water diffusivity        [NSOIL, JM, IM] (output)
    float* __restrict__ wcnd,           // WCND    soil hydraulic conductivity   [NSOIL, JM, IM] (output)
    int im, int jm, int nsoil)
{
    POINT_IJL                              // l0 = soil layer (blockIdx.z)
    if (i0 >= im || j0 >= jm || l0 >= nsoil) return;
    const size_t c = IDX2(i0, j0, im);
    const size_t n = IDX3(i0, j0, l0, im, jm);
    float d = 0.0f, k = 0.0f;
    nam_wdfcnd(d, k, smc[n], smcmax[c], bexp[c], dksat[c], dwsat[c], sicemax[c]);
    wdf[n] = d;
    wcnd[n] = k;
}
