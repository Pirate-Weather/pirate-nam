// phys_noah_frh2o.cu -- FRH2O: unfrozen soil water (Noah LSM), one launch over every soil layer.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2644-2782 (nam.v4.2.11, 83df4e13c1).
// Executed statements: see phys_noah_frh2o.cuh (every term is there).
// Start point: namgpu/third_party/arwen/core/kernels/noah.cu noah_frh2o (Apache-2.0, ArWen). The NAM
// Fortran structure is kept as written (NLOG/KCOUNT loop, BX clamp, option 2 fallback).
//
// Layout: 3-D point kernel. One thread per (i, j, soil layer) with the soil layer on grid.z
// (grid_2d with lm = NSOIL). Soil arrays are [NSOIL, JM, IM] (layer 1 at the top, CONTRACT
// section 1); soil-type parameters SMCMAX, BEXP, PSIS are per column [JM, IM]. TKELV is per
// layer because NAM calls SNKSRC with TAVG = TMPAVG(...) for each soil layer K.
// Float32 throughout (Fortran REAL = float, kfpt = single).

#include "nmmb_common.cuh"
#include "phys_noah_frh2o.cuh"

#define NAM_NSOIL 4

extern "C" __global__ void phys_noah_frh2o(
    const float* __restrict__ tkelv,    // TAVG    soil layer temperature (K)   [NSOIL, JM, IM]
    const float* __restrict__ smc,      // SMC     total soil moisture          [NSOIL, JM, IM]
    const float* __restrict__ sh2o,     // SH2O    unfrozen soil moisture       [NSOIL, JM, IM]
    const float* __restrict__ smcmax,   // SMCMAX  saturation moisture (REDPRM) [JM, IM]
    const float* __restrict__ bexp,     // BEXP    soil B parameter (REDPRM)    [JM, IM]
    const float* __restrict__ psis,     // PSISAT  saturated matric potential   [JM, IM]
    float* __restrict__ freew,          // FREE    supercooled liquid water     [NSOIL, JM, IM] (output)
    int im, int jm)
{
    POINT_IJL                              // l0 = soil layer (blockIdx.z)
    if (i0 >= im || j0 >= jm || l0 >= NAM_NSOIL) return;
    const size_t c = IDX2(i0, j0, im);
    const size_t n = IDX3(i0, j0, l0, im, jm);
    freew[n] = nam_frh2o(tkelv[n], smc[n], sh2o[n], smcmax[c], bexp[c], psis[c]);
}
