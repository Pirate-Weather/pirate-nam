// phys_noah_tmpavg.cu -- TMPAVG: soil layer average temperature in the freezing/thawing layer (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:5095-5197 (nam.v4.2.11, 83df4e13c1).
//
// Start point: namgpu/third_party/arwen/core/kernels/noah.cu noah_tmpavg (Apache License 2.0, ArWen,
// namgpu/third_party/arwen/LICENSE). The branch structure is kept as NAM writes it (the eight TUP/TM/TDN
// sign cases with the X0/XUP/XDN interface crossings), statement for statement; the only change is
// the layout (one thread per (i, j, soil layer), Fortran K = k + 1) and the NAM T0 literal.
//
// Fortran TMPAVG(TAVG,TUP,TM,TDN,ZSOIL,NSOIL,K): K is 1-based; k = K-1 here (0..NSOIL-1).
// DZ = -ZSOIL(1) for K = 1, else ZSOIL(K-1) - ZSOIL(K). DZH = DZ*0.5. T0 = 273.15 K (PARAMETER).
// TUP is the temperature at the top boundary of layer K, TDN at its bottom boundary, TM the layer
// prognostic temperature. Every caller in SFLX passes TUP and TDN explicitly.
//
// Layout: TUP, TM, TDN, TAVG are soil fields [NSOIL, JM, IM] (CONTRACT section 1, layer 1 at the
// top, grid.z = layer). ZSOIL is the shared [NSOIL] layer-bottom depth vector (m, negative down).
// One launch via namgpu.kernels.grid_2d(IM, JM, NSOIL); no host sync, no device malloc.
// Float32 throughout (Fortran REAL, kfpt = single). Fast numerics (fmad on, IEEE division).

#include "nmmb_common.cuh"
#include "noah_soil_helpers.cuh"

// TMPAVG: nam_tmpavg from noah_soil_helpers.cuh (single source, shared with phys-noah-hrt).

extern "C" __global__ void phys_noah_tmpavg(
    const float* __restrict__ tup,    // TUP    temperature at top of layer (K)     [NSOIL, JM, IM]
    const float* __restrict__ tm,     // TM     layer prognostic temperature (K)    [NSOIL, JM, IM]
    const float* __restrict__ tdn,    // TDN    temperature at bottom of layer (K)  [NSOIL, JM, IM]
    const float* __restrict__ zsoil,  // ZSOIL  layer bottom depth (m)              [NSOIL]
    float* __restrict__ tavg,         // TAVG   layer average temperature (K) out   [NSOIL, JM, IM]
    int im, int jm, int nsoil)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int k0 = blockIdx.z;
    if (i0 >= im || j0 >= jm || k0 >= nsoil) return;
    const size_t p = IDX3(i0, j0, k0, im, jm);
    tavg[p] = nam_tmpavg(tup[p], tm[p], tdn[p], zsoil, k0);
}
