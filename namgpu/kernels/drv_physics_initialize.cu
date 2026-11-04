// nam-gpu unit drv-physics-initialize: PHYSICS_INITIALIZE selection and table kernels (NAM 12 km parent).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Notice kept from the Fortran modules (nam.v4.2.11, 83df4e13c1):
//   subroutine PHYSICS_INITIALIZE     (nmm/module_SOLVER_GRID_COMP.F90:9936-11062)
//   subroutine gpkap, fpkapx          (phys/funcphys.f:2181-2215, 2377-2385)
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_GRID_COMP.F90:10768-10789 (ICICE/ICLIQ special
// cases), nmm/module_SOLVER_GRID_COMP.F90:10811 (LOWLYR allocate, placeholder) and
// phys/funcphys.f:2181-2215 (GPKAP table, the pkap table read by FPKAP in radiation_gases_nmmb).
//
// Kernels:
//   physinit_select     one thread. Fortran 10768-10789. In p[7]: p[0] shortwave is rrtm, p[1] microphysics
//                       is thompson, p[2] np3d, p[3] ICICE_SW, p[4] ICICE_LW, p[5] ICLIQ_SW, p[6] ICLIQ_LW
//                       (the values rad_initialize_nmmb receives before the special cases). Out sel[5]:
//                       sel[0] error (1 = rrtm with thompson and np3d /= 8: NMMB_FINALIZE), sel[1] ICICE_SW,
//                       sel[2] ICICE_LW, sel[3] ICLIQ_SW, sel[4] ICLIQ_LW.
//   physinit_gpkap      one thread per table node jx = 0..NXPKAP-1 (Fortran jx-1). Fortran 2196-2212:
//                       tbpkap(jx) = fpkapx(xmin + (jx-1)*xinc), fpkapx(p) = (p/1e5)^kappa, kappa = con_rocp.
//   physinit_lowlyr     one thread per (i, j). Fortran 10811 ALLOCATE(LOWLYR(IMS:IME,JMS:JME)) with no value:
//                       the array is a placeholder filled by TURBULENCE. Zeroed here so it is never read
//                       uninitialised.
// Launch: one launch per kernel on the caller's stream, no host sync inside the kernels.

#include "nmmb_common.cuh"

#define PI_XMIN     0.0f
#define PI_XMAX     110000.0f
#define PI_CON_RD   2.8705e+2f
#define PI_CON_CP   1.0046e+3f
#define PI_CON_ROCP (PI_CON_RD / PI_CON_CP)

extern "C" __global__ void physinit_select(
    const int* __restrict__ p,      // [7] in
    int* __restrict__ sel)          // [5] out
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int rrtm_thompson = (p[0] != 0 && p[1] != 0) ? 1 : 0;
    const int np3d = p[2];

    int icice_sw = p[3];
    int icice_lw = p[4];
    int iclq_sw  = p[5];
    int iclq_lw  = p[6];

    sel[0] = (rrtm_thompson && np3d != 8) ? 1 : 0;     // NP3D /= 8 -> NMMB_FINALIZE (:10771)
    if (rrtm_thompson) {                               // :10777-10778
        icice_sw = 4;
        icice_lw = 4;
    }
    if (np3d == 5) {                                   // :10786-10789 GFDL-type diagnostic
        iclq_sw = 0;
        iclq_lw = 0;
    }
    sel[1] = icice_sw;
    sel[2] = icice_lw;
    sel[3] = iclq_sw;
    sel[4] = iclq_lw;
}

extern "C" __global__ void physinit_gpkap(
    float* __restrict__ tbpkap,     // [NXPKAP] out, index jx-1
    const int n)                    // NXPKAP
{
    const int jx = blockIdx.x * blockDim.x + threadIdx.x;   // zero-based node
    if (jx >= n) return;
    // xinc = (xmax - xmin)/(nxpkap - 1): funcphys.f:2213
    const float xinc = (PI_XMAX - PI_XMIN) / (float)(n - 1);
    const float x = PI_XMIN + (float)jx * xinc;             // funcphys.f:2215 with jx-1 zero-based
    tbpkap[jx] = powf(x / 1.0e5f, PI_CON_ROCP);             // fpkapx(p), funcphys.f:2377-2385
}

extern "C" __global__ void physinit_lowlyr(
    int* __restrict__ lowlyr,       // [JM, IM] out
    const int im,
    const int jm)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= im || j >= jm) return;
    lowlyr[j * im + i] = 0;
}
