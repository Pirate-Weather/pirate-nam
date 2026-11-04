// phys_noah_snowz0.cu -- SNOWZ0: total roughness length over snow (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:4508-4526 (nam.v4.2.11, 83df4e13c1).
//
// Start point and notice: ArWen (Apache-2.0, third_party/arwen/LICENSE, Copyright 2026 ArWen
// contributors), core/kernels/noah.cu, function noah_snowz0. That copy is NOT carried over: it
// blends toward a burial-depth roughness (7*Z0BRD - SNOWH), which is not in NAM's SNOWZ0. The NAM
// routine is the one below, so the port follows NAM (CONTRACT section 7, copy-arwen: "bring it to
// the NAM Fortran").
//
// Fortran (SUBROUTINE SNOWZ0, lines 4508-4526), every executed statement in order:
//   Z0 = Z0BRD
// The blended form (1-SNCOVR)*Z0BRD + SNCOVR*Z0S with Z0S = 0.001 is commented out in the source
// (line 4520, "!m"), and the parameter Z0S is declared but unused. SNCOVR is INTENT(IN) and is
// not referenced by any executed statement, so the device function takes it for the calling
// convention but does not read it. No MPI, halo, OpenMP or print statements in this range.
// The caller (SFLX, module_LS_NOAHLSM.F90:2080) only calls SNOWZ0 when SNCOVR > 0; otherwise it
// sets Z0 = Z0BRD itself. The kernel is called for every point and returns Z0BRD either way,
// which is the same value on both paths.
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1. All arrays float32 [JM, IM].

#include "nmmb_common.cuh"

// Device function: the Fortran SNOWZ0 body.
__device__ __forceinline__ float noah_snowz0(float sncovr, float z0brd)
{
    (void)sncovr;
    return z0brd;
}

extern "C" __global__ void phys_noah_snowz0(
    const float* __restrict__ sncovr,  // SNCOVR fractional snow cover (INTENT IN, not read) [JM, IM]
    const float* __restrict__ z0brd,   // Z0BRD  background roughness length (m)              [JM, IM]
    float* __restrict__ z0,            // Z0     roughness length with snow (m), output      [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    z0[n] = noah_snowz0(sncovr[n], z0brd[n]);
}
