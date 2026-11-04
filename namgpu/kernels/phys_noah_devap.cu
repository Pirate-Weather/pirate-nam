// phys_noah_devap.cu -- DEVAP: direct bare-soil evaporation (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2514-2554 (nam.v4.2.11, 83df4e13c1).
//
// The point function is in phys_noah_devap.cuh (shared with the EVAPO kernel). Apache-2.0 notice of
// the ArWen-derived starting point: namgpu/third_party/arwen/LICENSE (Copyright 2026 ArWen contributors,
// namgpu/third_party/arwen/README.md).
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3). Arrays are
// float32 [JM, IM] (C order, IDX2). SH2O is the soil field [NSOIL, JM, IM]; the caller passes its
// layer-1 slice (SH2O[0], contiguous). Limits are the caller's B1 range, 1-based (ITS_B1..ITE_B1 =
// 2..IM-1, JTS_B1..JTE_B1 = 2..JM-1, the NOAHLSM call in module_TURBULENCE.F90): only points inside
// the range are written; the rest of EDIR is not touched, as the Fortran does not touch them.
// Fortran REAL = float (kfpt=single).

#include "nmmb_common.cuh"
#include "phys_noah_devap.cuh"

extern "C" __global__ void phys_noah_devap(
    const float* __restrict__ etp1,    // ETP1    potential evaporation        [JM, IM]
    const float* __restrict__ sh2o,    // SH2O    layer-1 unfrozen soil water  [JM, IM]
    const float* __restrict__ shdfac,  // SHDFAC  green vegetation fraction    [JM, IM]
    const float* __restrict__ smcmax,  // SMCMAX  porosity                     [JM, IM]
    const float* __restrict__ smcdry,  // SMCDRY  dry-soil moisture limit      [JM, IM]
    const float* __restrict__ fxexp,   // FXEXP   soil evaporation exponent    [JM, IM]
    float* __restrict__ edir,          // EDIR    direct soil evaporation      [JM, IM] (output)
    int im, int jm,
    int its, int ite, int jts, int jte)   // 1-based B1 limits of the caller
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    // Thread -> zero-based point. Fortran limits are 1-based: i in [its, ite], j in [jts, jte].
    if (i0 < its - 1 || i0 > ite - 1) return;
    if (j0 < jts - 1 || j0 > jte - 1) return;
    const size_t n = IDX2(i0, j0, im);
    edir[n] = noah_devap(etp1[n], sh2o[n], shdfac[n], smcmax[n], smcdry[n], fxexp[n]);
}
