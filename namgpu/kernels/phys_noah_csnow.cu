// phys_noah_csnow.cu -- CSNOW: snow thermal conductivity (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The NOAA author/history
// header block of the source module is kept verbatim as the notice (module_LS_NOAHLSM.F90, lines 1-15):
//
// !-----------------------------------------------------------------------
// !
//       MODULE MODULE_LS_NOAHLSM
// !
// !-----------------------------------------------------------------------
// !***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
// !-----------------------------------------------------------------------
// !
// ! HISTORY LOG:
// !
// !   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
// !               Vasic - SOILW limitted to positive values.
// !
// !-----------------------------------------------------------------------
// !
//
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:2472-2511 (nam.v4.2.11, 83df4e13c1).
//
// Copied from ArWen gpuwm/core/kernels/noah.cu (Apache-2.0) for WOOF-NAM, 2026-11-04. The formula is
// ArWen's noah_csnow (2.0f * 0.11631f * (0.328f * powf(10.0f, 2.25f * dsnow))), the same arithmetic
// in the same order as the Fortran below. ArWen is Copyright 2026 ArWen contributors, licensed under
// the Apache License, Version 2.0 (namgpu/third_party/arwen/LICENSE; https://www.apache.org/licenses/LICENSE-2.0).
//
// Fortran (SUBROUTINE CSNOW(SNCOND,DSNOW)), every executed statement:
//   C = 0.328*10**(2.25*DSNOW)                     (line 2491; REAL**REAL, base 10 -> powf)
//   SNCOND = 2.0 * UNIT * C, UNIT = 0.11631        (line 2508; left-to-right, (2*UNIT)*C)
// Dyachkova (1960) form, valid for snow density 0.1-0.4. The commented-out CSNOW=UNIT*C,
// De Vaux (1933) and Andersen (Flerchinger) forms (lines 2494-2505) are not executed in NAM
// and are not ported. SNCOND is the output; DSNOW is the input snow density (SNDENS).
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3).
// Arrays are float32 [JM, IM] (C order, index IDX2). Fortran REAL = float (kfpt=single).
// The device function nam_csnow is the CONTRACT unit; phys_noah_csnow is the one launch that
// applies it to every column. The Fortran calls CSNOW only on the SNEQV /= 0 branch
// (module_LS_NOAHLSM.F90:1892-1905 and :1960). This kernel computes every cell; the gating is
// the SFLX unit's job and is an open item in namgpu/physics/noah/phys_noah_csnow.py.

#include "nmmb_common.cuh"

#define NAM_CSNOW_UNIT 0.11631f

__device__ inline float nam_csnow(float dsnow)
{
    const float c = 0.328f * powf(10.0f, 2.25f * dsnow);   // line 2491
    return 2.0f * NAM_CSNOW_UNIT * c;                         // line 2508
}

extern "C" __global__ void phys_noah_csnow(
    const float* __restrict__ dsnow,   // DSNOW   snow density SNDENS (INTENT IN)   [JM, IM]
    float* __restrict__ sncond,        // SNCOND  snow thermal conductivity (output) [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    sncond[n] = nam_csnow(dsnow[n]);
}
