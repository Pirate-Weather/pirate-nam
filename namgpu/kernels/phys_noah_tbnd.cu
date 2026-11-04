// phys_noah_tbnd.cu -- TBND: soil layer boundary temperature by linear interpolation (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:4945-4985 (nam.v4.2.11, 83df4e13c1).
//
// Start point and notice: ArWen (Apache-2.0, namgpu/third_party/arwen/LICENSE, Copyright 2026 ArWen
// contributors), core/kernels/noah.cu, function noah_tbnd (WRF-lineage Noah, same formula). The
// NAM routine is a SUBROUTINE with INTENT(OUT) TBND1; it is written here as a device function
// plus one point kernel that evaluates it for every soil interface in one launch (CONTRACT
// section 3). The ArWen copy already matches the Fortran term for term; only the names, the
// zero-based K and the SMP-free layout were changed.
//
// Fortran (SUBROUTINE TBND (TU,TB,ZSOIL,ZBOT,K,NSOIL,TBND1)), every executed statement in order:
//   IF (K == 1) THEN ZUP = 0. ELSE ZUP = ZSOIL(K-1)                  (K is 1-based here)
//   IF (K == NSOIL) THEN ZB = 2.*ZBOT - ZSOIL(K) ELSE ZB = ZSOIL(K+1)
//   TBND1 = TU + (TB - TU)* (ZUP - ZSOIL(K))/ (ZUP - ZB)
// Callers (SFLX): TBND(STC(1),STC(2),...,K=1) at :2890; TBND(STC(K),STC(K+1),...,K) at :2958 for
// K < NSOIL; TBND(STC(K),TBOT,...,K) at :2989 for K = NSOIL. The kernel therefore takes TU = STC(k),
// TB = STC(k+1) for k < NSOIL and TB = TBOT for the last layer, which reproduces all three calls.
// No MPI, halo, OpenMP or print statements in this range.
//
// Layout: 3-D point kernel. One thread per (i, j, k) with the 1-based interface index k - 1 on
// grid.z (namgpu.kernels.grid_2d with lm = NSOIL). STC and TBND are [NSOIL, JM, IM] (layer 1 at
// the top, CONTRACT section 1). TBOT is [JM, IM]. ZSOIL [NSOIL] and ZBOT are the same for every
// column (set by the SFLX caller from the Noah parameter tables), so they are passed once.
// Float32 throughout (Fortran REAL = float, kfpt = single).

#include "nmmb_common.cuh"
#include "noah_soil_helpers.cuh"

#define NAM_NSOIL 4

// TBND: nam_tbnd from noah_soil_helpers.cuh (single source, shared with phys-noah-hrt).

extern "C" __global__ void phys_noah_tbnd(
    const float* __restrict__ stc,    // STC   soil layer temperature (K)                 [NSOIL, JM, IM]
    const float* __restrict__ tbot,   // TBOT  bottom boundary temperature (K)            [JM, IM]
    const float* __restrict__ zsoil,  // ZSOIL soil layer bottom depths (m, negative)     [NSOIL]
    float zbot,                       // ZBOT  depth of the constant bottom temperature (m)
    float* __restrict__ tbnd,         // TBND  boundary temperature of interface k (K)    [NSOIL, JM, IM]
    int im, int jm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm) return;
    const int k = l0;                       // zero-based layer; Fortran K = k + 1
    const size_t n = IDX3(i0, j0, k, im, jm);
    const size_t col = IDX2(i0, j0, im);
    const float tu = stc[n];
    const float tb = (k < NAM_NSOIL - 1) ? stc[IDX3(i0, j0, k + 1, im, jm)] : tbot[col];
    tbnd[n] = nam_tbnd(tu, tb, zsoil, zbot, k);
}
