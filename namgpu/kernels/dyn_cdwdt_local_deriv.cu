// dyn_cdwdt_local_deriv.cu -- local time derivative of w on H points: DWDT = (W - DWDT) * RDT.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:3533-3552 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (inside CDWDT, after the lateral smoothing of w):
//     rdt = 1./dt
//     do l = 1, lm ; do j = jts, jte ; do i = its, ite
//         dwdt(i,j,l) = (w(i,j,l) - dwdt(i,j,l)) * rdt
//     enddo ; enddo ; enddo
// DWDT on entry holds the old w (stored by the caller); on exit it holds the local dw/dt.
//
// Mapping (CONTRACT.md section 1): Fortran A(i,j,l) -> A[(l-1), (j-1), (i-1)] with IDX3.
// Loop limits: the loop uses jts..jte and its..ite (the H-point tile limits, not the _b1 limits),
// which on the single full domain are the whole H grid 1..IM, 1..JM. So every H point is updated.
// Each point reads and writes only its own element; there is no recurrence along l or along i/j,
// so one thread per (i, j, l) point, level in blockIdx.z (grid_2d). Single precision, as kfpt.
// rdt is passed in from the wrapper as float(1.0) / float(dt), the same single-precision value
// the Fortran 1./dt produces.
// Fortran lines not mapped to a term: the !$omp directives (OpenMP only, dropped).

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_cdwdt_local_deriv(
    const int lm,                          // total number of levels (Fortran LM)
    const int im,                          // IM
    const int jm,                          // JM
    const float rdt,                       // 1/dt (Fortran rdt = 1./dt, kfpt)
    const float* __restrict__ w,           // [LM, JM, IM] current vertical velocity
    float* __restrict__ dwdt)              // [LM, JM, IM] in: old w; out: local dw/dt
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;

    const size_t ix = IDX3(i0, j0, l0, im, jm);
    dwdt[ix] = (w[ix] - dwdt[ix]) * rdt;   // dwdt(i,j,l) = (w(i,j,l) - dwdt(i,j,l)) * rdt
}
