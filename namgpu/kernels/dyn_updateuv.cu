// dyn_updateuv.cu -- apply the horizontal wind tendencies TCU, TCV to U, V, then limit the
// wind speed to the CFL bound and zero the tendencies.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This file has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:2461-2526 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine UPDATEUV(lm, u, v, tcu, tcv).
//
// Mapping (CONTRACT.md section 1): Fortran A(i,j,l) -> A[(l-1), (j-1), (i-1)] with IDX3.
//   U, V, TCU, TCV: LM levels, V-point arrays with the H shape (CONTRACT section 1).
//
// Loop limits (module_DM_PARALLEL.F90 looplimits, single full domain ids = 1, ide = IM):
//   its_b1 = max(its, ids+1) = 2,  ite_b2 = min(ite, ide-2) = IM-2
//   jts_b1 = max(jts, jds+1) = 2,  jte_b2 = min(jte, jde-2) = JM-2
// So the Fortran loop runs 1-based i in [2, IM-2], j in [2, JM-2] for every level. Zero-based:
// i0 in [1, IM-3], j0 in [1, JM-3]. Points outside that range are not touched (tcu/tcv keep their
// values there, as in the Fortran).
//
// Per-point work: every (i, j, l) in the range is independent (no neighbour reads), so one thread
// per (i, j) with the level in blockIdx.z (CONTRACT section 3). The Fortran order of operations is
// kept: u = u + tcu, v = v + tcv, tcu = 0, tcv = 0, then the CFL test on the updated u, v.
// The OpenMP directives (!$omp parallel do) are host threading only and are not mapped.
//
// Single-precision constants: cflfc = 1./(140.*140.) is a kfpt parameter, so it is a float here.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_updateuv(
    const int lm,                          // total number of levels (Fortran LM)
    const int im,                          // IM
    const int jm,                          // JM
    float* __restrict__ u,                 // [LM, JM, IM] u, updated in place
    float* __restrict__ v,                 // [LM, JM, IM] v, updated in place
    float* __restrict__ tcu,               // [LM, JM, IM] u time change, zeroed in range
    float* __restrict__ tcv)               // [LM, JM, IM] v time change, zeroed in range
{
    POINT_IJL                              // i0 = thread x, j0 = thread y, l0 = blockIdx.z
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    // NMMB _b1 / _b2 limits for the full domain: 1-based i, j in [2, IM-2], [2, JM-2].
    if (i0 < 1 || i0 > im - 3 || j0 < 1 || j0 > jm - 3) return;

    const float cflfc = 1.0f / (140.0f * 140.0f);     // parameter cflfc (cfl limit), kfpt

    const size_t ix = IDX3(i0, j0, l0, im, jm);

    // Fortran lines 2477-2481: add the tendencies, then clear them.
    float uu = u[ix] + tcu[ix];
    float vv = v[ix] + tcv[ix];
    tcu[ix] = 0.0f;
    tcv[ix] = 0.0f;

    // Fortran lines 2483-2490: CFL limiter on the updated wind speed.
    const float cflc = (uu * uu + vv * vv) * cflfc;
    if (cflc > 1.0f) {
        const float rcflc = sqrtf(1.0f / cflc);
        uu = uu * rcflc;
        vv = vv * rcflc;
    }

    u[ix] = uu;
    v[ix] = vv;
}
