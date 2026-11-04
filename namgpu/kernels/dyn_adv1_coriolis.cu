// nam-gpu unit dyn-adv1-coriolis: Coriolis/curvature tendency on V points and the
// diagonally averaged mass fluxes used by ADV1 horizontal momentum advection.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:1838-1866 (nam.v4.2.11, 83df4e13c1),
// the body of subroutine adv1 (module_DYNAMICS_ROUTINES.F90:1233), inside the grand vertical loop over l.
//
// Fortran (per level l). Stage [0] is the Adams-Bashforth extrapolation at Fortran 1829-1833
// (u1d/v1d from u, v, up, vp; then up = u, vp = v), ported here because no other unit did it:
//   [0] j=jts_h2..jte_h2, i=its_h2..ite_h2 (single process: the full domain 1..IM, 1..JM):
//         u1d(i,j) = u(i,j,l)*cfc + up(i,j,l)*bfc;  v1d(i,j) = v(i,j,l)*cfc + vp(i,j,l)*bfc
//         up(i,j,l) = u(i,j,l);  vp(i,j,l) = v(i,j,l)
//   [A] below reads u1d/v1d at its own point, so the same thread sees the new values.
//   [A] j=jts_b1..jte_b2, i=its_b1..ite_b2 (1-based: 2..IM-2, 2..JM-2):
//         crv = curv(j)*dt;  fp = f(i,j)*dt;  fpp = u1d(i,j)*crv + fp
//         tcu(i,j,l) = fpp*v1d(i,j) + tcu(i,j,l)
//         tcv(i,j,l) = -fpp*u1d(i,j) + tcv(i,j,l)
//   [B] j=jts_b1..jte_b1_h1, i=its_b1..ite_b1_h1 (1-based: 2..IM-1, 2..JM-1):
//         pfxx1 (i,j) = pfx (i,j,l)   + pfx (i+1,j+1,l)
//         pfyx1 (i,j) = pfy (i,j,l)   + pfy (i+1,j+1,l)
//         pfnex1(i,j) = pfne(i,j,l)   + pfne(i+1,j+1,l)
//         pfnwx1(i,j) = pfnw(i,j,l)   + pfnw(i+1,j+1,l)
//         pfxy1 (i,j) = pfx (i+1,j,l) + pfx (i,j+1,l)
//         pfyy1 (i,j) = pfy (i+1,j,l) + pfy (i,j+1,l)
//         pfney1(i,j) = pfne(i+1,j,l) + pfne(i,j+1,l)
//         pfnwy1(i,j) = pfnw(i+1,j,l) + pfnw(i,j+1,l)
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2, ite_b2 = IM-2, jts_b1 = 2, jte_b2 = JM-2, ite_b1_h1 = IM-1, jte_b1_h1 = JM-1.
//   0-based: [A] i0 in [1, IM-3], j0 in [1, JM-3];  [B] i0 in [1, IM-2], j0 in [1, JM-2].
//   The MPI/OpenMP looplimits split is dropped (one card, one thread per point).
//
// Layout: 3-D arrays [LM,JM,IM] via IDX3; 2-D F [JM,IM] via IDX2; curv is a row vector [JM].
// Level l is blockIdx.z (grid_2d). Each level depends only on its own level's inputs, so the
// whole vertical loop is one launch. [A] and [B] write disjoint arrays (tcu/tcv vs pf*1) and
// read only inputs, so they are fused in one kernel with no ordering hazard.
// The Fortran 2-D scratch (u1d, v1d, pfxx1, ...) is held as level-indexed 3-D scratch so one launch
// covers all levels; the VRAM cost is stated in the wrapper.
// Fortran `tcu`, `tcv` are read-modify-write (inout). `pfx/pfy/pfne/pfnw` are inputs from dyn-pgforce-pgf.
// Floating point: float throughout (kfpt=single), fmad left on (fast numerics).
// No line of the Fortran is skipped. Nothing here is MPI, OpenMP, halo or print.

#include "nmmb_common.cuh"

extern "C" __global__ void adv1_coriolis(
    const int IM, const int JM, const int LM,
    const float dt,
    const float cfc, const float bfc,   // Adams-Bashforth weights (1.533, 1.-cfc), kfpt=single
    const float* __restrict__ u,        // U      [LM,JM,IM]     in   Fortran u(i,j,l)
    const float* __restrict__ v,        // V      [LM,JM,IM]     in   Fortran v(i,j,l)
    float* __restrict__ up,             // UP     [LM,JM,IM]     inout Fortran up(i,j,l)
    float* __restrict__ vp,             // VP     [LM,JM,IM]     inout Fortran vp(i,j,l)
    const float* __restrict__ curv,     // CURV   [JM]           Fortran curv(j)
    const float* __restrict__ f,        // F      [JM,IM]        Fortran f(i,j), Coriolis parameter
    float* __restrict__ u1d,            // U1D    [LM,JM,IM]     out  Fortran u1d(i,j), stage [0]
    float* __restrict__ v1d,            // V1D    [LM,JM,IM]     out  Fortran v1d(i,j), stage [0]
    float* __restrict__ tcu,            // TCU    [LM,JM,IM]     inout  Fortran tcu(i,j,l)
    float* __restrict__ tcv,            // TCV    [LM,JM,IM]     inout  Fortran tcv(i,j,l)
    const float* __restrict__ pfx,      // PFX    [LM,JM,IM]     in     Fortran pfx(i,j,l)
    const float* __restrict__ pfy,      // PFY    [LM,JM,IM]     in
    const float* __restrict__ pfne,     // PFNE   [LM,JM,IM]     in
    const float* __restrict__ pfnw,     // PFNW   [LM,JM,IM]     in
    float* __restrict__ pfxx1,          // PFXX1  [LM,JM,IM]     out
    float* __restrict__ pfyx1,          // PFYX1  [LM,JM,IM]     out
    float* __restrict__ pfnex1,         // PFNEX1 [LM,JM,IM]     out
    float* __restrict__ pfnwx1,         // PFNWX1 [LM,JM,IM]     out
    float* __restrict__ pfxy1,          // PFXY1  [LM,JM,IM]     out
    float* __restrict__ pfyy1,          // PFYY1  [LM,JM,IM]     out
    float* __restrict__ pfney1,         // PFNEY1 [LM,JM,IM]     out
    float* __restrict__ pfnwy1)         // PFNWY1 [LM,JM,IM]     out
{
    POINT_IJL                           // i0 = x-thread, j0 = y-thread, l0 = blockIdx.z
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;

    // [0] Adams-Bashforth extrapolation of U and V, Fortran 1829-1833 (full H domain).
    // Each thread reads its own old UP/VP before overwriting them, so no ordering hazard.
    {
        const size_t ij3 = IDX3(i0, j0, l0, IM, JM);
        const float uc = u[ij3], vc = v[ij3];
        u1d[ij3] = uc * cfc + up[ij3] * bfc;
        v1d[ij3] = vc * cfc + vp[ij3] * bfc;
        up[ij3] = uc;
        vp[ij3] = vc;
    }

    // [A] Coriolis and curvature tendency. Fortran i = 2..IM-2, j = 2..JM-2 (1-based).
    if (i0 >= 1 && i0 <= IM - 3 && j0 >= 1 && j0 <= JM - 3) {
        const size_t ij3 = IDX3(i0, j0, l0, IM, JM);
        const float crv = curv[j0] * dt;                 // Fortran crv = curv(j)*dt
        const float fp  = f[IDX2(i0, j0, IM)] * dt;      // Fortran fp  = f(i,j)*dt
        const float fpp = u1d[ij3] * crv + fp;           // Fortran fpp = u1d*crv + fp
        tcu[ij3] = fpp * v1d[ij3] + tcu[ij3];            // Fortran tcu = fpp*v1d + tcu
        tcv[ij3] = -fpp * u1d[ij3] + tcv[ij3];           // Fortran tcv = -fpp*u1d + tcv
    }

    // [B] Diagonally averaged fluxes on V points. Fortran i = 2..IM-1, j = 2..JM-1 (1-based).
    if (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2) {
        const size_t ij3   = IDX3(i0, j0, l0, IM, JM);
        const size_t ip_j  = IDX3(i0 + 1, j0,     l0, IM, JM);   // (i+1, j)
        const size_t i_jp  = IDX3(i0,     j0 + 1, l0, IM, JM);   // (i, j+1)
        const size_t ip_jp = IDX3(i0 + 1, j0 + 1, l0, IM, JM);   // (i+1, j+1)

        pfxx1[ij3]  = pfx[ij3]  + pfx[ip_jp];
        pfyx1[ij3]  = pfy[ij3]  + pfy[ip_jp];
        pfnex1[ij3] = pfne[ij3] + pfne[ip_jp];
        pfnwx1[ij3] = pfnw[ij3] + pfnw[ip_jp];

        pfxy1[ij3]  = pfx[ip_j]  + pfx[i_jp];
        pfyy1[ij3]  = pfy[ip_j]  + pfy[i_jp];
        pfney1[ij3] = pfne[ip_j] + pfne[i_jp];
        pfnwy1[ij3] = pfnw[ip_j] + pfnw[i_jp];
    }
}
