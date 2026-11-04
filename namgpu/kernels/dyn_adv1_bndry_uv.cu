// WOOF-NAM unit dyn-adv1-bndry-uv: upstream advection of momentum along the four lateral
// boundaries (regional branch of ADV1), accumulated into TCU and TCV.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:2022-2186 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (last block of vertical_loop2, l = 1..lm; executed when .not.global and adv_upstream):
//   envp = -dt*rdyv
//   [S] south rows   j = max(jts,jds+1) .. min(jte,jds+lnsad),  i = its_b1 .. ite_b2   emvp = -dt*rdxv(j)
//   [N] north rows   j = max(jts,jde-1-lnsad) .. jte_b2,        i = its_b1 .. ite_b2
//   [W] west columns j = max(jts,jds+1+lnsad) .. min(jte,jde-2-lnsad), i = its_b1 .. ids+lnsad
//   [E] east columns same j range,                              i = max(its,ide-1-lnsad) .. ite_b2
//   each region: pp = u*emvp, qq = v*envp, sign-selected upwind offsets iap, jap, and
//     tcu = (u(i+iap)-u)*pp + (u(j+jap)-u)*qq + (u - u(i+iap) - u(j+jap) + u(i+iap,j+jap))*pp*qq + tcu
//     tcv = same with v                                       (the Fortran expression, same order)
//
// Index limits (module_DM_PARALLEL.F90 looplimits, 1-based), single process, full domain:
//   ids=its=1, ide=ite=IM, jds=jts=1, jde=jte=JM, lnsad = lnsh+2 = 7 (module_CONTROL.F90 regional
//   branch, lnsad=lnsh+2).  its_b1 = 2, ite_b2 = IM-2, jte_b2 = JM-2.
//
// Order: one thread per (i,j,l) runs the four regions in the Fortran order S, N, W, E, and each
// region updates a cell only when it lies in it.  The regions are disjoint for lnsad=7 on this
// domain, but if they ever overlap, the cell gets each contribution in the Fortran order (the
// update reads U and V only, never TCU or TCV, so no cross-cell ordering matters).
//
// Fortran-to-CUDA review: every assignment in 2022-2186 (tcu and tcv in the four regions, the
// envp and emvp scalars, the iap/jap/pp/qq sign logic, the overlap-free loop limits) has a
// counterpart below.  The MPI looplimits split and the OpenMP directives are dropped (one card,
// full domain).  Not in this range: the T update of ADV1 (dyn-adv1-bndry-t) and the horizontal
// interior advection (dyn-adv1-horiz-uv); the purpose line in UNITS.json mentions T, but the
// Fortran lines 2022-2186 touch only TCU and TCV.
//
// Layout: U, V, TCU, TCV are [LM,JM,IM] (IDX3, l0 = blockIdx.z).  RDXV is [JM].
// Fast numerics (fmad on), single precision.

#include "nmmb_common.cuh"

__device__ __forceinline__ void adv1_bndry_uv_point(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ tcu,
    float* __restrict__ tcv,
    const int IM, const int JM,
    const int i0, const int j0, const int l0,
    const float emvp, const float envp)
{
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    const float uc = u[c];
    const float vc = v[c];

    float pp = uc * emvp;
    float qq = vc * envp;

    int iap, jap;
    if (pp <= 0.0f) {
        iap = -1;
        pp = -pp;
    } else {
        iap = 1;
    }
    if (qq <= 0.0f) {
        jap = -1;
        qq = -qq;
    } else {
        jap = 1;
    }

    const size_t cx  = IDX3(i0 + iap, j0,       l0, IM, JM);
    const size_t cy  = IDX3(i0,       j0 + jap, l0, IM, JM);
    const size_t cxy = IDX3(i0 + iap, j0 + jap, l0, IM, JM);

    const float ux = u[cx], uy = u[cy], uxy = u[cxy];
    const float vx = v[cx], vy = v[cy], vxy = v[cxy];

    tcu[c] = (ux - uc) * pp
           + (uy - uc) * qq
           + (uc - ux - uy + uxy) * pp * qq
           + tcu[c];
    tcv[c] = (vx - vc) * pp
           + (vy - vc) * qq
           + (vc - vx - vy + vxy) * pp * qq
           + tcv[c];
}

extern "C" __global__ void dyn_adv1_bndry_uv(
    const int IM, const int JM, const int LM,
    const int lnsad,
    const float dt,
    const float rdyv,
    const float* __restrict__ rdxv,      // RDXV [JM]      Fortran rdxv(1:jm), 1/deltax per row
    const float* __restrict__ u,         // U    [LM,JM,IM]
    const float* __restrict__ v,         // V    [LM,JM,IM]
    float* __restrict__ tcu,             // TCU  [LM,JM,IM] inout
    float* __restrict__ tcv,             // TCV  [LM,JM,IM] inout
    const int JOFF, const int JMG)       // row slab (nmmb_common.cuh): domain tests on the global row
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    if (l0 >= LM) return;
    const int jg = j0 + JOFF;
    if (NMMB_SLAB_EDGE(j0, JM, jg, JMG, 1)) return;   // reads j0-1..j0+1

    const int I = i0 + 1;               // Fortran 1-based indices
    const int J = jg + 1;               // global Fortran j
    const float envp = -(dt * rdyv);    // envp=-dt*rdyv

    // Single-process limits (see header).  ids+lnsad = 1+lnsad, jds+1+lnsad = 2+lnsad, etc.
    const int jw_lo = max(1, 2 + lnsad);            // max(jts, jds+1+lnsad)
    const int jw_hi = min(JMG, JMG - 2 - lnsad);      // min(jte, jde-2-lnsad)
    const int js_hi = min(JMG, 1 + lnsad);          // min(jte, jds+lnsad)
    const int jn_lo = max(1, JMG - 1 - lnsad);       // max(jts, jde-1-lnsad)
    const int jn_hi = JMG - 2;                     // jte_b2
    const int i_lo_b1 = 2;                          // its_b1
    const int i_hi_b2 = IM - 2;                     // ite_b2
    const int iw_hi = min(IM, 1 + lnsad);           // ids+lnsad
    const int ie_lo = max(1, IM - 1 - lnsad);       // max(its, ide-1-lnsad)

    // [S] southern rows: emvp = -dt*rdxv(j)
    if (J >= 2 && J <= js_hi && I >= i_lo_b1 && I <= i_hi_b2) {
        const float emvp = -(dt * rdxv[j0]);
        adv1_bndry_uv_point(u, v, tcu, tcv, IM, JM, i0, j0, l0, emvp, envp);
    }
    // [N] northern rows
    if (J >= jn_lo && J <= jn_hi && I >= i_lo_b1 && I <= i_hi_b2) {
        const float emvp = -(dt * rdxv[j0]);
        adv1_bndry_uv_point(u, v, tcu, tcv, IM, JM, i0, j0, l0, emvp, envp);
    }
    // [W] western columns
    if (J >= jw_lo && J <= jw_hi && I >= i_lo_b1 && I <= iw_hi) {
        const float emvp = -(dt * rdxv[j0]);
        adv1_bndry_uv_point(u, v, tcu, tcv, IM, JM, i0, j0, l0, emvp, envp);
    }
    // [E] eastern columns
    if (J >= jw_lo && J <= jw_hi && I >= ie_lo && I <= i_hi_b2) {
        const float emvp = -(dt * rdxv[j0]);
        adv1_bndry_uv_point(u, v, tcu, tcv, IM, JM, i0, j0, l0, emvp, envp);
    }
}
