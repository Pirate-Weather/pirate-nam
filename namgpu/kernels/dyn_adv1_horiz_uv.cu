// WOOF-NAM unit dyn-adv1-horiz-uv: horizontal momentum advection inside NMMB ADV1 (secadv = true, second
// order), the 4th-order diagonal branch (secadv = false) and the Adams-Bashforth advection of u1d/v1d.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:1867-2021 (nam.v4.2.11, 83df4e13c1),
// the body of subroutine adv1 inside the grand vertical loop over l (vertical_loop2).
//
// Fortran, per level l (1-based indices below; kernels use 0-based i0 = i-1, j0 = j-1, l0 = l-1):
//   [A] if(.not.secadv): u2d/v2d on i=2..IM-1, j=2..JM-1 (1-based):
//         u2d = (u1d(i,j-1)+u1d(i-1,j)+u1d(i+1,j)+u1d(i,j+1))*sfc + u1d(i,j)*pfc
//         (same for v2d from v1d), pfc = 1+4/6, sfc = -1/6.
//   [A'] regional boundary copies (s_bdy, n_bdy, w_bdy, e_bdy, in that order; the tile is the whole
//        domain so all four are true):  south row 1, north rows JM-1 and JM (from JM-1), west column 1,
//        east columns IM-1 and IM (from IM-1), for u2d and v2d.  Kernel adv1_uv_u4_edge, one launch per
//        edge in that order, so the corner overwrites come out in Fortran order.
//   [A''] u1d = u2d, v1d = v2d over the full domain (kernel adv1_uv_u4_copy).  Writes U1D/V1D in place.
//   [B] The flux scratch pfxx1..pfnwy1 come from dyn-adv1-coriolis (inputs here).  The 16 momentum
//       fluxes ufxx1..vfnwy1 are NOT stored: each is (difference of u1d or v1d) * pf at the point it is used,
//       recomputed in the advection kernel from the same float32 operands (the same values as the Fortran
//       2-D scratch, without 16 x 191 MB of level scratch).  Helper fx() below.
//   [C] if(adv_standard): i = ibeg..iend, j = jbeg..jend (host limits from module_CONTROL.F90:250-275 with
//       lnsad = lnsh+2 = 7; upper bound ide-2-lnsad as written at module_DYNAMICS_ROUTINES.F90:1982):
//         dxody = dxv(j)*rdyv,  dyodx = dyv*rdxv(j),  fadp = fad(j)
//         pdop(i,j) = PDOP_UV(i,j): the 8-point value written by dyn-adv1-vert-uv (Fortran 1731-1732).
//         It is NOT (pd+pdo)*0.5 here: that is the T-section formula (Fortran ~1427).
//         fdpp = fadp/(dsg2(l)*pdop + pdsg1(l))
//         dux1 = (ufnex1(i+1,j+1)+ufnex1(i,j)+ufnwx1(i,j+1)+ufnwx1(i+1,j))*0.25
//              + (ufxx1(i,j)+ufxx1(i+1,j)+ufyx1(i,j)+ufyx1(i,j+1))          (dvx1: same with v1d)
//         duy1 = (ufney1(i+1,j+1)+ufney1(i,j)+ufnwy1(i,j+1)+ufnwy1(i+1,j))*0.25
//              + (ufxy1(i,j)+ufxy1(i+1,j)+ufyy1(i,j)+ufyy1(i,j+1))          (dvy1: same with v1d)
//         tcu(i,j,l) = ((dvx1-dvy1)*dxody + (dux1+duy1))*fdpp + tcu(i,j,l)
//         tcv(i,j,l) = ((dux1-duy1)*dyodx + (dvx1+dvy1))*fdpp + tcv(i,j,l)
//
// Not ported, global-only (CONTRACT section 2; the wrapper refuses global = true):
//   swapwn/polewn of u2d/v2d, the global zeroing of pfyx1/pfyy1/pfnex1/pfnwx1/pfney1/pfnwy1 rows.
// The MPI/OpenMP looplimits split is dropped (one card, one thread per point).
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   [A] interior: 0-based i0 in [1, IM-2], j0 in [1, JM-2].
//   [C] passed in as 0-based inclusive ib..ie, jb..je by the wrapper.
//
// Layout: 3-D arrays [LM,JM,IM] via IDX3 (CONTRACT section 1); level is blockIdx.z (grid_2d). Each level
// depends only on its own level, so every launch covers all levels at once. The copy kernels run as
// separate launches on one stream, so no kernel reads a value another thread of the same launch writes.
// Floating point: float throughout (kfpt = single), fmad left on (fast numerics).

#include "nmmb_common.cuh"

// One momentum flux at point (p,q): (a(p+ai,q+aj) - a(p+bi,q+bj)) * pf(p,q), level l0.
__device__ __forceinline__ float fx(const float* __restrict__ a, const float* __restrict__ pf,
                                    const int IM, const int JM, const int l0,
                                    const int p, const int q,
                                    const int ai, const int aj, const int bi, const int bj)
{
    return (a[IDX3(p + ai, q + aj, l0, IM, JM)] - a[IDX3(p + bi, q + bj, l0, IM, JM)])
           * pf[IDX3(p, q, l0, IM, JM)];
}

// The four momentum-flux terms of one component (u or v) at (p,q). The x component uses the
// (ne, nw, xx, yx) set; the y component uses the (ney, nwy, xy, yy) set.
__device__ __forceinline__ float mom_comp(const float* __restrict__ a,
                                          const float* __restrict__ pf_ne,
                                          const float* __restrict__ pf_nw,
                                          const float* __restrict__ pf_xx,
                                          const float* __restrict__ pf_yx,
                                          const int IM, const int JM, const int l0,
                                          const int p, const int q)
{
    const float diag =
          (fx(a, pf_ne, IM, JM, l0, p + 1, q + 1, 0, 0, -1, -1)
         + fx(a, pf_ne, IM, JM, l0, p,     q,     0, 0, -1, -1)
         + fx(a, pf_nw, IM, JM, l0, p,     q + 1, -1, 0, 0, -1)
         + fx(a, pf_nw, IM, JM, l0, p + 1, q,     -1, 0, 0, -1)) * 0.25f;
    const float axis =
          fx(a, pf_xx, IM, JM, l0, p,     q,     0, 0, -1, 0)
        + fx(a, pf_xx, IM, JM, l0, p + 1, q,     0, 0, -1, 0)
        + fx(a, pf_yx, IM, JM, l0, p,     q,     0, 0, 0, -1)
        + fx(a, pf_yx, IM, JM, l0, p,     q + 1, 0, 0, 0, -1);
    return diag + axis;
}

// [A] 4th-order diagonal average, interior. Fortran i = 2..IM-1, j = 2..JM-1 (1-based).
extern "C" __global__ void adv1_uv_u4_diag(
    const int IM, const int JM, const int LM,
    const float* __restrict__ u1d,      // U1D [LM,JM,IM]  in
    const float* __restrict__ v1d,      // V1D [LM,JM,IM]  in
    float* __restrict__ u2d,            // U2D [LM,JM,IM]  out (scratch)
    float* __restrict__ v2d)            // V2D [LM,JM,IM]  out (scratch)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;
    const float pfc = 1.0f + 4.0f / 6.0f;   // Fortran pfc = 1.+4./6.   (module_DYNAMICS_ROUTINES:1254)
    const float sfc = -1.0f / 6.0f;         // Fortran sfc = -1./6.     (module_DYNAMICS_ROUTINES:1255)
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    const size_t s = IDX3(i0,     j0 - 1, l0, IM, JM);
    const size_t w = IDX3(i0 - 1, j0,     l0, IM, JM);
    const size_t e = IDX3(i0 + 1, j0,     l0, IM, JM);
    const size_t n = IDX3(i0,     j0 + 1, l0, IM, JM);
    u2d[c] = (u1d[s] + u1d[w] + u1d[e] + u1d[n]) * sfc + (u1d[c] * pfc);
    v2d[c] = (v1d[s] + v1d[w] + v1d[e] + v1d[n]) * sfc + (v1d[c] * pfc);
}

// [A'] regional boundary copies. edge: 0 = south (s_bdy), 1 = north (n_bdy), 2 = west (w_bdy),
// 3 = east (e_bdy). Launched once per edge, in that order, on one stream.
//   south: u2d(i,jds) = u1d(i,jds)                              -> row 0
//   north: u2d(i,jde-1) = u1d(i,jde-1); u2d(i,jde) = u1d(i,jde-1) -> rows JM-2 and JM-1
//   west : u2d(ids,j) = u1d(ids,j)                              -> column 0
//   east : u2d(ide-1,j) = u1d(ide-1,j); u2d(ide,j) = u1d(ide-1,j) -> columns IM-2 and IM-1
// Thread t runs over i (edges 0, 1) or j (edges 2, 3); blockIdx.z is the level.
extern "C" __global__ void adv1_uv_u4_edge(
    const int IM, const int JM, const int LM, const int edge,
    const float* __restrict__ u1d,
    const float* __restrict__ v1d,
    float* __restrict__ u2d,
    float* __restrict__ v2d)
{
    const int t  = blockIdx.x * blockDim.x + threadIdx.x;
    const int l0 = blockIdx.z;
    if (l0 >= LM) return;
    const int n = (edge == 0 || edge == 1) ? IM : JM;
    if (t >= n) return;
    if (edge == 0) {
        const size_t c = IDX3(t, 0, l0, IM, JM);
        u2d[c] = u1d[c];
        v2d[c] = v1d[c];
    } else if (edge == 1) {
        const size_t src = IDX3(t, JM - 2, l0, IM, JM);
        const size_t d1  = IDX3(t, JM - 2, l0, IM, JM);
        const size_t d2  = IDX3(t, JM - 1, l0, IM, JM);
        u2d[d1] = u1d[src];
        u2d[d2] = u1d[src];
        v2d[d1] = v1d[src];
        v2d[d2] = v1d[src];
    } else if (edge == 2) {
        const size_t c = IDX3(0, t, l0, IM, JM);
        u2d[c] = u1d[c];
        v2d[c] = v1d[c];
    } else {
        const size_t src = IDX3(IM - 2, t, l0, IM, JM);
        const size_t d1  = IDX3(IM - 2, t, l0, IM, JM);
        const size_t d2  = IDX3(IM - 1, t, l0, IM, JM);
        u2d[d1] = u1d[src];
        u2d[d2] = u1d[src];
        v2d[d1] = v1d[src];
        v2d[d2] = v1d[src];
    }
}

// [A''] u1d = u2d, v1d = v2d over the full domain (Fortran its_h1..ite_h1, jts_h1..jte_h1 = 1..IM, 1..JM).
extern "C" __global__ void adv1_uv_u4_copy(
    const int IM, const int JM, const int LM,
    const float* __restrict__ u2d,
    const float* __restrict__ v2d,
    float* __restrict__ u1d,            // U1D [LM,JM,IM]  inout
    float* __restrict__ v1d)            // V1D [LM,JM,IM]  inout
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    u1d[c] = u2d[c];
    v1d[c] = v2d[c];
}

// [C] Momentum advection. Range ib..ie, jb..je (0-based, inclusive) = Fortran ibeg..iend, jbeg..jend.
// Read-modify-write of tcu/tcv; reads u1d, v1d and the flux scratch only.
extern "C" __global__ void adv1_uv_horiz(
    const int IM, const int JM, const int LM,
    const int ib, const int ie, const int jb, const int je,
    const float dyv,                    // DYV    scalar   deltay
    const float rdyv,                   // RDYV   scalar   1/deltay
    const float* __restrict__ dxv,      // DXV    [JM]     dxv(j)
    const float* __restrict__ rdxv,     // RDXV   [JM]     1/dxv(j)
    const float* __restrict__ fad,      // FAD    [JM]     grid factor fad(j)
    const float* __restrict__ dsg2,     // DSG2   [LM]     dsg2(l)
    const float* __restrict__ pdsg1,    // PDSG1  [LM]     pdsg1(l)
    const float* __restrict__ pdop,     // PDOP_UV [JM,IM]  from dyn-adv1-vert-uv
    const float* __restrict__ u1d,      // U1D    [LM,JM,IM]
    const float* __restrict__ v1d,      // V1D    [LM,JM,IM]
    const float* __restrict__ pfxx1,    // PFXX1  [LM,JM,IM]  from dyn-adv1-coriolis
    const float* __restrict__ pfyx1,    // PFYX1  [LM,JM,IM]
    const float* __restrict__ pfnex1,   // PFNEX1 [LM,JM,IM]
    const float* __restrict__ pfnwx1,   // PFNWX1 [LM,JM,IM]
    const float* __restrict__ pfxy1,    // PFXY1  [LM,JM,IM]
    const float* __restrict__ pfyy1,    // PFYY1  [LM,JM,IM]
    const float* __restrict__ pfney1,   // PFNEY1 [LM,JM,IM]
    const float* __restrict__ pfnwy1,   // PFNWY1 [LM,JM,IM]
    float* __restrict__ tcu,            // TCU    [LM,JM,IM]  inout
    float* __restrict__ tcv)            // TCV    [LM,JM,IM]  inout
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    if (i0 < ib || i0 > ie || j0 < jb || j0 > je) return;

    const float dxody = dxv[j0] * rdyv;        // Fortran dxody = dxv(j)*rdyv
    const float dyodx = dyv * rdxv[j0];        // Fortran dyodx = dyv*rdxv(j)
    const float fadp  = fad[j0];               // Fortran fadp  = fad(j)

    const size_t c   = IDX3(i0, j0, l0, IM, JM);
    const size_t c2  = IDX2(i0, j0, IM);
    const float fdpp = fadp / (dsg2[l0] * pdop[c2] + pdsg1[l0]);        // Fortran fdpp

    const float dux1 = mom_comp(u1d, pfnex1, pfnwx1, pfxx1, pfyx1, IM, JM, l0, i0, j0);
    const float dvx1 = mom_comp(v1d, pfnex1, pfnwx1, pfxx1, pfyx1, IM, JM, l0, i0, j0);
    const float duy1 = mom_comp(u1d, pfney1, pfnwy1, pfxy1, pfyy1, IM, JM, l0, i0, j0);
    const float dvy1 = mom_comp(v1d, pfney1, pfnwy1, pfxy1, pfyy1, IM, JM, l0, i0, j0);

    tcu[c] = ((dvx1 - dvy1) * dxody + (dux1 + duy1)) * fdpp + tcu[c];
    tcv[c] = ((dux1 - duy1) * dyodx + (dvx1 + dvy1)) * fdpp + tcv[c];
}
