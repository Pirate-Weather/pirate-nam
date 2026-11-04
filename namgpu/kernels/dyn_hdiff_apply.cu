// dyn_hdiff_apply.cu -- second-order horizontal diffusion of T, Q, CW, Q2, U, V (NAM NMMB dynamics).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone; NOTICE at the
// repo root). The Fortran has no history block of its own; its module header is kept as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:2834-2915 (nam.v4.2.11, 83df4e13c1),
// the body of subroutine HDIFF inside vertical_loop_3 (do l = 1, lm). The per-level slope masks
// fmlx/fmly (Fortran 2799-2829) are produced once by dyn-hdiff-coef (FMLX_3D, FMLY_3D) and read
// here; this kernel does not recompute them. dyn-hdiff-coef produces def3d, fmlx, fmly.
//
// Fortran HDIFF apply, per level l (1-based):
//   hkx(i,j) = def3d(i-1,j,l)+def3d(i,j,l);  hky(i,j) = def3d(i,j-1,l)+def3d(i,j,l)
//   hkfx = hkx*fmlx;  hkfy = hky*fmly
//   tx = (t(i,j)-t(i-1,j))*hkfx;  ty = (t(i,j)-t(i,j-1))*hkfy   (same for q, cw, q2)
//   ux = (u(i,j)-u(i-1,j))*hky(i,j+1);  uy = (u(i,j)-u(i,j-1))*hkx(i+1,j)   (same for v)
//   tdif = (tx(i+1,j)-tx(i,j))*hdacx(i,j) + (ty(i,j+1)-ty(i,j))*hdacy(i,j)   (q, cw, q2 alike)
//   udif = (ux(i+1,j)-ux(i,j))*hdacvx(i,j) + (uy(i,j+1)-uy(i,j))*hdacvy(i,j)  (v alike)
//   t += tdif; q += facdif*qdif; cw += facdif*cdif; q2 += q2dif; u += udif; v += vdif
//
// Loop limits (module_DM_PARALLEL.F90 looplimits, ids=1, ide=IM, jds=1, jde=JM, single process):
//   mask / hk / tx / ty:   i in [2, IM],   j in [2, JM]     (its_b1 .. ite_h2, jts_b1 .. jte_h2)
//   (checked against module_DM_PARALLEL.F90:799-821 for one process: jte_b1 = min(jte, jde-1) and
//    jte_b2 = min(jte, jde-2) give the same zero-based bounds below, jte_b1 included.)
//   ux / uy / vx / vy:     i in [2, IM-1], j in [2, JM-1]   (its_b1 .. ite_b1_h1, jts_b1 .. jte_b1_h1)
//   tdif/qdif/cdif/q2dif:  i in [2, IM-1], j in [2, JM-1]   (its_b1 .. ite_b1, jts_b1 .. jte_b1)
//   udif/vdif, u/v update: i in [2, IM-2], j in [2, JM-2]   (its_b1 .. ite_b2, jts_b1 .. jte_b2)
//   t/q/cw/q2 update:      i in [2, IM-1], j in [2, JM-1]
// Zero-based (i0 = i-1, j0 = j-1, l0 = l-1) the test is i0 in [1, IM-2] and so on.
//
// Stencil dependence: tdif(i,j) reads tx(i+1,j), i.e. t(i+1,j), so no t may be updated before every
// tdif has been formed. The work is therefore two launches on one stream inside one wrapper call:
//   1. dyn_hdiff_apply_diff:   all six increments of all levels, from the OLD fields only. The
//      tx/ty/ux/uy/vx/vy values at the stencil neighbours are formed on the fly from def3d and the
//      coef masks (one load per term), so no neighbour is read after it was updated;
//   2. dyn_hdiff_apply_update: the six updates, reading the increment arrays.
//
// Fortran lines not mapped to a term: the ENABLE_SMP OpenMP directives (2793-2796, 2909-2912) are
// host threading only. The level loop is the grid z axis (levels are independent in HDIFF). FACDIF is
// 1.0 for the regional run (non-global branch, Fortran 2659-2662) and is passed in.
//
// Layout: CONTRACT.md section 1. 3-D fields [LM, JM, IM]; 2-D H fields [JM, IM]; RDXH [JM].
// z is read at l0 < LM only; a Z array with LM+1 levels has the same strides.

#include "nmmb_common.cuh"

// Fortran local constant in HDIFF (slopec = .05, critical slope).
#define HDIFF_SLOPEC 0.05f

// hkx(a,b) = def3d(a-1,b) + def3d(a,b), needs a >= 1. Fortran 2838.
__device__ __forceinline__ float hd_hkx(const float* __restrict__ def3d, int a, int b, int l0, int im, int jm)
{
    return def3d[IDX3(a - 1, b, l0, im, jm)] + def3d[IDX3(a, b, l0, im, jm)];
}

// hky(a,b) = def3d(a,b-1) + def3d(a,b), needs b >= 1. Fortran 2839.
__device__ __forceinline__ float hd_hky(const float* __restrict__ def3d, int a, int b, int l0, int im, int jm)
{
    return def3d[IDX3(a, b - 1, l0, im, jm)] + def3d[IDX3(a, b, l0, im, jm)];
}

// tx(a,b) = (x(a,b)-x(a-1,b)) * hkfx,  hkfx = hkx*fmlx (fmlx from dyn-hdiff-coef). Fortran 2840, 2844-2850.
__device__ __forceinline__ float hd_tx(const float* __restrict__ x, const float* __restrict__ def3d,
                                       const float* __restrict__ fmlx,
                                       int a, int b, int l0, int im, int jm)
{
    const float hkfx = hd_hkx(def3d, a, b, l0, im, jm) * fmlx[IDX3(a, b, l0, im, jm)];
    return (x[IDX3(a, b, l0, im, jm)] - x[IDX3(a - 1, b, l0, im, jm)]) * hkfx;
}

// ty(a,b) = (x(a,b)-x(a,b-1)) * hkfy,  hkfy = hky*fmly (fmly from dyn-hdiff-coef). Fortran 2841, 2844-2850.
__device__ __forceinline__ float hd_ty(const float* __restrict__ x, const float* __restrict__ def3d,
                                       const float* __restrict__ fmly,
                                       int a, int b, int l0, int im, int jm)
{
    const float hkfy = hd_hky(def3d, a, b, l0, im, jm) * fmly[IDX3(a, b, l0, im, jm)];
    return (x[IDX3(a, b, l0, im, jm)] - x[IDX3(a, b - 1, l0, im, jm)]) * hkfy;
}

// ux(a,b) = (u(a,b)-u(a-1,b)) * hky(a,b+1). Fortran 2853. The hky here is the unmasked sum.
__device__ __forceinline__ float hd_ux(const float* __restrict__ u, const float* __restrict__ def3d,
                                       int a, int b, int l0, int im, int jm)
{
    return (u[IDX3(a, b, l0, im, jm)] - u[IDX3(a - 1, b, l0, im, jm)]) * hd_hky(def3d, a, b + 1, l0, im, jm);
}

// uy(a,b) = (u(a,b)-u(a,b-1)) * hkx(a+1,b). Fortran 2854. The hkx here is the unmasked sum.
__device__ __forceinline__ float hd_uy(const float* __restrict__ u, const float* __restrict__ def3d,
                                       int a, int b, int l0, int im, int jm)
{
    return (u[IDX3(a, b, l0, im, jm)] - u[IDX3(a, b - 1, l0, im, jm)]) * hd_hkx(def3d, a + 1, b, l0, im, jm);
}

// Pass 1: all six increments of all levels, from the OLD fields. Grid: grid_2d(IM, JM, LM).
// Scalar increments on zero-based [1, IM-2] x [1, JM-2]; velocity increments on [1, IM-3] x [1, JM-3].
extern "C" __global__ void dyn_hdiff_apply_diff(
    const int lm,                          // total number of levels (Fortran LM); thread guard
    const int im,                          // IM
    const int jm,                          // JM
    const float* __restrict__ fmlx,        // [LM, JM, IM] x-slope mask FMLX_3D (dyn-hdiff-coef)
    const float* __restrict__ fmly,        // [LM, JM, IM] y-slope mask FMLY_3D (dyn-hdiff-coef)
    const float* __restrict__ t,           // [LM, JM, IM]
    const float* __restrict__ q,           // [LM, JM, IM]
    const float* __restrict__ cw,          // [LM, JM, IM]
    const float* __restrict__ q2,          // [LM, JM, IM]
    const float* __restrict__ u,           // [LM, JM, IM]
    const float* __restrict__ v,           // [LM, JM, IM]
    const float* __restrict__ def3d,       // [LM, JM, IM] deformation (dyn-hdiff-coef)
    const float* __restrict__ hdacx,       // [JM, IM]
    const float* __restrict__ hdacy,       // [JM, IM]
    const float* __restrict__ hdacvx,      // [JM, IM]
    const float* __restrict__ hdacvy,      // [JM, IM]
    float* __restrict__ tdif,              // [LM, JM, IM] out
    float* __restrict__ qdif,              // [LM, JM, IM] out
    float* __restrict__ cdif,              // [LM, JM, IM] out
    float* __restrict__ q2dif,             // [LM, JM, IM] out
    float* __restrict__ udif,              // [LM, JM, IM] out
    float* __restrict__ vdif)              // [LM, JM, IM] out
{
    POINT_IJL
    if (l0 >= lm || i0 >= im || j0 >= jm) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);
    const size_t h = IDX2(i0, j0, im);

    // Scalars: Fortran i in [2, IM-1], j in [2, JM-1]  ->  zero-based a in [1, IM-2], b in [1, JM-2].
    if (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2) {
        const int a = i0, b = j0;
        // tdif (Fortran 2868-2869): tx(i+1,j), tx(i,j), ty(i,j+1), ty(i,j)
        const float tx_p = hd_tx(t, def3d, fmlx, a + 1, b, l0, im, jm);
        const float tx_0 = hd_tx(t, def3d, fmlx, a, b, l0, im, jm);
        const float ty_p = hd_ty(t, def3d, fmly, a, b + 1, l0, im, jm);
        const float ty_0 = hd_ty(t, def3d, fmly, a, b, l0, im, jm);
        tdif[n] = (tx_p - tx_0) * hdacx[h] + (ty_p - ty_0) * hdacy[h];

        // qdif (Fortran 2870-2871)
        const float qx_p = hd_tx(q, def3d, fmlx, a + 1, b, l0, im, jm);
        const float qx_0 = hd_tx(q, def3d, fmlx, a, b, l0, im, jm);
        const float qy_p = hd_ty(q, def3d, fmly, a, b + 1, l0, im, jm);
        const float qy_0 = hd_ty(q, def3d, fmly, a, b, l0, im, jm);
        qdif[n] = (qx_p - qx_0) * hdacx[h] + (qy_p - qy_0) * hdacy[h];

        // cdif (Fortran 2872-2873)
        const float cx_p = hd_tx(cw, def3d, fmlx, a + 1, b, l0, im, jm);
        const float cx_0 = hd_tx(cw, def3d, fmlx, a, b, l0, im, jm);
        const float cy_p = hd_ty(cw, def3d, fmly, a, b + 1, l0, im, jm);
        const float cy_0 = hd_ty(cw, def3d, fmly, a, b, l0, im, jm);
        cdif[n] = (cx_p - cx_0) * hdacx[h] + (cy_p - cy_0) * hdacy[h];

        // q2dif (Fortran 2874-2875)
        const float q2x_p = hd_tx(q2, def3d, fmlx, a + 1, b, l0, im, jm);
        const float q2x_0 = hd_tx(q2, def3d, fmlx, a, b, l0, im, jm);
        const float q2y_p = hd_ty(q2, def3d, fmly, a, b + 1, l0, im, jm);
        const float q2y_0 = hd_ty(q2, def3d, fmly, a, b, l0, im, jm);
        q2dif[n] = (q2x_p - q2x_0) * hdacx[h] + (q2y_p - q2y_0) * hdacy[h];
    }

    // Velocities: Fortran i in [2, IM-2], j in [2, JM-2]  ->  zero-based a in [1, IM-3], b in [1, JM-3].
    if (i0 >= 1 && i0 <= im - 3 && j0 >= 1 && j0 <= jm - 3) {
        const int a = i0, b = j0;
        // udif (Fortran 2881-2882): ux(i+1,j), ux(i,j), uy(i,j+1), uy(i,j)
        const float ux_p = hd_ux(u, def3d, a + 1, b, l0, im, jm);
        const float ux_0 = hd_ux(u, def3d, a, b, l0, im, jm);
        const float uy_p = hd_uy(u, def3d, a, b + 1, l0, im, jm);
        const float uy_0 = hd_uy(u, def3d, a, b, l0, im, jm);
        udif[n] = (ux_p - ux_0) * hdacvx[h] + (uy_p - uy_0) * hdacvy[h];

        // vdif (Fortran 2883-2884)
        const float vx_p = hd_ux(v, def3d, a + 1, b, l0, im, jm);
        const float vx_0 = hd_ux(v, def3d, a, b, l0, im, jm);
        const float vy_p = hd_uy(v, def3d, a, b + 1, l0, im, jm);
        const float vy_0 = hd_uy(v, def3d, a, b, l0, im, jm);
        vdif[n] = (vx_p - vx_0) * hdacvx[h] + (vy_p - vy_0) * hdacvy[h];
    }
}

// Pass 2: the six updates (Fortran 2891-2906). Runs after pass 1 on the same stream, so every
// increment was formed from the old fields. Grid: grid_2d(IM, JM, LM).
extern "C" __global__ void dyn_hdiff_apply_update(
    const int lm,                          // total number of levels (Fortran LM); thread guard
    const int im,                          // IM
    const int jm,                          // JM
    const float facdif,                    // 1.0 for the regional run (Fortran 2659-2662)
    float* __restrict__ t,                 // [LM, JM, IM] updated in place
    float* __restrict__ q,                 // [LM, JM, IM] updated in place
    float* __restrict__ cw,                // [LM, JM, IM] updated in place
    float* __restrict__ q2,                // [LM, JM, IM] updated in place
    float* __restrict__ u,                 // [LM, JM, IM] updated in place
    float* __restrict__ v,                 // [LM, JM, IM] updated in place
    const float* __restrict__ tdif,
    const float* __restrict__ qdif,
    const float* __restrict__ cdif,
    const float* __restrict__ q2dif,
    const float* __restrict__ udif,
    const float* __restrict__ vdif)
{
    POINT_IJL
    if (l0 >= lm || i0 >= im || j0 >= jm) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);

    // Fortran i in [2, IM-1], j in [2, JM-1]
    if (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2) {
        t[n]  = t[n] + tdif[n];
        q[n]  = q[n] + facdif * qdif[n];       // Fortran 2895
        cw[n] = cw[n] + facdif * cdif[n];      // Fortran 2896
        q2[n] = q2[n] + q2dif[n];
    }
    // Fortran i in [2, IM-2], j in [2, JM-2]
    if (i0 >= 1 && i0 <= im - 3 && j0 >= 1 && j0 <= jm - 3) {
        u[n] = u[n] + udif[n];
        v[n] = v[n] + vdif[n];
    }
}
