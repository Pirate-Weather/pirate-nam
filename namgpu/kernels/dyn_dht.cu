// nam-gpu unit dyn-dht: horizontal mass divergence and pressure-advection term of the T equation.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:498-796 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine dht, regional configuration (global = .false.; the zero-divergence ring is applied).
// Sections of the Fortran and where they live here:
//   [pre]  pdx, pdy, pdne, pdnw (2-D pre-pass, its_b1..ite_h2 x jts_b1..jte_h2): evaluated inline in
//          dht_flux at each (i,j,l). Same expressions and order.
//   [flux] per level, j = 2..JM, i = 2..IM: pfx, pfy, pfne, pfnw written to state.
//          tx, ty, tne, tnw are the Fortran 2-D level temporaries; they are not stored here but
//          recomputed (same expressions) by dht_tx/dht_ty/dht_tne/dht_tnw wherever dht_div reads them.
//   [div]  per level, j = 2..JM-1, i = 2..IM-1: div (in place), tdiv = div, omgalf (dht_div).
//   [bdy]  s_bdy, n_bdy, w_bdy, e_bdy are all true for the single-process regional layout: the outer
//          ring of div and tdiv is set to 0 (dht_div, perimeter branch).
//   [cum]  tdiv(i,j,l) = tdiv(i,j,l-1) + tdiv(i,j,l), l = 2..LM, i = 2..IM-1, j = 2..JM-1 (dht_cum).
//
// Index mapping (1-based Fortran -> 0-based CUDA): fortran (i,j,l) = cuda (i-1, j-1, l-1).
// The helpers fat/f2 take 1-based i, j, l so every line reads like the Fortran.
//
// Launches on one stream, in order: dht_flux (grid_2d over i, j, l), dht_div (grid_2d), dht_cum (columns).
// No host sync, no allocation. OpenMP (ENABLE_SMP) and halo exchange code is not ported: one card,
// whole domain in memory.

#include "nmmb_common.cuh"

// module_DYNAMICS_ROUTINES dht: cfc = 1.533 (Adams-Bashforth positioning), bfc = 1 - cfc, both kfpt=single.
#define DHT_CFC 1.533f
#define DHT_BFC (1.0f - 1.533f)

// 1-based Fortran access: 3-D field A(i,j,l) and 2-D field A(i,j).
__device__ __forceinline__ float fat(const float* a, int IM, int JM, int i, int j, int l) {
    return a[IDX3(i - 1, j - 1, l - 1, IM, JM)];
}
__device__ __forceinline__ float f2(const float* a, int IM, int i, int j) {
    return a[IDX2(i - 1, j - 1, IM)];
}

// dxv(j-1), 1-based Fortran index: dxv is dimensioned jds:jde, so dxv(j-1) is dxv[j-2] (0-based).
__device__ __forceinline__ float dht_dxp1(const float* dxv, int j) {
    return dxv[j - 2];
}

// tx(i,j) = pcx(i,j,l) * udy(i,j),  udy = (u(i-1,j-1,l)+u(i-1,j,l))*dyp
__device__ __forceinline__ float dht_tx(const float* u, const float* pcx, int IM, int JM,
                                        int i, int j, int l, float dyp) {
    const float udy = (fat(u, IM, JM, i - 1, j - 1, l) + fat(u, IM, JM, i - 1, j, l)) * dyp;
    return fat(pcx, IM, JM, i, j, l) * udy;
}

// ty(i,j) = pcy(i,j,l) * vdx,  vdx = (v(i-1,j-1,l)+v(i,j-1,l))*dxp,  dxp = dxv(j-1)*0.5 on row j
__device__ __forceinline__ float dht_ty(const float* v, const float* pcy, const float* dxv, int IM, int JM,
                                        int i, int j, int l) {
    const float dxp = dht_dxp1(dxv, j) * 0.5f;
    const float vdx = (fat(v, IM, JM, i - 1, j - 1, l) + fat(v, IM, JM, i, j - 1, l)) * dxp;
    return fat(pcy, IM, JM, i, j, l) * vdx;
}

// udy1 = u(i-1,j-1,l)*dyp1,  vdx1 = v(i-1,j-1,l)*dxp1 (row j)
// tne(i,j) = pcne(i,j,l)*(udy1+vdx1),  tnw(i,j) = pcnw(i,j,l)*(-udy1+vdx1)
__device__ __forceinline__ float dht_tne(const float* u, const float* v, const float* pcne, const float* dxv,
                                         int IM, int JM, int i, int j, int l, float dyp1) {
    const float udy1 = fat(u, IM, JM, i - 1, j - 1, l) * dyp1;
    const float vdx1 = fat(v, IM, JM, i - 1, j - 1, l) * dht_dxp1(dxv, j);
    return fat(pcne, IM, JM, i, j, l) * (udy1 + vdx1);
}
__device__ __forceinline__ float dht_tnw(const float* u, const float* v, const float* pcnw, const float* dxv,
                                         int IM, int JM, int i, int j, int l, float dyp1) {
    const float udy1 = fat(u, IM, JM, i - 1, j - 1, l) * dyp1;
    const float vdx1 = fat(v, IM, JM, i - 1, j - 1, l) * dht_dxp1(dxv, j);
    return fat(pcnw, IM, JM, i, j, l) * (-udy1 + vdx1);
}

// [flux] Fortran dht, the mass and pressure fluxes on H points, i = 2..IM, j = 2..JM, every level.
extern "C" __global__ void dht_flux(
    const int IM, const int JM, const int LM,
    const float dyv,                       // DYV: deltay, v point (dyp1 = dyv)
    const float* __restrict__ dsg2,        // DSG2 [LM]    delta sigmas
    const float* __restrict__ pdsg1,       // PDSG1 [LM]   delta pressures
    const float* __restrict__ dxv,         // DXV [JM]     deltax, v point
    const float* __restrict__ pd,          // PD [JM,IM]
    const float* __restrict__ pdo,         // PDO [JM,IM]
    const float* __restrict__ u,           // U [LM,JM,IM]
    const float* __restrict__ v,           // V [LM,JM,IM]
    float* __restrict__ pfx,               // PFX [LM,JM,IM]   out
    float* __restrict__ pfy,               // PFY [LM,JM,IM]   out
    float* __restrict__ pfne,              // PFNE [LM,JM,IM]  out
    float* __restrict__ pfnw)              // PFNW [LM,JM,IM]  out
{
    POINT_IJL
    if (i0 < 1 || i0 > IM - 1 || j0 < 1 || j0 > JM - 1 || l0 >= LM) return;
    const int i = i0 + 1, j = j0 + 1, l = l0 + 1;

    const float dyp1 = dyv;                 // dyp1 = dyv
    const float dyp = dyp1 * 0.5f;          // dyp  = dyp1*0.5
    const float dxp1 = dht_dxp1(dxv, j);    // dxp1 = dxv(j-1)
    const float dxp = dxp1 * 0.5f;          // dxp  = dxp1*0.5
    const float pdp = pdsg1[l0];            // pdp = pdsg1(l)
    const float dsg = dsg2[l0];             // dsg2(l)

    // pre-pass values at (i,j): pdx, pdy, pdne, pdnw
    const float pdx  = ((f2(pd, IM, i - 1, j) + f2(pd, IM, i, j)) * DHT_CFC
                      + (f2(pdo, IM, i - 1, j) + f2(pdo, IM, i, j)) * DHT_BFC) * 0.5f;
    const float pdy  = ((f2(pd, IM, i, j - 1) + f2(pd, IM, i, j)) * DHT_CFC
                      + (f2(pdo, IM, i, j - 1) + f2(pdo, IM, i, j)) * DHT_BFC) * 0.5f;
    const float pdne = ((f2(pd, IM, i - 1, j - 1) + f2(pd, IM, i, j)) * DHT_CFC
                      + (f2(pdo, IM, i - 1, j - 1) + f2(pdo, IM, i, j)) * DHT_BFC) * 0.5f;
    const float pdnw = ((f2(pd, IM, i, j - 1) + f2(pd, IM, i - 1, j)) * DHT_CFC
                      + (f2(pdo, IM, i, j - 1) + f2(pdo, IM, i - 1, j)) * DHT_BFC) * 0.5f;

    const float pdxp  = dsg * pdx + pdp;
    const float pdyp  = dsg * pdy + pdp;
    const float pdnep = dsg * pdne + pdp;
    const float pdnwp = dsg * pdnw + pdp;

    const float udy = (fat(u, IM, JM, i - 1, j - 1, l) + fat(u, IM, JM, i - 1, j, l)) * dyp;
    const float vdx = (fat(v, IM, JM, i - 1, j - 1, l) + fat(v, IM, JM, i, j - 1, l)) * dxp;

    const size_t c = IDX3(i0, j0, l0, IM, JM);
    pfx[c] = udy * pdxp;
    pfy[c] = vdx * pdyp;

    const float udy1 = fat(u, IM, JM, i - 1, j - 1, l) * dyp1;
    const float vdx1 = fat(v, IM, JM, i - 1, j - 1, l) * dxp1;

    pfne[c] = (udy1 + vdx1) * pdnep;
    pfnw[c] = (-udy1 + vdx1) * pdnwp;
}

// [div] Fortran dht, divergence and omega-alpha on the interior, zero ring on the regional boundary.
extern "C" __global__ void dht_div(
    const int IM, const int JM, const int LM,
    const float dyv,
    const float* __restrict__ dsg2,        // DSG2 [LM]
    const float* __restrict__ pdsg1,       // PDSG1 [LM]
    const float* __restrict__ dxv,         // DXV [JM]
    const float* __restrict__ fcp,         // FCP [JM]     (row constant, init-grid-consts)
    const float* __restrict__ fdiv,        // FDIV [JM]    (row constant, init-grid-consts)
    const float* __restrict__ pd,          // PD [JM,IM]
    const float* __restrict__ u,           // U [LM,JM,IM]
    const float* __restrict__ v,           // V [LM,JM,IM]
    const float* __restrict__ pcx,         // PCX [LM,JM,IM]
    const float* __restrict__ pcy,         // PCY [LM,JM,IM]
    const float* __restrict__ pcne,        // PCNE [LM,JM,IM]
    const float* __restrict__ pcnw,        // PCNW [LM,JM,IM]
    const float* __restrict__ pfx,         // PFX [LM,JM,IM]   from dht_flux
    const float* __restrict__ pfy,
    const float* __restrict__ pfne,
    const float* __restrict__ pfnw,
    float* __restrict__ div,               // DIV [LM,JM,IM]   inout
    float* __restrict__ tdiv,              // TDIV [LM,JM,IM]  out
    float* __restrict__ omgalf)            // OMGALF [LM,JM,IM] out (interior only, as Fortran)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);

    // regional boundary ring (global = .false., s/n/w/e_bdy all true): div = tdiv = 0
    if (i0 == 0 || i0 == IM - 1 || j0 == 0 || j0 == JM - 1) {
        div[c] = 0.0f;
        tdiv[c] = 0.0f;
        return;
    }

    // interior: i = 2..IM-1, j = 2..JM-1 (1-based)
    const int i = i0 + 1, j = j0 + 1, l = l0 + 1;
    const float dyp1 = dyv;
    const float dyp = dyp1 * 0.5f;
    const float fdp = fdiv[j0];             // fdp  = fdiv(j)
    const float fcpp = fcp[j0];             // fcpp = fcp(j)

    // div(i,j,l) = ((pfx(i+1,j)-pfx(i,j) + pfy(i,j+1)-pfy(i,j)) + (pfne(i+1,j+1)-pfne(i,j) + pfnw(i,j+1)-pfnw(i+1,j))*0.25)*fdp + div(i,j,l)
    const float hdiv =
        ((fat(pfx, IM, JM, i + 1, j, l) - fat(pfx, IM, JM, i, j, l)
        + fat(pfy, IM, JM, i, j + 1, l) - fat(pfy, IM, JM, i, j, l))
        + (fat(pfne, IM, JM, i + 1, j + 1, l) - fat(pfne, IM, JM, i, j, l)
        + fat(pfnw, IM, JM, i, j + 1, l) - fat(pfnw, IM, JM, i + 1, j, l)) * 0.25f) * fdp;
    const float dnew = hdiv + div[c];
    div[c] = dnew;
    tdiv[c] = dnew;

    // omgalf(i,j,l) = ((tx(i,j)+tx(i+1,j)+ty(i,j)+ty(i,j+1)) + (tne(i+1,j+1)+tne(i,j)+tnw(i,j+1)+tnw(i+1,j))*0.25) * fcpp / (dsg2(l)*pd(i,j)+pdsg1(l))
    const float tsum =
        dht_tx(u, pcx, IM, JM, i, j, l, dyp) + dht_tx(u, pcx, IM, JM, i + 1, j, l, dyp)
        + dht_ty(v, pcy, dxv, IM, JM, i, j, l) + dht_ty(v, pcy, dxv, IM, JM, i, j + 1, l);
    const float tnsum =
        dht_tne(u, v, pcne, dxv, IM, JM, i + 1, j + 1, l, dyp1) + dht_tne(u, v, pcne, dxv, IM, JM, i, j, l, dyp1)
        + dht_tnw(u, v, pcnw, dxv, IM, JM, i, j + 1, l, dyp1) + dht_tnw(u, v, pcnw, dxv, IM, JM, i + 1, j, l, dyp1);
    omgalf[c] = (tsum + tnsum * 0.25f) * fcpp / (dsg2[l0] * f2(pd, IM, i, j) + pdsg1[l0]);
}

// [cum] Fortran dht, cumulative horizontal divergence tdiv down each column, interior columns.
extern "C" __global__ void dht_cum(
    const int IM, const int JM, const int LM,
    float* __restrict__ tdiv)              // TDIV [LM,JM,IM] inout
{
    COLUMN_IJ(IM, JM)
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;
    for (int l0 = 1; l0 < LM; ++l0) {
        tdiv[IDX3(i0, j0, l0, IM, JM)] = tdiv[IDX3(i0, j0, l0 - 1, IM, JM)] + tdiv[IDX3(i0, j0, l0, IM, JM)];
    }
}
