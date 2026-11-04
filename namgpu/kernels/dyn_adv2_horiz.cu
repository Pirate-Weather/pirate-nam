// dyn_adv2_horiz.cu -- unit dyn-adv2-horiz: tracer fluxes on H points and horizontal advection of
// species (secadv 2nd order, Adams-Bashforth positioning of sqrt(s)), NMMB module_DYNAMICS_ROUTINES ADV2.
//
// Notice: nmm/module_DYNAMICS_ROUTINES.F90 carries no PROGRAM HISTORY LOG / author block in the NAM
// source at 83df4e13c1; the NAM notice in NOTICE applies (NOAA/NCEP/EMC, US Government work).
//
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:4570-4599 (nam.v4.2.11, 83df4e13c1),
// with the species/level prologue of the same loop nest (4560-4569) as kernel adv2_h_ss1.
//
// Fortran source, per species ks (outer loop), per level l (inner loop):
//   [1] (4563-4568) ss1(i,j) = s1(i,j,l,ks)*cfc + sp(i,j,l,ks)*bfc     (all h points, full domain)
//                   sp(i,j,l,ks) = s1(i,j,l,ks)
//   [2] (4571-4579) ssx(i,j)  = (ss1(i,j)-ss1(i-1,j))  *pfx(i,j,l)
//                   ssy(i,j)  = (ss1(i,j)-ss1(i,j-1))  *pfy(i,j,l)
//                   ssne(i,j) = (ss1(i,j)-ss1(i-1,j-1))*pfne(i,j,l)
//                   ssnw(i,j) = (ss1(i-1,j)-ss1(i,j-1))*pfnw(i,j,l)     (i in its_b1..ite_h1, j in jts_b1..jte_h1)
//   [3] (4581-4599) if(adv_standard): ibeg=max(its,ids+1+lnsad)..iend=min(ite,ide-1-lnsad), same for j:
//                   fahp = fah(j)*idtadt
//                   tcs(i,j,l,ks) = ((((ssx(i,j)+ssx(i+1,j)+ssy(i,j)+ssy(i,j+1))
//                                    +(ssne(i+1,j+1)+ssne(i,j)+ssnw(i,j+1)+ssnw(i+1,j))*0.25)*fahp)
//                                   /(dsg2(l)*pdop(i,j)+pdsg1(l)) + tcs(i,j,l,ks)
//
// Scope of this unit: items [1]-[3] of the horizontal section (the regional upstream branch
// `if(.not.global.and.adv_upstream)` at 4600+ is unit dyn-adv2-bndry, not here).
//
// Two kernels, one launch each per species ks (CONTRACT section 3, routine split): item [1] is a
// point kernel that fills the scratch SS1 (needed by the neighbours) and updates SP; item [2]+[3]
// recomputes the four flux terms inline (same operand order as Fortran) and updates TCS.
//
// Index ranges: item [3] standard region is ibeg..iend, jbeg..jend (0-based, inclusive) passed by the
// wrapper from the tile limits and the module_CONTROL lnsad gate (adv_standard); the wrapper does not
// launch it when adv_standard is false (Fortran :4581).
//
// Layout (CONTRACT section 1): 3-D A(i,j,l) -> A[l0][j0][i0] via IDX3; 4-D TRACERS-type arrays
// [n0,l0,j0,i0] via IDX4 (n0 = ks-1); 2-D PDOP -> IDX2; FAH [JM], DSG2 [LM], PDSG1 [LM] 1-D.
//
// Constants: cfc = 1.533 (Adams-Bashforth, adv2 parameter), bfc = 1.-cfc, both kfpt=single.
// Fast numerics: FMA contraction on (kernels.OPTIONS --fmad=true); the Fortran expression order is kept.

#include "nmmb_common.cuh"

#define ADV2_CFC 1.533f
#define ADV2_BFC (1.0f - 1.533f)

// item [1]: Adams-Bashforth positioning of sqrt(s) for species n0, and SP update. One thread per cell.
extern "C" __global__ void adv2_h_ss1(
    const int IM, const int JM, const int LM, const int n0,
    const float* __restrict__ s1,      // TRACERS_SQRT [NTR,LM,JM,IM] in (sqrt(s), from dyn-adv2-q2-mid)
    float* __restrict__ sp,            // TRACERS_PREV [NTR,LM,JM,IM] inout (set to s1)
    float* __restrict__ ss1)           // SS1 [LM,JM,IM] out, scratch for the species being advected
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t c4 = IDX4(i0, j0, l0, n0, IM, JM, LM);
    const size_t c3 = IDX3(i0, j0, l0, IM, JM);
    const float sv = s1[c4];
    ss1[c3] = sv * ADV2_CFC + sp[c4] * ADV2_BFC;
    sp[c4] = sv;
}

// item [2] helpers on one level: pointers already offset to level l0; index j0*IM+i0 (IDX2).
__device__ __forceinline__ float adv2_ssx(const float* ssl, const float* pfxl, int i0, int j0, int IM) {
    return (ssl[IDX2(i0, j0, IM)] - ssl[IDX2(i0 - 1, j0, IM)]) * pfxl[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv2_ssy(const float* ssl, const float* pfyl, int i0, int j0, int IM) {
    return (ssl[IDX2(i0, j0, IM)] - ssl[IDX2(i0, j0 - 1, IM)]) * pfyl[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv2_ssne(const float* ssl, const float* pfnel, int i0, int j0, int IM) {
    return (ssl[IDX2(i0, j0, IM)] - ssl[IDX2(i0 - 1, j0 - 1, IM)]) * pfnel[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv2_ssnw(const float* ssl, const float* pfnwl, int i0, int j0, int IM) {
    return (ssl[IDX2(i0 - 1, j0, IM)] - ssl[IDX2(i0, j0 - 1, IM)]) * pfnwl[IDX2(i0, j0, IM)];
}

// items [2]+[3]: fluxes and the standard-region second-order horizontal advection of species n0 into TCS.
extern "C" __global__ void adv2_h_flux_std(
    const int IM, const int JM, const int LM, const int ibeg, const int iend, const int jbeg, const int jend,
    const int idtadt, const int n0,
    const float* __restrict__ ss1,     // SS1  [LM,JM,IM] in (from adv2_h_ss1)
    const float* __restrict__ pfx,     // PFX  [LM,JM,IM] in
    const float* __restrict__ pfy,     // PFY  [LM,JM,IM] in
    const float* __restrict__ pfne,    // PFNE [LM,JM,IM] in
    const float* __restrict__ pfnw,    // PFNW [LM,JM,IM] in
    const float* __restrict__ fah,     // FAH  [JM] in, Fortran fah(j), grid factor
    const float* __restrict__ dsg2,    // DSG2 [LM] in
    const float* __restrict__ pdsg1,   // PDSG1[LM] in
    const float* __restrict__ pdop,    // PDOP [JM,IM] in, (pd+pdo)*0.5 from the ADV2 section before
    float* __restrict__ tcs)           // TRACERS_TEND [NTR,LM,JM,IM] inout (added to, as in Fortran)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;

    // item [3] standard region: 0-based inclusive ibeg..iend, jbeg..jend, computed on the host from the
    // tile limits (max(its,ids+1+lnsad)..min(ite,ide-1-lnsad) in Fortran, minus 1). Passed in, not assumed.
    if (i0 < ibeg || i0 > iend || j0 < jbeg || j0 > jend) return;

    const size_t lo = (size_t)l0 * JM * IM;           // level offset for 3-D arrays
    const float* ssl   = ss1 + lo;
    const float* pfxl  = pfx + lo;
    const float* pfyl  = pfy + lo;
    const float* pfnel = pfne + lo;
    const float* pfnwl = pfnw + lo;

    const int i1 = i0 + 1;
    const int j1 = j0 + 1;

    // item [2]: the eight flux terms of the expression, in Fortran's operand order
    const float ssx_ij    = adv2_ssx(ssl, pfxl, i0, j0, IM);
    const float ssx_i1j   = adv2_ssx(ssl, pfxl, i1, j0, IM);
    const float ssy_ij    = adv2_ssy(ssl, pfyl, i0, j0, IM);
    const float ssy_ij1   = adv2_ssy(ssl, pfyl, i0, j1, IM);
    const float ssne_i1j1 = adv2_ssne(ssl, pfnel, i1, j1, IM);
    const float ssne_ij   = adv2_ssne(ssl, pfnel, i0, j0, IM);
    const float ssnw_ij1  = adv2_ssnw(ssl, pfnwl, i0, j1, IM);
    const float ssnw_i1j  = adv2_ssnw(ssl, pfnwl, i1, j0, IM);

    // item [3]: ((((ssx+ssx+ssy+ssy)+(ssne+ssne+ssnw+ssnw)*0.25)*fahp)/(dsg2*pdop+pdsg1)) + tcs
    const float fahp  = fah[j0] * (float)idtadt;
    const float sum_x = ((ssx_ij + ssx_i1j) + ssy_ij) + ssy_ij1;
    const float sum_n = ((ssne_i1j1 + ssne_ij) + ssnw_ij1) + ssnw_i1j;
    const float flux  = (sum_x + sum_n * 0.25f) * fahp;
    const size_t c4 = IDX4(i0, j0, l0, n0, IM, JM, LM);
    const float den = dsg2[l0] * pdop[IDX2(i0, j0, IM)] + pdsg1[l0];
    tcs[c4] = flux / den + tcs[c4];
}
