// dyn_adv1_horiz_t.cu -- unit dyn-adv1-horiz-t: temperature horizontal advection (secadv, 2nd order)
// with the Adams-Bashforth time positioning of T, NMMB module_DYNAMICS_ROUTINES.F90 ADV1.
//
// Notice: nmm/module_DYNAMICS_ROUTINES.F90 carries no PROGRAM HISTORY LOG / author block in the
// NAM source at 83df4e13c1; the NAM notice in NOTICE applies (NOAA/NCEP/EMC, US Government work).
//
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:1539-1570 (nam.v4.2.11, 83df4e13c1).
//
// Fortran source (ADV1, vertical_loop1 body, one level l):
//   [1] t1(i,j)  = t(i,j,l)*cfc + tp(i,j,l)*bfc          (all h points, j in jts_h1..jte_h1)
//       tp(i,j,l) = t(i,j,l)                              (same loop, after t1)
//   [2] tx(i,j)  = (t1(i,j)-t1(i-1,j))*pfx(i,j,l)
//       ty(i,j)  = (t1(i,j)-t1(i,j-1))*pfy(i,j,l)
//       tne(i,j) = (t1(i,j)-t1(i-1,j-1))*pfne(i,j,l)
//       tnw(i,j) = (t1(i-1,j)-t1(i,j-1))*pfnw(i,j,l)     (i in its_b1..ite_h1, j in jts_b1..jte_h1)
//   [3] if(adv_standard): ibeg=max(its,ids+1+lnsad) .. iend=min(ite,ide-1-lnsad), same for j:
//       tct(i,j,l) = (((tx(i,j)+tx(i+1,j)+ty(i,j)+ty(i,j+1))
//                     +(tne(i+1,j+1)+tne(i,j)+tnw(i,j+1)+tnw(i+1,j))*0.25)*fah(j))
//                    /(dsg2(l)*pdop(i,j)+pdsg1(l)) + omgalf(i,j,l) + tct(i,j,l)
//
// Three kernels, one launch each, same wrapper call (CONTRACT section 3, routine split):
//   adv1_pdop_h     -> PDOP = (pd+pdo)*0.5 on H points (Fortran 1427-1431), run first each step.
//   adv1_t_ab_col   -> item [1]: needs no neighbours, so it runs first and fills T1 (scratch).
//   adv1_t_horiz_std-> items [2]+[3]: reads T1 at (i-1, j-1); recomputes the four flux terms
//                      inline with the same expressions and operation order as the Fortran.
// The regional upstream branch (ADV1, `if(.not.global.and.adv_upstream)`, Fortran 1573+) is a
// different unit and is not in this file; it runs after this one and overwrites its rows of TCT.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, regional, ids=1 ide=IM jds=1 jde=JM,
// 1-based -> 0-based here):
//   item [1] h1 range: full domain  i0 in [0, IM-1], j0 in [0, JM-1]
//   item [2] b1 range: i0 in [1, IM-1], j0 in [1, JM-1]  (tx/ty/tne/tnw need i0-1, j0-1 >= 0)
//   item [3] standard region: i0 in [1+lnsad, IM-2-lnsad], j0 in [1+lnsad, JM-2-lnsad]
//
// Layout (CONTRACT section 1): 3-D A(i,j,l) -> A[l0][j0][i0] via IDX3; 2-D PDOP -> IDX2; FAH and
// DSG2/PDSG1 are 1-D per-j and per-l arrays (Fortran fah(jds:jde), dsg2(1:lm), pdsg1(1:lm)).
//
// Constants: cfc = 1.533 (Adams-Bashforth), bfc = 1.-cfc, both kfpt=single (module adv1 parameters).
// Fast numerics: FMA contraction is on (--fmad=true via kernels.OPTIONS); the Fortran expression
// is written in the same order, so contraction is the only difference the compiler may introduce.

#include "nmmb_common.cuh"

#define ADV1_CFC 1.533f
#define ADV1_BFC (1.0f - 1.533f)

// item [1]: AB positioning of T for the horizontal advection, and TP update. One thread per cell.
extern "C" __global__ void adv1_t_ab_col(
    const int IM, const int JM, const int LM,
    const float* __restrict__ t,       // T    [LM,JM,IM] in
    float* __restrict__ tp,            // TP   [LM,JM,IM] inout (set to T)
    float* __restrict__ t1)            // T1   [LM,JM,IM] out, scratch
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    const float tv = t[c];
    t1[c] = tv * ADV1_CFC + tp[c] * ADV1_BFC;
    tp[c] = tv;
}

// PDOP producer (Fortran ADV1, module_DYNAMICS_ROUTINES.F90:1427-1431, before the vertical loop):
//   pdop(i,j) = (pd(i,j)+pdo(i,j))*0.5 over the H points, j in jts..jte, i in its..ite.
// Single process, regional or global: the full [0,IM-1] x [0,JM-1] domain. One thread per cell.
extern "C" __global__ void adv1_pdop_h(
    const int IM, const int JM,
    const float* __restrict__ pd,      // PD   [JM,IM] in
    const float* __restrict__ pdo,     // PDO  [JM,IM] in
    float* __restrict__ pdop)          // PDOP [JM,IM] out
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const size_t c2 = IDX2(i0, j0, IM);
    pdop[c2] = (pd[c2] + pdo[c2]) * 0.5f;
}

// flux helpers on one level: pointers already offset to level l0; index j0*IM+i0 (IDX2).
__device__ __forceinline__ float adv1_tx(const float* t1l, const float* pfxl, int i0, int j0, int IM) {
    return (t1l[IDX2(i0, j0, IM)] - t1l[IDX2(i0 - 1, j0, IM)]) * pfxl[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv1_ty(const float* t1l, const float* pfyl, int i0, int j0, int IM) {
    return (t1l[IDX2(i0, j0, IM)] - t1l[IDX2(i0, j0 - 1, IM)]) * pfyl[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv1_tne(const float* t1l, const float* pfnel, int i0, int j0, int IM) {
    return (t1l[IDX2(i0, j0, IM)] - t1l[IDX2(i0 - 1, j0 - 1, IM)]) * pfnel[IDX2(i0, j0, IM)];
}
__device__ __forceinline__ float adv1_tnw(const float* t1l, const float* pfnwl, int i0, int j0, int IM) {
    return (t1l[IDX2(i0 - 1, j0, IM)] - t1l[IDX2(i0, j0 - 1, IM)]) * pfnwl[IDX2(i0, j0, IM)];
}

// items [2]+[3]: fluxes and the standard-region second-order horizontal advection of T into TCT.
extern "C" __global__ void adv1_t_horiz_std(
    const int IM, const int JM, const int LM, const int lnsad,
    const float* __restrict__ t1,      // T1   [LM,JM,IM] in (from adv1_t_ab_col)
    const float* __restrict__ pfx,     // PFX  [LM,JM,IM] in
    const float* __restrict__ pfy,     // PFY  [LM,JM,IM] in
    const float* __restrict__ pfne,    // PFNE [LM,JM,IM] in
    const float* __restrict__ pfnw,    // PFNW [LM,JM,IM] in
    const float* __restrict__ fah,     // FAH  [JM] in, Fortran fah(j), grid factor
    const float* __restrict__ dsg2,    // DSG2 [LM] in
    const float* __restrict__ pdsg1,   // PDSG1[LM] in
    const float* __restrict__ pdop,    // PDOP [JM,IM] in, (pd+pdo)*0.5 from the adv1 section before
    const float* __restrict__ omgalf,  // OMGALF [LM,JM,IM] in
    float* __restrict__ tct,           // TCT  [LM,JM,IM] inout (added to, as in Fortran)
    const int JOFF, const int JMG)     // row slab (nmmb_common.cuh): domain tests on the global row
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const int jg = j0 + JOFF;
    if (NMMB_SLAB_EDGE(j0, JM, jg, JMG, 1)) return;   // reads j0-1..j0+1

    // item [3] standard region (Fortran ibeg..iend, jbeg..jend with ids=jds=1)
    const int ibeg = 1 + lnsad;
    const int iend = IM - 2 - lnsad;
    const int jbeg = 1 + lnsad;
    const int jend = JMG - 2 - lnsad;
    if (i0 < ibeg || i0 > iend || jg < jbeg || jg > jend) return;

    const size_t lo = (size_t)l0 * JM * IM;           // level offset
    const float* t1l  = t1  + lo;
    const float* pfxl = pfx + lo;
    const float* pfyl = pfy + lo;
    const float* pfnel = pfne + lo;
    const float* pfnwl = pfnw + lo;

    const int i1 = i0 + 1;
    const int j1 = j0 + 1;

    // item [2]: the eight flux terms of the expression, in Fortran's operand order
    const float tx_ij  = adv1_tx(t1l, pfxl, i0, j0, IM);
    const float tx_i1j = adv1_tx(t1l, pfxl, i1, j0, IM);
    const float ty_ij  = adv1_ty(t1l, pfyl, i0, j0, IM);
    const float ty_ij1 = adv1_ty(t1l, pfyl, i0, j1, IM);
    const float tne_i1j1 = adv1_tne(t1l, pfnel, i1, j1, IM);
    const float tne_ij   = adv1_tne(t1l, pfnel, i0, j0, IM);
    const float tnw_ij1  = adv1_tnw(t1l, pfnwl, i0, j1, IM);
    const float tnw_i1j  = adv1_tnw(t1l, pfnwl, i1, j0, IM);

    // item [3]: (((tx+tx+ty+ty)+(tne+tne+tnw+tnw)*0.25)*fah(j)) / (dsg2*pdop+pdsg1) + omgalf + tct
    const float fahp = fah[j0];
    const float sum_x = ((tx_ij + tx_i1j) + ty_ij) + ty_ij1;
    const float sum_n = ((tne_i1j1 + tne_ij) + tnw_ij1) + tnw_i1j;
    const float flux  = ((sum_x + sum_n * 0.25f) * fahp);
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    const float den = dsg2[l0] * pdop[IDX2(i0, j0, IM)] + pdsg1[l0];
    tct[c] = flux / den + omgalf[c] + tct[c];
}
