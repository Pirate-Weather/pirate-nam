// dyn_adv2_bndry.cu -- unit dyn-adv2-bndry: regional upstream advection of the tracer group along
// the four lateral boundary strips (ADV2, regional branch), NMMB module_DYNAMICS_ROUTINES.F90.
//
// Notice: nmm/module_DYNAMICS_ROUTINES.F90 carries no PROGRAM HISTORY LOG / author block in the
// NAM source at 83df4e13c1; the NAM notice in NOTICE applies (NOAA/NCEP/EMC, US Government work).
//
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:4609-4763 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (adv2, after the species loop, `if(.not.global.and.adv_upstream)`), one level l:
//   enhp = -dt*rdyh*0.25*idtadt                                   (once per call)
//   Four loops over the boundary strips, applied in this order (S, N, W, E); each cell of a strip
//   gets one update per strip that contains it (the Fortran loops accumulate with +tcs):
//     S : j = jts_b1 .. min(jte, jds+lnsad),          i = its_b1 .. ite_b1
//     N : j = max(jts, jde-lnsad) .. jte_b1,          i = its_b1 .. ite_b1
//     W : j = max(jts, jds+1+lnsad) .. min(jte, jde-1-lnsad),  i = its_b1 .. min(ite, ids+lnsad)
//     E : j = as W,                                    i = max(its, ide-lnsad) .. ite_b1
//   per cell (i,j), with emhp = -dt*rdxh(j)*0.25*idtadt:
//     pp = (up(i-1,j-1)+up(i,j-1)+up(i-1,j)+up(i,j))*emhp
//     qq = (vp(i-1,j-1)+vp(i,j-1)+vp(i-1,j)+vp(i,j))*enhp
//     iap = -1 if pp<=0 (pp = -pp), else +1;  jap = -1 if qq<=0 (qq = -qq), else +1
//     for ks = kss..kse:
//       tcs(i,j,l,ks) = (s1(i+iap,j)-s1(i,j))*pp + (s1(i,j+jap)-s1(i,j))*qq
//                     + (s1(i,j)-s1(i+iap,j)-s1(i,j+jap)+s1(i+iap,j+jap))*pp*qq + tcs(i,j,l,ks)
// Single process, full domain: ids = its = 1, ide = ite = IM, jds = jts = 1, jde = jte = JM (1-based),
// so its_b1 = 2, ite_b1 = IM-1, jts_b1 = 2, jte_b1 = JM-1 (module_DM_PARALLEL.F90:799-804). No halo
// exchange (CONTRACT section 1). The regional branch is entered only when adv_upstream is true
// (module_CONTROL.F90:264-269); the wrapper checks that and the global flag.
//
// Arrays (CONTRACT section 1): UP, VP [LM,JM,IM] (H-shaped, Fortran up/vp); RDXH [JM] (1-based j ->
// j0 = j-1); S1 = TRACERS_SQRT [NT,LM,JM,IM] (the intermediate species of the horizontal ADV2 step);
// TCS = TRACERS_TEND [NT,LM,JM,IM] (added to). kss, kse are 1-based species indices as in Fortran.
// Thread mapping: one thread per (i, j) point, level in blockIdx.z (grid_2d). Each thread writes only
// its own cell, so there are no races. No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_adv2_bndry_upstream(
    const int im, const int jm, const int lm,
    const int kss, const int kse, const int lnsad,
    const float dt, const float rdyh, const int idtadt,
    const float* __restrict__ rdxh,      // [JM]
    const float* __restrict__ up,        // [LM,JM,IM]
    const float* __restrict__ vp,        // [LM,JM,IM]
    const float* __restrict__ s1,        // [NT,LM,JM,IM]
    float* tcs,                          // [NT,LM,JM,IM], added to
    const int JOFF, const int JMG)       // row slab (nmmb_common.cuh): domain tests on the global row
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= im || j0 >= jm) return;
    const int jg = j0 + JOFF;
    if (NMMB_SLAB_EDGE(j0, jm, jg, JMG, 1)) return;   // reads j0-1..j0+1

    // 1-based Fortran indices and single-tile limits (module_DM_PARALLEL looplimits, its = ids etc.)
    const int i = i0 + 1;
    const int j = jg + 1;                // global Fortran j
    const int ids = 1, ide = im, jds = 1, jde = JMG;
    const int its = 1, ite = im, jts = 1, jte = JMG;
    const int its_b1 = max(its, ids + 1), ite_b1 = min(ite, ide - 1);
    const int jts_b1 = max(jts, jds + 1), jte_b1 = min(jte, jde - 1);

    // Membership in the four strips (module_DYNAMICS_ROUTINES.F90:4628, 4680, 4714, 4748 loop bounds).
    const bool in_ib = (i >= its_b1 && i <= ite_b1);
    const bool inS = in_ib && (j >= jts_b1 && j <= min(jte, jds + lnsad));
    const bool inN = in_ib && (j >= max(jts, jde - lnsad) && j <= jte_b1);
    const bool in_jwe = (j >= max(jts, jds + 1 + lnsad) && j <= min(jte, jde - 1 - lnsad));
    const bool inW = in_jwe && (i >= its_b1 && i <= min(ite, ids + lnsad));
    const bool inE = in_jwe && (i >= max(its, ide - lnsad) && i <= ite_b1);
    const int nseg = (int)inS + (int)inN + (int)inW + (int)inE;
    if (nseg == 0) return;

    // enhp is the same for every strip and cell (computed once in Fortran; same operation order here).
    const float enhp = -dt * rdyh * 0.25f * (float)idtadt;
    const float emhp = -dt * rdxh[j0] * 0.25f * (float)idtadt;

    // Upstream velocity-weighted factors and the upwind direction (Fortran pp, qq, iap, jap).
    float pp = (up[IDX3(i0 - 1, j0 - 1, l0, im, jm)] + up[IDX3(i0, j0 - 1, l0, im, jm)]
              + up[IDX3(i0 - 1, j0, l0, im, jm)] + up[IDX3(i0, j0, l0, im, jm)]) * emhp;
    float qq = (vp[IDX3(i0 - 1, j0 - 1, l0, im, jm)] + vp[IDX3(i0, j0 - 1, l0, im, jm)]
              + vp[IDX3(i0 - 1, j0, l0, im, jm)] + vp[IDX3(i0, j0, l0, im, jm)]) * enhp;
    int iap, jap;
    if (pp <= 0.0f) { iap = -1; pp = -pp; } else { iap = 1; }
    if (qq <= 0.0f) { jap = -1; qq = -qq; } else { jap = 1; }

    // Fortran applies the update once per strip that holds the cell; the strips do not overlap
    // for NAM sizes, so this loops nseg (0..1 in practice) times in the Fortran S, N, W, E order.
    for (int seg = 0; seg < nseg; ++seg) {
        for (int ks = kss; ks <= kse; ++ks) {
            const int n0 = ks - 1;
            const float s_c  = s1[IDX4(i0,       j0,       l0, n0, im, jm, lm)];
            const float s_x  = s1[IDX4(i0 + iap, j0,       l0, n0, im, jm, lm)];
            const float s_y  = s1[IDX4(i0,       j0 + jap, l0, n0, im, jm, lm)];
            const float s_xy = s1[IDX4(i0 + iap, j0 + jap, l0, n0, im, jm, lm)];
            const size_t t = IDX4(i0, j0, l0, n0, im, jm, lm);
            tcs[t] = (((s_x - s_c) * pp + (s_y - s_c) * qq)
                      + (s_c - s_x - s_y + s_xy) * pp * qq)
                     + tcs[t];
        }
    }
}
