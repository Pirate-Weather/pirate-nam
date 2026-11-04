// WOOF-NAM unit dyn-adv1-bndry-t: regional upstream advection of T along the lateral boundary strips.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:1571-1705 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine adv1, inside vertical_loop1 do l=1,lm, regional branch, adv_upstream true):
//   enhp = -dt*rdyh*0.25
//   [S] j = jts_b1 .. min(jte, jds+lnsad),        i = its_b1 .. ite_b1   (southern rows)
//   [N] j = max(jts, jde-lnsad) .. jte_b1,        i = its_b1 .. ite_b1   (northern rows)
//   [W] j = max(jts, jds+1+lnsad) .. min(jte, jde-1-lnsad),
//                                                  i = its_b1 .. min(ite, ids+lnsad)   (western rows)
//   [E] j = max(jts, jds+1+lnsad) .. min(jte, jde-1-lnsad),
//                                                  i = max(its, ide-lnsad) .. ite_b1  (eastern rows)
//   each (i,j) in those ranges, with emhp = -dt*rdxh(j)*0.25:
//     pp = (u(i-1,j-1)+u(i,j-1)+u(i-1,j)+u(i,j))*emhp
//     qq = (v(i-1,j-1)+v(i,j-1)+v(i-1,j)+v(i,j))*enhp
//     iap = -1 if pp<=0 (pp=-pp) else +1;  jap = -1 if qq<=0 (qq=-qq) else +1
//     tct = (t(i+iap,j)-t(i,j))*pp + (t(i,j+jap)-t(i,j))*qq
//           + (t(i,j)-t(i+iap,j)-t(i,j+jap)+t(i+iap,j+jap))*pp*qq + omgalf(i,j) + tct
//
// Index ranges: single process, regional, ids=1 ide=IM jds=1 jde=JM its=1 ite=IM jts=1 jte=JM
// (module_DM_PARALLEL looplimits; its_b1 = 2, ite_b1 = IM-1, jts_b1 = 2, jte_b1 = JM-1, 1-based).
// lnsad = lnsh + 2 (module_CONTROL.F90:260, regional) = 7 with lnsh = 5 (nam_configfile).
// adv_upstream is true on this domain (module_CONTROL.F90:264-270); the wrapper enforces the guard.
// The four strips are disjoint for IM, JM > 2*lnsad+1 (wrapper checks), so one thread per (i,j,l)
// that falls in any strip reproduces the Fortran exactly; each point is written once.
//
// Layout (CONTRACT.md section 1): X[l0, j0, i0] with IDX3; RDXH [JM] indexed by j0 (Fortran rdxh(j)).
// Thread mapping: point kernel, one thread per (i, j), level in blockIdx.z (grid_2d).
// No line of the Fortran is skipped; the OpenMP directives (ENABLE_SMP) are not ported (no effect on state).

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_adv1_bndry_t(
    const int IM, const int JM, const int LM,
    const int lnsad,
    const float dt,
    const float rdyh,                       // RDYH scalar (1/dyh)
    const float* __restrict__ rdxh,         // RDXH [JM]          Fortran rdxh(j)
    const float* __restrict__ u,            // U    [LM,JM,IM]    in
    const float* __restrict__ v,            // V    [LM,JM,IM]    in
    const float* __restrict__ t,            // T    [LM,JM,IM]    in
    const float* __restrict__ omgalf,       // OMGALF [LM,JM,IM]  in
    float* __restrict__ tct,                // TCT  [LM,JM,IM]    inout (in-place accumulate)
    const int JOFF, const int JMG)          // row slab (nmmb_common.cuh): domain tests on the global row
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const int jg = j0 + JOFF;
    if (NMMB_SLAB_EDGE(j0, JM, jg, JMG, 1)) return;   // reads j0-1..j0+1

    // 1-based Fortran indices and the single-process regional ranges (see header).
    const int i = i0 + 1;
    const int j = jg + 1;                   // global Fortran j
    const int jsouth_hi = (1 + lnsad < JMG) ? (1 + lnsad) : JMG;          // min(jte, jds+lnsad)
    const int jnorth_lo = (JMG - lnsad > 1) ? (JMG - lnsad) : 1;         // max(jts, jde-lnsad)
    const int jside_lo  = (2 + lnsad > 1) ? (2 + lnsad) : 1;            // max(jts, jds+1+lnsad)
    const int jside_hi  = (JMG - 1 - lnsad < JMG) ? (JMG - 1 - lnsad) : JMG; // min(jte, jde-1-lnsad)
    const int iwest_hi  = (1 + lnsad < IM) ? (1 + lnsad) : IM;          // min(ite, ids+lnsad)
    const int ieast_lo  = (IM - lnsad > 1) ? (IM - lnsad) : 1;          // max(its, ide-lnsad)

    const bool south = (j >= 2 && j <= jsouth_hi) && (i >= 2 && i <= IM - 1);
    const bool north = (j >= jnorth_lo && j <= JMG - 1) && (i >= 2 && i <= IM - 1);
    const bool west  = (j >= jside_lo && j <= jside_hi) && (i >= 2 && i <= iwest_hi);
    const bool east  = (j >= jside_lo && j <= jside_hi) && (i >= ieast_lo && i <= IM - 1);
    if (!(south || north || west || east)) return;

    // emhp = -dt*rdxh(j)*0.25 ;  enhp = -dt*rdyh*0.25   (Fortran evaluation, single precision)
    const float emhp = -dt * rdxh[j0] * 0.25f;
    const float enhp = -dt * rdyh * 0.25f;

    const size_t c = IDX3(i0, j0, l0, IM, JM);

    float pp = (u[IDX3(i0 - 1, j0 - 1, l0, IM, JM)] + u[IDX3(i0, j0 - 1, l0, IM, JM)]
              + u[IDX3(i0 - 1, j0, l0, IM, JM)] + u[c]) * emhp;
    float qq = (v[IDX3(i0 - 1, j0 - 1, l0, IM, JM)] + v[IDX3(i0, j0 - 1, l0, IM, JM)]
              + v[IDX3(i0 - 1, j0, l0, IM, JM)] + v[c]) * enhp;

    int iap, jap;
    if (pp <= 0.0f) { iap = -1; pp = -pp; } else { iap = 1; }
    if (qq <= 0.0f) { jap = -1; qq = -qq; } else { jap = 1; }

    const float tc  = t[c];
    const float tip = t[IDX3(i0 + iap, j0, l0, IM, JM)];
    const float tjp = t[IDX3(i0, j0 + jap, l0, IM, JM)];
    const float tij = t[IDX3(i0 + iap, j0 + jap, l0, IM, JM)];

    tct[c] = (tip - tc) * pp
           + (tjp - tc) * qq
           + (tc - tip - tjp + tij) * pp * qq
           + omgalf[c]
           + tct[c];
}
