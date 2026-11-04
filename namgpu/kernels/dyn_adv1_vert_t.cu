// WOOF-NAM unit dyn-adv1-vert-t: ADV1 setup and Crank-Nicolson (uncentred w1/w2) vertical advection of T.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:1233-1538 (nam.v4.2.11, 83df4e13c1).
//   This unit covers the vertical Crank-Nicolson block of subroutine adv1 only: the pdop setup
//   (Fortran lines 1292-1297) and the rcmt/crt/rstt forward sweep and back substitution that write
//   tct (Fortran lines 1298-1357, the non-SMP path). The horizontal T advection that follows
//   (vertical_loop1 and the tct accumulation at Fortran lines 1561-1693) belongs to the horizontal
//   ADV1 units and is not in this kernel.
//
// Fortran (subroutine adv1, lm = LM, dt = DT, dsg2/pdsg1 per level, psgdt = PSGDT(1:lm-1)):
//   pdop(i,j)   = (pd(i,j) + pdo(i,j)) * 0.5                         (computed per column here)
//   dtq         = dt * 0.25
//   [F1] l=1:   pvvlo = psgdt(1)*dtq; vvlo = pvvlo/(dsg2(1)*pdop+pdsg1(1))
//               cmt = -vvlo*w2 + 1; rcmt(1) = 1/cmt; crt(1) = vvlo*w2
//               rstt(1) = (-vvlo*w1)*(t(2)-t(1)) + t(1)
//   [F2] l=2..lm-1: rdp = 1/(dsg2(l)*pdop+pdsg1(l)); pvvup = pvvlo; pvvlo = psgdt(l)*dtq
//               vvup = pvvup*rdp; vvlo = pvvlo*rdp
//               cf = -vvup*w2*rcmt(l-1); cmt = -crt(l-1)*cf + ((vvup-vvlo)*w2 + 1)
//               rcmt(l) = 1/cmt; crt(l) = vvlo*w2
//               rstt(l) = -rstt(l-1)*cf + t(l) - ((t(l)-t(l-1))*vvup + (t(l+1)-t(l))*vvlo)*w1
//   [F3] l=lm:  pvvup = pvvlo; vvup = pvvup/(dsg2(lm)*pdop+pdsg1(lm))
//               cf = -vvup*w2*rcmt(lm-1); cmt = -crt(lm-1)*cf + (vvup*w2 + 1)
//               rcmt(lm) = 1/cmt; crt(lm) = 0
//               rstt(lm) = -rstt(lm-1)*cf + t(lm) - (t(lm)-t(lm-1))*vvup*w1
//               tct(lm) = rstt(lm)*rcmt(lm) - t(lm)
//   [F4] l=lm-1..1 (back substitution):
//               tct(l) = (-crt(l)*(t(l+1)+tct(l+1)) + rstt(l))*rcmt(l) - t(l)
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, regional, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = max(its, ids+1) = 2 .. ite_b1 = min(ite, ide-1) = IM-1   (1-based) -> i0 in [1, IM-2]
//   jts_b1 = 2 .. jte_b1 = JM-1 (1-based)                                       -> j0 in [1, JM-2]
//   Boundary rows and columns are not written by the Fortran; this kernel does not write them.
//   The MPI/OpenMP looplimits split (ENABLE_SMP) is dropped: one thread per column.
//
// Scratch (CONTRACT section 1): rcmt and crt are [LM,JM,IM] scratch the wrapper allocates once
// (ADV1_RCMT, ADV1_CRT). The Fortran rstt is stored in the output tct array during F1/F2/F3: tct(l)
// holds rstt(l) until F4 reads it and overwrites it, level by level from the top. This avoids a
// third 191 MB scratch array with the same arithmetic.
//
// Constants: w1 = 0.9 (crank-nicholson uncentering), w2 = 2 - w1 (module_DYNAMICS_ROUTINES parameters),
// single precision as kfpt=single. The Fortran `if(abs(cmt).lt.epscm)` floor is commented out in the
// source and is not ported.
// Fast numerics: FMA contraction on (NVRTC --fmad=true), IEEE division.
//
// Ordering note: the horizontal part of ADV1 later accumulates into tct (`tct = ... + omgalf + tct`),
// so this kernel must run before the horizontal T units in the same step.

#include "nmmb_common.cuh"

extern "C" __global__ void adv1_vert_t(
    const int IM, const int JM, const int LM,
    const float dt,
    const float* __restrict__ t,       // T         [LM,JM,IM]    in   (Fortran t(i,j,l))
    const float* __restrict__ psgdt,   // PSGDT     [LM+1,JM,IM]  in   levels 0..LM-2 (Fortran psgdt(1:lm-1))
    const float* __restrict__ pd,      // PD        [JM,IM]       in
    const float* __restrict__ pdo,     // PDO       [JM,IM]       in
    const float* __restrict__ dsg2,    // DSG2      [LM]          in   delta sigmas
    const float* __restrict__ pdsg1,   // PDSG1     [LM]          in   delta pressures
    float* __restrict__ rcmt,          // ADV1_RCMT [LM,JM,IM]    scratch (1/cmt per level)
    float* __restrict__ crt,           // ADV1_CRT  [LM,JM,IM]    scratch (vvlo*w2 per level)
    float* __restrict__ tct)           // TCT       [LM,JM,IM]    out  (vertical T tendency)
{
    COLUMN_IJ(IM, JM)

    // b1 range: boundary rows and columns are not written by the Fortran
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const float w1  = 0.9f;                // crank-nicholson uncentering
    const float w2  = 2.0f - w1;           // = 2.-w1
    const float dtq = dt * 0.25f;          // dtq = dt*0.25

    const float pdop = (pd[IDX2(i0, j0, IM)] + pdo[IDX2(i0, j0, IM)]) * 0.5f;

    // ---- [F1] level 1 (Fortran l=1 -> l0=0) -------------------------------------------------
    float pvvlo = psgdt[IDX3(i0, j0, 0, IM, JM)] * dtq;
    float vvlo  = pvvlo / (dsg2[0] * pdop + pdsg1[0]);
    float cmt   = -vvlo * w2 + 1.0f;
    rcmt[IDX3(i0, j0, 0, IM, JM)] = 1.0f / cmt;
    crt[IDX3(i0, j0, 0, IM, JM)]  = vvlo * w2;
    const float t_1 = t[IDX3(i0, j0, 0, IM, JM)];
    const float t_2 = t[IDX3(i0, j0, 1, IM, JM)];
    float rstt_prev = (-vvlo * w1) * (t_2 - t_1) + t_1;
    tct[IDX3(i0, j0, 0, IM, JM)] = rstt_prev;      // holds rstt(1) until F4

    // ---- [F2] levels 2..lm-1 (l0 = 1..LM-2) -------------------------------------------------
    for (int l0 = 1; l0 < LM - 1; ++l0) {
        const float rdp   = 1.0f / (dsg2[l0] * pdop + pdsg1[l0]);
        const float pvvup = pvvlo;
        pvvlo = psgdt[IDX3(i0, j0, l0, IM, JM)] * dtq;
        const float vvup = pvvup * rdp;
        vvlo = pvvlo * rdp;
        const float cf = -vvup * w2 * rcmt[IDX3(i0, j0, l0 - 1, IM, JM)];
        cmt = -crt[IDX3(i0, j0, l0 - 1, IM, JM)] * cf + ((vvup - vvlo) * w2 + 1.0f);
        rcmt[IDX3(i0, j0, l0, IM, JM)] = 1.0f / cmt;
        crt[IDX3(i0, j0, l0, IM, JM)]  = vvlo * w2;

        const float t_bel = t[IDX3(i0, j0, l0 - 1, IM, JM)];
        const float t_cur = t[IDX3(i0, j0, l0, IM, JM)];
        const float t_abv = t[IDX3(i0, j0, l0 + 1, IM, JM)];
        const float rstt = -rstt_prev * cf + t_cur
                         - ((t_cur - t_bel) * vvup + (t_abv - t_cur) * vvlo) * w1;
        tct[IDX3(i0, j0, l0, IM, JM)] = rstt;      // holds rstt(l) until F4
        rstt_prev = rstt;
    }

    // ---- [F3] level lm (l0 = LM-1) ----------------------------------------------------------
    {
        const int lt = LM - 1;
        const float pvvup = pvvlo;
        const float vvup  = pvvup / (dsg2[lt] * pdop + pdsg1[lt]);
        const float cf    = -vvup * w2 * rcmt[IDX3(i0, j0, lt - 1, IM, JM)];
        cmt = -crt[IDX3(i0, j0, lt - 1, IM, JM)] * cf + (vvup * w2 + 1.0f);
        rcmt[IDX3(i0, j0, lt, IM, JM)] = 1.0f / cmt;
        crt[IDX3(i0, j0, lt, IM, JM)]  = 0.0f;

        const float t_top = t[IDX3(i0, j0, lt, IM, JM)];
        const float t_bel = t[IDX3(i0, j0, lt - 1, IM, JM)];
        const float rstt_lm = -rstt_prev * cf + t_top - (t_top - t_bel) * vvup * w1;
        tct[IDX3(i0, j0, lt, IM, JM)] = rstt_lm * rcmt[IDX3(i0, j0, lt, IM, JM)] - t_top;
    }

    // ---- [F4] back substitution, levels lm-1..1 (l0 = LM-2..0) ------------------------------
    for (int l0 = LM - 2; l0 >= 0; --l0) {
        const size_t c    = IDX3(i0, j0, l0, IM, JM);
        const float rstt_l = tct[c];               // rstt(l), stored in F1/F2
        tct[c] = (-crt[c] * (t[IDX3(i0, j0, l0 + 1, IM, JM)] + tct[IDX3(i0, j0, l0 + 1, IM, JM)])
                  + rstt_l) * rcmt[c] - t[c];
    }
}
