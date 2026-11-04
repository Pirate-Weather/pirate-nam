// nam-gpu unit dyn-vsound: implicit vertically propagating sound waves (one tridiagonal per column).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3904-4234 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine vsound; hydro = false, ntsd >= 2, global = false, single process):
//   [A] l=1 init: pone = pnp1 = pstr = pt  (its_b1..ite_b1 x jts_b1..jte_b1)
//   [B] l=2..lm+1: dppl, rdpp(l-1), dpstr, pstr(l), pp1, pone(l)=pint(l) (old), dp (wght=0.35),
//       pnp1(l), tfc, fcc, cofl, coff(l-1), dfrh(l)
//   [C] l=2..lm:   b1, b2, b3, c0 (tridiagonal coefficients)
//   [D] b2(lm) += b3(lm); forward elimination l=3..lm (tmp, b2, c0)
//   [E] chi(1)=0, chi(lm)=c0(lm)/b2(lm), chi(lm+1)=chi(lm); back substitution l=lm-1..2
//   [F] l=1..lm+1: pnp1 = chi + pstr; pint = pnp1
//   [G] l=1..lm:   t, delp, w (w = (delp-dwdt)*gdt + w), w_tot = w + baro, dwdt = delp
//   [H] not global: dwdt = 1 on H points of the regional boundary strips (s, n, w, e)
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2 .. ite_b1 = IM-1 and jts_b1 = 2 .. jte_b1 = JM-1 (1-based), so the solve runs for
//   zero-based i0 in [1, IM-2], j0 in [1, JM-2]. The boundary override [H] (H points, its..ite x
//   jts..jte) runs for every column of the strips, after the solve in the same thread (Fortran order).
//
// Scratch: the Fortran per-column work arrays (pone, pstr, pnp1, dfrh, chi, coff, rdpp, b1, b2,
// b3, c0) are per-thread locals indexed by the Fortran level l (index 0 unused), so the l-loops keep
// the Fortran order. VS_LMAX bounds LM; the wrapper refuses larger LM.
//
// Outputs: PINT (LM+1 levels), W, T, DWDT (inout) and W_TOT (out). The Fortran writes DWDT and W_TOT
// too, so both are written here (CONTRACT section 2 lists PINT/W/T; the extra two are Fortran writes).
//
// Layout: PINT [LM+1,JM,IM]; W, T, DWDT, W_TOT, CW, Q, RTOP [LM,JM,IM]; PD, BARO [JM,IM];
// DSG2, PDSG1 [LM] (CONTRACT section 1). Fortran level l (1-based) is memory index l-1.
//
// Constants: r = R_D, g = G, cp = CP from NMMB module_CONSTANTS (nmmb_common.cuh).
// Fast numerics (fmad on): the Fortran expression order is kept, FMA contraction left to the compiler.

#include "nmmb_common.cuh"

#define VS_LMAX 64
#define VS_WGHT 0.35f

// Fortran 1-based level access into a [LM,JM,IM] array, and into a [LM] vector.
#define VS_F3(arr, l)  (arr)[IDX3(i0, j0, (l) - 1, IM, JM)]
#define VS_F1(arr, l)  (arr)[(l) - 1]

extern "C" __global__ void vsound_col(
    const int IM, const int JM, const int LM,
    const float dt,                       // DT   (dynamics step, s)
    const float pt,                       // PT   (pressure at top, Pa)
    const int s_bdy, const int n_bdy,     // BDY flags: task on south / north domain boundary
    const int w_bdy, const int e_bdy,     // BDY flags: task on west / east domain boundary
    const int global_flag,                // 1 = global (boundary override skipped), 0 = regional
    const float* __restrict__ dsg2,       // DSG2  [LM]          in
    const float* __restrict__ pdsg1,      // PDSG1 [LM]          in
    const float* __restrict__ pd,         // PD    [JM,IM]       in
    const float* __restrict__ cw,         // CW    [LM,JM,IM]    in
    const float* __restrict__ q,          // Q     [LM,JM,IM]    in
    const float* __restrict__ rtop,       // RTOP  [LM,JM,IM]    in (rt/p)
    const float* __restrict__ baro,       // BARO  [JM,IM]       in
    float* __restrict__ dwdt,             // DWDT  [LM,JM,IM]    inout
    float* __restrict__ t,                // T     [LM,JM,IM]    inout
    float* __restrict__ w,                // W     [LM,JM,IM]    inout
    float* __restrict__ w_tot,            // W_TOT [LM,JM,IM]    out
    float* __restrict__ pint)             // PINT  [LM+1,JM,IM]  inout
{
    COLUMN_IJ(IM, JM)

    const float r = NMMB_R_D;
    const float g = NMMB_G;
    const float cp = NMMB_CP;
    const float cappa = r / cp;
    const float gdt = g * dt;
    const float gdt2 = gdt * gdt;
    const float ffc = -r * 0.25f / gdt2;
    const float rcph = 0.5f / cp;

    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);

    if (interior) {
        const size_t c2 = IDX2(i0, j0, IM);
        const float p = pd[c2];

        // Column work arrays, indexed by Fortran level (index 0 unused).
        float pone[VS_LMAX + 2], pstr[VS_LMAX + 2], pnp1[VS_LMAX + 2], dfrh[VS_LMAX + 2];
        float chi[VS_LMAX + 2], coff[VS_LMAX + 2], rdpp[VS_LMAX + 2];
        float b1[VS_LMAX + 2], b2[VS_LMAX + 2], b3[VS_LMAX + 2], c0[VS_LMAX + 2];

        // [A]
        pone[1] = pt;
        pstr[1] = pt;
        pnp1[1] = pt;

        // [B] l = 2 .. lm+1
        for (int l = 2; l <= LM + 1; ++l) {
            const float dppl = VS_F1(dsg2, l - 1) * p + VS_F1(pdsg1, l - 1);
            rdpp[l - 1] = 1.0f / dppl;
            const float dpstr = VS_F3(dwdt, l - 1) * dppl;
            pstr[l] = pstr[l - 1] + dpstr;
            const float pp1 = pnp1[l - 1] + dpstr;
            pone[l] = VS_F3(pint, l);
            const float dp = (pp1 - pone[l]) * VS_WGHT;
            pnp1[l] = pone[l] + dp;
            const float tfc = VS_F3(q, l - 1) * 0.608f - VS_F3(cw, l - 1) + 1.0f;
            const float fcc = (1.0f - cappa * tfc) * tfc * ffc;
            const float avg = (pnp1[l - 1] + pnp1[l]) * 0.5f;
            const float cofl = VS_F3(t, l - 1) * dppl * fcc / (avg * avg);
            coff[l - 1] = cofl;
            dfrh[l] = (pstr[l - 1] + pstr[l] - pone[l - 1] - pone[l]) * cofl;
        }

        // [C] l = 2 .. lm
        for (int l = 2; l <= LM; ++l) {
            const float rdpdn = rdpp[l];
            const float rdpup = rdpp[l - 1];
            b1[l] = coff[l - 1] + rdpup;
            b2[l] = coff[l - 1] + coff[l] - rdpup - rdpdn;
            b3[l] = coff[l] + rdpdn;
            c0[l] = -dfrh[l] - dfrh[l + 1];
        }
        // [D]
        b2[LM] = b2[LM] + b3[LM];
        for (int l = 3; l <= LM; ++l) {
            const float tmp = -b1[l] / b2[l - 1];
            b2[l] = b3[l - 1] * tmp + b2[l];
            c0[l] = c0[l - 1] * tmp + c0[l];
        }
        // [E]
        chi[1] = 0.0f;
        chi[LM] = c0[LM] / b2[LM];
        chi[LM + 1] = chi[LM];
        for (int l = LM - 1; l >= 2; --l) {
            chi[l] = (-b3[l] * chi[l + 1] + c0[l]) / b2[l];
        }
        // [F]
        for (int l = 1; l <= LM + 1; ++l) {
            pnp1[l] = chi[l] + pstr[l];
            pint[IDX3(i0, j0, l - 1, IM, JM)] = pnp1[l];   // PINT has LM+1 levels
        }
        // [G]
        float dptu = 0.0f;
        for (int l = 1; l <= LM; ++l) {
            const float dptl = pnp1[l + 1] - pone[l + 1];
            VS_F3(t, l) = (dptu + dptl) * VS_F3(rtop, l) * rcph + VS_F3(t, l);
            const float delp = (pnp1[l + 1] - pnp1[l]) * rdpp[l];
            VS_F3(w, l) = (delp - VS_F3(dwdt, l)) * gdt + VS_F3(w, l);
            VS_F3(w_tot, l) = VS_F3(w, l) + baro[c2];
            VS_F3(dwdt, l) = delp;
            dptu = dptl;
        }
    }

    // [H] regional boundary strips: dwdt = 1 on H points (after the solve, as in Fortran).
    if (!global_flag) {
        const bool s_strip = (s_bdy != 0) && (j0 <= 1);
        const bool n_strip = (n_bdy != 0) && (j0 >= JM - 2);
        const bool w_strip = (w_bdy != 0) && (i0 <= 1);
        const bool e_strip = (e_bdy != 0) && (i0 >= IM - 2);
        if (s_strip || n_strip || w_strip || e_strip) {
            for (int l = 1; l <= LM; ++l) {
                VS_F3(dwdt, l) = 1.0f;
            }
        }
    }
}
