// nam-gpu unit dyn-cdwdt-filter: lateral spatial filter of DWDT, the cdwdt clamp and the
// time-smoothing blend, and DWDT = 1 on the regional boundary H rows.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3773-3899 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine cdwdt, lines 3773-3899), with g = gravity, wad, wp, epsn, epsp as declared
// at lines 3296-3315 (wad = 0, wp = 0, epsn = -9.80, epsp = +9.80):
//   [F1] if(wad.gt.0): for l=1..lm, b1 range (its_b1..ite_b1 x jts_b1..jte_b1):
//          ww(i,j)     = dwdt(i,j-1,l)+dwdt(i-1,j,l)+dwdt(i+1,j,l)+dwdt(i,j+1,l)-dwdt(i,j,l)*4.0
//          dwdt(i,j,l) = ww(i,j)*wad+dwdt(i,j,l)
//        ww is a per-level 2-D temporary read from the pre-update level, so the whole level's ww
//        is formed first (kernel dwdt_filter_ww, scratch CDWDT_WW) and then applied
//        (kernel dwdt_filter_apply). Level l reads only level l, so this equals the Fortran result.
//   [F2] if(global): swaphn/polehn. Global-only, not called in the regional configuration: skipped.
//   [F3] rg = 1./g; all i,j,l (full domain, its..ite x jts..jte):
//          dwdtp = dwdt; if(dwdtp.lt.epsn) dwdtp=epsn; if(dwdtp.gt.epsp) dwdtp=epsp;
//          dwdt(i,j,l) = (dwdtp*rg+1.)*(1.-wp)+pdwdt(i,j,l)*wp
//        The max/min tracking (dwdtmx, imx, ... kn, kp) only feeds the 1100/1200 format labels; the
//        routine has no WRITE of them, so it has no effect on state and is not ported.
//   [F4] if(.not.global) (regional), for every level, after F3:
//          s_bdy: j=jts..jts+1 (1-based 1,2)     all i  -> dwdt = 1.
//          n_bdy: j=jte-1..jte (JM-1, JM)        all i  -> dwdt = 1.
//          w_bdy: i=its..its+1 (1,2)             all j  -> dwdt = 1.
//          e_bdy: i=ite-1..ite (IM-1, IM)        all j  -> dwdt = 1.
//        The flags are true for the one-card, single-process layout (module_CONTROL s_bdy etc.:
//        its==ids, jts==jds, ...), passed in as 1.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, single process, ids=1 ide=IM jds=1 jde=JM):
//   its_b1 = 2 .. ite_b1 = IM-1 and jts_b1 = 2 .. jte_b1 = JM-1 (1-based)
//   -> zero-based interior i0 in [1, IM-2], j0 in [1, JM-2].
//
// Layout: DWDT, PDWDT, WW [LM,JM,IM] (CONTRACT section 1); l0 = 0 is the model top.
// F3 and F4 are pointwise in (i,j,l). The Fortran runs the boundary set after the clamp, so the
// kernel computes clamp+blend and then the boundary points take 1.0f: the same result.
// F1 is split into two launches because its stencil reads neighbours that the apply would
// overwrite. No host sync, no device allocation in the kernels.

#include "nmmb_common.cuh"

#define CDWDT_EPSN (-9.80f)   /* module_DYNAMICS_ROUTINES epsn = -epsfc, epsfc = 9.80 (line 3302) */
#define CDWDT_EPSP ( 9.80f)   /* epsp = epsfc (line 3303) */

// F1, stencil: ww(i,j) for the interior H points of level l0. Only the interior is written.
extern "C" __global__ void dwdt_filter_ww(
    const int IM, const int JM, const int LM,
    const float* __restrict__ dwdt, float* __restrict__ ww)
{
    POINT_IJL
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2 || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    ww[c] = dwdt[IDX3(i0, j0 - 1, l0, IM, JM)] + dwdt[IDX3(i0 - 1, j0, l0, IM, JM)]
          + dwdt[IDX3(i0 + 1, j0, l0, IM, JM)] + dwdt[IDX3(i0, j0 + 1, l0, IM, JM)]
          - dwdt[c] * 4.0f;
}

// F1, apply: dwdt = ww*wad + dwdt on the interior (only launched when wad > 0).
extern "C" __global__ void dwdt_filter_apply(
    const int IM, const int JM, const int LM, const float wad,
    const float* __restrict__ ww, float* __restrict__ dwdt)
{
    POINT_IJL
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2 || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);
    dwdt[c] = ww[c] * wad + dwdt[c];
}

// F3 + F4: clamp to [epsn, epsp], blend with the previous correction factor PDWDT (wp), then the
// regional boundary H rows and columns are set to 1.0. rg = 1./g, computed on the host in single.
extern "C" __global__ void cdwdt_clamp_bdy(
    const int IM, const int JM, const int LM,
    const float rg, const float wp,
    const int s_bdy, const int n_bdy, const int w_bdy, const int e_bdy,
    float* __restrict__ dwdt, const float* __restrict__ pdwdt)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t c = IDX3(i0, j0, l0, IM, JM);

    float dwdtp = dwdt[c];
    if (dwdtp < CDWDT_EPSN) dwdtp = CDWDT_EPSN;
    if (dwdtp > CDWDT_EPSP) dwdtp = CDWDT_EPSP;
    const float v = (dwdtp * rg + 1.0f) * (1.0f - wp) + pdwdt[c] * wp;

    const int on_bdy = (s_bdy && j0 < 2) || (n_bdy && j0 >= JM - 2)
                    || (w_bdy && i0 < 2) || (e_bdy && i0 >= IM - 2);
    dwdt[c] = on_bdy ? 1.0f : v;
}
