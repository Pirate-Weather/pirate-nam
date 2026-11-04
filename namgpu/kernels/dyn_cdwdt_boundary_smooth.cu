// dyn_cdwdt_boundary_smooth.cu -- CDWDT setup, no w diffusion in the backward DFI step, and
// smoothing of w on H points along the lateral boundaries.
//
// NOTICE: NAM source file nmm/module_DYNAMICS_ROUTINES.F90 (NOAA/NCEP/EMC, US Government work,
// no licence file). This routine has no author/history block of its own; see NOTICE at the repo root.
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:3278-3532 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine CDWDT, lines 3278-3532 only (the local dw/dt derivative after 3532 belongs to a
// later unit). Arguments used in this range: global, hydro, restart, lm, ntsd, dt (only for diffw,
// which is used after 3532, not here), dsg2, pdsg1, pd, pdo (not read), dwdt, pdwdt, w, pint.
//
// Three kernels, launched by the wrapper:
//   cdwdt_setup        (Fortran 3388-3413, the early-return branch):  one thread per column.
//       dwdt = 1, pdwdt = 1, pint(l+1) = dsg2(l)*pd + pdsg1(l) + pint(l), for l = 1..LM in order.
//       The l-recurrence on pint runs inside the thread, in the Fortran order.
//   cdwdt_smooth_ww    (Fortran 3424-3510, phase 1 of one ks iteration): stencil into ww on the
//   cdwdt_smooth_copy  (phase 2, the copy back into w): on the four lateral strips.
//       The Fortran computes every ww value for the level before copying any w, so the stencil
//       reads the un-smoothed w. Two launches keep that order across threads.
//
// Strip membership (zero-based, single full domain, module_DM_PARALLEL looplimits, ids=1 ide=IM):
//   south  j0 = 1,       i0 in [1, IM-2]   (Fortran j = jts+1..lnsnh, i = its_b1..ite_b1)
//   north  j0 = JM-2,    i0 in [1, IM-2]   (Fortran j = jte-lnsnh+1..jte-1, same i range)
//   west   i0 = 1,       j0 in [2, JM-3]   (Fortran i = its+1..its-1+lnsnh, j = max(jts,jds+lnsnh)..min(jte,jde-lnsnh))
//   east   i0 = IM-2,    j0 in [2, JM-3]   (Fortran i = ite-lnsnh+1..ite-1, same j range)
// The copy-back ranges are the same sets, so one membership test serves both phases.
//
// Stencil (Fortran order kept): ww = (w(i,j-1)+w(i-1,j)+w(i+1,j)+w(i,j+1))*wa + w(i,j)*wb,
// wa = 0.125, wb = 0.5 (module_DYNAMICS_ROUTINES cdwdt, kfpt single).
//
// Dead code in the configured run: nsmud is a PARAMETER equal to 0 in the Fortran, so the
// smoothing loop `do ks=1,nsmud` never runs. The wrapper keeps NSMUD = 0 and launches nothing for
// the smoothing; the kernels stay so the path is complete if nsmud is ever changed.
// wad = 0 and wp = 0 (parameters): the lateral smoothing weight and the time smoothing weight have
// no term here.
//
// Flags: bit0 south, bit1 north, bit2 west, bit3 east. Each is the Fortran s_bdy/n_bdy/w_bdy/e_bdy
// logical, true for every side on one card (the whole domain is one process).
//
// Not-mapped Fortran (no term in this kernel):
//   - OpenMP ENABLE_SMP directives (host threading only);
//   - the `global` test is handled by the wrapper (`.not.global` gate) since it is a host logical;
//   - the diffw logical (3424-3427) is set from dt but first used after 3532.
//
// Launch: no host sync, no device malloc. The ww scratch is allocated by the wrapper only when
// nsmud > 0.

#include "nmmb_common.cuh"

#define CDWDT_WA 0.125f
#define CDWDT_WB 0.5f
#define CDWDT_FLAG_S 1
#define CDWDT_FLAG_N 2
#define CDWDT_FLAG_W 4
#define CDWDT_FLAG_E 8

// cdwdt setup: the early-return branch (hydro, or cold start with ntsd < 2).
extern "C" __global__ void cdwdt_setup(
    const int IM, const int JM, const int LM,
    const float* __restrict__ dsg2,    // [LM]    delta sigma (DSG2)
    const float* __restrict__ pdsg1,   // [LM]    delta pressure (PDSG1, grid.pdsg1)
    const float* __restrict__ pd,      // [JM,IM] sigma-range pressure difference
    float* __restrict__ dwdt,          // [LM,JM,IM]   out: 1
    float* __restrict__ pdwdt,         // [LM,JM,IM]   out: 1
    float* __restrict__ pint)          // [LM+1,JM,IM] inout (levels 2..LM+1 recomputed)
{
    COLUMN_IJ(IM, JM)
    const float pdv = pd[IDX2(i0, j0, IM)];
    for (int l0 = 0; l0 < LM; ++l0) {
        dwdt[IDX3(i0, j0, l0, IM, JM)]  = 1.0f;
        pdwdt[IDX3(i0, j0, l0, IM, JM)] = 1.0f;
        pint[IDX3(i0, j0, l0 + 1, IM, JM)] =
            dsg2[l0] * pdv + pdsg1[l0] + pint[IDX3(i0, j0, l0, IM, JM)];
    }
}

// Strip membership for the lateral smoothing of w (zero-based indices).
__device__ __forceinline__ bool cdwdt_in_strip(const int i0, const int j0,
                                               const int IM, const int JM, const int flags)
{
    const bool xs = (i0 >= 1 && i0 <= IM - 2);     // its_b1 .. ite_b1
    const bool yw = (j0 >= 2 && j0 <= JM - 3);     // max(jts,jds+lnsnh) .. min(jte,jde-lnsnh)
    if ((flags & CDWDT_FLAG_S) && j0 == 1 && xs) return true;
    if ((flags & CDWDT_FLAG_N) && j0 == JM - 2 && xs) return true;
    if ((flags & CDWDT_FLAG_W) && i0 == 1 && yw) return true;
    if ((flags & CDWDT_FLAG_E) && i0 == IM - 2 && yw) return true;
    return false;
}

// Phase 1: stencil into ww on the strips, reading the un-smoothed w.  Grid z = level.
extern "C" __global__ void cdwdt_smooth_ww(
    const int IM, const int JM, const int LM, const int flags,
    const float* __restrict__ w,       // [LM,JM,IM]
    float* __restrict__ ww)            // [LM,JM,IM] scratch
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    if (!cdwdt_in_strip(i0, j0, IM, JM, flags)) return;
    ww[IDX3(i0, j0, l0, IM, JM)] =
        (w[IDX3(i0, j0 - 1, l0, IM, JM)] + w[IDX3(i0 - 1, j0, l0, IM, JM)]
       + w[IDX3(i0 + 1, j0, l0, IM, JM)] + w[IDX3(i0, j0 + 1, l0, IM, JM)]) * CDWDT_WA
       + w[IDX3(i0, j0, l0, IM, JM)] * CDWDT_WB;
}

// Phase 2: copy the smoothed strip values back into w.
extern "C" __global__ void cdwdt_smooth_copy(
    const int IM, const int JM, const int LM, const int flags,
    float* __restrict__ w,             // [LM,JM,IM] inout
    const float* __restrict__ ww)      // [LM,JM,IM]
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    if (!cdwdt_in_strip(i0, j0, IM, JM, flags)) return;
    w[IDX3(i0, j0, l0, IM, JM)] = ww[IDX3(i0, j0, l0, IM, JM)];
}
