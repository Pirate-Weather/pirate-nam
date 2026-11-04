// WOOF-NAM unit phys-bmj-ttblex: TTBLEX, table look-up of the moist-adiabat temperature from the
// BMJ tables (ttbl) for one parcel. Used by the BMJ column scheme (bmjdrv and its cloud-base,
// cloud-top and deep/shallow iteration units) and by nothing else.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran module has no licence block; its module header is kept here as the notice:
//   MODULE MODULE_CU_BMJ (Betts-Miller-Janjic convection, NMMB, nam.v4.2.11).
// Ported to CUDA for WOOF-NAM from sorc/nam_nems_nmmb.fd/src/atmos/phys/module_CU_BMJ.F90:2071-2154 (nam.v4.2.11, 83df4e13c1,
// NAM_SRC=<a NAM clone>). The cmp copy has no TTBLEX; do not diff against it.
//
// Two entry points:
//   bmj_ttblex            __device__ function, the exact Fortran TTBLEX body. Called by the
//                         bmjdrv column kernels (one thread per column, no host round trip).
//   phys_bmj_ttblex_columns  extern "C" __global__, one thread per column, evaluates TTBLEX for
//                         a 2-D field of parcel pressures and theta-e values. It is the unit's
//                         launchable form (wrapper namgpu.physics.bmj.phys_bmj_ttblex) and is
//                         what the done-rule check drives. It is not a separate Fortran routine.
//
// Fortran dummy arguments -> device arguments (Fortran 1-based indices kept in the body):
//   itbx, jtbx        -> itbx, jtbx (table dimensions; ttbl is (jtbx, itbx) in Fortran)
//   plx, rdpx, rdthex -> plx, rdpx, rdthex (scalars: table base pressure and scalings)
//   prsmid            -> prsmid (per column, the parcel pressure pk)
//   thesp             -> thesp  (per column, the parcel theta-e)
//   sthe, the0        -> sthe[itbx], the0[itbx] (1-D table columns, Fortran index iptb)
//   ttbl              -> ttbl, CuPy shape [itbx, jtbx]: Fortran ttbl(ithtb, iptb) is
//                        ttbl[(iptb-1)*jtbx + (ithtb-1)]
//   tref              -> tref (per column output)
//
// Index handling: iptb and ithtb are kept as 1-based integers exactly as in Fortran, including
// the two clamps (below 1, and at or above the table size minus one, with the weight reset to 0),
// and converted to 0-based array indices at each access. int() and aint() are truncation toward
// zero in both languages (C cast and truncf). Evaluation order of the four-point blend follows
// the Fortran expression left to right.
//
// Layout (CONTRACT sections 1 and 3): per-column [JM, IM] arrays use IDX2(i0, j0, IM); the table
// vectors are 1-D. Single process, no halos (CONTRACT section 1).
//
// Numerics: float32 (NMMB kfpt = single), FMA contraction on (fast numerics, CONTRACT section 3).
// No host sync, no printf, no device allocation.

#pragma once
#include "nmmb_common.cuh"

__device__ __forceinline__ float bmj_ttblex(
    const int itbx, const int jtbx,
    const float plx, const float prsmid, const float rdpx, const float rdthex, const float thesp,
    const float* __restrict__ sthe, const float* __restrict__ the0,
    const float* __restrict__ ttbl)
{
    // ----------------scaling pressure & tt table index----------------------
    const float pk = prsmid;
    const float tpk = (pk - plx) * rdpx;
    float qq = tpk - truncf(tpk);                 // aint(tpk) subtracted
    int iptb = (int)tpk + 1;                      // int(tpk)+1 (1-based)

    // ----------------keeping indices within the table-----------------------
    if (iptb < 1) {
        iptb = 1;
        qq = 0.0f;
    }
    if (iptb >= itbx) {
        iptb = itbx - 1;
        qq = 0.0f;
    }

    // ----------------base and scaling factor for thetae---------------------
    const float bthe00k = the0[iptb - 1];
    const float sthe00k = sthe[iptb - 1];
    const float bthe10k = the0[iptb];
    const float sthe10k = sthe[iptb];

    // ----------------scaling the & tt table index---------------------------
    const float bthk = (bthe10k - bthe00k) * qq + bthe00k;
    const float sthk = (sthe10k - sthe00k) * qq + sthe00k;
    const float tthk = (thesp - bthk) / sthk * rdthex;
    float pp = tthk - truncf(tthk);
    int ithtb = (int)tthk + 1;

    // ----------------keeping indices within the table-----------------------
    if (ithtb < 1) {
        ithtb = 1;
        pp = 0.0f;
    }
    if (ithtb >= jtbx) {
        ithtb = jtbx - 1;
        pp = 0.0f;
    }

    // ----------------temperature at four surrounding tt table pts.----------
    const float* __restrict__ col_lo = ttbl + (size_t)(iptb - 1) * (size_t)jtbx;  // iptb
    const float* __restrict__ col_hi = ttbl + (size_t)(iptb)     * (size_t)jtbx;  // iptb+1
    const float t00k = col_lo[ithtb - 1];
    const float t10k = col_lo[ithtb];
    const float t01k = col_hi[ithtb - 1];
    const float t11k = col_hi[ithtb];

    // ----------------parcel temperature-------------------------------------
    return (t00k + (t10k - t00k) * pp + (t01k - t00k) * qq
            + (t00k - t10k - t01k + t11k) * pp * qq);
}

// One thread per column (CONTRACT section 3, grid_cols). prsmid and thesp are [JM, IM].
extern "C" __global__ void phys_bmj_ttblex_columns(
    const int IM, const int JM,
    const int itbx, const int jtbx,
    const float plx, const float rdpx, const float rdthex,
    const float* __restrict__ prsmid,
    const float* __restrict__ thesp,
    const float* __restrict__ the0,
    const float* __restrict__ sthe,
    const float* __restrict__ ttbl,
    float* __restrict__ tref)
{
    COLUMN_IJ(IM, JM)
    const size_t p2 = IDX2(i0, j0, IM);
    tref[p2] = bmj_ttblex(itbx, jtbx, plx, prsmid[p2], rdpx, rdthex, thesp[p2],
                          sthe, the0, ttbl);
}
