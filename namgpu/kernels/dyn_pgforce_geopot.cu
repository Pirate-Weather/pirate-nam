// dyn_pgforce_geopot.cu -- PGFORCE setup and the geopotential / APEL / DFI vertical grand loop.
//
// Notice: module_DYNAMICS_ROUTINES.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1,
// NMMB dynamics, NOAA/NCEP/EMC) has no author or history header block of its own; the NOAA
// source notice in NOTICE applies. NAM is US Government work with no licence file.
//
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:51-248 (nam.v4.2.11, 83df4e13c1).
//
// Fortran source covered (subroutine pgforce, lines 51-248):
//   175-185  filo(i,j) = fis(i,j)                       -> filo register init
//   199-222  vertical grand loop do l=lm,1,-1: apelp, apel_3d, dfdp, rtop, dfip, fiup,
//            dfi_3d, fim_3d, filo update                -> per-column recursion below
//   240-247  per-level staging copy apel(i,j)=apel_3d(i,j,l), fim, dfi into 2-D scratch
//            -> not needed here: the PGF unit (dyn-pgforce-pgf, lines 249-486) reads the 3-D
//               APEL_3D/FIM_3D/DFI_3D arrays directly, so no 2-D staging is done in this kernel.
//   Not in this unit: first/global/restart/dt/ntimestep/rdyv/rdxv/wpdar (used by other pgforce
//            units), the ENABLE_SMP OpenMP split (dropped: one column thread per cell), and the
//            pgx/pgy/pgne/pgnw/pcx/pcy/tcu/tcv terms (other units).
//
// Halo-free layout (CONTRACT.md 1): the recursion is purely column-local (filo, fis, pd, t, q,
// cw, pint at the same i,j), so NMMB's halo-extended loop limits (its_h2..ite_h2) reduce to the
// full domain 1..IM, 1..JM. One thread per column; the l-loop runs upward from the surface
// (Fortran l = LM down to 1, zero-based l0 = LM-1 down to 0) exactly as the Fortran does.
//
// Term-by-term (Fortran -> CUDA, zero-based i0 j0 l0):
//   apelp  = (pint(i,j,l) + pint(i,j,l+1)) * 0.5        -> (PINT[l0] + PINT[l0+1]) * 0.5f
//   apel_3d(i,j,l) = apelp                               -> apel_3d[IDX3(l0)] = apelp
//   dfdp   = (q*0.608 + (1-cw)) * t * r / apelp          -> Fortran operation order kept,
//            r = 287.04 (module_CONSTANTS, NMMB_R_D)
//   rtop(i,j,l) = dfdp                                   -> rtop[IDX3(l0)] = dfdp
//   dfip   = dfdp * (dsg2(l)*pd(i,j) + pdsg1(l))         -> dsg2[l0], pdsg1[l0], pd[IDX2]
//   fiup   = filo + dfip
//   dfi_3d(i,j,l) = dfip * 0.5                           -> dfi_3d[IDX3(l0)]
//   fim_3d(i,j,l) = (filo + fiup) * 0.5                  -> fim_3d[IDX3(l0)]
//   filo   = fiup                                        -> register carried up the column
//
// Fast numerics: fmad contraction on (NVRTC --fmad=true), IEEE division, single precision
// (module_CONTROL kfpt = single). No host sync, no malloc, no printf.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_pgforce_geopot(
    const float* __restrict__ fis,     // [JM, IM]        surface geopotential
    const float* __restrict__ pd,      // [JM, IM]        sigma-range pressure difference
    const float* __restrict__ t,       // [LM, JM, IM]    temperature
    const float* __restrict__ q,       // [LM, JM, IM]    specific humidity
    const float* __restrict__ cw,      // [LM, JM, IM]    condensate
    const float* __restrict__ pint,    // [LM+1, JM, IM]  pressure at interfaces
    const float* __restrict__ dsg2,    // [LM]            delta sigmas
    const float* __restrict__ pdsg1,   // [LM]            delta pressures
    float* __restrict__ rtop,          // [LM, JM, IM]    out: RT/p (dfdp)
    float* __restrict__ apel_3d,       // [LM, JM, IM]    out: pressure at layer middle
    float* __restrict__ fim_3d,        // [LM, JM, IM]    out: geopotential at layer middle
    float* __restrict__ dfi_3d,        // [LM, JM, IM]    out: half layer geopotential thickness
    const int im,
    const int jm,
    const int lm)
{
    COLUMN_IJ(im, jm)                   // col_, i0, j0; returns for threads past IM*JM

    // filo(i,j) = fis(i,j)  (Fortran 175-185)
    float filo = fis[IDX2(i0, j0, im)];
    const float pdij = pd[IDX2(i0, j0, im)];

    // Vertical grand loop, Fortran l = LM down to 1 (zero-based l0 = LM-1 down to 0).
    for (int l0 = lm - 1; l0 >= 0; --l0) {
        const size_t k      = IDX3(i0, j0, l0, im, jm);       // layer l
        const size_t k_lo   = IDX3(i0, j0, l0, im, jm);       // interface l   (pint level l0)
        const size_t k_hi   = IDX3(i0, j0, l0 + 1, im, jm);   // interface l+1 (pint level l0+1)

        // apelp = (pint(i,j,l) + pint(i,j,l+1)) * 0.5
        const float apelp = (pint[k_lo] + pint[k_hi]) * 0.5f;
        apel_3d[k] = apelp;

        // dfdp = (q*0.608 + (1-cw)) * t * r / apelp   (Fortran left-to-right order kept)
        const float dfdp = (q[k] * 0.608f + (1.0f - cw[k])) * t[k] * NMMB_R_D / apelp;

        // dfip = dfdp * (dsg2(l)*pd(i,j) + pdsg1(l))
        const float dfip = dfdp * (dsg2[l0] * pdij + pdsg1[l0]);
        rtop[k] = dfdp;

        // fiup = filo + dfip ; dfi_3d = dfip*0.5 ; fim_3d = (filo+fiup)*0.5 ; filo = fiup
        const float fiup = filo + dfip;
        dfi_3d[k] = dfip * 0.5f;
        fim_3d[k] = (filo + fiup) * 0.5f;
        filo = fiup;
    }
}
