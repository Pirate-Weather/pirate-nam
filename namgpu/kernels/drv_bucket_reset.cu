// nam-gpu unit drv-bucket-reset: accumulator resets by NCLOD, NRDLW, NRDSW, NSRFC, NPREC.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone; see NOTICE
//   at the repo root). The module header of nmm/module_SOLVER_GRID_COMP.F90 (module name, purpose
//   and HISTORY LOG, the notice for this port) is reproduced here:
//      MODULE module_SOLVER_GRID_COMP
// 
// -----------------------------------------------------------------------
// 
// ***  This module holds the Solver component's  Register, Init, Run, 
// ***  and Finalize routines.  They are called from the DOMAIN component
// ***  (DOMAIN_INITIALIZE calls SOLVER_INITIALIZE, etc.) 
// ***  in MODULE_DOMAIN_GRID_COMP.F90.
// 
// -----------------------------------------------------------------------
//  HISTORY LOG:
// 
//    2008-07-30  Janjic - Add CONVECTION='none' to OPERATIONAL_PHYSICS.
//                Janjic - Fix lower J limit in FFTFHN(WATER).
//    2008-08-23  Janjic - General pressure-sigma hybrid
//                Janjic - Consistent nonhydrostatic correction in the
//                         first term of the pressure gradient force
//    2008-09-03  Black  - Added initialization of boundary arrays
//                         for nests.
//    2009-03-12  Black  - Changes for general hybrid coordinate.
//    2009-11     Jovic  - Modified for ownership/import/export specification
//    2010-11-03  Pyle   - Modifications/corrections for digital filter.
//    2011-02     Yang   - Updated to use both the ESMF 4.0.0rp2 library,
//                         ESMF 5 series library and the the
//                         ESMF 3.1.0rp2 library.
//    2011-05-12  Yang   - Modified for using the ESMF 5.2.0r_beta_snapshot_07.
//    2011-12-22  Jovic  - Combined Dyn and Phy into single component.
// 
//    2012-02-08  Yang   - Modified for using the ESMF 5.2.0rp1 library.
//    2012-04-06  Juang  - add passing argument for gbphys for idea
//    2012-07-20  Black  - Modified for generational usage.
//    2013-09-09 Moorthi - Adding SR, DTDT, and TRIGGERPERTS for GBPHYS call
//    2013-11-09 Xingren Wu - Adding DUSFCI/DVSFCI for GBPHYS call
//    2014-03-28 Xingren Wu - Add "_CPL" field for GBPHYS call
//    2014-05-14 J. Wang - Adding cgwf,prslrd0 and levr to gbphys call
//    2014-06-26 Weiguo Wang -- Add HURRICANE PBL and SFCLAY calls
//    2016-02-16 J. Wang - change newsas/sashal from logical to integer
//    2016-05-09 Ferrier/Janjic - Constants epsq2,epsl function of level,
//                        added subroutine TQadjust
// -----------------------------------------------------------------------
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_GRID_COMP.F90:5367-5470 and 6348-6384
// (nam.v4.2.11, 83df4e13c1).
//
// Fortran structure. Two places zero the accumulators. Each is a set of independent IF blocks,
// each a DO J / DO I loop over ITS..ITE x JTS..JTE (one card: the whole domain, CONTRACT.md section 1).
//   Site A, before turbulence (SOLVER_RUN 5367-5470):
//     5367-5376  IF(MOD(NTIMESTEP,NCLOD)==0): ACFRST, ACFRCV = 0.; NCFRST, NCFRCV = 0   (4)
//     5418-5428  IF(MOD(NTIMESTEP,NRDLW)==0): ALWIN, ALWOUT, ALWTOA, ARDLW = 0.          (4)
//     5430-5440  IF(MOD(NTIMESTEP,NRDSW)==0): ASWIN, ASWOUT, ASWTOA, ARDSW = 0.          (4)
//     5442-5454  IF(MOD(NTIMESTEP,NSRFC)==0): SFCSHX, SFCLHX, SUBSHX, SNOPCX, POTFLX, ASRFC = 0. (6)
//     5456-5468  IF(MOD(NTIMESTEP,NPREC)==0): ACSNOW, ACSNOM, SSROFF, BGROFF, SFCEVP, POTEVP = 0. (6)
//   Site B, after the microphysics and CLTEND calls and before the GFS physics branch
//   (SOLVER_RUN 6348-6384). TURBL (5490) adds to ARDLW, ARDSW and ASRFC, so this second reset is
//   what stops those counters carrying turbulence increments across a reset interval:
//     6348-6357  IF(MOD(NTIMESTEP,NRDLW)==0): ALWIN, ALWOUT, ALWTOA, ARDLW = 0.          (4)
//     6359-6368  IF(MOD(NTIMESTEP,NRDSW)==0): ASWIN, ASWOUT, ASWTOA, ARDSW = 0.          (4)
//     6370-6384  IF(MOD(NTIMESTEP,NSRFC)==0): ACSNOW, POTEVP, SFCEVP, SFCLHX, SFCSHX, SUBSHX,
//                                              BGROFF, SSROFF, ASRFC = 0.               (9)
//   Site B's NSRFC set is not Site A's NSRFC set; both are kept as written.
//
// The MOD tests are done on the host by the wrapper (drv_bucket_reset.py) from NTIMESTEP and the
// bucket intervals; the kernel receives one flag per block. The 2-D fields are [JM, IM] (CONTRACT
// section 1). NCFRST/NCFRCV are int32. Each thread writes only its own (i, j), so no two blocks
// race. ARDLW, ARDSW and ASRFC are the Fortran "artificial 2-D" counters (ESMF scalar workaround).

#include "nmmb_common.cuh"

// Site A (5367-5468). The kernel flags select the blocks that fire on this step.
extern "C" __global__ void bucket_reset(const int im, const int jm,
                                        const int do_nclod, const int do_nrdlw,
                                        const int do_nrdsw, const int do_nsrfc,
                                        const int do_nprec,
                                        float* __restrict__ ACFRST,
                                        float* __restrict__ ACFRCV,
                                        int*   __restrict__ NCFRST,
                                        int*   __restrict__ NCFRCV,
                                        float* __restrict__ ALWIN,
                                        float* __restrict__ ALWOUT,
                                        float* __restrict__ ALWTOA,
                                        float* __restrict__ ARDLW,
                                        float* __restrict__ ASWIN,
                                        float* __restrict__ ASWOUT,
                                        float* __restrict__ ASWTOA,
                                        float* __restrict__ ARDSW,
                                        float* __restrict__ SFCSHX,
                                        float* __restrict__ SFCLHX,
                                        float* __restrict__ SUBSHX,
                                        float* __restrict__ SNOPCX,
                                        float* __restrict__ POTFLX,
                                        float* __restrict__ ASRFC,
                                        float* __restrict__ ACSNOW,
                                        float* __restrict__ ACSNOM,
                                        float* __restrict__ SSROFF,
                                        float* __restrict__ BGROFF,
                                        float* __restrict__ SFCEVP,
                                        float* __restrict__ POTEVP)
{
    // 2-D point kernel: grid.z is 1, so l0 is 0 and unused.
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);

    // Block 5367-5376: NCLOD (cloud fraction accumulators).
    if (do_nclod) {
        ACFRST[n] = 0.0f;
        ACFRCV[n] = 0.0f;
        NCFRST[n] = 0;
        NCFRCV[n] = 0;
    }

    // Block 5418-5428: NRDLW (LW radiation accumulators).
    if (do_nrdlw) {
        ALWIN[n]  = 0.0f;
        ALWOUT[n] = 0.0f;
        ALWTOA[n] = 0.0f;
        ARDLW[n]  = 0.0f;
    }

    // Block 5430-5440: NRDSW (SW radiation accumulators).
    if (do_nrdsw) {
        ASWIN[n]  = 0.0f;
        ASWOUT[n] = 0.0f;
        ASWTOA[n] = 0.0f;
        ARDSW[n]  = 0.0f;
    }

    // Block 5442-5454: NSRFC (surface energy and evaporation accumulators).
    if (do_nsrfc) {
        SFCSHX[n] = 0.0f;
        SFCLHX[n] = 0.0f;
        SUBSHX[n] = 0.0f;
        SNOPCX[n] = 0.0f;
        POTFLX[n] = 0.0f;
        ASRFC[n]  = 0.0f;
    }

    // Block 5456-5468: NPREC (precipitation and runoff accumulators).
    if (do_nprec) {
        ACSNOW[n] = 0.0f;
        ACSNOM[n] = 0.0f;
        SSROFF[n] = 0.0f;
        BGROFF[n] = 0.0f;
        SFCEVP[n] = 0.0f;
        POTEVP[n] = 0.0f;
    }
}

// Site B (6348-6384), after the microphysics and CLTEND calls. Separate kernel because its NSRFC
// set differs from Site A's (9 fields here, 6 there).
extern "C" __global__ void bucket_reset_post(const int im, const int jm,
                                             const int do_nrdlw, const int do_nrdsw,
                                             const int do_nsrfc,
                                             float* __restrict__ ALWIN,
                                             float* __restrict__ ALWOUT,
                                             float* __restrict__ ALWTOA,
                                             float* __restrict__ ARDLW,
                                             float* __restrict__ ASWIN,
                                             float* __restrict__ ASWOUT,
                                             float* __restrict__ ASWTOA,
                                             float* __restrict__ ARDSW,
                                             float* __restrict__ ACSNOW,
                                             float* __restrict__ POTEVP,
                                             float* __restrict__ SFCEVP,
                                             float* __restrict__ SFCLHX,
                                             float* __restrict__ SFCSHX,
                                             float* __restrict__ SUBSHX,
                                             float* __restrict__ BGROFF,
                                             float* __restrict__ SSROFF,
                                             float* __restrict__ ASRFC)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);

    // Block 6348-6357: NRDLW.
    if (do_nrdlw) {
        ALWIN[n]  = 0.0f;
        ALWOUT[n] = 0.0f;
        ALWTOA[n] = 0.0f;
        ARDLW[n]  = 0.0f;
    }

    // Block 6359-6368: NRDSW.
    if (do_nrdsw) {
        ASWIN[n]  = 0.0f;
        ASWOUT[n] = 0.0f;
        ASWTOA[n] = 0.0f;
        ARDSW[n]  = 0.0f;
    }

    // Block 6370-6384: NSRFC.
    if (do_nsrfc) {
        ACSNOW[n] = 0.0f;
        POTEVP[n] = 0.0f;
        SFCEVP[n] = 0.0f;
        SFCLHX[n] = 0.0f;
        SFCSHX[n] = 0.0f;
        SUBSHX[n] = 0.0f;
        BGROFF[n] = 0.0f;
        SSROFF[n] = 0.0f;
        ASRFC[n]  = 0.0f;
    }
}
