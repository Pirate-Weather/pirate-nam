// WOOF-NAM unit drv-dfi-filt-vars: classify each DFI filter/restore table entry.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone; see NOTICE
// at the repo root). Notice kept from the Fortran routine (module module_DOMAIN_GRID_COMP,
// nam.v4.2.11, 83df4e13c1):
//   SUBROUTINE BUILD_FILT_BUNDLE
//   For digital filtering purposes, the model needs to know both which variables should be
//   filtered, and which variables need to be saved, before filtering begins and restored to their
//   original state after filtering has occurred. Insert the appropriate internal states into the
//   appropriate bundles.
//   PROGRAM HISTORY LOG (module): 2010-11-03  Pyle  - Revised for digital filters.
// Ported to CUDA for WOOF-NAM from sorc/nam_nems_nmmb.fd/src/atmos/nmm/module_DOMAIN_GRID_COMP.F90:5844-6055 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (BUILD_FILT_BUNDLE, per table line after the comment and blank skip):
//   IF (CH_FILTREST(1:1)=='F')      FILT_TYP_INT = 1                 (filter)
//   ELSEIF (CH_FILTREST(1:1)=='R')  FILT_TYP_INT = 2                 (restore)
//   ELSEIF (CH_FILTREST(1:1)=='B')  FILT_TYP_INT = RESTART ? 2 : 1   (both)
//   ELSE                            FILT_TYP_INT = -999              (ignored: no bundle entry)
//
// Input  codes[n]  (int32, host-filled from the table in drv_dfi_filt_vars.py):
//        1 = 'F', 2 = 'R', 3 = 'B', 0 = any other first character (ignored by the Fortran).
// Input  n, restart (int32, 0 or 1; RESTART from the configfile).
// Output action[n] (int32): 0 = ignore, 1 = filter (DFI bundle), 2 = restore (restore bundle).
//
// One thread per table entry, one launch at init. No host sync; the wrapper keeps the device
// array for device-side consumers and mirrors the same numbers on the host.

#include "nmmb_common.cuh"

extern "C" __global__ void dfi_filt_classify(
    const int* __restrict__ codes,      // [n]  in
    int n,
    int restart,
    int* __restrict__ action)           // [n]  out
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    const int c = codes[k];
    int a = 0;                          // ignored (Fortran FILT_TYP_INT = -999)
    if (c == 1) {
        a = 1;                          // 'F' -> filter
    } else if (c == 2) {
        a = 2;                          // 'R' -> restore
    } else if (c == 3) {
        a = restart ? 2 : 1;            // 'B' -> restore if restart, else filter
    }
    action[k] = a;
}
