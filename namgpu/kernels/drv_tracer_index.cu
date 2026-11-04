// nam-gpu unit drv-tracer-index: tracer slot indices of the NMMB TRACERS array.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Notice kept from the Fortran module (nam.v4.2.11, 83df4e13c1):
//   module module_SOLVER_INTERNAL_STATE  (nmm/module_SOLVER_INTERNAL_STATE.F90)
//   module module_VARS_STATE             (nmm/module_VARS_STATE.F90)
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_INTERNAL_STATE.F90:864-953 (NUM_WATER, P_*,
// NUM_TRACERS_MET/TOTAL), nmm/module_SOLVER_INTERNAL_STATE.F90:1379-1453 (INDX_Q, INDX_CW, INDX_Q2,
// INDX_WATER_START/END, INDX_QC..INDX_NR) and nmm/module_VARS_STATE.F90:797 (INDX_Q2 = 3).
//
// Fortran routine (module_SOLVER_INTERNAL_STATE, microphysics branch and TRACERS slot setup):
//   NUM_TRACERS_MET   = 3                         (line 943, sets the configfile value 4 aside)
//   NUM_TRACERS_TOTAL = MET + CHEM + NUM_WATER
//   INDX_Q = 1, INDX_CW = 2, INDX_Q2 = 3
//   INDX_WATER_START  = MET + CHEM + 1
//   INDX_WATER_END    = INDX_WATER_START + NUM_WATER - 1
//   INDX_x            = INDX_WATER_START - 1 + P_x   when P_x > 0, else 0 (declared default)
//
// Input  params[10] (int32, host-filled from the microphysics table in drv_tracer_index.py):
//   [0] NUM_TRACERS_MET  [1] NUM_TRACERS_CHEM  [2] NUM_WATER
//   [3] P_QC [4] P_QI [5] P_QR [6] P_QS [7] P_QG [8] P_NI [9] P_NR
// Output indx[13] (int32), FORTRAN 1-BASED slot numbers (INDX_Q = 1, not 0). TRACERS is zero-based
// in its slot axis: a consumer indexing TRACERS[n] subtracts 1. Order: [0] INDX_Q [1] INDX_CW [2] INDX_Q2 [3] INDX_QC [4] INDX_QI [5] INDX_QR
//   [6] INDX_QS [7] INDX_QG [8] INDX_NI [9] INDX_NR [10] INDX_WATER_START [11] INDX_WATER_END
//   [12] NUM_TRACERS_TOTAL
//
// Single thread, one launch at init: the Fortran is straight-line assignments with no loop over
// cells. It is not on the step path (no host sync is needed; the wrapper caches the device table).

#include "nmmb_common.cuh"

extern "C" __global__ void tracer_index_build(
    const int* __restrict__ params,     // [10]  in
    int* __restrict__ indx)             // [13]  out
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int met   = params[0];
    const int chem  = params[1];
    const int nw    = params[2];

    indx[0] = 1;                        // INDX_Q
    indx[1] = 2;                        // INDX_CW
    indx[2] = 3;                        // INDX_Q2 (module_VARS_STATE.F90:797)

    const int wstart = met + chem + 1;  // INDX_WATER_START
    indx[10] = wstart;
    indx[11] = wstart + nw - 1;         // INDX_WATER_END
    indx[12] = met + chem + nw;         // NUM_TRACERS_TOTAL

    // Water species: P_x > 0 gives INDX_x = WATER_START - 1 + P_x, otherwise the declared 0.
    for (int k = 0; k < 7; ++k) {       // QC QI QR QS QG NI NR
        const int p = params[3 + k];
        indx[3 + k] = (p > 0) ? (wstart - 1 + p) : 0;
    }
}
