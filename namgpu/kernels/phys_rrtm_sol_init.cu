// phys_rrtm_sol_init.cu -- SOL_INIT_NMMB: solar-constant selection (module_radiation_astronomy_nmmb).
//
// NOTICE: NAM source file phys/radiation_astronomy_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). The subroutine header is kept here as the notice:
//   subroutine sol_init_nmmb (me): initialize astronomy process, set up module constants.
//   inputs: ISOL - =0 fixed solar constant (con_solr_old); =1 NOAA annual-mean TSI (ABS scale) table;
//           =2 NOAA annual-mean TSI (TIM scale); =3 CMIP5 annual-mean TSI (TIM); =4 CMIP5 monthly-mean TSI;
//           =10 fixed new solar constant (con_solr); anything else is a selection error.
//   outputs (module variables): isolflg (solar constant scheme control flag), solc0 (solar constant, W/m2).
//   mar 2012 - y.-t. hou: sol_init sets module parameters at start of run; sol_update runs in the time loop.
// Ported to CUDA for nam-gpu from phys/radiation_astronomy_nmmb.f:113-259 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (lines 157-257), every assignment and its counterpart here:
//   isolflg = isolar                                  -> out_isolflg = isolar           (157)
//   solc0   = con_solr                                -> out_solc0 = CON_SOLR           (158)
//   solar_fname = solar_file                          -> host side (wrapper), file name only
//   if isolar == 0:      solc0 = con_solr_old        -> out_solc0 = CON_SOLR_OLD       (isolar==0 branch)
//   elseif isolar == 10: no change                    -> (kept)
//   elseif isolar in 1,2,3,4:
//       solar_fname(15:26) = table name               -> host side (wrapper)
//       inquire(file=solar_fname); if missing:
//           isolflg = 0; solc0 stays con_solr          -> out_isolflg = 0 when file_exists == 0
//   else (selection error): isolflg = 0; solc0 stays con_solr  -> out_isolflg = 0
// The `me == 0` print statements are host diagnostics for MPI rank 0 and write no state; they are not
// ported (the wrapper returns the same information to the host instead).
// Precision: con_solr, con_solr_old are kind_phys (real*8 in physparam.f), so the outputs are double.
// Scalar work, one thread, one launch, no host sync.

// physcons.f:50-51 (the active lines; the other values are commented out in the Fortran).
#define CON_SOLR_OLD 1.3660e+3
#define CON_SOLR     ((double)1.3608e+3f)   // physcons.f:51 default-REAL literal, stored as kind_phys (double) after float32 rounding

extern "C" __global__ void sol_init_nmmb(
    const int isolar,         // isolar (physparam.f:102), configured solar constant scheme
    const int file_exists,    // 1 if the table named for isolar 1..4 exists in the working directory (host inquire)
    double* __restrict__ out_solc0,   // solc0 (module variable)
    int* __restrict__ out_isolflg)    // isolflg (module variable)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int isolflg = isolar;                 // line 157
    double solc0 = CON_SOLR;              // line 158

    if (isolar == 0) {
        solc0 = CON_SOLR_OLD;
    } else if (isolar == 10) {
        // fixed new solar constant: solc0 stays CON_SOLR
    } else if (isolar == 1 || isolar == 2 || isolar == 3 || isolar == 4) {
        if (file_exists == 0) {
            isolflg = 0;                  // missing table: fall back, solc0 stays CON_SOLR
        }
    } else {
        isolflg = 0;                      // selection error: fall back, solc0 stays CON_SOLR
    }

    out_solc0[0] = solc0;
    out_isolflg[0] = isolflg;
}
