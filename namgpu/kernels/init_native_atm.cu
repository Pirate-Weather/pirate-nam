// nam-gpu unit init-native-atm: map the decoded NAM bgrd3d00 fields onto the NMMB state
// (cold start). Each term follows the NMM cold-start reader and the NPS (nemsinterp) writer
// that produced the file the NMM reader consumes, cited per term.
//
// Terms (one thread per (i, j, l); l = 1 is the model top, as in NMMB):
//   T   <- TMP    direct_temp=.true. (NAM production, parm/nam_namelist.nps): NPS does not
//                 devirtualise, so T = TMP. direct_temp=.false.: NPS applies
//                 T = T/(1+0.608Q) on the interior only (module_vinterp_routines.F90, block
//                 under .NOT. direct_temp, loops to IDE-1/JDE-1); the last row and column stay
//                 virtual, as NPS leaves them. The NMM cold reader copies 'tmp' into
//                 int_state%t (module_INIT_READ_NEMSIO.F90:305-330).
//   Q   <- SPFH   module_INIT_READ_NEMSIO.F90 cold branch 'spfh' copied to int_state%q.
//   U,V <- UGRD,VGRD  NPS writes 'ugrd'/'vgrd' (output_nemsinit.F90:142,150) with no rotation
//                 call in the writer; the NMM cold reader copies them to u/v at the same (i,j)
//                 (module_INIT_READ_NEMSIO.F90:274-305). Direct copy kept.
//   CW  <- CWM    NPS CWM block (module_vinterp_routines.F90, "CWM definition") runs only when
//                 do_clouds AND maxval(CLWMR) > 0; otherwise CWM = 0 everywhere. The gate is a
//                 device flag (cw_on), no host sync. Inside the gate, each species below 1e-11
//                 is zeroed, then CWM = SNMR + CLWMR + RWMR + CICE in that order, on the
//                 interior only (I < IDE-1, J < JDE-1); the last row and column stay 0.
//                 NAM CONUS coldstart has do_clouds=.false. (exnam_coldstart_prep_nest*.sh), so
//                 CW = 0 there.
//   F_ICE, F_RAIN, F_RIMEF  NMM cold start initialises them to 0 in the "normal 3D arrays"
//                 block (module_SOLVER_GRID_COMP.F90:937-939). The cold reader never reads
//                 f_ice, f_rain or rime, and NPS never writes them. The runtime fractions in
//                 module_CONVECTION.F90:592-595 are not a cold-start rule.
//   Q2  <- const  NMM cold start sets Q2 = 0.02 everywhere (module_SOLVER_GRID_COMP.F90:941).
//                 The cold reader does not read 'q2' (the 'q2' and 'vvel' reads are in the
//                 restart branch, module_INIT_READ_NEMSIO.F90:857 onward).
//   W   <- 0      The cold reader never writes int_state%w (its only 'vvel' read is in the
//                 restart branch). Value 0 is taken as the cold-start state.
//   PD  <- PSFC   NPS computes PD = PSFC - PT when vcoord = 1 (NAM), PSFC - PDTOP - PT when
//                 vcoord = 0 (module_vinterp_routines.F90:1989-1992). NPS writes it as 'dpres'
//                 (output_nemsinit.F90:138), which the NMM cold reader loads into int_state%pd
//                 (module_INIT_READ_NEMSIO.F90:257-264). This kernel computes the same value
//                 directly, so PD is not an algebraic inverse of PINT.
//                 Open: NPS's own loop writes PD only up to IDE-1/JDE-1; this kernel writes the
//                 formula at the edges too (edge value in NPS is not traced).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_GRID_COMP.F90:937-941,
// nmm/module_INIT_READ_NEMSIO.F90 (cold branch 137-856), nam_nps.fd NMMB_init/NPS/nemsinterp
// (output_nemsinit.F90, module_vinterp_routines.F90) (nam.v4.2.11, 83df4e13c1).
//
// Launch: grid_2d(IM, JM, LM), BLOCK_2D. No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

#define INIT_CUT_SPECIES 1.0e-11f   /* module_vinterp_routines.F90 species floor */

extern "C" __global__ void init_native_atm(
    const float* __restrict__ tmp,
    const float* __restrict__ spfh,
    const float* __restrict__ ugrd,
    const float* __restrict__ vgrd,
    const float* __restrict__ clmr,
    const float* __restrict__ cice,
    const float* __restrict__ rwmr,
    const float* __restrict__ snmr,
    const float* __restrict__ psfc,
    const int* __restrict__ cw_on,
    float* __restrict__ T,
    float* __restrict__ Q,
    float* __restrict__ U,
    float* __restrict__ V,
    float* __restrict__ CW,
    float* __restrict__ W,
    float* __restrict__ Q2,
    float* __restrict__ F_ICE,
    float* __restrict__ F_RAIN,
    float* __restrict__ F_RIMEF,
    float* __restrict__ PD,
    const float pt,
    const float pdtop,
    const float q2_default,
    const int vcoord,
    const int direct_temp,
    const int im,
    const int jm,
    const int lm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t n = IDX3(i0, j0, l0, im, jm);
    /* NPS loop bounds: I < IDE-1, J < JDE-1 (IDE = IM, JDE = JM). */
    const int interior = (i0 < im - 1) && (j0 < jm - 1);

    const float q = spfh[n];
    float t = tmp[n];
    if (!direct_temp && interior) t = t / (1.0f + 0.608f * q);
    T[n] = t;
    Q[n] = q;
    U[n] = ugrd[n];
    V[n] = vgrd[n];
    W[n] = 0.0f;
    Q2[n] = q2_default;
    F_ICE[n] = 0.0f;
    F_RAIN[n] = 0.0f;
    F_RIMEF[n] = 0.0f;

    float cwm = 0.0f;
    if (cw_on[0] && interior) {
        /* Condensate species with the NPS floor (zero below 1e-11), then CWM summed in NPS order. */
        const float c_clw = (clmr[n] < INIT_CUT_SPECIES) ? 0.0f : clmr[n];
        const float c_sno = (snmr[n] < INIT_CUT_SPECIES) ? 0.0f : snmr[n];
        const float c_rai = (rwmr[n] < INIT_CUT_SPECIES) ? 0.0f : rwmr[n];
        const float c_ice = (cice[n] < INIT_CUT_SPECIES) ? 0.0f : cice[n];
        cwm = c_sno + c_clw + c_rai + c_ice;
    }
    CW[n] = cwm;

    if (l0 == 0) {
        const size_t p = IDX2(i0, j0, im);
        PD[p] = (vcoord == 0) ? ((psfc[p] - pdtop) - pt) : (psfc[p] - pt);
    }
}
