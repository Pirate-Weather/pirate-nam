// phys_noah_penman.cu -- PENMAN: potential evaporation (Noah LSM), NAM nam-gpu port.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:3390-3468 (SUBROUTINE PENMAN).
// Start point: ArWen third_party/arwen/core/kernels/noah.cu (noah_penman, Apache-2.0,
// WRF-lineage transcription). ArWen blends latent heat and emissivity on SNCOVR; NAM's
// PENMAN switches on SNEQV == 0 (the Fortran is the referee here), so the switch is NAM's.
//
// Fortran (SUBROUTINE PENMAN), every executed statement, in order:
//   if SNEQV == 0:  EMISSI = EMISSI_IN, ELCP1 = ELCP, LVS = LSUBC
//   else:           EMISSI = 0.95 (snow), ELCP1 = ELCP*LSUBS/LSUBC, LVS = LSUBS
//   FLX2 = 0
//   DELTA = ELCP1*DQSDT2
//   T24 = SFCTMP*SFCTMP*SFCTMP*SFCTMP
//   RR = EMISSI*T24*6.48E-8/(SFCPRS*CH) + 1
//   RHO = SFCPRS/(RD*T2V)
//   RCH = RHO*CP*CH
//   if not SNOWNG: if PRCP > 0: RR = RR + CPH2O*PRCP/RCH
//   else:          RR = RR + CPICE*PRCP/RCH
//   FNET = FDOWN - EMISSI*SIGMA*T24 - SSOIL
//   if FRZGRA: FLX2 = -LSUBF*PRCP; FNET = FNET - FLX2
//   RAD = FNET/RCH + TH2 - SFCTMP
//   A = ELCP1*(Q2SAT - Q2)
//   EPSCA = (A*RR + RAD*DELTA)/(DELTA + RR)
//   ETP = EPSCA*RCH/LVS
// Left-to-right evaluation order is kept as the Fortran writes it. Float32 throughout
// (kfpt=single); fast numerics (FMA contraction on, the NVRTC default).
//
// Constants: ELCP, LSUBC, LSUBS, CP are PENMAN's own PARAMETERs (lines 3418-3419 of the
// module); RD, SIGMA, CPH2O, CPICE, LSUBF are the module's PARAMETERs (module header).
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3).
// All fields float32 [JM, IM] (C order, IDX2). SNOWNG and FRZGRA are LOGICAL, passed as
// int32 0/1 [JM, IM]. The device function nam_penman is the CONTRACT unit; kernel
// phys_noah_penman applies it to every column in one launch.

#include "nmmb_common.cuh"

#define NAM_PEN_ELCP    2.4888e+3f   // PENMAN ELCP
#define NAM_PEN_LSUBC  2.501000e+6f  // PENMAN LSUBC
#define NAM_PEN_LSUBS  2.83e+6f      // PENMAN LSUBS
#define NAM_PEN_CP     1004.6f       // PENMAN CP
#define NAM_PEN_EMISS_SNOW 0.95f     // PENMAN EMISSI for snow (SNEQV /= 0)
#define NAM_RD         287.04f       // module_LS_NOAHLSM RD
#define NAM_SIGMA      5.67e-8f      // module_LS_NOAHLSM SIGMA
#define NAM_CPH2O      4.218e+3f     // module_LS_NOAHLSM CPH2O
#define NAM_CPICE      2.106e+3f     // module_LS_NOAHLSM CPICE
#define NAM_LSUBF      3.335e+5f     // module_LS_NOAHLSM LSUBF

__device__ inline void nam_penman(
    float sfctmp, float sfcprs, float ch, float t2v, float th2, float prcp,
    float fdown, float ssoil, float q2, float q2sat, float dqsdt2,
    int snowng, int frzgra, float emissi_in, float sneqv,
    float& t24, float& etp, float& rch, float& rr, float& epsca, float& flx2)
{
    float emissi, elcp1, lvs;
    if (sneqv == 0.0f) {
        emissi = emissi_in;
        elcp1  = NAM_PEN_ELCP;
        lvs    = NAM_PEN_LSUBC;
    } else {
        emissi = NAM_PEN_EMISS_SNOW;
        elcp1  = NAM_PEN_ELCP * NAM_PEN_LSUBS / NAM_PEN_LSUBC;
        lvs    = NAM_PEN_LSUBS;
    }

    flx2 = 0.0f;
    float delta = elcp1 * dqsdt2;
    t24 = sfctmp * sfctmp * sfctmp * sfctmp;
    rr = emissi * t24 * 6.48e-8f / (sfcprs * ch) + 1.0f;
    float rho = sfcprs / (NAM_RD * t2v);
    rch = rho * NAM_PEN_CP * ch;

    if (snowng == 0) {
        if (prcp > 0.0f) rr = rr + NAM_CPH2O * prcp / rch;
    } else {
        rr = rr + NAM_CPICE * prcp / rch;
    }

    float fnet = fdown - emissi * NAM_SIGMA * t24 - ssoil;
    if (frzgra != 0) {
        flx2 = -NAM_LSUBF * prcp;
        fnet = fnet - flx2;
    }

    float rad = fnet / rch + th2 - sfctmp;
    float a = elcp1 * (q2sat - q2);
    epsca = (a * rr + rad * delta) / (delta + rr);
    etp = epsca * rch / lvs;
}

extern "C" __global__ void phys_noah_penman(
    const float* __restrict__ sfctmp,   // SFCTMP  surface air temperature      [JM, IM]
    const float* __restrict__ sfcprs,   // SFCPRS  surface pressure             [JM, IM]
    const float* __restrict__ ch,       // CH      surface exchange coefficient [JM, IM]
    const float* __restrict__ t2v,      // T2V     virtual T at 2 m             [JM, IM]
    const float* __restrict__ th2,      // TH2     potential T at 2 m           [JM, IM]
    const float* __restrict__ prcp,     // PRCP    precipitation rate           [JM, IM]
    const float* __restrict__ fdown,    // FDOWN   downward radiation           [JM, IM]
    const float* __restrict__ ssoil,    // SSOIL   soil heat flux               [JM, IM]
    const float* __restrict__ q2,       // Q2      specific humidity at 2 m     [JM, IM]
    const float* __restrict__ q2sat,    // Q2SAT   saturation humidity          [JM, IM]
    const float* __restrict__ dqsdt2,   // DQSDT2  d(qsat)/dT                   [JM, IM]
    const int*   __restrict__ snowng,   // SNOWNG  LOGICAL (0/1)                [JM, IM]
    const int*   __restrict__ frzgra,   // FRZGRA  LOGICAL (0/1)                [JM, IM]
    const float* __restrict__ emissi_in,// EMISSI_IN emissivity (no snow)       [JM, IM]
    const float* __restrict__ sneqv,    // SNEQV   snow water equivalent        [JM, IM]
    float* __restrict__ t24,            // T24     output                       [JM, IM]
    float* __restrict__ etp,            // ETP     output                       [JM, IM]
    float* __restrict__ rch,            // RCH     output                       [JM, IM]
    float* __restrict__ rr,             // RR      output                       [JM, IM]
    float* __restrict__ epsca,          // EPSCA   output                       [JM, IM]
    float* __restrict__ flx2,           // FLX2    output                       [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    float o_t24, o_etp, o_rch, o_rr, o_epsca, o_flx2;
    nam_penman(sfctmp[n], sfcprs[n], ch[n], t2v[n], th2[n], prcp[n],
               fdown[n], ssoil[n], q2[n], q2sat[n], dqsdt2[n],
               snowng[n], frzgra[n], emissi_in[n], sneqv[n],
               o_t24, o_etp, o_rch, o_rr, o_epsca, o_flx2);
    t24[n] = o_t24;
    etp[n] = o_etp;
    rch[n] = o_rch;
    rr[n] = o_rr;
    epsca[n] = o_epsca;
    flx2[n] = o_flx2;
}
