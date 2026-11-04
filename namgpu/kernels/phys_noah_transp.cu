// phys_noah_transp.cu -- TRANSP: transpiration for the vegetation class (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:5200-5309 (nam.v4.2.11, 83df4e13c1).
// Derived from the ArWen Noah kernel (third_party/arwen/core/kernels/noah.cu, Apache-2.0, see
// third_party/arwen/LICENSE). Its noah_transp body was the starting point; it matches the NAM
// routine term for term apart from names, layout and single precision.
//
// Source: SUBROUTINE TRANSP (ET, NSOIL, ETP1, SMC, CMC, ZSOIL, SHDFAC, SMCWLT, CMCMAX, PC, CFACTR,
// SMCREF, SFCTMP, Q2, NROOT, RTDIS, VEGTYP, STC). The only call (EVAPO, module_LS_NOAHLSM.F90:2613)
// passes SH2O as the dummy SMC, so the kernel's smc input is SH2O. Every executed statement:
//   ET(K) = 0 for K = 1..NSOIL
//   ETP1A = SHDFAC*PC*ETP1*(1.0-(CMC/CMCMAX)**CFACTR)   if CMC /= 0, else SHDFAC*PC*ETP1
//   GX(I) = (SMC(I)-SMCWLT)/(SMCREF-SMCWLT); GX(I) = MAX(MIN(GX(I),1),0); SGX += GX(I)   I=1..NROOT
//   SGX = SGX/NROOT
//   RTX = RTDIS(I)+GX(I)-SGX; GX(I) = GX(I)*MAX(RTX,0); DENOM += GX(I)                  I=1..NROOT
//   IF DENOM <= 0: DENOM = 1
//   ET(I) = ETP1A*GX(I)/DENOM                                                            I=1..NROOT
// SFCTMP, Q2, ZSOIL, VEGTYP and STC are dummy arguments that the executed code never reads (the
// ZSOIL and PART variants are commented out in the Fortran) and are not passed. The commented
// variable-root-distribution forms are not executed and are not ported.
//
// Caller gate, reproduced here so the unit is correct on its own: EVAPO calls TRANSP only inside
// IF (ETP1 > 0.0) and IF (SHDFAC > 0.0) (module_LS_NOAHLSM.F90:2600-2613), and sets ET(K) = 0
// before (2590-2593). A column failing either test writes ET = 0 here. The IF tests are written
// as !(x > 0) so NaN fails them, as in Fortran.
//
// NROOT contract: 1 <= NROOT <= NSOIL (module_LS_NOAHLSM.F90:2360). The VEGPARM table loader
// enforces it (phys_noah_parm_tables.py, NoahParmTableError). The kernel does not index out of
// range on bad input: an active column with NROOT outside 1..NSOIL writes NaN to its ET, so the
// fault is visible in the output (the canres convention). NaN clamp semantics match gfortran:
// fminf/fmaxf drop a NaN argument, and gfortran MAX(MIN(NaN,1.),0.) gives 1.0 too (probed).
//
// Layout: 2-D point kernel, one thread per (i, j) that runs the soil-layer loops for its column,
// grid.z = 1 (CONTRACT section 3). Arrays: SMC, RTDIS, ET are [NSOIL, JM, IM] (layer 1 at the
// top); ETP1, CMC, SHDFAC, SMCWLT, CMCMAX, PC, CFACTR, SMCREF are [JM, IM]; NROOT is int32 [JM, IM]
// (per-column root depth from the vegetation table). No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

#ifndef NSOIL
#define NSOIL 4
#endif

// --------------------------------------------------------------- TRANSP (device function)
__device__ static void nam_transp(float* et, float etp1,
                                  const float* smc, float cmc,
                                  float shdfac, float smcwlt,
                                  float cmcmax, float pc, float cfactr,
                                  float smcref, int nroot,
                                  const float* rtdis)
{
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    // Caller gate (EVAPO 2600-2613): TRANSP is not called unless ETP1 > 0 and SHDFAC > 0.
    if (!(etp1 > 0.0f) || !(shdfac > 0.0f)) return;
    // NROOT contract (1 <= NROOT <= NSOIL): poison the column rather than index past gx[].
    if (nroot < 1 || nroot > NSOIL) {
        const float nan_v = __int_as_float(0x7fc00000);
        for (int k = 0; k < NSOIL; ++k) et[k] = nan_v;
        return;
    }
    float etp1a;
    if (cmc != 0.0f)
        etp1a = shdfac * pc * etp1 * (1.0f - powf(cmc / cmcmax, cfactr));
    else
        etp1a = shdfac * pc * etp1;
    float gx[NSOIL];
    float sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (smc[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = fmaxf(fminf(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (float)nroot;
    float denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        float rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * fmaxf(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

// --------------------------------------------------------------- one launch, one thread per column
extern "C" __global__ void phys_noah_transp(
    const float* __restrict__ smc,      // [NSOIL, JM, IM]  SH2O at the call site (dummy SMC)
    const float* __restrict__ etp1,     // [JM, IM]         ETP * 0.001 (NOPAC 3247)
    const float* __restrict__ cmc,      // [JM, IM]
    const float* __restrict__ shdfac,   // [JM, IM]
    const float* __restrict__ smcwlt,   // [JM, IM]
    const float* __restrict__ cmcmax,   // [JM, IM]
    const float* __restrict__ pc,       // [JM, IM]
    const float* __restrict__ cfactr,   // [JM, IM]
    const float* __restrict__ smcref,   // [JM, IM]
    const int* __restrict__ nroot,      // [JM, IM]
    const float* __restrict__ rtdis,    // [NSOIL, JM, IM]
    float* __restrict__ et,             // [NSOIL, JM, IM]  output
    int im, int jm)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;

    const size_t p = IDX2(i0, j0, im);
    float smc_c[NSOIL], rtdis_c[NSOIL], et_c[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        const size_t q = IDX3(i0, j0, k, im, jm);
        smc_c[k] = smc[q];
        rtdis_c[k] = rtdis[q];
    }
    nam_transp(et_c, etp1[p], smc_c, cmc[p], shdfac[p], smcwlt[p], cmcmax[p],
               pc[p], cfactr[p], smcref[p], nroot[p], rtdis_c);
    for (int k = 0; k < NSOIL; ++k)
        et[IDX3(i0, j0, k, im, jm)] = et_c[k];
}
