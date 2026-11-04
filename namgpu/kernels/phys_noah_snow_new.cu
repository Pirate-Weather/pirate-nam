// phys_noah_snow_new.cu -- SNOW_NEW: new-snow density and snow depth (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The module's NOAA
// header (MODULE MODULE_LS_NOAHLSM, module_LS_NOAHLSM.F90) is kept as the notice.
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:4530-4600 (nam.v4.2.11, 83df4e13c1).
// Start point: ArWen third_party/arwen/core/kernels/noah.cu (noah_snow_new, Apache-2.0,
// notice kept in the ArWen tree). The ArWen copy has no RIMEF1 densification block (the NAM
// 2013 addition), so that block is ported from the NAM Fortran.
//
// Fortran (SUBROUTINE SNOW_NEW(TEMP,T1,FFROZP,PRCP,RIMEF1,NEWSN,SNOWH,SNDENS)), every executed
// statement:
//   SNOWHC = SNOWH*100.; NEWSNC = NEWSN*100.
//   TEMPC = TEMP - 273.15
//   DSNEW = 0.05 if TEMPC <= -15, else 0.05 + 0.0017*(TEMPC+15)**1.5
//   RIMEF1 block: <2 keep DSNEW; <5 min(0.9, 2*DSNEW);
//                 <20 min(0.9, 4*DSNEW); else min(0.9, 10*DSNEW)
//   HNEWC = NEWSNC / DSNEW
//   if SNOWHC + HNEWC < 1.0E-3: SNDENS = MAX(DSNEW, SNDENS)
//   else: SNDENS = (SNOWHC*SNDENS + HNEWC*DSNEW) / (SNOWHC + HNEWC)
//   SNOWHC = SNOWHC + HNEWC; SNOWH = SNOWHC*0.01
// The Promeroy et al. (1998) DSNEW form is commented out in the Fortran and is not ported.
// The dummy arguments T1, FFROZP and PRCP are not referenced by the executed body, so they are
// not in this kernel's signature.
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3).
// Arrays are float32 [JM, IM] (C order, IDX2). Fortran REAL = float (kfpt=single).
// SNOWH and SNDENS are read and written in place (INTENT INOUT in the Fortran).

#include "nmmb_common.cuh"

__device__ inline void nam_snow_new(float temp, float rimef1, float newsn,
                                    float& snowh, float& sndens)
{
    float snowhc = snowh * 100.0f;                 // SNOWHC = SNOWH * 100.
    const float newsnc = newsn * 100.0f;           // NEWSNC = NEWSN * 100.
    const float tempc = temp - 273.15f;            // TEMPC  = TEMP - 273.15

    float dsnew;
    if (tempc <= -15.0f) {
        dsnew = 0.05f;
    } else {
        dsnew = 0.05f + 0.0017f * powf(tempc + 15.0f, 1.5f);
    }

    // 2013 RIMEF1 densification (NAM Fortran, SNOW_NEW body).
    if (rimef1 < 2.0f) {
        dsnew = dsnew;
    } else if (rimef1 < 5.0f) {
        dsnew = fminf(0.9f, 2.0f * dsnew);
    } else if (rimef1 < 20.0f) {
        dsnew = fminf(0.9f, 4.0f * dsnew);
    } else {
        dsnew = fminf(0.9f, 10.0f * dsnew);
    }

    const float hnewc = newsnc / dsnew;            // HNEWC = NEWSNC / DSNEW
    if (snowhc + hnewc < 1.0e-3f) {
        sndens = fmaxf(dsnew, sndens);             // SNDENS = MAX(DSNEW, SNDENS)
    } else {
        sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc);
    }
    snowhc = snowhc + hnewc;                       // SNOWHC = SNOWHC + HNEWC
    snowh = snowhc * 0.01f;                        // SNOWH = SNOWHC * 0.01
}

extern "C" __global__ void phys_noah_snow_new(
    const float* __restrict__ temp,     // TEMP    air temperature K (INTENT IN)   [JM, IM]
    const float* __restrict__ rimef1,   // RIMEF1  riming enhancement factor (IN)  [JM, IM]
    const float* __restrict__ newsn,    // NEWSN   new snowfall m (INTENT IN)      [JM, IM]
    float* __restrict__ snowh,          // SNOWH   snow depth m (INOUT)            [JM, IM]
    float* __restrict__ sndens,         // SNDENS  snow density fraction (INOUT)   [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    nam_snow_new(temp[n], rimef1[n], newsn[n], snowh[n], sndens[n]);
}
