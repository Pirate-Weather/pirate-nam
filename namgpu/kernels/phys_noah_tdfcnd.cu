// phys_noah_tdfcnd.cu -- TDFCND: soil thermal diffusivity (Noah LSM), Peters-Lidard approach.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The NOAA module header of
// the source file is kept as the notice (phys/module_LS_NOAHLSM.F90, MODULE MODULE_LS_NOAHLSM):
//
//   !-----------------------------------------------------------------------
//   !
//         MODULE MODULE_LS_NOAHLSM
//   !
//   !-----------------------------------------------------------------------
//   !***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
//   !-----------------------------------------------------------------------
//   ! HISTORY LOG:
//   !
//   !   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
//   !               Vasic - SOILW limitted to positive values.
//
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:4989-5092 (nam.v4.2.11, 83df4e13c1).
// Start point was the ArWen copy (namgpu/third_party/arwen/core/kernels/noah.cu, noah_tdfcnd, Apache-2.0,
// Copyright 2026 ArWen contributors). That routine is WRF's TDFCND (opt_thcnd branch, Campbell
// psisat/bexp form), a different formulation and interface from NAM's Peters-Lidard TDFCND, so
// the body here is transcribed from the NAM Fortran, statement by statement; only the layout
// (CONTRACT section 1) and the launch shape (section 3) come from the ArWen file.
//
// Fortran SUBROUTINE TDFCND(DF, SMC, QZ, SMCMAX, SH2O), every executed statement:
//   SATRATIO = SMC / SMCMAX
//   THKICE = 2.2 ; THKW = 0.57 ; THKO = 2.0 ; THKQTZ = 7.7
//   THKS   = (THKQTZ**QZ) * (THKO**(1. - QZ))
//   XUNFROZ = SH2O / SMC ; XU = XUNFROZ * SMCMAX
//   THKSAT = THKS**(1. - SMCMAX) * THKICE**(SMCMAX - XU) * THKW**(XU)
//   GAMMD  = (1. - SMCMAX) * 2700.
//   THKDRY = (0.135*GAMMD + 64.7) / (2700. - 0.947*GAMMD)
//   IF ((SH2O + 0.0005) < SMC) THEN AKE = SATRATIO                     (frozen)
//   ELSE IF (SATRATIO > 0.1) THEN AKE = LOG10(SATRATIO) + 1.0           (unfrozen, fine Kersten)
//        ELSE AKE = 0.0
//   DF = AKE * (THKSAT - THKDRY) + THKDRY
// The commented-out THKO = 3.0 branch (QZ <= 0.2) is not executed in NAM and is not ported.
//
// Caller-side branches around the TDFCND calls are part of the kernel's inputs, selected by MODE
// (the Fortran callers, not TDFCND itself, apply them; they are transcribed here so every caller
// gets the same numbers):
//   MODE_SFLX_TOP  (SFLX, module_LS_NOAHLSM.F90:2003-2035 and 3350-3352): layer 1 only.
//                  ICE == 1: DF = 2.2 (sea ice, TDFCND not called). Else TDFCND, then if
//                  VEGTYP == IW: DF = 3.24 (urban, IW = 13 for IVEGSRC 1, else 1).
//   MODE_HRT       (HRT, module_LS_NOAHLSM.F90:2938-2941 and 2970-2973): every layer.
//                  VEGTYP == IW: DF = 3.24 after each TDFCND call. (HRTICE is not called here.)
//   MODE_NOPAC_TOP (NOPAC, module_LS_NOAHLSM.F90:3350-3352): layer 1 only. The Fortran tests the
//                  literal VEGTYP == 1 there, not IW, so the class is 1 for every IVEGSRC.
//
// Layout: point kernel, one thread per (i, j), the NSOIL = 4 soil layers looped inside the
// thread (CONTRACT section 3). SMC, SH2O, DF are float32 [NSOIL, JM, IM] (layer 1 at the top),
// indexed IDX3(i0, j0, k0, IM, JM). QZ and SMCMAX are float32 [JM, IM] (soil-type dependent,
// constant over the layers, as in the Fortran call sites). VEGTYP and ICE are int32 [JM, IM].
// Fortran REAL = float (kfpt=single), Fortran "**" with a REAL exponent is powf. fmad is left on
// (fast numerics).

#include "nmmb_common.cuh"

#define NSOIL 4

#define TDF_THKICE  2.2f
#define TDF_THKW    0.57f
#define TDF_THKO    2.0f
#define TDF_THKQTZ  7.7f
#define TDF_SEAICE_DF 2.2f
#define TDF_URBAN_DF  3.24f

#define TDF_MODE_SFLX_TOP  0
#define TDF_MODE_HRT       1
#define TDF_MODE_NOPAC_TOP 2

// Device function: TDFCND for one soil layer. Arguments as in the Fortran (QZ, SMC, SMCMAX, SH2O).
__device__ __forceinline__ float nam_tdfcnd(float qz, float smc, float smcmax, float sh2o)
{
    const float satratio = smc / smcmax;
    const float thks = powf(TDF_THKQTZ, qz) * powf(TDF_THKO, 1.0f - qz);
    const float xunfroz = sh2o / smc;
    const float xu = xunfroz * smcmax;
    const float thksat = powf(thks, 1.0f - smcmax)
                         * powf(TDF_THKICE, smcmax - xu)
                         * powf(TDF_THKW, xu);
    const float gammd = (1.0f - smcmax) * 2700.0f;
    const float thkdry = (0.135f * gammd + 64.7f) / (2700.0f - 0.947f * gammd);

    float ake;
    if ((sh2o + 0.0005f) < smc) {
        ake = satratio;                                        // frozen
    } else {
        if (satratio > 0.1f) ake = log10f(satratio) + 1.0f;   // unfrozen, Kersten (fine)
        else ake = 0.0f;
    }
    return ake * (thksat - thkdry) + thkdry;
}

extern "C" __global__ void phys_noah_tdfcnd(
    const float* __restrict__ smc,     // SMC     total soil moisture         [NSOIL, JM, IM]
    const float* __restrict__ sh2o,    // SH2O    unfrozen soil moisture      [NSOIL, JM, IM]
    const float* __restrict__ qz,      // QZ      quartz content (soil type)  [JM, IM]
    const float* __restrict__ smcmax,  // SMCMAX  porosity (soil type, SFLX urban rule) [JM, IM]
    const int* __restrict__ vegtyp,    // VEGTYP  IGBP/USGS class             [JM, IM]
    const int* __restrict__ ice,       // ICE     sea-ice flag                [JM, IM]
    float* __restrict__ df,            // DF      thermal diffusivity         [NSOIL, JM, IM] (out)
    int mode,                          // TDF_MODE_*: which caller's branches apply
    int iw,                            // IW: urban class (13 for IVEGSRC 1, else 1)
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n2 = IDX2(i0, j0, im);
    const float q = qz[n2];
    const float smx = smcmax[n2];
    const int vt = vegtyp[n2];
    const int ic = ice[n2];
    for (int k0 = 0; k0 < NSOIL; ++k0) {
        const size_t n3 = IDX3(i0, j0, k0, im, jm);
        float v = nam_tdfcnd(q, smc[n3], smx, sh2o[n3]);
        if (mode == TDF_MODE_SFLX_TOP) {
            if (k0 == 0) {
                if (ic == 1) v = TDF_SEAICE_DF;
                else if (vt == iw) v = TDF_URBAN_DF;
            }
        } else if (mode == TDF_MODE_HRT) {
            if (vt == iw) v = TDF_URBAN_DF;
        } else if (mode == TDF_MODE_NOPAC_TOP) {
            if (k0 == 0 && vt == 1) v = TDF_URBAN_DF;
        }
        df[n3] = v;
    }
}
