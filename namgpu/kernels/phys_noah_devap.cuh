// phys_noah_devap.cuh -- DEVAP point function, shared by the DEVAP unit (phys_noah_devap.cu) and
// the EVAPO kernel (phys_noah_evapo.cu), which calls it inside its own gate.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the source
// module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Source: phys/module_LS_NOAHLSM.F90:2514-2554 (SUBROUTINE DEVAP), frozen-soil form (SH2O(1)).
//
// Apache-2.0 notice for the ArWen-derived starting point (noah_devap in
// namgpu/third_party/arwen/core/kernels/noah.cu): Copyright 2026 ArWen contributors. The full licence text
// is namgpu/third_party/arwen/LICENSE (Apache License, Version 2.0), and the copyright line is in
// namgpu/third_party/arwen/README.md. This header is not a copy of that licence; the notice is that file.
//
// Fortran (every executed statement):
//   SRATIO = (SH2O - SMCDRY) / (SMCMAX - SMCDRY)            (line 2540)
//   IF (SRATIO > 0.) THEN                                    (line 2541)
//     FX = SRATIO**FXEXP                                     (line 2542)
//     FX = MAX ( MIN ( FX, 1. ) ,0. )                        (line 2543)
//   ELSE
//     FX = 0.                                                (line 2545)
//   ENDIF
//   EDIR = FX * ( 1.0 - SHDFAC ) * ETP1                      (line 2551)
//
// Gate (unit boundary). The Fortran DEVAP has no gate: EVAPO calls it only when ETP1 > 0 and
// SHDFAC < 1 (module_LS_NOAHLSM.F90:2602-2603). The same test is applied here, so a direct call with
// ETP1 <= 0 or SHDFAC >= 1 returns EDIR = 0 (the value EVAPO's EDIR = 0 start gives) instead of a
// negative flux that NAM cannot produce. For every call NAM makes, the result is unchanged.
//
// Fast numerics: fmad on, IEEE division, powf for the Fortran exponentiation.

#ifndef PHYS_NOAH_DEVAP_CUH
#define PHYS_NOAH_DEVAP_CUH

__device__ static inline float noah_devap(float etp1, float sh2o1, float shdfac,
                                          float smcmax, float smcdry, float fxexp)
{
    if (!(etp1 > 0.0f) || !(shdfac < 1.0f)) return 0.0f;   // EVAPO gate, unit boundary
    const float sratio = (sh2o1 - smcdry) / (smcmax - smcdry);   // line 2540
    float fx;
    if (sratio > 0.0f) {                                          // line 2541
        fx = powf(sratio, fxexp);                                 // line 2542
        fx = fmaxf(fminf(fx, 1.0f), 0.0f);                        // line 2543
    } else {
        fx = 0.0f;                                                // line 2545
    }
    return fx * (1.0f - shdfac) * etp1;                           // EDIR (line 2551)
}

#endif
