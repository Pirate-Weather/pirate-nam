// phys_rad_radconst.cu -- RADCONST: solar declination and solar constant for the forecast date.
// DEAD IN NAM: nothing in nam.v4.2.11 calls radconst (definition at module_RADIATION.F90:1000-1042 only).
// Kept as a faithful port with no consumer. The live solar path is sol_update_nmmb (radiation_astronomy_nmmb.f).
//
// NOTICE: header block of nmm/module_RADIATION.F90 (NOAA/NCEP/EMC, US Government work, no licence file),
// nam.v4.2.11 83df4e13c1, kept verbatim as the notice (CONTRACT section 3):
// !-----------------------------------------------------------------------
// !
//       MODULE MODULE_RADIATION
// !
// !-----------------------------------------------------------------------
// !
// !***  THE RADIATION DRIVERS AND PACKAGES
// !
// !---------------------
// !--- Modifications ---
// !---------------------
// ! 2010-04-02 Vasic - Removed WFR driver
// !-----------------------------------------------------------------------
// !
//
// Ported to CUDA for WOOF-NAM from nmm/module_RADIATION.F90:1000-1042 (nam.v4.2.11, 83df4e13c1).
//
// Fortran: SUBROUTINE radconst(XTIME, DECLIN, SOLCON, JULIAN)
// Every statement of lines 1000-1042 and its counterpart here:
//   REAL DEGRAD = 3.1415926/180.     -> DEGRAD: the Fortran single literal, NOT pi/180 from nmmb_common.cuh
//   REAL DPD = 360./365.             -> DPD (local with initialiser, implicit SAVE; constant value)
//   DECLIN = 0.; SOLCON = 0.         -> out[0] = out[1] = 0 first (overwritten below, as in Fortran)
//   OBECL = 23.5*DEGRAD; SINOB = SIN(OBECL)                 -> obecl, sinob
//   IF(JULIAN.GE.80.) SXLONG = DPD*(JULIAN-80.)             -> sxlong branch 1
//   IF(JULIAN.LT.80.) SXLONG = DPD*(JULIAN+285.)            -> sxlong branch 2
//   SXLONG = SXLONG*DEGRAD; ARG = SINOB*SIN(SXLONG)         -> sxlong, arg
//   DECLIN = ASIN(ARG)                                       -> out[0] (radians)
//   DECDEG = DECLIN/DEGRAD            -> Fortran computes it but never reads it; no output, omitted
//   DJUL = JULIAN*360./365.; RJUL = DJUL*DEGRAD             -> djul, rjul
//   ECCFAC = 1.000110 + 0.034221*COS(RJUL) + 0.001280*SIN(RJUL)
//          + 0.000719*COS(2*RJUL) + 0.000077*SIN(2*RJUL)     -> eccfac (Paltridge and Platt 1976)
//   SOLCON = 1370.*ECCFAC                                    -> out[1] (W m-2)
// XTIME is an argument the Fortran body never references; it is kept in the signature only.
//
// Scalar routine (no loops): one thread, one launch, no host sync, no allocation.
// Output: out[0] = DECLIN [rad], out[1] = SOLCON [W m-2], both float32 (Fortran REAL).

extern "C" __global__ void radconst(const float xtime, const float julian, float* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)xtime;   // not referenced by the Fortran routine

    const float DEGRAD = 3.1415926f / 180.0f;
    const float DPD = 360.0f / 365.0f;

    float declin = 0.0f;
    float solcon = 0.0f;

    // Obliquity = 23.5 degrees.
    const float obecl = 23.5f * DEGRAD;
    const float sinob = sinf(obecl);

    // Longitude of the sun from vernal equinox.
    float sxlong = 0.0f;
    if (julian >= 80.0f) sxlong = DPD * (julian - 80.0f);
    if (julian < 80.0f)  sxlong = DPD * (julian + 285.0f);

    sxlong = sxlong * DEGRAD;
    const float arg = sinob * sinf(sxlong);
    declin = asinf(arg);

    // Solar constant eccentricity factor (Paltridge and Platt 1976).
    const float djul = julian * 360.0f / 365.0f;
    const float rjul = djul * DEGRAD;
    const float eccfac = 1.000110f + 0.034221f * cosf(rjul) + 0.001280f * sinf(rjul)
                       + 0.000719f * cosf(2.0f * rjul) + 0.000077f * sinf(2.0f * rjul);
    solcon = 1370.0f * eccfac;

    out[0] = declin;
    out[1] = solcon;
}
