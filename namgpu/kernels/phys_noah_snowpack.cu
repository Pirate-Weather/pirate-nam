// phys_noah_snowpack.cu -- SNOWPACK: snowpack compaction (Noah LSM).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:4378-4505 (nam.v4.2.11, 83df4e13c1).
// Derived from the ArWen Noah kernel (namgpu/third_party/arwen/core/kernels/noah.cu, Apache-2.0, see
// namgpu/third_party/arwen/LICENSE). Its noah_snowpack body was the starting point; the NAM differences
// listed below are applied so the kernel follows the NAM Fortran, not the WRF-lineage code.
//
// Source: SUBROUTINE SNOWPACK (ESD, DTSEC, SNOWH, SNDENS, TSNOW, TSOIL). Called from SNOPAC
// (module_LS_NOAHLSM.F90:4364-4366, IF (ESD > 0.) THEN CALL SNOWPACK (ESD,DT,SNOWH,SNDENS,T1,YY)),
// with YY as TSOIL. The SNOPAC ELSE branch (ESD = 0, SNOWH = 0, SNDENS = 0) is ported here as the
// call-site gate, so the kernel is safe to launch on any column. SNCOND = 1 and SNCOVR = 0 from
// that branch are SNOPAC outputs, not SNOWPACK state, and belong to the SNOPAC unit. Every executed
// statement of the routine is ported:
//   SNOWHC = SNOWH*100; ESDC = ESD*100; DTHR = DTSEC/3600; TSNOWC = TSNOW-273.15; TSOILC = TSOIL-273.15
//   TAVGC = 0.5*(TSNOWC+TSOILC)
//   ESDCX = ESDC if ESDC > 1E-2 else 1E-2
//   BFAC = DTHR*C1*EXP(0.08*TAVGC - C2*SNDENS)           (C1 = 0.01, C2 = 21.0)
//   PEXP = 0; DO J = IPOL,1,-1 (IPOL = 4): PEXP = (1+PEXP)*BFAC*ESDCX/REAL(J+1); PEXP = PEXP+1
//   DSX = SNDENS*PEXP; clamp DSX to [0.05, 0.9]          (NAM 2013 limit 0.9, not ArWen 0.40)
//   SNDENS = DSX
//   IF TSNOWC >= 0: DW = 0.13*DTHR/24; SNDENS = SNDENS*(1-DW)+DW; clamp SNDENS >= 0.9 to 0.9
//   SNOWHC = ESDC/SNDENS; SNOWH = SNOWHC*0.01
// The Coley/Koren commented-out forms in the Fortran are not executed and are not ported.
// The J loop is kept as a per-column loop; the IPOL=4 polynomial is evaluated in the same
// order as the Fortran (left to right), in single precision (kfpt=single).
//
// Layout: 2-D point kernel, one thread per (i, j), grid.z = 1 (CONTRACT section 3).
// Arrays are float32 [JM, IM] (C order, IDX2). Scalar DTSEC is float. TSNOW, TSOIL are inputs;
// ESD, SNOWH and SNDENS are read and written in place (INTENT INOUT, as in SNOPAC). No host sync.

#include "nmmb_common.cuh"

#define NAM_SNOWPACK_C1   0.01f
#define NAM_SNOWPACK_C2   21.0f
#define NAM_SNOWPACK_IPOL 4

extern "C" __global__ void phys_noah_snowpack(
    float* __restrict__ esd,           // ESD     water equivalent of snow (m) (INOUT) [JM, IM]
    float dtsec,                       // DTSEC   time step (s)                         (scalar)
    float* __restrict__ snowh,         // SNOWH   snow depth (m)                        (INOUT) [JM, IM]
    float* __restrict__ sndens,        // SNDENS  snow density (fraction of H2O)        (INOUT) [JM, IM]
    const float* __restrict__ tsnow,   // TSNOW   snow surface temperature (K) (IN)     [JM, IM]
    const float* __restrict__ tsoil,   // TSOIL   soil surface temperature (K) (IN)     [JM, IM]
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);

    // SNOPAC call-site gate (module_LS_NOAHLSM.F90:4364-4370): SNOWPACK runs only when ESD > 0.
    // The ELSE branch resets ESD, SNOWH and SNDENS to zero. NaN fails the test and takes the ELSE
    // branch, as in Fortran.
    if (!(esd[n] > 0.0f)) {
        esd[n] = 0.0f;
        snowh[n] = 0.0f;
        sndens[n] = 0.0f;
        return;
    }

    // CONVERSION INTO SIMULATION UNITS
    const float esdc = esd[n] * 100.0f;        // ESDC
    const float dthr = dtsec / 3600.0f;        // DTHR
    const float tsnowc = tsnow[n] - 273.15f;   // TSNOWC
    const float tsoilc = tsoil[n] - 273.15f;   // TSOILC

    // AVERAGE TEMPERATURE OF SNOW PACK
    const float tavgc = 0.5f * (tsnowc + tsoilc);

    float esdcx;
    if (esdc > 1.0e-2f) esdcx = esdc;
    else esdcx = 1.0e-2f;

    // COMPACTION: (e**x-1)/x replaced by the IPOL-term polynomial (Fortran lines 4440-4460)
    const float sd = sndens[n];
    const float bfac = dthr * NAM_SNOWPACK_C1 * expf(0.08f * tavgc - NAM_SNOWPACK_C2 * sd);
    float pexp = 0.0f;
    for (int j = NAM_SNOWPACK_IPOL; j >= 1; --j)
        pexp = (1.0f + pexp) * bfac * esdcx / (float)(j + 1);
    pexp = pexp + 1.0f;

    // SET UPPER/LOWER LIMIT ON SNOW DENSITY (NAM: 0.9 upper limit, 2013 change)
    float dsx = sd * pexp;
    if (dsx > 0.9f) dsx = 0.9f;
    if (dsx < 0.05f) dsx = 0.05f;

    // MELT: 13% of liquid water stored per day till density 0.9 (NAM: 0.9 cap)
    float sdn = dsx;
    if (tsnowc >= 0.0f) {
        const float dw = 0.13f * dthr / 24.0f;
        sdn = sdn * (1.0f - dw) + dw;
        if (sdn >= 0.9f) sdn = 0.9f;
    }
    sndens[n] = sdn;

    // SNOW DEPTH (cm) from SWE and density, converted to m
    const float snowhc = esdc / sdn;
    snowh[n] = snowhc * 0.01f;
}
