// phys_gwd_init.cu -- unit phys-gwd-init: GWD_init, gravity-wave-drag constants and grid rotation
// (WOOF-NAM), CONTRACT.md section 3.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   MODULE_GWD (nmm/module_GWD.F90, Ferrier/Janjic GWD + mountain blocking), nam.v4.2.11.
// Ported to CUDA for WOOF-NAM from nmm/module_GWD.F90:32-119 (nam.v4.2.11, 83df4e13c1).
//
// Fortran GWD_init, module_GWD.F90:32-119 (line numbers checked against the NAM source):
//  - NMTVR = 14 (:79), DELTIM = DTPHS (:80), RDELTIM = 1./DTPHS (:81): module-level PRIVATE scalars
//    declared REAL(KIND=KIND_PHYS) (:20, 8-byte). Not device work: the wrapper writes them as float64
//    into ctx.tables["gwd"].
//  - DPHD0 = 0.108 (:73), single precision, used by CLEFF below.
//  - DTR = ACOS(-1.)/180. (:86), TPH0 = CEN_LAT*DTR, TLM0 = CEN_LON*DTR (:87-88), CTPH0 = COS(TPH0),
//    STPH0 = SIN(TPH0) (:89-90). Recomputed per thread here with the same single-precision operations.
//  - Rotation loop, DO J=JTS,JTE / DO I=ITS,ITE (:91-92 and the body to the ENDDO): RELM = GLON-TLM0,
//    SRLM, CRLM, SPH, CPH, TPH = ASIN(CTPH0*SPH - STPH0*CPH*CRLM), RCTPH = 1.0/COS(TPH),
//    SROT = STPH0*SRLM*RCTPH, CROT = (CTPH0*CPH + STPH0*SPH*CRLM)*RCTPH.
//  - RTD = 1./DTR (:105) and the HANGL loop (:106-108): HANGL = RTD*HANGL (INOUT, radians to degrees).
//    The Fortran runs the rotation loop and the HANGL loop separately; they touch disjoint arrays
//    so they share one launch here.
//  - CLEFF = CLEFFAMP*1.E-5*SQRT(DPHD/DPHD0) (:116): one scalar, one thread (phys_gwd_init_cleff).
//  - Loop bounds: the kernel covers i0 < IM, j0 < JM. That equals ITS..ITE, JTS..JTE only for the
//    single-tile layout of CONTRACT.md section 1 (one card holds the whole domain, ITS=IDS, ITE=IDE,
//    JTS=JDS, JDE). The wrapper cannot see the Fortran patch limits, so this is an assumption of the
//    layout, not a value the check script shows.
//  - Not ported, dropped with reason: the commented-out write(0,*) diagnostic (:117) and the
//    RESTRT argument, which the routine does not read.

// Outputs (device, float32, CONTRACT layout [JM, IM]):
//   CROT, SROT, HANGL (HANGL in degrees, in place)
//   cleff[1] (device scalar)

#include "nmmb_common.cuh"

#define GWD_PI      3.14159274f     /* ACOS(-1.) in single precision */
#define GWD_DPHD0   0.108f          /* module_GWD.F90 DPHD0, nominal 12-km grid resolution */

// One thread per H point (i0 = blockIdx.x*blockDim.x+threadIdx.x, j0 likewise).
// cen_lat, cen_lon in degrees (configfile tph0d, tlm0d). glat, glon in radians.
extern "C" __global__ void phys_gwd_init_rot(
    const float* __restrict__ glat,
    const float* __restrict__ glon,
    float* __restrict__ crot,
    float* __restrict__ srot,
    float* __restrict__ hangl,
    const float cen_lat,
    const float cen_lon,
    const int im,
    const int jm)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t p = IDX2(i0, j0, im);

    const float dtr = GWD_PI / 180.0f;           /* DTR = ACOS(-1.)/180. */
    const float rtd = 1.0f / dtr;                /* RTD = 1./DTR */
    const float tph0 = cen_lat * dtr;
    const float tlm0 = cen_lon * dtr;
    const float ctph0 = cosf(tph0);
    const float stph0 = sinf(tph0);

    const float relm = glon[p] - tlm0;
    const float srlm = sinf(relm);
    const float crlm = cosf(relm);
    const float sph = sinf(glat[p]);
    const float cph = cosf(glat[p]);
    const float tph = asinf(ctph0 * sph - stph0 * cph * crlm);
    const float rctph = 1.0f / cosf(tph);
    srot[p] = stph0 * srlm * rctph;
    crot[p] = (ctph0 * cph + stph0 * sph * crlm) * rctph;

    hangl[p] = rtd * hangl[p];                   /* HANGL = RTD*HANGL, INOUT */
}

// CLEFF = CLEFFAMP*1.E-5*SQRT(DPHD/DPHD0), left to right as in the Fortran.
extern "C" __global__ void phys_gwd_init_cleff(
    float* __restrict__ cleff,
    const float cleffamp,
    const float dphd)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cleff[0] = cleffamp * 1.0e-5f * sqrtf(dphd / GWD_DPHD0);
    }
}
