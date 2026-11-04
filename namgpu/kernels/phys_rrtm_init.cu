// phys_rrtm_init.cu -- unit phys-rrtm-init: RRTM_INIT + GAUSIN, nam-gpu.
//
// NOTICE: NOAA/NCEP NMMB source, module MODULE_RA_RRTM (phys/module_RA_RRTM.F90, nam.v4.2.11,
//   83df4e13c1). US government work; no licence file in the NAM clone. Notice block copied from the
//   Fortran (the clone has no author header block for this file). Verbatim Fortran lines:
//     module_RA_RRTM.F90:1424-1426  SUBROUTINE RRTM_INIT(EMISS,SFULL,SHALF,PPTOP, ...
//     module_RA_RRTM.F90:1499       and xsd # of standard deviation (03/22/2004  Hsin-mu Lin)
//   The GAUSIN author credit is Hsin-Mu Lin (2004).
// Ported to CUDA for nam-gpu from phys/module_RA_RRTM.F90:1424-1517 (nam.v4.2.11, 83df4e13c1).
//
// Fortran RRTM_INIT / GAUSIN, every term accounted for:
//  - Module constants (module_RA_RRTM.F90:44-47): XSDmax = 3.1, DXSD = .01, NXSD = XSDmax/DXSD = 310
//    (integer truncation of the single-precision quotient, which rounds to 310.0). AXSD(NXSD) and RSQR
//    are module SAVE variables; here they are device arrays in ctx.tables["rrtm_init"].
//  - Module arrays LTOP(3) and PTOPC(4) (module_RA_RRTM.F90:51-52), also SAVE, also in ctx.tables.
//  - EMISS(I,J) = 1.0 over the tile (:1475-1479). Single card, single tile: ITE=IDE=IM, JTE=JDE=JM
//    (the wrapper states the caller limits and their single-tile equality).
//  - SQR2PI = SQRT(2.*PI) and RSQR = 1./SQR2PI (:1482-1483). PI is the MODULE_CONSTANTS real(kfpt)
//    parameter 3.141592653589793 (share/module_CONSTANTS.F90:29), so single precision is 3.14159274.
//  - AXSD(I) = GAUSIN(REAL(I)*DXSD), I = 1..NXSD (:1484-1486). GAUSIN (:1494-1517) is the Gaussian
//    area between the mean and xsd standard deviations, evaluated by the series in the Fortran loop
//    (crit = 1.e-3, do-while on b2 > crit). The FUNCTION body is line for line in rrtm_gausin().
//  - LTOP (:1452-1460): for N = 1..KTE (KTE = LM = 60, from the caller's KTS..KTE = 1..LM),
//    PCLD = (SSLP - PPTOP*10)*SHALF(N) + PPTOP*10 with SSLP = 1013.25, and LTOP(k) = N whenever PCLD
//    passes the k-th threshold. Each Fortran IF overwrites LTOP(k) with the current N, so the final
//    value is the LAST N that passes, i.e. the largest N that passes. That is an atomicMax over N, so
//    the loop is parallel here and gives the same result. LTOP must be zeroed before the launch (the
//    wrapper does that, matching :1452-1454).
//  - PTOPC (:1466-1469): PLBTM = 105000, PTOP_LO*100 = 64200, PTOP_MID*100 = 35000, PTOP_HI*100 = 15000.
//  - The Fortran PRINT at :1461 is a commented-out line in module_RA_RRTM.F90 (`!       PRINT *,N,PCLD,...`).
//    Nothing is dropped; the statement does not exist in the compiled code and there is no print here.
//  - Not written: the RRTM_INIT dummy argument SFULL is used only to build SHALF (caller side,
//    :10265-10269). The RRTM_INIT body itself never reads SFULL, so no term is ported from it.
//
// SHALF input: SMID_FLIP as the caller builds it from SFULL = NAM header SGM (see the wrapper). A stand-in
// for SGM is refused there, not used here.
//
// Outputs (device, float32 unless noted):
//   emiss[JM, IM]           : 1.0 everywhere (ctx.st["EMISS"])
//   axsd[NXSD = 310]        : Gaussian area table
//   rsqr[1]                 : 1/sqrt(2*pi)
//   ltop[3] (int32, 1-based): LTOP(1..3)
//   ptopc[4]                : PTOPC(1..4) in Pa
// Inputs: shalf[LM] (SMID_FLIP as the caller builds it, see the wrapper), pptop (PT_CB, kPa-scale value
// from the caller's PT*1.0E-3), kte = LM.

#include "nmmb_common.cuh"

#define RRTM_NXSD       310                 /* NXSD = XSDmax/DXSD, module_RA_RRTM.F90:44-45 */
#define RRTM_DXSD       0.01f               /* module_RA_RRTM.F90:44 */
#define RRTM_PI         3.14159274f         /* MODULE_CONSTANTS pi (real kfpt) */
#define RRTM_SSLP       1013.25f            /* RRTM_INIT local SSLP (:1437) */
#define RRTM_PTOP_HI    150.0f              /* RRTM_INIT PARAMETER (:1438) */
#define RRTM_PTOP_MID   350.0f
#define RRTM_PTOP_LO    642.0f
#define RRTM_PLBTM      105000.0f

// GAUSIN(xsd), module_RA_RRTM.F90:1494-1517, with RSQR the module variable.
__device__ __forceinline__ float rrtm_gausin(const float xsd, const float rsqr)
{
    const float crit = 1.0e-3f;                 /* REAL PARAMETER crit (:1496) */
    const float a1 = xsd * rsqr;                /* A1 = xsd*RSQR */
    const float a2 = expf(-0.5f * (xsd * xsd)); /* A2 = exp(-0.5*xsd**2) */
    float rn = 1.0f, b1 = 1.0f, b2 = 1.0f, b3 = 1.0f, sum = 1.0f;
    do {                                        /* do while (b2 .gt. crit) */
        rn = rn + 1.0f;
        b2 = (xsd * xsd) / (2.0f * rn - 1.0f);  /* B2 = xsd**2/(2.*rn-1.) */
        b3 = b1 * b2;
        sum = sum + b3;
        b1 = b3;
    } while (b2 > crit);
    return a1 * a2 * sum;                       /* GAUSIN = a1*a2*sum */
}

// Block 0 thread 0 also writes RSQR (SQR2PI = SQRT(2.*PI), RSQR = 1./SQR2PI). Every thread that
// needs RSQR derives the same value from the same expression, so the table is exact per thread.
extern "C" __global__ void phys_rrtm_init_gauss(float* __restrict__ axsd,
                                                float* __restrict__ rsqr_out)
{
    const float sqr2pi = sqrtf(2.0f * RRTM_PI); /* SQR2PI = SQRT(2.*PI) */
    const float rsqr = 1.0f / sqr2pi;           /* RSQR = 1./SQR2PI */
    if (blockIdx.x == 0 && threadIdx.x == 0) rsqr_out[0] = rsqr;

    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;   /* Fortran I = i0+1 */
    if (i0 >= RRTM_NXSD) return;
    const float xsd = (float)(i0 + 1) * RRTM_DXSD;          /* XSD = REAL(I)*DXSD */
    axsd[i0] = rrtm_gausin(xsd, rsqr);                      /* AXSD(I) = GAUSIN(XSD) */
}

// LTOP and PTOPC. One thread per level N = n0+1 (N = 1..KTE). ltop must be zeroed by the caller.
extern "C" __global__ void phys_rrtm_init_ltop(const float* __restrict__ shalf,
                                               const int kte,
                                               const float pptop,
                                               int* __restrict__ ltop,
                                               float* __restrict__ ptopc)
{
    const int n0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n0 == 0) {                              /* PTOPC (:1466-1469) */
        ptopc[0] = RRTM_PLBTM;
        ptopc[1] = RRTM_PTOP_LO * 100.0f;
        ptopc[2] = RRTM_PTOP_MID * 100.0f;
        ptopc[3] = RRTM_PTOP_HI * 100.0f;
    }
    if (n0 >= kte) return;
    const int N = n0 + 1;
    const float pcld = (RRTM_SSLP - pptop * 10.0f) * shalf[n0] + pptop * 10.0f;   /* :1457 */
    if (pcld >= RRTM_PTOP_LO)  atomicMax(&ltop[0], N);   /* LTOP(1)=N (:1458) */
    if (pcld >= RRTM_PTOP_MID) atomicMax(&ltop[1], N);   /* LTOP(2)=N (:1459) */
    if (pcld >= RRTM_PTOP_HI)  atomicMax(&ltop[2], N);   /* LTOP(3)=N (:1460) */
}

// EMISS(I,J) = 1.0 (:1475-1479). One thread per (i, j) point.
extern "C" __global__ void phys_rrtm_init_emiss(float* __restrict__ emiss, const int im, const int jm)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm) return;
    emiss[IDX2(i0, j0, im)] = 1.0f;
}
