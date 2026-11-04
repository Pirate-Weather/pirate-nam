// phys_jsfc_init.cu -- unit phys-jsfc-init: JSFC_INIT (surface-layer integral-function tables,
// initial USTAR), WOOF-NAM.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   MODULE_SF_JSFC (Janjic J surface scheme, module_SF_JSFC.F90), nam.v4.2.11.
// Ported to CUDA for WOOF-NAM from phys/module_SF_JSFC.F90:1014-1251 (nam.v4.2.11, 83df4e13c1).
//
// Fortran JSFC_INIT, every term accounted for:
//  - Stability-function tables (module arrays KZTM=10001 entries, module_SF_JSFC.F90:65-69):
//      ZTMIN1=ZTMIN2=-5, ZTMAX1=ZTMAX2=9 (:1143-1144; the earlier variants at :1133-1138 are
//      commented out), ZRNG = ZTMAX - ZTMIN = 14 (:1148), DZETA = ZRNG/(KZTM-1) (:1151).
//      Loop K=1..KZTM: zeta = ZETA1 (resp. ZETA2) by repeated single-precision addition of DZETA,
//      unstable (zeta<0) Paulson 1970 PSIM/PSIH, stable (zeta>=0) Holtslag and de Bruin 1988
//      PSIM = PSIH = 0.7 z + 0.75 z (6 - 0.35 z) exp(-0.35 z). At K=KZTM, ZTMAX = zeta (:1238);
//      after the loop ZTMAX = ZTMAX - EPS (EPS=1.E-6, :1247).
//      Each table is built by one thread with the serial accumulation, so the zeta sequence is the
//      Fortran's float rounding sequence. This runs once at init, off the step path.
//  - FH01 = FH02 = 1 (:1130; FH02 likewise).
//  - Initial USTAR (cold start only, :1120): USTAR(I,J) = 0.1 for J = JTS..JTE, I = ITS..ITF with
//      ITF = MIN(ITE, IDE-1) = IM-1 on the full domain (single card), so column i = IM is not set.
//  - Not written, dropped with reason: the MPI_ALLREDUCE of MAXVAL(IVGTYP) (:1086) is a
//      diagnostic with no output, and the VZ0TBL/VZ0TBL_24 Z0 assignments are commented out in the
//      source (`!zj Z0(I,J)=...` at :1096 and :1107), so LOWLYR and Z0 are not written. The RESTART
//      branch skips the USTAR write exactly as the Fortran does.
//
// Outputs (device, float32):
//   psim1, psih1, psim2, psih2 : float[kztm]   (the JSFC module's own PSIM1/PSIH1/PSIM2/PSIH2)
//   scal[8] : [0] ztmin1, [1] ztmax1 (after -eps), [2] ztmin2, [3] ztmax2 (after -eps),
//             [4] dzeta1, [5] dzeta2, [6] fh01, [7] fh02
//   USTAR[JM, IM] (CONTRACT layout, [j][i]) : 0.1 on i0 < IM-1 when restart is 0.

#include "nmmb_common.cuh"

#define JSFC_KZTM     10001
#define JSFC_EPS      1.0e-6f
#define JSFC_PIHF     (3.1415926f / 2.0f)     /* JSFC_INIT local PIHF = 3.1415926/2. */

__device__ __forceinline__ float jsfc_psim(const float zeta)
{
    if (zeta < 0.0f) {                                      /* unstable, Paulson 1970 */
        const float x = sqrtf(sqrtf(1.0f - 16.0f * zeta));
        return -2.0f * logf((x + 1.0f) / 2.0f) - logf((x * x + 1.0f) / 2.0f)
               + 2.0f * atanf(x) - JSFC_PIHF;
    }
    /* stable, Holtslag and de Bruin 1988 (Fortran evaluates left to right) */
    return 0.7f * zeta + 0.75f * zeta * (6.0f - 0.35f * zeta) * expf(-0.35f * zeta);
}

__device__ __forceinline__ float jsfc_psih(const float zeta)
{
    if (zeta < 0.0f) {                                      /* unstable, Paulson 1970 */
        const float x = sqrtf(sqrtf(1.0f - 16.0f * zeta));
        return -2.0f * logf((x * x + 1.0f) / 2.0f);
    }
    /* stable, Holtslag and de Bruin 1988 (PSIH = PSIM in the stable range) */
    return 0.7f * zeta + 0.75f * zeta * (6.0f - 0.35f * zeta) * expf(-0.35f * zeta);
}

// One thread per table (threadIdx.x = 0 -> ZETA1/PSIM1/PSIH1, 1 -> ZETA2/PSIM2/PSIH2).
// Serial over K so the zeta accumulation matches the Fortran rounding sequence.
extern "C" __global__ void phys_jsfc_init_tables(
    float* __restrict__ psim1,      // [KZTM]
    float* __restrict__ psih1,      // [KZTM]
    float* __restrict__ psim2,      // [KZTM]
    float* __restrict__ psih2,      // [KZTM]
    float* __restrict__ scal)       // [8]
{
    const int which = threadIdx.x;  // 0 or 1; launched with blockDim.x == 2
    if (blockIdx.x != 0 || which > 1) return;

    const float ztmin = -5.0f;      // ZTMIN1 / ZTMIN2
    const float ztmax_init = 9.0f;  // ZTMAX1 / ZTMAX2 before the loop
    const float zrng = ztmax_init - ztmin;
    const float dzeta = zrng / (float)(JSFC_KZTM - 1);

    float* psim = which == 0 ? psim1 : psim2;
    float* psih = which == 0 ? psih1 : psih2;

    float zeta = ztmin;
    float ztmax = ztmax_init;
    for (int k0 = 0; k0 < JSFC_KZTM; ++k0) {
        psim[k0] = jsfc_psim(zeta);
        psih[k0] = jsfc_psih(zeta);
        if (k0 == JSFC_KZTM - 1) ztmax = zeta;   /* Fortran IF(K==KZTM) ZTMAX = ZETA */
        zeta = zeta + dzeta;                     /* Fortran ZETA = ZETA + DZETA */
    }
    ztmax = ztmax - JSFC_EPS;

    if (which == 0) {
        scal[0] = ztmin;   /* ztmin1 */
        scal[1] = ztmax;   /* ztmax1 */
        scal[4] = dzeta;   /* dzeta1 */
        scal[6] = 1.0f;    /* fh01 */
        scal[7] = 1.0f;    /* fh02 */
    } else {
        scal[2] = ztmin;   /* ztmin2 */
        scal[3] = ztmax;   /* ztmax2 */
        scal[5] = dzeta;   /* dzeta2 */
    }
}

// Cold-start USTAR: one thread per (i, j) on the full 2-D grid (grid_2d with lm = 1).
// Fortran loops are DO J=JTS,JTE / DO I=ITS,ITF with ITF = MIN(ITE, IDE-1) = IM-1.
extern "C" __global__ void phys_jsfc_init_ustar(
    float* __restrict__ ustar,      // [JM, IM]
    const int im,
    const int jm,
    const int itf,                  // ITF = IM-1 on the full domain
    const int restart)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (restart) return;
    if (i0 >= itf || j0 >= jm) return;
    ustar[IDX2(i0, j0, im)] = 0.1f;
}
