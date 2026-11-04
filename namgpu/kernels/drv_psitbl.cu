// WOOF-NAM unit drv-psitbl: surface-layer integral stability functions (PSITBL) table.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran file has no PROGRAM HISTORY LOG block; its module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_CONTROL.F90:2189-2300 (nam.v4.2.11, 83df4e13c1).
//
// Fortran subroutine psitbl (responsible person z.janjic) fills, on the sea (1) and land (2) tables:
//   psim1, psih1, psim2, psih2 (1:kztm, kfpt = single) on z/L in [ztmin, ztmax] with step dzeta,
//   ztmin1 = ztmin2 = -5, ztmax1 = ztmax2 = 1 (before the loop), fh01 = fh02 = 1, and
//   kztm2 = kztm - 2 (module_CONTROL.F90:34 kztm = 10001). After the loop ztmax1/2 are set to the
//   value zeta had at k = kztm, minus eps (eps = 1e-6, the routine's parameter).
// Stability functions, per table point with z = z/L:
//   z < 0 (unstable, Paulson 1970):  x = sqrt(sqrt(1 - 16 z))
//        psim = -2 log((x+1)/2) - log((x*x+1)/2) + 2 atan(x) - pi/2
//        psih = -2 log((x*x+1)/2)
//   z >= 0 (stable, Holtslag and de Bruin 1988):
//        psim = psih = 0.7 z + 0.75 z (6 - 0.35 z) exp(-0.35 z)
// Both land and sea tables use the same functions; the two tables differ only in ztmin/ztmax
// bookkeeping, which here is identical (the Fortran also sets them identically).
//
// Mapping of the Fortran loop to threads: one thread per table point k (0-based k0 = k-1). The
// Fortran builds zeta by repeated single-precision addition, zeta = zeta + dzeta, once per k, so the
// value at k0 is the float sum of k0 additions from ztmin. Each thread replays exactly that sequence
// for its own k0 (at most kztm-1 adds, init only, no shared state, no sync), so every zeta, and
// therefore every table value, is the Fortran's float value at that index. dzeta is the IEEE quotient
// zrng/(kztm-1) (__fdiv_rn), as the Fortran computes it.
// ztmax (the accumulated zeta at k = kztm, i.e. after kztm-1 additions) minus eps is written by the
// thread for k0 = kztm-1 into scal[1] / scal[3].
//
// Outputs (device, float32):
//   psim1, psih1, psim2, psih2 : float[kztm]
//   scal[8] : [0] ztmin1, [1] ztmax1 (after -eps), [2] ztmin2, [3] ztmax2 (after -eps),
//             [4] dzeta1, [5] dzeta2, [6] fh01, [7] fh02
// The wrapper keeps kztm and kztm2 as host integers.
//
// Fortran constants in kfpt = single: pi = 3.141592653589793 -> float; pihf = 0.5*pi.

#include "nmmb_common.cuh"

#define PSITBL_EPS   1.0e-6f
#define PSITBL_PIHF  (0.5f * 3.1415926535897931f)

__device__ __forceinline__ float psitbl_psim(const float z)
{
    if (z < 0.0f) {                                   // unstable range, Paulson 1970
        const float x = sqrtf(sqrtf(1.0f - 16.0f * z));
        return -2.0f * logf((x + 1.0f) / 2.0f) - logf((x * x + 1.0f) / 2.0f)
               + 2.0f * atanf(x) - PSITBL_PIHF;
    }
    // stable range, Holtslag and de Bruin 1988
    return 0.7f * z + 0.75f * z * (6.0f - 0.35f * z) * expf(-0.35f * z);
}

__device__ __forceinline__ float psitbl_psih(const float z)
{
    if (z < 0.0f) {                                   // unstable range, Paulson 1970
        const float x = sqrtf(sqrtf(1.0f - 16.0f * z));
        return -2.0f * logf((x * x + 1.0f) / 2.0f);
    }
    // stable range, Holtslag and de Bruin 1988
    return 0.7f * z + 0.75f * z * (6.0f - 0.35f * z) * expf(-0.35f * z);
}

extern "C" __global__ void drv_psitbl_table(
    float* __restrict__ psim1,   // [kztm] sea momentum
    float* __restrict__ psih1,   // [kztm] sea heat
    float* __restrict__ psim2,   // [kztm] land momentum
    float* __restrict__ psih2,   // [kztm] land heat
    float* __restrict__ scal,    // [8]    ztmin1, ztmax1, ztmin2, ztmax2, dzeta1, dzeta2, fh01, fh02
    const int kztm)
{
    const int k0 = blockIdx.x * blockDim.x + threadIdx.x;

    // Fortran: ztmin1 = -5, ztmax1 = 1, same for 2; zrng = ztmax - ztmin = 6; dzeta = zrng/(kztm-1)
    const float ztmin1 = -5.0f;
    const float ztmax1 = 1.0f;
    const float ztmin2 = -5.0f;
    const float ztmax2 = 1.0f;
    const float zrng1 = ztmax1 - ztmin1;
    const float zrng2 = ztmax2 - ztmin2;
    const float dzeta1 = __fdiv_rn(zrng1, (float)(kztm - 1));
    const float dzeta2 = __fdiv_rn(zrng2, (float)(kztm - 1));

    if (k0 == 0) {
        scal[0] = ztmin1;
        scal[2] = ztmin2;
        scal[4] = dzeta1;
        scal[5] = dzeta2;
        scal[6] = 1.0f;                               // fh01
        scal[7] = 1.0f;                               // fh02
    }
    if (k0 >= kztm) return;

    // Fortran zeta at table index k0: zeta = ztmin, then k0 times zeta = zeta + dzeta.
    float zeta1 = ztmin1;
    float zeta2 = ztmin2;
    for (int j = 0; j < k0; ++j) {
        zeta1 = zeta1 + dzeta1;
        zeta2 = zeta2 + dzeta2;
    }

    psim1[k0] = psitbl_psim(zeta1);
    psih1[k0] = psitbl_psih(zeta1);
    psim2[k0] = psitbl_psim(zeta2);
    psih2[k0] = psitbl_psih(zeta2);

    if (k0 == kztm - 1) {                             // Fortran: if (k.eq.kztm) ztmax = zeta
        scal[1] = zeta1 - PSITBL_EPS;
        scal[3] = zeta2 - PSITBL_EPS;
    }
}
