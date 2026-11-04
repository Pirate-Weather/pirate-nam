// nam-gpu unit phys-myj-vdifv: vertical diffusion of the horizontal velocity components (MYJ PBL).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Header kept
// as the notice:
//   MODULE MODULE_BL_MYJPBL  (NMMB, nam.v4.2.11), THE MYJ PBL SCHEME
//   SUBROUTINE VDIFV(LMH,DTDIF,UZ0,VZ0,RKMS,D,U,V,RKM,Z,RHO,I,J,LM)  (phys/module_BL_MYJPBL.F90)
// Ported to CUDA for nam-gpu from phys/module_BL_MYJPBL.F90:1527-1607 (nam.v4.2.11, 83df4e13c1).
// The arithmetic is the NAM routine's, in its order: the tridiagonal coefficients CR and CM, the
// forward sweep for RSU/RSV, the top (LMH) row with the surface-exchange terms UZ0/VZ0, and the back
// substitution. Lower-case k below is the NAM level index: k0 = K-1, with K = 1 the MODEL TOP.
// The ArWen MYJ kernel (namgpu/third_party/arwen/core/kernels/myjpbl.cu, Apache-2.0, LICENSE in that tree)
// carries the same VDIFV algebra inside a bottom-up column kernel; this file is written top-down
// from the NAM Fortran, not copied across.
//
// Fortran -> kernel mapping (K 1-based in the Fortran, k0 = K-1 here):
//   DTOZ(K)=DTDIF/(Z(K)-Z(K+1)),  CR(K)=-DTOZ(K)*RKM(K)           K=1..LMH-1    (1537-1541)
//   RHOK=RHO(1); CM(1)=DTOZ(1)*RKM(1)+RHOK; RSU(1)=U(1)*RHOK; RSV(1)=V(1)*RHOK     (1544-1547)
//   K=2..LMH-1: CF=-DTOZL*RKM(K-1)/CM(K-1); RHOK=RHO(K);
//      CM(K)=-CR(K-1)*CF+(RKM(K-1)+RKM(K)+D(K)*RKMS)*DTOZL+RHOK;
//      RSU(K)=-RSU(K-1)*CF+U(K)*RHOK; RSV likewise                                 (1550-1558)
//   DTOZS=DTDIF/(Z(LMH)-Z(LMH+1)); RKMH=RKM(LMH-1); CF=-DTOZS*RKMH/CM(LMH-1);
//      RHOK=RHO(LMH); RCMVB=1/((RKMH+RKMS)*DTOZS-CR(LMH-1)*CF+RHOK); DTOZAK=DTOZS*RKMS  (1563-1570)
//   U(LMH)=(DTOZAK*UZ0-RSU(LMH-1)*CF+U(LMH)*RHOK)*RCMVB; V(LMH) likewise               (1575-1578)
//   K=LMH-1..1: RCML=1/CM(K); U(K)=(-CR(K)*U(K+1)+RSU(K))*RCML; V likewise            (1581-1585)
// Levels above LMH are not touched (the Fortran writes U and V only on 1..LMH).
// LMH is per column (the caller computes it per column, as the NAM driver does at 364/605), so it
// is an int field [JM, IM], not a scalar. A column with LMH < 2 is refused (the NAM routine would
// read CM(0) and RKM(0), which do not exist); the wrapper checks it on the host.
//
// Layout (CONTRACT section 1): 3-D fields [LM, JM, IM] with l = 0 the model top; Z is [LM+1, JM, IM];
// 2-D fields [JM, IM]. Thread per column (CONTRACT section 3, grid_cols). Fast numerics (fmad on),
// float throughout (kfpt = single), IEEE division (NVRTC default). No host sync, no device malloc,
// no printf. Column scratch is per-thread (5 x VDIFV_LMAX floats).

#include "nmmb_common.cuh"

#define VDIFV_LMAX 64   /* must equal LMAX in namgpu/physics/myj/phys_myj_vdifv.py (LM <= 64) */

extern "C" __global__ void phys_myj_vdifv_columns(
    int IM, int JM, int LM, float dtdif,
    const int* __restrict__ lmh_arr,        /* LMH [JM, IM] */
    const float* __restrict__ uz0,          /* UZ0 [JM, IM] */
    const float* __restrict__ vz0,          /* VZ0 [JM, IM] */
    const float* __restrict__ rkms,         /* AKMS_DENS [JM, IM], RKMS of the call */
    const float* __restrict__ D,            /* D [LM, JM, IM], levels 1..LM-1 used */
    const float* __restrict__ RKM,          /* RKM [LM, JM, IM], levels 1..LM-1 used */
    const float* __restrict__ Z,            /* Z [LM+1, JM, IM] (interfaces, Fortran Z(1..LM+1)) */
    const float* __restrict__ RHO,          /* RHO [LM, JM, IM] */
    float* __restrict__ U,                  /* U [LM, JM, IM], in/out */
    float* __restrict__ V)                  /* V [LM, JM, IM], in/out */
{
    COLUMN_IJ(IM, JM)
    const size_t c2 = IDX2(i0, j0, IM);
    const size_t S  = (size_t)JM * IM;       /* stride between levels in [l, j, i] */
    const int L = lmh_arr[c2];               /* Fortran LMH */
    /* Guard, reason: NAM VDIFV reads DTOZ(0) and RKM(0) (undefined) when LMH < 2. The MYJPBL caller
       sets LMH = LM (module_BL_MYJPBL.F90:364, 605) and refuses LM < 4 (phys_myj_myjpbl.py), so this
       branch is unreachable from it; the host wrapper refuses LM < 2 before launch. L > LM and
       LM > VDIFV_LMAX are likewise refused on the host (vdifv_columns). */
    if (L < 2 || L > LM || LM > VDIFV_LMAX) return;

    const float RKMS = rkms[c2];
    const float UZ0 = uz0[c2];
    const float VZ0 = vz0[c2];

    float dtoz[VDIFV_LMAX], cr[VDIFV_LMAX], cm[VDIFV_LMAX], rsu[VDIFV_LMAX], rsv[VDIFV_LMAX];

    /* DO K=1,LMH-1: DTOZ, CR (1537-1541) */
    for (int k0 = 0; k0 <= L - 2; ++k0) {
        dtoz[k0] = dtdif / (Z[k0 * S + c2] - Z[(k0 + 1) * S + c2]);
        cr[k0] = -dtoz[k0] * RKM[k0 * S + c2];
    }

    /* first level (1544-1547) */
    float RHOK = RHO[c2];
    cm[0] = dtoz[0] * RKM[c2] + RHOK;
    rsu[0] = U[c2] * RHOK;
    rsv[0] = V[c2] * RHOK;

    /* DO K=2,LMH-1: forward sweep (1550-1558) */
    for (int k0 = 1; k0 <= L - 2; ++k0) {
        const float DTOZL = dtoz[k0];
        const float CF = -DTOZL * RKM[(k0 - 1) * S + c2] / cm[k0 - 1];
        RHOK = RHO[k0 * S + c2];
        cm[k0] = -cr[k0 - 1] * CF
               + (RKM[(k0 - 1) * S + c2] + RKM[k0 * S + c2] + D[k0 * S + c2] * RKMS) * DTOZL
               + RHOK;
        rsu[k0] = -rsu[k0 - 1] * CF + U[k0 * S + c2] * RHOK;
        rsv[k0] = -rsv[k0 - 1] * CF + V[k0 * S + c2] * RHOK;
    }

    /* top row LMH with the surface exchange terms (1563-1578) */
    const float DTOZS = dtdif / (Z[(L - 1) * S + c2] - Z[L * S + c2]);
    const float RKMH = RKM[(L - 2) * S + c2];
    const float CFH = -DTOZS * RKMH / cm[L - 2];
    RHOK = RHO[(L - 1) * S + c2];
    const float RCMVB = 1.0f / ((RKMH + RKMS) * DTOZS - cr[L - 2] * CFH + RHOK);
    const float DTOZAK = DTOZS * RKMS;
    U[(L - 1) * S + c2] = (DTOZAK * UZ0 - rsu[L - 2] * CFH + U[(L - 1) * S + c2] * RHOK) * RCMVB;
    V[(L - 1) * S + c2] = (DTOZAK * VZ0 - rsv[L - 2] * CFH + V[(L - 1) * S + c2] * RHOK) * RCMVB;

    /* DO K=LMH-1,1,-1: back substitution (1581-1585) */
    for (int k0 = L - 2; k0 >= 0; --k0) {
        const float RCML = 1.0f / cm[k0];
        U[k0 * S + c2] = (-cr[k0] * U[(k0 + 1) * S + c2] + rsu[k0]) * RCML;
        V[k0 * S + c2] = (-cr[k0] * V[(k0 + 1) * S + c2] + rsv[k0]) * RCML;
    }
}
