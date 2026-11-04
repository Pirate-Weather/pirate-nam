// WOOF-NAM unit phys-bmj-driver: CUCNVC glue for every NPRECIP (BMJ convection branch).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The Fortran module has no licence block; its module header is kept here as the notice:
//   MODULE MODULE_CONVECTION (NMMB convection drivers and packages, nam.v4.2.11).
// Ported to CUDA for WOOF-NAM from nmm/module_CONVECTION.F90:48-658 (nam.v4.2.11, 83df4e13c1).
//
// Scope: the CUCNVC body for CONVECTION == "bmj" (CU_PHYSICS = BMJSCHEME = 2, nam_configfile
// `convection: bmj`). The BMJ call itself (bmjdrv, module_CU_BMJ.F90:117-395) is unit
// phys-bmj-bmjdrv, run between the two kernels below by the wrapper. The KF (W0AVG reset),
// SAS and SASHUR branches (sasdrv / sasdrv_hur and their tendency arrays) are not in this
// configuration and are refused in the wrapper; they are not silently skipped.
//
// Fortran -> kernel map (CUCNVC line numbers in module_CONVECTION.F90):
//   rad_reset : 281-290  HTOP=0, HBOT=LM+1, CUPPT=0 on NRADS/NRADL multiples (before the NCNVC return)
//   prep      : 335-336  AVCNVC, ACUTIM increments
//               344-362  PHINT(LM+1), XLAND, NCA, SFCZ, CUTOP/CUBOT=999, KPBL=LPBL
//               364-377  PHINT, PHMID, RR, EXNER, TH per level (T=TL is a no-op)
//               380-411  U_PHY/V_PHY 4-point average (interior b1 range); zero elsewhere;
//                        DUDT/DVDT, RTHCUTEN, RQCUTEN zeroed (full domain)
//               413-436  DZ: top level uses raw Q, other levels MAX(Q,EPSQ)
//   (bmjdrv)  : 462-476  BMJ call -> unit phys-bmj-bmjdrv (wrapper, not in this file)
//   post      : 487-500  CNVBOT/CNVTOP/HBOTD/HTOPD/HBOTS/HTOPS reset (full domain, MNTO condition)
//               520-536  T, Q, TCUCN update per level (interior b1 range only)
//               552-589  PREC, ACPREC, ACPREC_TOT, CUPREC, CUPPT, CPRATE; CUTOP/CUBOT limits
//                        into HTOP/HBOT, CNVTOP/CNVBOT, HTOPD/HBOTD or HTOPS/HBOTS
//
// Index ranges: single process, regional, ids=1 ide=IM jds=1 jde=JM (CONTRACT section 1).
// its_b1 = 2, ite_b1 = IM-1 (1-based) -> interior i0 in [1, IM-2]; same for j. The full domain
// (ims..ime = 1..IM) is every (i0, j0). One thread owns one column, so each column's vertical
// loops and precipitation accumulation run in order in one thread.
//
// Layout (CONTRACT sections 1 and 3): 3-D A[l0, j0, i0] via IDX3, 2-D [j0, i0] via IDX2,
// level-interface arrays PHINT [LM+1, JM, IM]. SG2, PSG1 are [LM+1]; SGML2, PSGML1 are [LM].
// Fortran K maps to k0 = K-1.
//
// Numerics: float32 throughout (NMMB kfpt = single). Literals are the Fortran single literals:
// .608 (not NMMB_P608), 1.E-5, 1.E-3, 0.25, 1. NMMB constants come from nmmb_common.cuh.
// FMA contraction on (fast numerics, CONTRACT section 3). No host sync, no printf.

#include "nmmb_common.cuh"

// ---------------------------------------------------------------------------------------------
// Kernel 1: radiation-step reset of the convective cloud arrays used by RRTM
// (Fortran 281-290). Full domain. Runs before the NCNVC return on radiation steps.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_driver_rad_reset(
    const int IM, const int JM, const int LM,
    float* __restrict__ HTOP,
    float* __restrict__ HBOT,
    float* __restrict__ CUPPT)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    HTOP[p2] = 0.0f;
    HBOT[p2] = (float)(LM + 1);
    CUPPT[p2] = 0.0f;
}

// ---------------------------------------------------------------------------------------------
// Kernel 2: column preparation before the BMJ call (Fortran 335-436). One thread per column.
// Writes the working arrays bmjdrv reads: PHINT, PHMID, EXNER, TH, RR, DZ, U_PHY, V_PHY,
// RTHCUTEN, RQCUTEN (zeroed), DUDT_PHY, DVDT_PHY (zeroed), RAINCV (zeroed), CUTOP, CUBOT (999),
// KPBL (copy of LPBL), XLAND, NCA, RAINC, SFCZ.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_driver_prep(
    const int IM, const int JM, const int LM,
    const float* __restrict__ PD,          // [JM, IM]
    const float* __restrict__ T,           // [LM, JM, IM]
    const float* __restrict__ Q,           // [LM, JM, IM]
    const float* __restrict__ U,           // [LM, JM, IM]
    const float* __restrict__ V,           // [LM, JM, IM]
    const float* __restrict__ FIS,         // [JM, IM]
    const float* __restrict__ SM,          // [JM, IM]
    const int*   __restrict__ LPBL,        // [JM, IM]
    const float* __restrict__ SG2,         // [LM+1]
    const float* __restrict__ PSG1,        // [LM+1]
    const float* __restrict__ SGML2,       // [LM]
    const float* __restrict__ PSGML1,      // [LM]
    float* __restrict__ PHINT,             // [LM+1, JM, IM]
    float* __restrict__ PHMID,             // [LM, JM, IM]
    float* __restrict__ EXNER,             // [LM, JM, IM]
    float* __restrict__ TH,                // [LM, JM, IM]
    float* __restrict__ RR,                // [LM, JM, IM]
    float* __restrict__ DZ,                // [LM, JM, IM]
    float* __restrict__ U_PHY,             // [LM, JM, IM]
    float* __restrict__ V_PHY,             // [LM, JM, IM]
    float* __restrict__ DUDT_PHY,          // [LM, JM, IM]
    float* __restrict__ DVDT_PHY,          // [LM, JM, IM]
    float* __restrict__ RTHCUTEN,          // [LM, JM, IM]
    float* __restrict__ RQCUTEN,           // [LM, JM, IM]
    float* __restrict__ RAINCV,            // [JM, IM]
    float* __restrict__ RAINC,             // [JM, IM]
    float* __restrict__ NCA,               // [JM, IM]
    float* __restrict__ XLAND,             // [JM, IM]
    float* __restrict__ SFCZ,              // [JM, IM]
    float* __restrict__ CUTOP,             // [JM, IM]
    float* __restrict__ CUBOT,             // [JM, IM]
    int*   __restrict__ KPBL,              // [JM, IM]
    float* __restrict__ AVCNVC,            // [JM, IM]
    float* __restrict__ ACUTIM)            // [JM, IM]
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);  // its_b1..ite_b1

    // Fortran 331-332: G_INV = 1/G (SFCZ only). DTCNVC = NCNVC*DT is formed in the post kernel.
    const float g_inv = 1.0f / NMMB_G;

    // Fortran 335-336: counters
    AVCNVC[p2] = AVCNVC[p2] + 1.0f;
    ACUTIM[p2] = ACUTIM[p2] + 1.0f;

    // Fortran 344-362: column scalars
    const float pdsl = PD[p2];
    RAINCV[p2] = 0.0f;
    RAINC[p2] = 0.0f;
    // PHINT(I,J,LM+1) = SG2(LM+1)*PDSL + PSG1(LM+1)   (Fortran index LM+1 -> 0-based LM)
    PHINT[IDX3(i0, j0, LM, IM, JM)] = SG2[LM] * pdsl + PSG1[LM];
    XLAND[p2] = SM[p2] + 1.0f;
    NCA[p2] = 0.0f;
    SFCZ[p2] = FIS[p2] * g_inv;
    CUTOP[p2] = 999.0f;
    CUBOT[p2] = 999.0f;
    KPBL[p2] = LPBL[p2];   // KPBL = LPBL (layer counted upward from the ground in bmjdrv)

    // Fortran 364-377: per-level pressure, density, exner and theta. T is assigned to itself
    // at Fortran 377 (T(I,J,K)=TL with TL=T), so no store is needed.
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        PHINT[pk] = SG2[k0] * pdsl + PSG1[k0];
        PHMID[pk] = SGML2[k0] * pdsl + PSGML1[k0];

        const float ql = fmaxf(Q[pk], NMMB_EPSQ);
        const float tl = T[pk];
        RR[pk] = PHMID[pk] / (NMMB_R_D * tl * (0.608f * ql + 1.0f));

        EXNER[pk] = powf(PHMID[pk] * 1.0e-5f, NMMB_CAPPA);
        TH[pk] = tl / EXNER[pk];
    }

    // Fortran 380-411: velocity at mass points (4-point average over the interior b1 range),
    // and the full-domain zeroing of the tendencies. Each level is independent.
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        if (interior) {
            // Fortran order: ((u(i,j)+u(i-1,j)) + u(i,j-1)) + u(i-1,j-1), then *0.25
            U_PHY[pk] = (U[pk]
                         + U[IDX3(i0 - 1, j0,     k0, IM, JM)]
                         + U[IDX3(i0,     j0 - 1, k0, IM, JM)]
                         + U[IDX3(i0 - 1, j0 - 1, k0, IM, JM)]) * 0.25f;
            V_PHY[pk] = (V[pk]
                         + V[IDX3(i0 - 1, j0,     k0, IM, JM)]
                         + V[IDX3(i0,     j0 - 1, k0, IM, JM)]
                         + V[IDX3(i0 - 1, j0 - 1, k0, IM, JM)]) * 0.25f;
        } else {
            U_PHY[pk] = 0.0f;
            V_PHY[pk] = 0.0f;
        }
        DUDT_PHY[pk] = 0.0f;
        DVDT_PHY[pk] = 0.0f;
        RTHCUTEN[pk] = 0.0f;
        RQCUTEN[pk] = 0.0f;
    }

    // Fortran 413-436: DZ, layer thickness from the hypsometric relation.
    // Top-most level (Fortran LM, 0-based LM-1) uses the raw Q; the others use MAX(Q, EPSQ).
    {
        const size_t pl = IDX3(i0, j0, LM - 1, IM, JM);
        DZ[pl] = T[pl] * (0.608f * Q[pl] + 1.0f) * NMMB_R_D
                 * (PHINT[IDX3(i0, j0, LM, IM, JM)] - PHINT[pl])
                 / (PHMID[pl] * NMMB_G);
    }
    for (int k0 = LM - 2; k0 >= 0; --k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        const float ql_k = fmaxf(Q[pk], NMMB_EPSQ);
        DZ[pk] = T[pk] * (0.608f * ql_k + 1.0f) * NMMB_R_D
                 * (PHINT[IDX3(i0, j0, k0 + 1, IM, JM)] - PHINT[pk])
                 / (PHMID[pk] * NMMB_G);
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel 3: post-BMJ update (Fortran 487-589). One thread per column.
//   out_reset (MNTO condition, Fortran 487-500): full-domain reset of CNVBOT, CNVTOP,
//   HBOTD, HTOPD, HBOTS, HTOPS. Applied before the interior updates in the same thread.
//   Interior b1 range: T, Q, TCUCN tendencies from RTHCUTEN (theta over exner) and RQCUTEN;
//   precipitation accumulation and cloud top/base bookkeeping.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void phys_bmj_driver_post(
    const int IM, const int JM, const int LM,
    const int ncnvc, const float dt,
    const int out_reset,
    const float* __restrict__ EXNER,       // [LM, JM, IM]
    const float* __restrict__ RTHCUTEN,    // [LM, JM, IM]
    const float* __restrict__ RQCUTEN,     // [LM, JM, IM]
    float* __restrict__ T,                 // [LM, JM, IM]
    float* __restrict__ Q,                 // [LM, JM, IM]
    float* __restrict__ TCUCN,             // [LM, JM, IM]
    const float* __restrict__ RAINCV,      // [JM, IM]
    const float* __restrict__ CUTOP,       // [JM, IM]
    const float* __restrict__ CUBOT,       // [JM, IM]
    float* __restrict__ PREC,              // [JM, IM]
    float* __restrict__ ACPREC,            // [JM, IM]
    float* __restrict__ ACPREC_TOT,        // [JM, IM]
    float* __restrict__ CUPREC,            // [JM, IM]
    float* __restrict__ CUPREC_TOT,        // [JM, IM] convective total since start (port addition)
    float* __restrict__ CUPPT,             // [JM, IM]
    float* __restrict__ CPRATE,            // [JM, IM]
    float* __restrict__ HTOP,              // [JM, IM]
    float* __restrict__ HBOT,              // [JM, IM]
    float* __restrict__ CNVBOT,            // [JM, IM]
    float* __restrict__ CNVTOP,            // [JM, IM]
    float* __restrict__ HBOTD,             // [JM, IM]
    float* __restrict__ HTOPD,             // [JM, IM]
    float* __restrict__ HBOTS,             // [JM, IM]
    float* __restrict__ HTOPS)             // [JM, IM]
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= IM * JM) return;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);

    // Fortran 487-500: output-window reset (full domain)
    if (out_reset) {
        CNVBOT[p2] = (float)(LM + 1);
        CNVTOP[p2] = 0.0f;
        HBOTD[p2] = (float)(LM + 1);
        HTOPD[p2] = 0.0f;
        HBOTS[p2] = (float)(LM + 1);
        HTOPS[p2] = 0.0f;
    }
    if (!interior) return;

    // Fortran 520-536: T, Q, TCUCN tendencies. DTDT = RTHCUTEN*EXNER (before the update).
    // The CONVECTION=="sas" block (QC/QR/QI/QS/QG, CWM, F_ICE, F_RAIN) is not reachable for
    // CONVECTION=="bmj" and is not ported here; the wrapper refuses SAS.
    const float dtcnvc = (float)ncnvc * dt;
    for (int k0 = 0; k0 < LM; ++k0) {
        const size_t pk = IDX3(i0, j0, k0, IM, JM);
        const float dtdt = RTHCUTEN[pk] * EXNER[pk];
        T[pk] = T[pk] + dtdt * dtcnvc;
        Q[pk] = Q[pk] + RQCUTEN[pk] * dtcnvc;
        TCUCN[pk] = TCUCN[pk] + dtdt;
    }

    // Fortran 552-560: precipitation accumulation. RAINCV*1.E-3*NCNVC (left to right).
    const float pcpcol = RAINCV[p2] * 1.0e-3f * (float)ncnvc;
    PREC[p2] = PREC[p2] + pcpcol;
    ACPREC[p2] = ACPREC[p2] + pcpcol;
    ACPREC_TOT[p2] = ACPREC_TOT[p2] + pcpcol;
    CUPREC[p2] = CUPREC[p2] + pcpcol;
    CUPREC_TOT[p2] = CUPREC_TOT[p2] + pcpcol;
    CUPPT[p2] = CUPPT[p2] + pcpcol;
    CPRATE[p2] = pcpcol;

    // Fortran 563-589: cloud top (HTOP/CNVTOP, deep or shallow by the sign of PCPCOL) and base
    // (HBOT/CNVBOT). NINT is round-half-away-from-zero, which roundf implements.
    const int ncutop = (int)roundf(CUTOP[p2]);
    const int ncubot = (int)roundf(CUBOT[p2]);

    if (ncutop > 1 && ncutop < LM + 1) {
        HTOP[p2] = fmaxf(CUTOP[p2], HTOP[p2]);
        CNVTOP[p2] = fmaxf(CUTOP[p2], CNVTOP[p2]);
        if (pcpcol > 0.0f) {
            HTOPD[p2] = fmaxf(CUTOP[p2], HTOPD[p2]);
        } else {
            HTOPS[p2] = fmaxf(CUTOP[p2], HTOPS[p2]);
        }
    }
    if (ncubot > 0 && ncubot < LM + 1) {
        HBOT[p2] = fminf(CUBOT[p2], HBOT[p2]);
        CNVBOT[p2] = fminf(CUBOT[p2], CNVBOT[p2]);
        if (pcpcol > 0.0f) {
            HBOTD[p2] = fminf(CUBOT[p2], HBOTD[p2]);
        } else {
            HBOTS[p2] = fminf(CUBOT[p2], HBOTS[p2]);
        }
    }
}
