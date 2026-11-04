// phys_fer_driver.cu -- WOOF-NAM unit phys-fer-driver: GSMDRIVE glue for NPRECIP microphysics
// (fer_hires route): column preparation, the FER_HIRES call (its own unit, launched from the
// wrapper between the two kernels here), precipitation accumulation and the T / TRAIN update.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Header of nmm/module_MICROPHYSICS.F90 (MODULE MODULE_MICROPHYSICS_NMM, comment block), verbatim
// authorship and history lines:
//   11-06-2009 W. Wang - Put NAM/Ferrier microphysics into a single module.
//   02-10-2010 W. Wang added wsm6
// Ported to CUDA for WOOF-NAM from nmm/module_MICROPHYSICS.F90:60-558 (nam.v4.2.11, 83df4e13c1),
// SUBROUTINE GSMDRIVE. The fer_hires branch (lines 362-382) is the FER_HIRES call and is not in
// this file.
//
// Fortran constants used (nmm/module_CONSTANTS.F90 lines 26-52):
//   EPSQ = 1.e-12, G = 9.8060226, R_D = 287.04, R_V = 461.6, P608 = R_V/R - 1, CAPPA = R/CP,
//   CP = 1004.6. R_G = 1./G (module_MICROPHYSICS.F90 line 51). Single precision (kfpt = single).
//
// Kernels (one launch each; no host sync, no device malloc, no printf):
//   phys_fer_driver_prep    grid_2d(IM, JM, LM): column fill, one thread per (i, j, k)
//                           (lines 222-263); the k0 == 0 thread of each (i, j) also does LOWLYR = 1,
//                           RAINNC = 0, RAINNCV = 0 (column-fill loop) and AVRAIN += 1 (lines 204-206).
//   phys_fer_driver_post    grid_2d(IM, JM, LM): TRAIN and T update after FER_HIRES (lines 511-530),
//                           B1 interior (its_b1..ite_b1 = 2..IM-1 1-based, zero-based 1..IM-2).
//   phys_fer_driver_precip  grid_2d(IM, JM, 1): PREC / ACPREC / ACPREC_TOT (lines 543-546), B1 interior.
//
// Not in this file, on purpose (no counterpart on the fer_hires route):
//   - USE_QV branch (line 276 and the Q/CW update after the micro select): only wsm6 and thompson set USE_QV.
//   - XLAND = SM + 1 (line 227): read by no fer_hires code.
//   - CUBOT, CUTOP: declared, never assigned on any route. SNOWNCV, GRAUPELNCV zeroing (column-fill loop)
//     is consumed only by the thompson branch (lines 418-466).
//   - gfs, wsm6, thompson and etanew (fer) branches (lines 335-361 and 383-466): other units.
//   - acpcp_ra/sn/gr (thompson branch only); the WRITE(0,*) and NMMB_FINALIZE paths (host).
//
// Single-card, full-domain layout (CONTRACT.md section 1): IMS:IME = 1:IM, JMS:JME = 1:JM,
// KMS:KME = 1:LM. Layer 1 = model top (l0 = 0).
#include "nmmb_common.cuh"

#define PFD_R_G (1.0f / NMMB_G)          /* module_MICROPHYSICS.F90 R_G = 1./G */

// ---------------------------------------------------------------------------
// Column preparation (GSMDRIVE lines 222-263): one thread per (i, j, k).
//   PDSL = PD(I,J)
//   DPL = DSG2(K)*PDSL + PDSG1(K)
//   PHMID = SGML2(K)*PDSL + PSGML1(K)
//   TL = T ; QL = AMAX1(Q, EPSQ)
//   RR = PHMID / (R_D*TL*(P608*QL + 1))
//   PI_PHY = (PHMID*1.E-5)**CAPPA
//   TH_PHY = TL / PI_PHY ; P_PHY = PHMID
//   DZ8W = DPL*R_G/RR
// Arrays: pd, rainnc, rainncv, avrain, lowlyr are [JM, IM]; t, q, dz8w, rho_phy, p_phy, pi_phy,
// th_phy are [LM, JM, IM]; dsg2, sgml2, pdsg1, psgml1 are [LM] (grid constants, device copy).
extern "C" __global__ void phys_fer_driver_prep(
    const int IM, const int JM, const int LM,
    const float* __restrict__ pd,
    const float* __restrict__ t,
    const float* __restrict__ q,
    const float* __restrict__ dsg2,
    const float* __restrict__ sgml2,
    const float* __restrict__ pdsg1,
    const float* __restrict__ psgml1,
    float* __restrict__ dz8w,
    float* __restrict__ rho_phy,
    float* __restrict__ p_phy,
    float* __restrict__ pi_phy,
    float* __restrict__ th_phy,
    float* __restrict__ avrain,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    int* __restrict__ lowlyr)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t h = IDX2(i0, j0, IM);

    if (l0 == 0) {
        lowlyr[h] = 1;                 /* LOWLYR(I,J) = 1 (line 226) */
        rainnc[h] = 0.0f;              /* RAINNC = 0 (line 241) */
        rainncv[h] = 0.0f;             /* RAINNCV = 0 (column-fill loop) */
        avrain[h] = avrain[h] + 1.0f;  /* AVRAIN = AVRAIN + 1 (lines 204-206) */
    }

    const size_t p = IDX3(i0, j0, l0, IM, JM);
    const float pdsl = pd[h];
    const float dpl = dsg2[l0] * pdsl + pdsg1[l0];
    const float phmid = sgml2[l0] * pdsl + psgml1[l0];
    const float tl = t[p];
    const float ql = fmaxf(q[p], NMMB_EPSQ);
    const float rr = phmid / (NMMB_R_D * tl * (NMMB_P608 * ql + 1.0f));
    const float pi = powf(phmid * 1.0e-5f, NMMB_CAPPA);

    rho_phy[p] = rr;
    pi_phy[p] = pi;
    th_phy[p] = tl / pi;
    p_phy[p] = phmid;
    dz8w[p] = dpl * PFD_R_G / rr;
}

// ---------------------------------------------------------------------------
// Temperature update after FER_HIRES (GSMDRIVE lines 511-530), B1 interior only.
//   TNEW = TH_PHY*PI_PHY
//   TRAIN = TRAIN + (TNEW - T)*RDTPHS
//   USE_RADAR: MP_TTEN = (TNEW-T)*RDTPHS; if DFI_TTEN > MP_TTEN and DFI_TTEN < 0.01 and
//              MP_TTEN < 0.0018 then MP_TTEN = DFI_TTEN; T = T + MP_TTEN/RDTPHS
//   else T = TNEW
// rdtphs = 1 / (NPRECIP*DT) is computed on the host in single precision (lines 199-200).
extern "C" __global__ void phys_fer_driver_post(
    const int IM, const int JM, const int LM,
    const float* __restrict__ th_phy,
    const float* __restrict__ pi_phy,
    const float* __restrict__ dfi_tten,
    float* __restrict__ t,
    float* __restrict__ train,
    const float rdtphs,
    const int use_radar)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    /* ITS_B1..ITE_B1 = max(its, ids+1)..min(ite, ide-1): zero-based 1..IM-2; same in j. */
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const size_t p = IDX3(i0, j0, l0, IM, JM);
    const float tnew = th_phy[p] * pi_phy[p];
    train[p] = train[p] + (tnew - t[p]) * rdtphs;
    if (use_radar) {
        float mp_tten = (tnew - t[p]) * rdtphs;
        const float dfi = dfi_tten[p];
        if (dfi > mp_tten && dfi < 0.01f && mp_tten < 0.0018f) {
            mp_tten = dfi;
        }
        t[p] = t[p] + mp_tten / rdtphs;
    } else {
        t[p] = tnew;
    }
}

// ---------------------------------------------------------------------------
// Precipitation accumulation (GSMDRIVE lines 543-546), B1 interior, one thread per (i, j).
//   PCPCOL = RAINNCV*1.E-3 ; PREC += PCPCOL ; ACPREC += PCPCOL ; ACPREC_TOT += PCPCOL
extern "C" __global__ void phys_fer_driver_precip(
    const int IM, const int JM,
    const float* __restrict__ rainncv,
    float* __restrict__ prec,
    float* __restrict__ acprec,
    float* __restrict__ acprec_tot)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    if (i0 < 1 || i0 > IM - 2 || j0 < 1 || j0 > JM - 2) return;

    const size_t h = IDX2(i0, j0, IM);
    const float pcpcol = rainncv[h] * 1.0e-3f;
    prec[h] = prec[h] + pcpcol;
    acprec[h] = acprec[h] + pcpcol;
    acprec_tot[h] = acprec_tot[h] + pcpcol;
}
