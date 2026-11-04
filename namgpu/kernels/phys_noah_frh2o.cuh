// phys_noah_frh2o.cuh -- FRH2O device function: unfrozen soil water (Noah LSM), for reuse by
// every kernel that needs FRH2O (SNKSRC and the SFLX soil paths).
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90), HISTORY
// LOG 2008-07-28 Vasic. The routine's own comment block (Koren et al. 1999, JGR 104(D16),
// 19569-19585, eqn 17; Flerchinger explicit form) is kept in the Fortran source.
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2644-2782 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE FRH2O(FREE,TKELV,SMC,SH2O,SMCMAX,BEXP,PSIS), every executed statement:
//   BX = BEXP; IF (BEXP > BLIM) BX = BLIM                           (BLIM = 5.5)
//   NLOG = 0; KCOUNT = 0
//   IF (TKELV > T0 - 1.E-3) THEN FREE = SMC                          (T0 = 273.15)
//   ELSE
//     Option 1 (CK /= 0, CK = 8.0):
//       SWL = SMC - SH2O; SWL = min(SWL, SMC - 0.02); SWL = max(SWL, 0)
//       DO WHILE (NLOG < 10 .AND. KCOUNT == 0)                       (goto 1001 / 1002)
//         NLOG = NLOG + 1
//         DF    = ALOG((PSIS*GS/HLICE)*((1+CK*SWL)**2)*(SMCMAX/(SMC-SWL))**BX)
//                 - ALOG(-(TKELV-T0)/TKELV)
//         DENOM = 2*CK/(1+CK*SWL) + BX/(SMC-SWL)
//         SWLK  = SWL - DF/DENOM; clamp SWLK to [0, SMC-0.02]
//         DSWL  = |SWLK - SWL|; SWL = SWLK
//         IF (DSWL <= ERROR) KCOUNT = KCOUNT + 1                     (ERROR = 0.005)
//       END DO
//       FREE = SMC - SWL
//     Option 2 (explicit Flerchinger, applied when option 1 did not converge):
//       IF (KCOUNT == 0) THEN
//         FK = ((HLICE/(GS*(-PSIS)))*((TKELV-T0)/TKELV))**(-1/BX) * SMCMAX
//         FK = max(FK, 0.02); FREE = min(FK, SMC)
//       END IF
//   END IF
// Constants (PARAMETER): CK = 8.0, BLIM = 5.5, ERROR = 0.005, HLICE = 3.335E5, GS = 9.81,
// T0 = 273.15. DICE and DH2O are declared in the Fortran but unused in the executed statements.
// The CK = 0 branch is not executed (CK is a PARAMETER equal to 8.0); it is kept behind a
// compile-time test so the structure matches the Fortran, and it folds away.
//
// Single precision throughout (kfpt = single). Fortran ALOG = logf, ** on REAL = powf.
// PSIS is the POSITIVE saturated matric potential (REDPRM PSISAT, module_LS_NOAHLSM.F90:1865), so
// the option 2 base (HLICE/(GRAV*(-PSIS)))*((TKELV-T0)/TKELV) is positive for TKELV < T0 (both factors
// negative) and powf never sees a negative base. Option 1 logs are of positive arguments too. The
// Fortran MIN(FK,SMC) and fminf(FK,SMC) differ only when FK is NaN, which the valid-input paths do not
// produce (checked by tools/check-phys-noah-frh2o.py).
// Pure device function: no memory access, no host interaction.

#pragma once

#define NAM_FRH2O_CK      8.0f
#define NAM_FRH2O_BLIM    5.5f
#define NAM_FRH2O_ERROR   0.005f
#define NAM_FRH2O_HLICE   3.335E5f
#define NAM_FRH2O_GS      9.81f
#define NAM_FRH2O_T0      273.15f

__device__ __forceinline__ float nam_frh2o(float tkelv, float smc, float sh2o,
                                           float smcmax, float bexp, float psis)
{
    const float ck = NAM_FRH2O_CK;
    const float blim = NAM_FRH2O_BLIM;
    const float error = NAM_FRH2O_ERROR;
    const float hlice = NAM_FRH2O_HLICE;
    const float gs = NAM_FRH2O_GS;
    const float t0 = NAM_FRH2O_T0;

    float bx = bexp;
    if (bexp > blim) bx = blim;

    int nlog = 0;
    int kcount = 0;
    float freev = smc;
    float swl = 0.0f;

    if (tkelv > (t0 - 1.0E-3f)) {
        freev = smc;
    } else {
        if (ck != 0.0f) {
            swl = smc - sh2o;
            if (swl > (smc - 0.02f)) swl = smc - 0.02f;
            if (swl < 0.0f) swl = 0.0f;
            while (nlog < 10 && kcount == 0) {
                nlog += 1;
                const float q = 1.0f + ck * swl;     // (1.+CK*SWL)**2. as a multiply (gfortran expands **2.)
                float df = logf((psis * gs / hlice)
                                * (q * q)
                                * powf(smcmax / (smc - swl), bx))
                           - logf(-(tkelv - t0) / tkelv);
                float denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
                float swlk = swl - df / denom;
                if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
                if (swlk < 0.0f) swlk = 0.0f;
                float dswl = fabsf(swlk - swl);
                swl = swlk;
                if (dswl <= error) kcount += 1;
            }
            freev = smc - swl;
        }
        if (kcount == 0) {
            float fk = powf((hlice / (gs * (-psis))) * ((tkelv - t0) / tkelv), -1.0f / bx) * smcmax;
            if (fk < 0.02f) fk = 0.02f;
            freev = fminf(fk, smc);
        }
    }
    return freev;
}
