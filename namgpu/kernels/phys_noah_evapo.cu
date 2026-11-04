// phys_noah_evapo.cu -- EVAPO: total evapotranspiration (Noah LSM), with its callees DEVAP and TRANSP.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the source
// module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90),
//   HISTORY LOG: 2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
//                2008-07-28  Vasic - SOILW limitted to positive values.
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:2557-2641 (nam.v4.2.11, 83df4e13c1).
// The callees are carried in this file, because the unit is one launch:
//   SUBROUTINE DEVAP   phys/module_LS_NOAHLSM.F90:2514-2554  (called from EVAPO)
//   SUBROUTINE TRANSP  phys/module_LS_NOAHLSM.F90:5200-5309  (called from EVAPO)
// Structure follows ArWen namgpu/third_party/arwen/core/kernels/noah.cu (noah_devap, noah_transp,
// noah_evapo), Apache-2.0, adapted to the NAM call sites and the CONTRACT layout. Changes from the
// ArWen copy:
//   - DEVAP is called with SH2O(1) (EVAPO passes SH2O(1) to DEVAP, module_LS_NOAHLSM.F90 line 2586;
//     the SRATIO line 2515 is the "2015" form). ArWen passes SMC(1). This port follows NAM.
//   - Arguments that no executed statement of EVAPO, DEVAP or TRANSP reads (EVAPO SMC; DEVAP
//     BEXP, DKSAT, DWSAT, ZSOIL, SMCWLT, SMCREF; TRANSP SFCTMP, Q2, VEGTYP, STC, ZSOIL) are not
//     passed. They appear only in commented-out code or in the frozen-ground branch NAM does not run.
//   - Soil and vegetation parameters are per point arrays (the Noah parameter tables are looked up
//     per point by the init unit); the time step DT is one scalar.
//
// Executed statements of EVAPO (2557-2641) and their counterparts below:
//   EDIR = 0; EC = 0; ETT = 0; ET(K) = 0 for K = 1..NSOIL          -> zero-init
//   IF (ETP1 > 0.0):                                               -> if (etp1 > 0)
//     IF (SHDFAC < 1.): CALL DEVAP                                 -> e_dir = noah_devap
//     IF (SHDFAC > 0.0):
//       CALL TRANSP (ET), then ETT = sum over K = 1..NSOIL         -> noah_transp, e_tt loop
//       IF (CMC > 0.0) EC = SHDFAC*((CMC/CMCMAX)**CFACTR)*ETP1 ELSE EC = 0
//       CMC2MS = CMC/DT ; EC = MIN(CMC2MS, EC)
//   ETA1 = EDIR + ETT + EC                                         -> eta1
// DEVAP (2514-2554): SRATIO = (SH2O - SMCDRY)/(SMCMAX - SMCDRY); IF SRATIO > 0: FX = SRATIO**FXEXP
//   clamped to [0, 1], ELSE FX = 0; EDIR = FX*(1.0 - SHDFAC)*ETP1.
// TRANSP (5200-5309): ET(K) = 0; ETP1A = SHDFAC*PC*ETP1*(1.0 - (CMC/CMCMAX)**CFACTR) if CMC /= 0,
//   else SHDFAC*PC*ETP1; GX(I) = clamp((SH2O(I) - SMCWLT)/(SMCREF - SMCWLT), 0, 1), SGX = sum/NROOT;
//   RTX = RTDIS(I) + GX(I) - SGX; GX(I) = GX(I)*MAX(RTX, 0); DENOM = sum GX (DENOM = 1 if <= 0);
//   ET(I) = ETP1A*GX(I)/DENOM for I = 1..NROOT. The commented-out PART/ZSOIL/RTDIS-only weightings
//   are not executed and are not ported.
//
// Caller-side inputs (bound by the wrapper phys_noah_evapo.py from ctx.st):
//   ETP (kg m-2 s-1, PENMAN) is converted here to ETP1 = ETP * 0.001 (m s-1), the conversion both
//   callers make (NOPAC module_LS_NOAHLSM.F90:3247, SNOPAC 4147). DEW is the dew branch both callers
//   take when ETP <= 0 (NOPAC 3286-3290, SNOPAC 4138-4148): DEW = -ETP1, and EVAPO is not called.
//   DEW is 0 for ETP > 0. The ETP1 > 0 gate below then matches EVAPO's own gate (2602).
// Two EVAPO call sites exist in NAM: NOPAC (3264), which this launch is, and SNOPAC (4155). The
// SNOPAC call is nam_evapo inside phys_noah_snopac.cu, with the same argument order as the Fortran
// EVAPO argument list and ETP1 = ETP * 0.001 (4147); it is not this launch.
//
// Layout (CONTRACT sections 1 and 3): 2-D point kernel, one thread per (i, j), grid.z = 1.
// Per-point fields are float32 [JM, IM] (IDX2). Per-layer fields SH2O, RTDIS, ET are float32
// [NSOIL, JM, IM] (layer 1 first). NROOT is int32 [JM, IM]. DT is one scalar.
// Fast numerics: fmad on, IEEE division and sqrt, powf for the Fortran exponentiation.

#include "nmmb_common.cuh"
#include "phys_noah_devap.cuh"

#define NOAH_NSOIL 4   // NUM_SOIL_LAYERS (module_LS_NOAHLSM.F90 line 22)

// ------------------------------------------------------------------ TRANSP (5200-5309)
// sh2o: SH2O(1..NSOIL), passed by EVAPO as TRANSP's SMC argument.
__device__ static void noah_transp(float* et, float etp1, const float* sh2o, float cmc,
                                   float shdfac, float smcwlt, float cmcmax, float pc,
                                   float cfactr, float smcref, int nroot, const float* rtdis)
{
    for (int k = 0; k < NOAH_NSOIL; ++k) et[k] = 0.0f;                 // ET(K) = 0
    float etp1a;
    if (cmc != 0.0f)
        etp1a = shdfac * pc * etp1 * (1.0f - powf(cmc / cmcmax, cfactr));
    else
        etp1a = shdfac * pc * etp1;
    float gx[NOAH_NSOIL];
    float sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (sh2o[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = fmaxf(fminf(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (float)nroot;
    float denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        const float rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * fmaxf(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

// ------------------------------------------------------------------ EVAPO (2557-2641)
extern "C" __global__ void phys_noah_evapo(
    const float* __restrict__ etp,      // ETP     potential evaporation (kg m-2 s-1) [JM, IM]
    const float* __restrict__ cmc,      // CMC     canopy moisture                 [JM, IM]
    const float* __restrict__ sh2o,     // SH2O    unfrozen soil moisture          [NSOIL, JM, IM]
    const float* __restrict__ rtdis,    // RTDIS   root distribution by layer      [NSOIL, JM, IM]
    const float* __restrict__ shdfac,   // SHDFAC  green vegetation fraction       [JM, IM]
    const float* __restrict__ smcmax,   // SMCMAX  porosity                        [JM, IM]
    const float* __restrict__ smcwlt,   // SMCWLT  wilting point                   [JM, IM]
    const float* __restrict__ smcref,   // SMCREF  reference soil moisture         [JM, IM]
    const float* __restrict__ smcdry,   // SMCDRY  air-dry soil moisture           [JM, IM]
    const float* __restrict__ cmcmax,   // CMCMAX  maximum canopy water            [JM, IM]
    const float* __restrict__ pc,       // PC      plant coefficient               [JM, IM]
    const float* __restrict__ cfactr,   // CFACTR  canopy water exponent           [JM, IM]
    const float* __restrict__ fxexp,    // FXEXP   bare-soil evaporation exponent  [JM, IM]
    const int* __restrict__ nroot,      // NROOT   number of root layers           [JM, IM]
    float dt,                           // DT      physics time step (s)
    float* __restrict__ eta1,           // ETA1    total evapotranspiration        [JM, IM] (out)
    float* __restrict__ edir,           // EDIR    direct soil evaporation         [JM, IM] (out)
    float* __restrict__ ec,             // EC      canopy evaporation              [JM, IM] (out)
    float* __restrict__ ett,            // ETT     plant transpiration             [JM, IM] (out)
    float* __restrict__ et,             // ET      transpiration by layer          [NSOIL, JM, IM] (out)
    float* __restrict__ dew,            // DEW     dewfall (ETP <= 0), m s-1       [JM, IM] (out)
    int im, int jm)
{
    POINT_IJL
    (void)l0;
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);
    const size_t stride = (size_t)jm * (size_t)im;   // one layer of a [NSOIL, JM, IM] field

    const float p_etp = etp[n];
    const float p_etp1 = p_etp * 0.001f;             // ETP1 = ETP * 0.001 (NOPAC 3247, SNOPAC 4147)
    const float p_cmc = cmc[n];
    const float p_shdfac = shdfac[n];
    const float p_smcmax = smcmax[n];
    const float p_smcwlt = smcwlt[n];
    const float p_smcref = smcref[n];
    const float p_smcdry = smcdry[n];
    const float p_cmcmax = cmcmax[n];
    const float p_pc = pc[n];
    const float p_cfactr = cfactr[n];
    const float p_fxexp = fxexp[n];
    int p_nroot = nroot[n];
    // NROOT is at most NSOIL in NAM (the soil column has NSOIL layers). Clamp only an index that
    // would run past the layer arrays; values inside 1..NSOIL pass through unchanged.
    if (p_nroot > NOAH_NSOIL) p_nroot = NOAH_NSOIL;
    if (p_nroot < 0) p_nroot = 0;

    float sh[NOAH_NSOIL], rt[NOAH_NSOIL], etl[NOAH_NSOIL];
    for (int k = 0; k < NOAH_NSOIL; ++k) {
        sh[k] = sh2o[(size_t)k * stride + n];
        rt[k] = rtdis[(size_t)k * stride + n];
        etl[k] = 0.0f;                                // ET(K) = 0
    }

    float e_dir = 0.0f;                               // EDIR = 0
    float e_c = 0.0f;                                 // EC = 0
    float e_tt = 0.0f;                                // ETT = 0

    if (p_etp1 > 0.0f) {                              // IF (ETP1 > 0.0)
        if (p_shdfac < 1.0f) {                        // IF (SHDFAC < 1.)
            e_dir = noah_devap(p_etp1, sh[0], p_shdfac, p_smcmax, p_smcdry, p_fxexp);
        }
        if (p_shdfac > 0.0f) {                        // IF (SHDFAC > 0.0)
            noah_transp(etl, p_etp1, sh, p_cmc, p_shdfac, p_smcwlt, p_cmcmax, p_pc,
                        p_cfactr, p_smcref, p_nroot, rt);
            for (int k = 0; k < NOAH_NSOIL; ++k) e_tt = e_tt + etl[k];   // ETT = sum of ET(K)
            // Canopy evaporation; the CMC > 0 guard avoids the tangent-linear problem near CMC = 0.
            if (p_cmc > 0.0f)
                e_c = p_shdfac * powf(p_cmc / p_cmcmax, p_cfactr) * p_etp1;
            else
                e_c = 0.0f;
            const float cmc2ms = p_cmc / dt;          // CMC2MS
            e_c = fminf(cmc2ms, e_c);                 // EC = MIN(CMC2MS, EC)
        }
    }

    eta1[n] = e_dir + e_tt + e_c;                     // ETA1 = EDIR + ETT + EC
    edir[n] = e_dir;
    ec[n] = e_c;
    ett[n] = e_tt;
    for (int k = 0; k < NOAH_NSOIL; ++k) et[(size_t)k * stride + n] = etl[k];
    // Dew branch (NOPAC 3286-3290, SNOPAC 4138-4148): ETP <= 0 gives DEW = -ETP1. The Fortran does
    // not reset ETP1 here; EVAPO is simply not called in that branch, which the gate above does.
    dew[n] = (p_etp <= 0.0f) ? -p_etp1 : 0.0f;
}
