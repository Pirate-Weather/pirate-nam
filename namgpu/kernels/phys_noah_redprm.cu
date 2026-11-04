/* Module header, verbatim from NAM phys/module_LS_NOAHLSM.F90 lines 1-22 (nam.v4.2.11, 83df4e13c1):
!-----------------------------------------------------------------------
!
      MODULE MODULE_LS_NOAHLSM
!
!-----------------------------------------------------------------------
!***  THIS MODULE CONTAINS THE LAND SURFACE SCHEME.
!-----------------------------------------------------------------------
!
! HISTORY LOG:
!
!   2008-07-28  Vasic - SMSTAV, SMSTOT, SMOIS reset for plotting.
!               Vasic - SOILW limitted to positive values.
!
!-----------------------------------------------------------------------
!
      USE ESMF
      USE MODULE_SF_URBAN
      USE MODULE_INCLUDE
!
!-----------------------------------------------------------------------
!
      PRIVATE
*/
//
// phys_noah_redprm.cu -- REDPRM: per-point land-surface parameters for the Noah LSM.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. The module header above is
// kept verbatim as the notice. Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:3471-3664
// (nam.v4.2.11, 83df4e13c1).
//
// NOTICE, ArWen (Apache-2.0): the start point is ArWen gpuwm/third_party/arwen/core/kernels/noah.cu,
// the REDPRM block that reads packed soil and vegetation tables per column. The ArWen LICENSE
// (Apache-2.0) is the repo root LICENSE. noah.cu carries no per-file copyright or licence header
// (checked 2026-11-04), so none is reproduced here.
// Copied from ArWen gpuwm/third_party/arwen/core/kernels/noah.cu (Apache-2.0) for WOOF-NAM, 2026-11-04.
// Adapted to the NAM Fortran:
//   - Table indices are the Fortran 1-based VEGTYP / SOILTYP / SLOPETYP, stored 0-based by
//     phys-noah-parm-tables (Fortran index LC is at LC-1).
//   - The LOCAL branch of REDPRM is kept: ALBBRD, Z0BRD and SHDFAC come from the tables only when
//     LOCAL is true. NAM calls SFLX with LOCAL = .false. (module parameter, module_LS_NOAHLSM.F90
//     line 450), so the normal path leaves those three as the caller set them.
//   - The ArWen urban parameter overrides are not here; the NAM REDPRM has none.
//
// Executed statements, in Fortran order (each has a counterpart below):
//   CSOIL = CSOIL_DATA                        BEXP = BB(SOILTYP)          DKSAT = SATDK(SOILTYP)
//   DWSAT = SATDW(SOILTYP)                    F1 = F11(SOILTYP)           PSISAT = SATPSI(SOILTYP)
//   QUARTZ = QTZ(SOILTYP)                     SMCDRY = DRYSMC(SOILTYP)    SMCMAX = MAXSMC(SOILTYP)
//   SMCREF = REFSMC(SOILTYP)                  SMCWLT = WLTSMC(SOILTYP)
//   ZBOT = ZBOT_DATA   SALP = SALP_DATA   SBETA = SBETA_DATA   REFDK = REFDK_DATA
//   FRZK = FRZK_DATA   FXEXP = FXEXP_DATA   REFKDT = REFKDT_DATA   PTU = 0.
//   KDT = REFKDT * DKSAT / REFDK              CZIL = CZIL_DATA            SLOPE = SLOPE_DATA(SLOPETYP)
//   FRZFACT = (SMCMAX / SMCREF) * (0.412 / 0.468)     FRZX = FRZK * FRZFACT
//   TOPT = TOPT_DATA   CMCMAX = CMCMAX_DATA   CFACTR = CFACTR_DATA   RSMAX = RSMAX_DATA
//   NROOT = NROTBL(VEGTYP)   SNUP = SNUPTBL(VEGTYP)   RSMIN = RSTBL(VEGTYP)   RGL = RGLTBL(VEGTYP)
//   HS = HSTBL(VEGTYP)   LAI = LAITBL(VEGTYP)
//   IF (LOCAL): ALBBRD = ALBTBL(VEGTYP), Z0BRD = Z0TBL(VEGTYP), SHDFAC = SHDTBL(VEGTYP)
//   IF (VEGTYP .eq. BARE) SHDFAC = 0.0
//   IF (NROOT .gt. NSOIL): the Fortran WRITE(0,*) 'Error: too many root layers' (refused, see below)
//   DO I = 1, NROOT: RTDIS(I) = -SLDPTH(I) / ZSOIL(NROOT)
//
// NSOIL is the run's soil layer count (the length of SLDPTH, bound by the wrapper), not a constant.
// SLDPTH is the state's layer thickness; NOAH_LSM_INIT writes it from DZS (Fortran line 635).
//
// Refusals (no printf in kernels, and no silent clamping of a result that is used):
//   An out-of-range class index (VEGTYP outside 1..NLUS, SOILTYP outside 1..NSLTYPE, SLOPETYP outside
//   1..NSLOPE) or NROOT outside 1..NSOIL increments bad[0]. The wrapper reads bad[0] once at init and
//   raises. The index is clamped only so the table read stays in bounds; that point is never used,
//   because the run is refused.
//
// Choices where the Fortran is undefined (documented, not guessed at):
//   - RTDIS(I) for I > NROOT is INTENT(OUT) and never set in the Fortran. It is written as 0 here so the
//     device array is fully defined and deterministic. phys-noah-evapo reads RTDIS only in NROOT-bounded
//     loops (phys_noah_evapo.cu lines 72-85); the other consumers were not re-read for this note.
//
// Layout: one thread per (i, j), grid.z = 1 (CONTRACT section 3). Per-point arrays are float32
// [JM, IM] (IDX2) and int32 [JM, IM]; RTDIS is [NSOIL, JM, IM] (IDX3 with l0 = soil layer).
// Fortran REAL = float (kfpt=single). Table vectors are device arrays built once at init.

#include "nmmb_common.cuh"

#define REDPRM_NLUS    50   // module_LS_NOAHLSM.F90:151
#define REDPRM_NSLTYPE 30   // module_LS_NOAHLSM.F90:162
#define REDPRM_NSLOPE  30   // module_LS_NOAHLSM.F90:171

__device__ static inline int redprm_clamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

extern "C" __global__ void phys_noah_redprm(
    // per-point types [JM, IM]
    const int* __restrict__ vegtyp,       // VEGTYP    IGBP vegetation class, Fortran 1-based
    const int* __restrict__ soiltyp,      // SOILTYP   STATSGO soil class, Fortran 1-based
    const int* __restrict__ slopetyp,     // SLOPETYP  slope class, Fortran 1-based
    // vegetation tables, 0-based, [NLUS] (phys-noah-parm-tables)
    const float* __restrict__ albtbl,     // ALBTBL    background albedo (LOCAL only)
    const float* __restrict__ z0tbl,      // Z0TBL     background roughness (LOCAL only)
    const float* __restrict__ shdtbl,     // SHDTBL    green vegetation fraction (LOCAL only)
    const float* __restrict__ rstbl,      // RSTBL     minimum stomatal resistance
    const float* __restrict__ rgltbl,     // RGLTBL    radiation stress function parameter
    const float* __restrict__ hstbl,      // HSTBL     vapour pressure deficit parameter
    const float* __restrict__ snuptbl,    // SNUPTBL   threshold snow depth for 100 % cover
    const float* __restrict__ laitbl,     // LAITBL    leaf area index
    const int*   __restrict__ nrotbl,     // NROTBL    root layers
    // soil tables, 0-based, [NSLTYPE]
    const float* __restrict__ bb,         // BB        B exponent
    const float* __restrict__ drysmc,     // DRYSMC    air-dry moisture
    const float* __restrict__ f11,        // F11       thermal coefficient
    const float* __restrict__ maxsmc,     // MAXSMC    porosity
    const float* __restrict__ refsmc,     // REFSMC    field capacity
    const float* __restrict__ satpsi,     // SATPSI    saturated soil potential
    const float* __restrict__ satdk,      // SATDK     saturated hydraulic conductivity
    const float* __restrict__ satdw,      // SATDW     saturated diffusivity
    const float* __restrict__ wltsmc,     // WLTSMC    wilting point
    const float* __restrict__ qtz,        // QTZ       quartz content
    // general tables
    const float* __restrict__ slope_data, // SLOPE_DATA [NSLOPE], 0-based
    const float* __restrict__ sldpth,     // SLDPTH    layer thickness [NSOIL] (state, from DZS)
    const float* __restrict__ zsoil,      // ZSOIL     layer bottom depth, negative [NSOIL]
    // GENPARM scalars (module_LS_NOAHLSM.F90 SBETA_DATA ... RSMAX_DATA, TOPT_DATA)
    float sbeta_data, float fxexp_data, float csoil_data, float salp_data,
    float refdk_data, float refkdt_data, float frzk_data, float zbot_data, float czil_data,
    float topt_data, float cmcmax_data, float cfactr_data, float rsmax_data,
    int bare,                             // BARE      bare-soil class (Fortran 1-based)
    int local,                            // LOCAL     local-site flag (NAM: 0)
    int im, int jm,
    int nsoil,                            // soil layers of the run (SLDPTH length)
    // soil parameter outputs [JM, IM]
    float* __restrict__ CSOIL,
    float* __restrict__ BEXP,
    float* __restrict__ DKSAT,
    float* __restrict__ DWSAT,
    float* __restrict__ F1,
    float* __restrict__ PSISAT,
    float* __restrict__ QUARTZ,
    float* __restrict__ SMCDRY,
    float* __restrict__ SMCMAX,
    float* __restrict__ SMCREF,
    float* __restrict__ SMCWLT,
    float* __restrict__ ZBOT,
    float* __restrict__ SALP,
    float* __restrict__ SBETA,
    float* __restrict__ FXEXP,
    float* __restrict__ REFKDT,
    float* __restrict__ FRZX,
    float* __restrict__ KDT,
    float* __restrict__ SLOPE,
    float* __restrict__ CZIL,
    float* __restrict__ PTU,
    // vegetation outputs [JM, IM]
    float* __restrict__ TOPT,
    float* __restrict__ CMCMAX,
    float* __restrict__ CFACTR,
    float* __restrict__ RSMAX,
    float* __restrict__ SNUP,
    float* __restrict__ RSMIN,
    float* __restrict__ RGL,
    float* __restrict__ HS,
    float* __restrict__ LAI,
    int*   __restrict__ NROOT,
    // in/out: the Fortran leaves these as the caller set them unless LOCAL or BARE applies
    float* __restrict__ SHDFAC,           // [JM, IM]
    float* __restrict__ Z0BRD,            // [JM, IM]
    float* __restrict__ ALBBRD,           // [JM, IM]
    // root distribution [NSOIL, JM, IM]
    float* __restrict__ RTDIS,
    // refusal counter [1]: points with a class index or NROOT outside the valid range
    int*   __restrict__ bad)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm) return;
    const size_t n = IDX2(i0, j0, im);

    // Table indices (Fortran 1-based). Out of range is refused via bad[0]; the clamp only keeps the read in bounds.
    const int vraw = vegtyp[n];
    const int sraw = soiltyp[n];
    const int praw = slopetyp[n];
    if (vraw < 1 || vraw > REDPRM_NLUS || sraw < 1 || sraw > REDPRM_NSLTYPE ||
        praw < 1 || praw > REDPRM_NSLOPE) {
        atomicAdd(bad, 1);
    }
    const int vtype = redprm_clamp(vraw, 1, REDPRM_NLUS);
    const int stype = redprm_clamp(sraw, 1, REDPRM_NSLTYPE);
    const int sptype = redprm_clamp(praw, 1, REDPRM_NSLOPE);
    const int vt = vtype - 1;     // 0-based table index
    const int st = stype - 1;
    const int sp = sptype - 1;

    // ---- SET-UP SOIL PARAMETERS -----------------------------------------------------------
    CSOIL[n] = csoil_data;
    const float bexp = bb[st];
    const float dksat = satdk[st];
    const float dwsat = satdw[st];
    const float f1 = f11[st];
    const float psisat = satpsi[st];
    const float quartz = qtz[st];
    const float smcdry = drysmc[st];
    const float smcmax = maxsmc[st];
    const float smcref = refsmc[st];
    const float smcwlt = wltsmc[st];
    BEXP[n] = bexp;
    DKSAT[n] = dksat;
    DWSAT[n] = dwsat;
    F1[n] = f1;
    PSISAT[n] = psisat;
    QUARTZ[n] = quartz;
    SMCDRY[n] = smcdry;
    SMCMAX[n] = smcmax;
    SMCREF[n] = smcref;
    SMCWLT[n] = smcwlt;

    // ---- universal parameters (not dependent on SOILTYP, VEGTYP or SLOPETYP) --------------
    ZBOT[n] = zbot_data;
    SALP[n] = salp_data;
    SBETA[n] = sbeta_data;
    const float refdk = refdk_data;
    const float frzk = frzk_data;
    FXEXP[n] = fxexp_data;
    REFKDT[n] = refkdt_data;
    PTU[n] = 0.0f;                        // (not used yet) intent(out) fill, as in the Fortran
    KDT[n] = refkdt_data * dksat / refdk;
    CZIL[n] = czil_data;
    SLOPE[n] = slope_data[sp];

    // TO ADJUST FRZK PARAMETER TO ACTUAL SOIL TYPE: FRZK * FRZFACT
    const float frzfact = (smcmax / smcref) * (0.412f / 0.468f);
    FRZX[n] = frzk * frzfact;

    // ---- SET-UP VEGETATION PARAMETERS -----------------------------------------------------
    TOPT[n] = topt_data;
    CMCMAX[n] = cmcmax_data;
    CFACTR[n] = cfactr_data;
    RSMAX[n] = rsmax_data;
    const int nroot = nrotbl[vt];
    NROOT[n] = nroot;
    SNUP[n] = snuptbl[vt];
    RSMIN[n] = rstbl[vt];
    RGL[n] = rgltbl[vt];
    HS[n] = hstbl[vt];
    LAI[n] = laitbl[vt];

    if (local) {
        ALBBRD[n] = albtbl[vt];
        Z0BRD[n] = z0tbl[vt];
        SHDFAC[n] = shdtbl[vt];
    }
    if (vtype == bare) SHDFAC[n] = 0.0f;  // IF (VEGTYP .eq. BARE) SHDFAC = 0.0

    // ---- root distribution: uniform over the root layers, RTDIS = -SLDPTH / ZSOIL(NROOT) --
    if (nroot < 1 || nroot > nsoil) atomicAdd(bad, 1);   // the Fortran WRITE(0,*) error, refused
    const int nr = redprm_clamp(nroot, 1, nsoil);        // bounded read only; that point is refused
    for (int k = 0; k < nsoil; ++k) {
        float v = 0.0f;
        if (k < nr) v = -sldpth[k] / zsoil[nr - 1];
        RTDIS[IDX3(i0, j0, k, im, jm)] = v;
    }
}
