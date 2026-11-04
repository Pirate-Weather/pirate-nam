// phys_noah_snopac.cu -- SNOPAC: snow energy and water balance (Noah LSM), one thread per column.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:4050-4374 (nam.v4.2.11, 83df4e13c1).
// Executed statements: see phys_noah_snopac.cuh (every term is there).
// Start point: namgpu/third_party/arwen/core/kernels/noah.cu (SNOPAC section, Apache-2.0, ArWen).
//
// Layout (CONTRACT section 1): soil arrays [NSOIL, JM, IM] (layer 1 at the top), 2-D columns
// [JM, IM], integer flags int32. One thread per column (grid_cols); the NSOIL soil layers of a
// column are read and written by that thread. Float32 throughout (kfpt = single).
// No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"
#include "phys_noah_sflx.cuh"   // SNOPAC plus the NAM callee definitions it declares (EVAPO, SMFLX, SHFLX, SNOWPACK)

extern "C" __global__ void phys_noah_snopac(
    // in, [JM, IM] per column
    const float* __restrict__ etp,      // ETP      potential evap (kg m-2 s-1), read-only here
    const float* __restrict__ prcp,     // PRCP     precipitation rate
    const float* __restrict__ prcpf,    // PRCPF    precipitation for the surface balance
    const int*   __restrict__ snowng,   // SNOWNG   1 = snowing
    const float* __restrict__ smcmax,   // SMCMAX   REDPRM
    const float* __restrict__ smcwlt,   // SMCWLT   REDPRM
    const float* __restrict__ smcref,   // SMCREF   REDPRM
    const float* __restrict__ smcdry,   // SMCDRY   REDPRM
    const float* __restrict__ cmcmax,   // CMCMAX   REDPRM
    const float* __restrict__ sbeta,    // SBETA    REDPRM
    const float* __restrict__ df1,      // DF1      thermal conductivity
    const float* __restrict__ q2,       // Q2       2-m specific humidity (EVAPO)
    const float* __restrict__ sfctmp,   // SFCTMP   lowest-level air temperature
    const float* __restrict__ t24,      // T24      air temperature to the 4th power source
    const float* __restrict__ th2,      // TH2      2-m potential temperature
    const float* __restrict__ fdown,    // FDOWN    downward radiation
    const float* __restrict__ f1,       // F1       REDPRM
    const float* __restrict__ bexp,     // BEXP     REDPRM
    const float* __restrict__ pc,       // PC       plant coefficient
    const float* __restrict__ rch,      // RCH      surface exchange coefficient
    const float* __restrict__ rr,       // RR       surface resistance factor
    const float* __restrict__ cfactr,   // CFACTR   REDPRM
    const float* __restrict__ slope,    // SLOPE    REDPRM
    const float* __restrict__ kdt,      // KDT      REDPRM
    const float* __restrict__ frzfact,  // FRZFACT  REDPRM
    const float* __restrict__ psisat,   // PSISAT   REDPRM
    const float* __restrict__ dwsat,    // DWSAT    REDPRM
    const float* __restrict__ dksat,    // DKSAT    REDPRM
    const float* __restrict__ tbot,     // TBOT     deep soil temperature
    const float* __restrict__ zbot,     // ZBOT     depth of deep soil
    const float* __restrict__ shdfac,   // SHDFAC   green-vegetation fraction
    const int*   __restrict__ nroot,    // NROOT    root layers
    const int*   __restrict__ vegtyp,   // VEGTYP   vegetation type
    const int*   __restrict__ ice,      // ICE      1 = sea ice
    const float* __restrict__ quartz,   // QUARTZ   REDPRM
    const float* __restrict__ fxexp,    // FXEXP    REDPRM
    const float* __restrict__ csoil,    // CSOIL    REDPRM
    const float* __restrict__ emissi,   // EMISSI   surface emissivity
    const float* __restrict__ ribb,     // RIBB     bulk Richardson (unused in executed code)
    const float* __restrict__ flx2,     // FLX2     freezing rain latent heat flux
    // soil arrays [NSOIL, JM, IM]
    float* __restrict__ smc,            // SMC      total soil moisture (INOUT)
    float* __restrict__ sh2o,           // SH2O     unfrozen soil moisture (INOUT)
    float* __restrict__ stc,            // STC      soil temperature (INOUT)
    const float* __restrict__ zsoil,    // ZSOIL    layer depths
    const float* __restrict__ rtdis,    // RTDIS    root distribution
    float* __restrict__ et,             // ET       transpiration per layer (OUT)
    // INOUT scalars [JM, IM], read and updated in place
    float* __restrict__ cmc,            // CMC      canopy water
    float* __restrict__ t1,             // T1       skin temperature
    float* __restrict__ sncovr,         // SNCOVR   snow cover fraction
    float* __restrict__ esd,            // ESD      snow water equivalent
    float* __restrict__ sndens,         // SNDENS   snow density
    float* __restrict__ snowh,          // SNOWH    snow depth
    float* __restrict__ beta,           // BETA     evaporation efficiency
    // OUT scalars [JM, IM]
    float* __restrict__ dew,            // DEW      dewfall
    float* __restrict__ edir,           // EDIR     direct soil evaporation
    float* __restrict__ ec,             // EC       canopy evaporation
    float* __restrict__ ett,            // ETT      total transpiration
    float* __restrict__ etns,           // ETNS     total snow-free evaporation
    float* __restrict__ esnow,          // ESNOW    snow sublimation
    float* __restrict__ flx1,           // FLX1     precip-snow surface heat flux
    float* __restrict__ flx3,           // FLX3     snowmelt heat flux
    float* __restrict__ ssoil,          // SSOIL    soil heat flux
    float* __restrict__ snomlt,         // SNOMLT   snowmelt amount
    float* __restrict__ drip,           // DRIP     canopy drip
    float* __restrict__ runoff1,        // RUNOFF1  surface runoff
    float* __restrict__ runoff2,        // RUNOFF2  subsurface runoff
    float* __restrict__ runoff3,        // RUNOFF3  groundwater runoff
    float dt,                           // DT       time step (s), launch scalar
    int ivegsrc,                        // IVEGSRC  vegetation source (launch scalar)
    int im, int jm)
{
    COLUMN_IJ(im, jm)                  // i0, j0 from the flat column index
    const size_t c = IDX2(i0, j0, im);

    NamSnopacCol s{};                  // zero-initialised: no member is stored uninitialised
    s.etp = etp[c];
    s.prcp = prcp[c];
    s.prcpf = prcpf[c];
    s.snowng = snowng[c];
    s.smcmax = smcmax[c];
    s.smcwlt = smcwlt[c];
    s.smcref = smcref[c];
    s.smcdry = smcdry[c];
    s.cmcmax = cmcmax[c];
    s.dt = dt;
    s.sbeta = sbeta[c];
    s.df1 = df1[c];
    s.q2 = q2[c];
    s.sfctmp = sfctmp[c];
    s.t24 = t24[c];
    s.th2 = th2[c];
    s.fdown = fdown[c];
    s.f1 = f1[c];
    s.bexp = bexp[c];
    s.pc = pc[c];
    s.rch = rch[c];
    s.rr = rr[c];
    s.cfactr = cfactr[c];
    s.slope = slope[c];
    s.kdt = kdt[c];
    s.frzfact = frzfact[c];
    s.psisat = psisat[c];
    s.dwsat = dwsat[c];
    s.dksat = dksat[c];
    s.tbot = tbot[c];
    s.zbot = zbot[c];
    s.shdfac = shdfac[c];
    s.nroot = nroot[c];
    s.vegtyp = vegtyp[c];
    s.ice = ice[c];
    s.ivegsrc = ivegsrc;
    s.quartz = quartz[c];
    s.fxexp = fxexp[c];
    s.csoil = csoil[c];
    s.emissi = emissi[c];
    s.ribb = ribb[c];
    s.flx2 = flx2[c];
    s.cmc = cmc[c];
    s.t1 = t1[c];
    s.sncovr = sncovr[c];
    s.esd = esd[c];
    s.sndens = sndens[c];
    s.snowh = snowh[c];
    s.beta = beta[c];
    for (int k = 0; k < NAM_SNOPAC_NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        s.smc[k] = smc[n];
        s.sh2o[k] = sh2o[n];
        s.stc[k] = stc[n];
        s.zsoil[k] = zsoil[n];
        s.rtdis[k] = rtdis[n];
        s.et[k] = 0.0f;
    }

    nam_snopac(s);

    cmc[c] = s.cmc;
    t1[c] = s.t1;
    sncovr[c] = s.sncovr;
    esd[c] = s.esd;
    sndens[c] = s.sndens;
    snowh[c] = s.snowh;
    beta[c] = s.beta;
    dew[c] = s.dew;
    edir[c] = s.edir;
    ec[c] = s.ec;
    ett[c] = s.ett;
    etns[c] = s.etns;
    esnow[c] = s.esnow;
    flx1[c] = s.flx1;
    flx3[c] = s.flx3;
    ssoil[c] = s.ssoil;
    snomlt[c] = s.snomlt;
    drip[c] = s.drip;
    runoff1[c] = s.runoff1;
    runoff2[c] = s.runoff2;
    runoff3[c] = s.runoff3;
    for (int k = 0; k < NAM_SNOPAC_NSOIL; ++k) {
        const size_t n = IDX3(i0, j0, k, im, jm);
        smc[n] = s.smc[k];
        sh2o[n] = s.sh2o[k];
        stc[n] = s.stc[k];
        et[n] = s.et[k];
    }
}
