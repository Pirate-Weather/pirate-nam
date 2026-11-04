// phys_noah_driver.cu -- NOAHLSM driver for nam-gpu: per land point prep, the SFLX column call,
// and write-back of soil, snow, canopy, skin temperature, fluxes and runoff.
//
// Notice: module_LS_NOAHLSM.F90 in NAM nam.v4.2.11 (MatthewPyle-NOAA/NAM @ 83df4e13c1, NMMB
// physics, NOAA/NCEP/EMC) keeps its author headers. There is no licence file in the NAM tree;
// the NOAA source notice in NOTICE applies. NAM is US Government work.
//
// Ported to CUDA for nam-gpu from phys/module_LS_NOAHLSM.F90:188-1161 (nam.v4.2.11, 83df4e13c1),
// the NOAHLSM subroutine body. The SFLX call (module_LS_NOAHLSM.F90:688) is the phys-noah-sflx
// unit, called here as noah_sflx(...) from phys_noah_sflx.cuh with the Fortran argument order
// (prototype below). This kernel contains every other statement of the NOAHLSM body.
//
// Configuration (parm/nam_configfile): land_surface noah (SURFACE_PHYSICS = LSMSCHEME), ivegsrc 1
// (IGBP, IW = 13), ucmcall 0, frpcpn true (FFROZP = SR), num_soil_layers 4.
//
// Not ported, with the reason:
//   - the UCMCALL == 1 urban branches (VEGTYP/T1 prep, urban() call and write-back): not reached
//     with ucmcall 0; the wrapper refuses ucmcall != 0.
//   - IPRINT debug prints (IPRINT = .false. in the source).
//   - MYJ, QZ0 (read only in the commented test at Fortran 499-500), ITIMESTEP (only in the
//     commented block at 457), GSW, CPM and ROVCP: passed by the caller, not read by the active body.
//   - ZNT: the caller's Z0 argument is not read by the active body (ZNT(I,J)=Z0K at 739 is
//     commented out).
//
// Values the Fortran leaves unassigned before the SFLX call, passed as 0.0 here. SFLX writes
// both before it reads them, so the 0.0 is never read (checked against module_LS_NOAHLSM.F90):
//   - Z0K (SFLX Z0): no assignment before the call (Fortran 673-695, call at 882). SFLX writes
//     Z0 before any read: CALL SNOWZ0 at 2080 or Z0=Z0BRD at 2082 (SFLX 1550-2278).
//   - ALBEDOK (SFLX ALBEDO): set only under UCMCALL == 1 (834), so 0.0 on this path. SFLX writes
//     ALBEDO before any read: ALB at 1979, CALL ALCALC at 1994, or 0.65 at 2001 (SFLX 1550-2278).
//   - DUMMY (SFCSPD, COSZ, PRCPRAIN, SOLARDIRECT) and the CM slot: the SFLX header marks COSZ,
//     PRCPRAIN and SOLARDIRECT "not used for now".
//
// Index convention (CONTRACT.md 1): Fortran (i,j,l) with 1-based l -> [l0, j0, i0], l0 = l-1.
// Fortran KTE (lowest layer) is l0 = LM-1; the surface interface KTE+1 is l0 = LM.
// Soil fields are [NSOIL, JM, IM]; layer 1 at the top is index 0.
//
// SFLX prototype expected from phys_noah_sflx.cuh (argument order = the Fortran CALL SFLX list;
// scalars by value, arrays and INOUT/OUT scalars by pointer):
//   __device__ void noah_sflx(
//       float FFROZP, int ICE, float DT, float ZLVL, int NSOIL, const float* SLDPTH,
//       int LOCAL, int IVEGSRC, int LUTYPE_ID, int SLTYPE_ID,
//       float LWDN, float SOLDN, float SOLNET, float SFCPRS, float PRCP, float SFCTMP, float Q2,
//       float SFCSPD, float* COSZ, float PRCPRAIN, float* SOLARDIRECT,
//       float TH2, float Q2SAT, float DQSDT2,
//       int VEGTYP, int SOILTYP, int SLOPETYP, float SHDFAC,
//       float ALB, float SNOALB, float TBOT, float Z0BRD, float* Z0, float EMISSI,
//       float* CMC, float* T1, float* STC, float* SMC, float* SH2O, float* SNOWH, float* SNEQV,
//       float* ALBEDO, float* CH, float* CM,
//       float* ETA, float* SHEAT, float* ETA_KINEMATIC, float* FDOWN,
//       float* EC, float* EDIR, float* ET, float* ETT, float* ESNOW, float* DRIP, float* DEW,
//       float* BETA, float* ETP, float* SSOIL,
//       float* FLX1, float* FLX2, float* FLX3,
//       float* SNOMLT, float* SNCOVR, float* RIMEF1,
//       float* RUNOFF1, float* RUNOFF2, float* RUNOFF3,
//       float* RC, float* PC, float* RSMIN, float* XLAI, float* RCS, float* RCT, float* RCQ,
//       float* RCSOIL, float* SOILW, float* SOILM, float* Q1,
//       float* SMCWLT, float* SMCDRY, float* SMCREF, float* SMCMAX, int* NROOT, float* RIBB);
//
// Launch: one thread per point over the B1 limits (its..ite, jts..jte, 1-based), one launch per
// NOAHLSM call, no host sync, no per-step allocation.

#include "nmmb_common.cuh"
#include "phys_noah_sflx.cuh"

// NOAHLSM module constants (module_LS_NOAHLSM.F90 PARAMETER block, single precision).
#define NOAH_A2        17.67f
#define NOAH_A3        273.15f
#define NOAH_A4        29.65f
#define NOAH_A23M4     (NOAH_A2 * (NOAH_A3 - NOAH_A4))   /* A23M4 = A2*(A3-A4) */
#define NOAH_ROW       1.0e3f
#define NOAH_XLF       3.50e5f                           /* module XLF */
#define NOAH_XLV       2.5e6f                            /* module XLV */
#define NOAH_RHOWATER  1.0e3f                            /* module rhowater */
#define NOAH_R_D       287.04f                           /* module r_d */
#define NOAH_CP        1004.6f                           /* module cp */
#define NOAH_CAPA      (NOAH_R_D / NOAH_CP)              /* local PARAMETER CAPA = R_D/CP */
#define NOAH_NSOIL_MAX 4

extern "C" __global__ void phys_noah_driver(
    // 3-D inputs, [LM, JM, IM] unless noted
    const float* __restrict__ DZ,          // DZ8W: layer thickness (m); lowest index LM-1
    const float* __restrict__ Q,           // Q3D: specific humidity (kg/kg)
    const float* __restrict__ PHINT,       // P8W3D: interface pressure, [LM+1, JM, IM] (Pa)
    const float* __restrict__ T,           // T3D: temperature (K)
    // 2-D, [JM, IM]
    float* __restrict__ TSFC,              // TSK (skin temperature), in/out
    float* __restrict__ TWBS,              // HFX (sensible heat flux, W m-2), out
    float* __restrict__ QWBS,              // QFX (moisture flux, kg m-2 s-1), out
    float* __restrict__ ELFLX,             // LH (latent heat flux, W m-2), out
    float* __restrict__ GRNFLX,            // GRDFLX (ground heat flux, W m-2), out
    const float* __restrict__ QGH,         // QGH (saturated specific humidity at 2 m)
    const float* __restrict__ RLW_DN_SFC,  // GLW (downward longwave, W m-2)
    const float* __restrict__ RSW_DN_SFC,  // SWDOWN (downward shortwave, W m-2)
    float* __restrict__ SMSTAV,            // out
    float* __restrict__ SMSTOT,            // out
    float* __restrict__ SSROFF,            // SFCRUNOFF, inout (accumulated, mm)
    float* __restrict__ BGROFF,            // UDRUNOFF, inout (accumulated, mm)
    const int* __restrict__ IVGTYP,        // vegetation type
    const int* __restrict__ ISLTYP,        // soil type
    const float* __restrict__ VGFRCK,      // VEGFRA (percent)
    float* __restrict__ ALBEDO,            // in/out, surface albedo
    const float* __restrict__ ALBASE,      // ALBBCK (background albedo)
    const float* __restrict__ Z0BASE,      // Z0 (background roughness, m)
    const float* __restrict__ TG,          // TMN (deep soil temperature, K)
    const float* __restrict__ XLAND,       // land mask (1 land, 2 water)
    const float* __restrict__ SICE,        // XICE (sea-ice fraction)
    const float* __restrict__ EPSR,        // EMISS (surface emissivity)
    float* __restrict__ SNOWC,             // in/out, snow cover fraction
    float* __restrict__ QSH,               // QSFC, in/out
    const float* __restrict__ RAINBL,      // precipitation per DT (mm)
    const float* __restrict__ DZSOIL,      // [NSOIL] soil layer thickness (SLDPTH)
    float* __restrict__ SMC,               // [NSOIL, JM, IM] SMOIS in/out
    float* __restrict__ STC,               // [NSOIL, JM, IM] TSLB in/out
    float* __restrict__ SNOW,              // SNO (liquid water snow, mm), in/out
    float* __restrict__ CMC,               // CANWAT (canopy water, mm), in/out
    const float* __restrict__ CHS,         // exchange coefficient for heat
    float* __restrict__ CHS2,              // 2 m exchange coefficient for heat, out
    float* __restrict__ CQS2,              // 2 m exchange coefficient for moisture, in/out
    const float* __restrict__ SR,          // frozen precipitation fraction
    const float* __restrict__ RIMEF,       // rime factor
    float* __restrict__ CHKLOWQ,           // out
    float* __restrict__ SH2O,              // [NSOIL, JM, IM] in/out
    float* __restrict__ SNOWH,             // snow depth (m), in/out
    const float* __restrict__ MXSNAL,      // SNOALB (upper bound on deep-snow albedo)
    float* __restrict__ ACSNOM,            // in/out, accumulated snow melt (mm)
    float* __restrict__ ACSNOW,            // in/out, accumulated snow (mm)
    float* __restrict__ SNOPCX,            // in/out, snow phase change flux (W m-2)
    float* __restrict__ POTEVP,            // in/out, accumulated potential evaporation
    float* __restrict__ RIB,               // bulk Richardson number (read as RIBB)
    // scalars
    float dt, int nsoil, int ivegsrc, int ucmcall, int frpcpn,
    int lutype_id, int sltype_id,
    int its, int ite, int jts, int jte,    // 1-based B1 limits
    int im, int jm, int lm,
    // REDPRM tables (phys-noah-parm-tables), read by SFLX through NoahTbl
    const float* __restrict__ albtbl, const float* __restrict__ z0tbl, const float* __restrict__ shdtbl,
    const float* __restrict__ rstbl, const float* __restrict__ rgltbl, const float* __restrict__ hstbl,
    const float* __restrict__ snuptbl, const float* __restrict__ laitbl, const int* __restrict__ nrotbl,
    const float* __restrict__ bb, const float* __restrict__ drysmc, const float* __restrict__ f11,
    const float* __restrict__ maxsmc, const float* __restrict__ refsmc, const float* __restrict__ satpsi,
    const float* __restrict__ satdk, const float* __restrict__ satdw, const float* __restrict__ wltsmc,
    const float* __restrict__ qtz, const float* __restrict__ slope_data,
    float sbeta_data, float fxexp_data, float csoil_data, float salp_data, float refdk_data,
    float refkdt_data, float frzk_data, float zbot_data, float czil_data, float topt_data,
    float cmcmax_data, float cfactr_data, float rsmax_data, int bare)
{
    NoahTbl tb;
    tb.albtbl = albtbl; tb.z0tbl = z0tbl; tb.shdtbl = shdtbl; tb.rstbl = rstbl; tb.rgltbl = rgltbl;
    tb.hstbl = hstbl; tb.snuptbl = snuptbl; tb.laitbl = laitbl; tb.nrotbl = nrotbl;
    tb.bb = bb; tb.drysmc = drysmc; tb.f11 = f11; tb.maxsmc = maxsmc; tb.refsmc = refsmc;
    tb.satpsi = satpsi; tb.satdk = satdk; tb.satdw = satdw; tb.wltsmc = wltsmc; tb.qtz = qtz;
    tb.slope_data = slope_data;
    tb.sbeta_data = sbeta_data; tb.fxexp_data = fxexp_data; tb.csoil_data = csoil_data;
    tb.salp_data = salp_data; tb.refdk_data = refdk_data; tb.refkdt_data = refkdt_data;
    tb.frzk_data = frzk_data; tb.zbot_data = zbot_data; tb.czil_data = czil_data;
    tb.topt_data = topt_data; tb.cmcmax_data = cmcmax_data; tb.cfactr_data = cfactr_data;
    tb.rsmax_data = rsmax_data; tb.bare = bare;

    const int ni = ite - its + 1;
    const int nj = jte - jts + 1;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni <= 0 || nj <= 0 || col >= ni * nj) return;
    const int i0 = (its - 1) + col % ni;
    const int j0 = (jts - 1) + col / ni;

    const size_t h = IDX2(i0, j0, im);             // 2-D index
    const int kl = lm - 1;                         // Fortran KTE (lowest layer), zero-based
    const int ks = lm;                             // Fortran KTE+1 (surface interface), zero-based
    const float fdtliw = dt / (NOAH_ROW * NOAH_XLF);       // FDTLIW = DT/ROWLIW
    const float fdtw = dt / (NOAH_XLV * NOAH_RHOWATER);    // FDTW = DT/(XLV*RHOWATER)
    const int iw = (ivegsrc == 1) ? 13 : 1;        // IW (IVEGSRC 1 -> 13)

    // ---- label 50: water and sea-ice pre-set, per point ----
    if ((XLAND[h] - 1.5f) >= 0.f) {                // water point
        SMSTAV[h] = 0.f;
        SMSTOT[h] = 0.f;
        for (int ns = 0; ns < nsoil; ++ns) {
            SMC[IDX3(i0, j0, ns, im, jm)] = 1.f;
            SH2O[IDX3(i0, j0, ns, im, jm)] = 1.f;
            STC[IDX3(i0, j0, ns, im, jm)] = 273.16f;      // TSLB = 273.16 (!STEMP)
        }
    } else if (SICE[h] > 0.5f) {                   // land point with sea ice
        SMSTAV[h] = 0.f;
        SMSTOT[h] = 0.f;
        for (int ns = 0; ns < nsoil; ++ns) {
            SMC[IDX3(i0, j0, ns, im, jm)] = 1.f;
            SH2O[IDX3(i0, j0, ns, im, jm)] = 1.f;
        }
    }

    // ---- label 100: per point. The prelude runs for every point. ----
    const float rimef1 = RIMEF[h];
    const float psfc = PHINT[IDX3(i0, j0, ks, im, jm)];                  // P8W3D(KTE+1)
    const float sfcprs = (PHINT[IDX3(i0, j0, kl, im, jm)] + psfc) * 0.5f; // P8W3D(KTE), KTE+1
    const float q2k = Q[IDX3(i0, j0, kl, im, jm)];                        // Q3D(KTE)
    float q2sat = QGH[h];
    CHKLOWQ[h] = 1.f;
    const float sfctmp = T[IDX3(i0, j0, kl, im, jm)];                     // T3D(KTE)
    const float zlvl = 0.5f * DZ[IDX3(i0, j0, kl, im, jm)];               // 0.5*DZ8W(KTE)
    const float apes = powf(1.e5f / psfc, NOAH_CAPA);
    const float apelm = powf(1.e5f / sfcprs, NOAH_CAPA);
    const float sfcth2 = sfctmp * apelm;                                  // SFCTH2
    const float th2 = sfcth2 / apes;
    const float emissi = EPSR[h];
    const float lwdn = RLW_DN_SFC[h] * emissi;
    const float soldn = RSW_DN_SFC[h];
    const float solnet = soldn * (1.f - ALBEDO[h]);
    const float prcp = RAINBL[h] / dt;
    int vegtyp = IVGTYP[h];
    int soiltyp = ISLTYP[h];
    float shdfac = VGFRCK[h] / 100.f;
    float t1 = TSFC[h];
    float chk = CHS[h];                                                  // CHS: INOUT in SFLX (CH)
    const float snoalb1 = MXSNAL[h];
    float sneqv = SNOW[h] * 0.001f;
    float snowhk = SNOWH[h];
    const float ffrozp = (frpcpn != 0) ? SR[h]
                                        : ((sfctmp <= 273.15f) ? 1.f : 0.f);  // FFROZP

    if ((XLAND[h] - 1.5f) < 0.f) {                 // land point (ELSE of the water test)
        const int ice = (SICE[h] > 0.5f) ? 1 : 0;
        float dqsdt2 = q2sat * NOAH_A23M4 / ((sfctmp - NOAH_A4) * (sfctmp - NOAH_A4));

        if (SNOW[h] > 0.f) {                       // snow-covered point: saturation blend
            const float sfctsno = sfctmp;
            const float e2sat = 611.2f * expf(6174.f * (1.f / 273.15f - 1.f / sfctsno));
            float q2sati = 0.622f * e2sat / (psfc - e2sat);
            q2sati = q2sati / (1.0f + q2sati);      // spec. hum.
            if (t1 > 273.14f) {
                q2sat = q2sat * (1.f - SNOWC[h]) + q2sati * SNOWC[h];
                dqsdt2 = dqsdt2 * (1.f - SNOWC[h]) + q2sati * 6174.f / (sfctsno * sfctsno) * SNOWC[h];
            } else {
                q2sat = q2sati;
                dqsdt2 = q2sati * 6174.f / (sfctsno * sfctsno);
            }
            if (t1 > 273.14f && SNOWC[h] > 0.f) dqsdt2 = dqsdt2 * (1.f - SNOWC[h]);
        }

        const float tbot = (ice == 0) ? TG[h] : 271.16f;
        if (vegtyp == 25 || vegtyp == 26 || vegtyp == 27) shdfac = 0.f;
        if (soiltyp == 14 && SICE[h] == 0.f) soiltyp = 7;   // water soil type on a land point

        float cmc = CMC[h];
        const float albbrd = ALBASE[h];
        const float z0brd = Z0BASE[h];
        float ribb = RIB[h];

        float smc[NOAH_NSOIL_MAX], stc[NOAH_NSOIL_MAX], swc[NOAH_NSOIL_MAX], et[NOAH_NSOIL_MAX];
        for (int ns = 0; ns < nsoil; ++ns) {
            smc[ns] = SMC[IDX3(i0, j0, ns, im, jm)];
            stc[ns] = STC[IDX3(i0, j0, ns, im, jm)];
            swc[ns] = SH2O[IDX3(i0, j0, ns, im, jm)];
        }
        if ((sneqv != 0.f && snowhk == 0.f) || (snowhk <= sneqv)) {
            snowhk = 5.f * sneqv;
        }

        // UCMCALL /= 1 branch (the urban UCMCALL == 1 branch is not ported, see header).
        if (ucmcall != 1) {
            if (IVGTYP[h] == iw || IVGTYP[h] == 31 || IVGTYP[h] == 32 || IVGTYP[h] == 33) {
                vegtyp = 1;
                if (ivegsrc == 1) vegtyp = 13;
            }
        }

        // SFLX column call (phys-noah-sflx).
        float cosz = 0.f, solardirect = 0.f, sfcspd = 0.f, prcprain = 0.f;   // DUMMY slots
        float z0k = 0.f, cm = 0.f, albedok = 0.f;                            // undefined in Fortran
        float eta = 0.f, sheat = 0.f, eta_kin = 0.f, fdown = 0.f;
        float ec = 0.f, edir = 0.f, ett = 0.f, esnow = 0.f, drip = 0.f, dew = 0.f;
        float beta = 0.f, etp = 0.f, ssoil = 0.f;
        float flx1 = 0.f, flx2 = 0.f, flx3 = 0.f;
        float snomlt = 0.f, sncovr = 0.f, rimef1_out = rimef1;
        float runoff1 = 0.f, runoff2 = 0.f, runoff3 = 0.f;
        float rc = 0.f, pc = 0.f, rsmin = 0.f, xlai = 0.f, rcs = 0.f, rct = 0.f, rcq = 0.f, rcsoil = 0.f;
        float soilw = 0.f, soilm = 0.f, q1 = 0.f;
        float smcwlt = 0.f, smcdry = 0.f, smcref = 0.f, smcmax = 0.f;
        int nroot = 0;
        const int slopetyp = 1;                          // SLOPETYP = 1 (driver prologue)
        const int local = 0;                             // LOCAL = .false.

        noah_sflx(ffrozp, ice, dt, zlvl, nsoil, DZSOIL,
                  local, ivegsrc, lutype_id, sltype_id,
                  lwdn, soldn, solnet, sfcprs, prcp, sfctmp, q2k, sfcspd,
                  &cosz, prcprain, &solardirect,
                  th2, q2sat, dqsdt2,
                  vegtyp, soiltyp, slopetyp, shdfac,
                  albbrd, snoalb1, tbot, z0brd, &z0k, emissi,
                  &cmc, &t1, stc, smc, swc, &snowhk, &sneqv, &albedok, &chk, &cm,
                  &eta, &sheat, &eta_kin, &fdown,
                  &ec, &edir, et, &ett, &esnow, &drip, &dew,
                  &beta, &etp, &ssoil,
                  &flx1, &flx2, &flx3,
                  &snomlt, &sncovr, &rimef1_out,
                  &runoff1, &runoff2, &runoff3,
                  &rc, &pc, &rsmin, &xlai, &rcs, &rct, &rcq, &rcsoil,
                  &soilw, &soilm, &q1,
                  &smcwlt, &smcdry, &smcref, &smcmax, &nroot, &ribb, tb);

        // ---- write-back after SFLX ----
        CMC[h] = cmc;
        SNOW[h] = sneqv * 1000.f;
        SNOWH[h] = snowhk;
        ALBEDO[h] = albedok;
        TSFC[h] = t1;
        TWBS[h] = sheat;                               // HFX = SHEAT
        POTEVP[h] = POTEVP[h] + etp * fdtw;
        QWBS[h] = eta_kin;                             // QFX = ETA_KINEMATIC
        ELFLX[h] = eta;                                // LH = ETA
        GRNFLX[h] = ssoil;                             // GRDFLX = SSOIL
        SNOWC[h] = sncovr;
        CHS2[h] = CQS2[h];
        if (q1 > QSH[h]) {                             // Q1 > QSFC (before update)
            CQS2[h] = CHS[h];
        }
        QSH[h] = q1 / (1.0f - q1);
        for (int ns = 0; ns < nsoil; ++ns) {
            SMC[IDX3(i0, j0, ns, im, jm)] = smc[ns];
            STC[IDX3(i0, j0, ns, im, jm)] = stc[ns];
            SH2O[IDX3(i0, j0, ns, im, jm)] = swc[ns];
        }
        SMSTAV[h] = soilw;
        SMSTOT[h] = soilm * 1000.f;
        SSROFF[h] = SSROFF[h] + runoff1 * dt * 1000.0f;
        BGROFF[h] = BGROFF[h] + (runoff2 + runoff3) * dt * 1000.0f;
        if (ffrozp > 0.f) {
            ACSNOW[h] = ACSNOW[h] + ffrozp * prcp * dt;
        }
        if (SNOW[h] > 0.f) {                           // SNOW after the update above
            ACSNOM[h] = ACSNOM[h] + snomlt * 1000.f;
            SNOPCX[h] = SNOPCX[h] - snomlt / fdtliw;
        }
    }
}
