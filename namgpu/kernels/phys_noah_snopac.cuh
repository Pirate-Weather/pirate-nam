// phys_noah_snopac.cuh -- SNOPAC device function: snow energy and water balance (Noah LSM),
// for every column with a snow pack (ESD > 0), plus the no-snow-state EVAPO/SMFLX/SHFLX path
// the Fortran takes when SNOPAC is entered.
//
// NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), NOAA US government work. Author headers of the
// source module are kept as notices: MODULE MODULE_LS_NOAHLSM (module_LS_NOAHLSM.F90).
// Ported to CUDA for WOOF-NAM from phys/module_LS_NOAHLSM.F90:4050-4374 (nam.v4.2.11, 83df4e13c1).
// Start point: namgpu/third_party/arwen/core/kernels/noah.cu (SNOPAC section, Apache-2.0, ArWen). The ArWen
// section is the WRF-lineage SFLX body with a different argument list and different helper
// signatures (WRF EVAPO/SMFLX/SHFLX/SNOWPACK), so the NAM Fortran is the reference: every executed
// statement of SNOPAC is transcribed below in NAM order.
//
// Helpers called here are separate CONTRACT units, declared (not defined) in this header:
//   nam_evapo     EVAPO    module_LS_NOAHLSM.F90:2557-2641  (unit phys-noah-evapo)
//   nam_smflx     SMFLX    module_LS_NOAHLSM.F90:3799-3916  (unit phys-noah-smflx)
//   nam_shflx     SHFLX    module_LS_NOAHLSM.F90:3729-3796  (unit phys-noah-shflx)
//   nam_snowpack  SNOWPACK module_LS_NOAHLSM.F90:4378-4505  (unit phys-noah-snowpack)
// Each must be defined by its own header before this one is included in a translation unit.
//
// Fortran SUBROUTINE SNOPAC, executed statements (ESDMIN=1e-6, LSUBC=2.501e6, LSUBS=2.83e6,
// TFREEZ=273.15, SNOEXP=2.0, EMISSI_S=0.95 local; SIGMA=5.67e-8, CPH2O=4218, CPICE=2106,
// LSUBF=3.335e5 from the module PARAMETER block):
//   DEW=0 EDIR=0 EDIR1=0 EC1=0 EC=0 EMISSI_S=0.95; ET(k)=ET1(k)=0; ETT=ETT1=0; ETNS=ETNS1=0;
//   ESNOW=ESNOW1=ESNOW2=0; PRCP1=PRCPF*0.001; BETA=1
//   IF (ETP <= 0): IF (ETP == 0) BETA=0; ETP1=ETP*0.001; DEW=-ETP1; ESNOW2=ETP1*DT;
//                  ETANRG = ETP*((1-SNCOVR)*LSUBC + SNCOVR*LSUBS)
//   ELSE: ETP1=ETP*0.001
//         IF (ICE /= 1) THEN
//           IF (SNCOVR < 1): EVAPO(ETNS1,...,EDIR1,EC1,ET1,ETT1,...), then scale EDIR1, EC1,
//                            ET1(k), ETT1, ETNS1 by (1-SNCOVR); EDIR=EDIR1*1000; EC=EC1*1000;
//                            ET(k)=ET1(k)*1000; ETT=ETT1*1000; ETNS=ETNS1*1000
//         ENDIF
//         ESNOW=ETP*SNCOVR; ESNOW1=ESNOW*0.001; ESNOW2=ESNOW1*DT; ETANRG=ESNOW*LSUBS+ETNS*LSUBC
//   FLX1=0; IF (SNOWNG) FLX1=CPICE*PRCP*(T1-SFCTMP) ELSE IF (PRCP>0) FLX1=CPH2O*PRCP*(T1-SFCTMP)
//   DSOIL=-(0.5*ZSOIL(1)); DTOT=SNOWH+DSOIL; DENOM=1+DF1/(DTOT*RR*RCH)
//   T12A = ((FDOWN-FLX1-FLX2-((SNCOVR*EMISSI_S)+EMISSI*(1-SNCOVR))*SIGMA*T24)/RCH
//           + TH2 - SFCTMP - ETANRG/RCH)/RR
//   T12B = DF1*STC(1)/(DTOT*RR*RCH);  T12 = (SFCTMP+T12A+T12B)/DENOM
//   IF (T12 <= TFREEZ): T1=T12; SSOIL=DF1*(T1-STC(1))/DTOT; ESD=max(0,ESD-ESNOW2);
//                       FLX3=0; EX=0; SNOMLT=0
//   ELSE: T1 = TFREEZ*SNCOVR**2 + T12*(1-SNCOVR**2); BETA=1; SSOIL=DF1*(T1-STC(1))/DTOT
//         IF (ESD-ESNOW2 <= ESDMIN): ESD=0; EX=0; SNOMLT=0
//         ELSE: ESD=ESD-ESNOW2; SEH=RCH*(T1-TH2); T14=(T1*T1)*(T1*T1)
//               FLX3 = FDOWN-FLX1-FLX2-((SNCOVR*EMISSI_S)+EMISSI*(1-SNCOVR))*SIGMA*T14
//                      -SSOIL-SEH-ETANRG;  IF (FLX3<=0) FLX3=0
//               EX=FLX3*0.001/LSUBF; SNOMLT=EX*DT
//               IF (ESD-SNOMLT >= ESDMIN) ESD=ESD-SNOMLT
//               ELSE EX=ESD/DT; FLX3=EX*1000*LSUBF; SNOMLT=ESD; ESD=0
//   PRCP1 = PRCP1 + EX
//   IF (ICE /= 1) CALL SMFLX(SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,
//                            SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,
//                            EDIR1,EC1,ET1,DRIP)
//   ZZ1=1; YY=STC(1)-0.5*SSOIL*ZSOIL(1)*ZZ1/DF1; T11=T1
//   CALL SHFLX(SSOIL1,STC,SMC,SMCMAX,NSOIL,T11,DT,YY,ZZ1,ZSOIL,TBOT,ZBOT,SMCWLT,PSISAT,SH2O,
//              BEXP,F1,DF1,ICE,QUARTZ,CSOIL,VEGTYP,IVEGSRC)   (SSOIL1, T11 are dummies)
//   IF (ESD > 0): CALL SNOWPACK(ESD,DT,SNOWH,SNDENS,T1,YY)
//   ELSE: ESD=0; SNOWH=0; SNDENS=0; SNCOND=1; SNCOVR=0
//
// Dummy arguments not read by the executed statements: ETA (local in NAM, never read), EPSCA,
// SFCPRS, SOLDN, RIBB (its only use is in a commented-out block). ETP is read-only here (the
// commented-out block was the only writer). SNCOND is set and never read.
//
// Single precision throughout (Fortran REAL = float, kfpt = single). No host interaction.

#pragma once

#include "nmmb_common.cuh"

// Fixed constraint: the Fortran NSOIL is a runtime dummy (4050-4060); this port hardcodes 4
// (CONTRACT section 1, NSOIL = 4). A different soil depth needs a new build.
#define NAM_SNOPAC_NSOIL     4
#define NAM_SNOPAC_ESDMIN    1.0e-6f
#define NAM_SNOPAC_LSUBC     2.501000e+6f
#define NAM_SNOPAC_LSUBS     2.83e+6f
#define NAM_SNOPAC_TFREEZ    273.15f
#define NAM_SNOPAC_EMISSI_S  0.95f
#define NAM_SNOPAC_SIGMA     5.67E-8f     // module_LS_NOAHLSM SIGMA
#define NAM_SNOPAC_CPH2O     4.218E+3f    // module_LS_NOAHLSM CPH2O
#define NAM_SNOPAC_CPICE     2.106E+3f    // module_LS_NOAHLSM CPICE
#define NAM_SNOPAC_LSUBF     3.335E+5f    // module_LS_NOAHLSM LSUBF

// Helpers defined in their own unit headers (listed above). Declared here only.
__device__ void nam_evapo(float& eta1, float* smc, int nsoil, float& cmc, float etp1, float dt,
                          const float* zsoil, const float* sh2o, float smcmax, float bexp,
                          float pc, float smcwlt, float dksat, float dwsat, float smcref,
                          float shdfac, float cmcmax, float smcdry, float cfactr,
                          float& edir, float& ec, float* et, float& ett, float sfctmp,
                          float q2, int nroot, const float* rtdis, float fxexp, int vegtyp,
                          const float* stc);
__device__ void nam_smflx(float* smc, int nsoil, float& cmc, float dt, float prcp1,
                          const float* zsoil, float* sh2o, float slope, float kdt, float frzfact,
                          float smcmax, float bexp, float smcwlt, float dksat, float dwsat,
                          float shdfac, float cmcmax, float& runoff1, float& runoff2,
                          float& runoff3, float& edir, float& ec, float* et, float& drip);
__device__ void nam_shflx(float& ssoil, float* stc, float* smc, float smcmax, int nsoil,
                          float& t1, float dt, float yy, float zz1, const float* zsoil,
                          float tbot, float zbot, float smcwlt, float psisat, float* sh2o,
                          float bexp, float f1, float df1, int ice, float quartz, float csoil,
                          int vegtyp, int ivegsrc);
__device__ void nam_snowpack(float& esd, float dtsec, float& snowh, float& sndens,
                             float& tsnow, float tsoil);

// Per-column SNOPAC state. The kernel loads each field from its CONTRACT array, the device
// function updates the INOUT/OUT members in place, and the kernel stores them back.
struct NamSnopacCol {
    // in
    float etp, prcp, prcpf;
    int   snowng;
    float smcmax, smcwlt, smcref, smcdry, cmcmax;
    float dt, sbeta, df1, q2, sfctmp, t24, th2, fdown, f1;
    float bexp, pc, rch, rr, cfactr;
    float slope, kdt, frzfact, psisat, dwsat, dksat, tbot, zbot, shdfac;
    int   nroot, vegtyp, ice, ivegsrc;
    float quartz, fxexp, csoil, emissi;
    // soil columns [NSOIL]
    float smc[NAM_SNOPAC_NSOIL];      // INOUT (SMFLX, SHFLX)
    float sh2o[NAM_SNOPAC_NSOIL];     // INOUT (SMFLX, SHFLX)
    float stc[NAM_SNOPAC_NSOIL];      // INOUT (SHFLX)
    float zsoil[NAM_SNOPAC_NSOIL];
    float rtdis[NAM_SNOPAC_NSOIL];
    float et[NAM_SNOPAC_NSOIL];       // OUT
    // INOUT scalars
    float cmc, t1, sncovr, esd, sndens, snowh, beta, flx2, ribb;
    // OUT scalars
    float dew, edir, ec, ett, etns, esnow, flx1, flx3, ssoil, snomlt, drip;
    float runoff1, runoff2, runoff3;
};

// Device function: SNOPAC for one column. Transcription of the statements listed in the header.
__device__ __forceinline__ void nam_snopac(NamSnopacCol& s)
{
    const int nsoil = NAM_SNOPAC_NSOIL;
    float edir1 = 0.0f, ec1 = 0.0f, ett1 = 0.0f, etns1 = 0.0f;
    float et1[NAM_SNOPAC_NSOIL];
    float ex = 0.0f, etanrg = 0.0f, etp1 = 0.0f;
    float ssoil1 = 0.0f, t11 = 0.0f, zz1 = 1.0f, yy = 0.0f;

    // Native leaves FLX3 (ESD-ESNOW2 <= ESDMIN branch), and DRIP and RUNOFF1-3 (ICE == 1, SMFLX not
    // called), unset. They are defined as 0 here so no indeterminate value is stored.
    s.flx3 = 0.0f;
    s.drip = 0.0f;
    s.runoff1 = 0.0f;
    s.runoff2 = 0.0f;
    s.runoff3 = 0.0f;
    s.dew = 0.0f;
    s.edir = 0.0f;
    s.ec = 0.0f;
    const float emissi_s = NAM_SNOPAC_EMISSI_S;
    for (int k = 0; k < nsoil; ++k) { s.et[k] = 0.0f; et1[k] = 0.0f; }
    s.ett = 0.0f;
    s.etns = 0.0f;
    s.esnow = 0.0f;
    float esnow2 = 0.0f;

    float prcp1 = s.prcpf * 0.001f;
    s.beta = 1.0f;

    if (s.etp <= 0.0f) {
        if (s.etp == 0.0f) s.beta = 0.0f;
        etp1 = s.etp * 0.001f;
        s.dew = -etp1;
        esnow2 = etp1 * s.dt;
        etanrg = s.etp * ((1.0f - s.sncovr) * NAM_SNOPAC_LSUBC + s.sncovr * NAM_SNOPAC_LSUBS);
    } else {
        etp1 = s.etp * 0.001f;
        if (s.ice != 1) {
            if (s.sncovr < 1.0f) {
                nam_evapo(etns1, s.smc, nsoil, s.cmc, etp1, s.dt, s.zsoil, s.sh2o,
                          s.smcmax, s.bexp, s.pc, s.smcwlt, s.dksat, s.dwsat, s.smcref,
                          s.shdfac, s.cmcmax, s.smcdry, s.cfactr,
                          edir1, ec1, et1, ett1, s.sfctmp, s.q2, s.nroot, s.rtdis,
                          s.fxexp, s.vegtyp, s.stc);
                edir1 = edir1 * (1.0f - s.sncovr);
                ec1 = ec1 * (1.0f - s.sncovr);
                for (int k = 0; k < nsoil; ++k) et1[k] = et1[k] * (1.0f - s.sncovr);
                ett1 = ett1 * (1.0f - s.sncovr);
                etns1 = etns1 * (1.0f - s.sncovr);
                s.edir = edir1 * 1000.0f;
                s.ec = ec1 * 1000.0f;
                for (int k = 0; k < nsoil; ++k) s.et[k] = et1[k] * 1000.0f;
                s.ett = ett1 * 1000.0f;
                s.etns = etns1 * 1000.0f;
            }
        }
        s.esnow = s.etp * s.sncovr;
        const float esnow1 = s.esnow * 0.001f;
        esnow2 = esnow1 * s.dt;
        etanrg = s.esnow * NAM_SNOPAC_LSUBS + s.etns * NAM_SNOPAC_LSUBC;
    }

    s.flx1 = 0.0f;
    if (s.snowng) {
        s.flx1 = NAM_SNOPAC_CPICE * s.prcp * (s.t1 - s.sfctmp);
    } else {
        if (s.prcp > 0.0f) s.flx1 = NAM_SNOPAC_CPH2O * s.prcp * (s.t1 - s.sfctmp);
    }

    const float dsoil = -(0.5f * s.zsoil[0]);
    const float dtot = s.snowh + dsoil;
    const float denom = 1.0f + s.df1 / (dtot * s.rr * s.rch);
    const float emis_mix = (s.sncovr * emissi_s) + s.emissi * (1.0f - s.sncovr);
    const float t12a = ((s.fdown - s.flx1 - s.flx2 - emis_mix * NAM_SNOPAC_SIGMA * s.t24)
                        / s.rch + s.th2 - s.sfctmp - etanrg / s.rch) / s.rr;
    const float t12b = s.df1 * s.stc[0] / (dtot * s.rr * s.rch);
    const float t12 = (s.sfctmp + t12a + t12b) / denom;

    if (t12 <= NAM_SNOPAC_TFREEZ) {
        s.t1 = t12;
        s.ssoil = s.df1 * (s.t1 - s.stc[0]) / dtot;
        s.esd = fmaxf(0.0f, s.esd - esnow2);
        s.flx3 = 0.0f;
        ex = 0.0f;
        s.snomlt = 0.0f;
    } else {
        const float sncov2 = s.sncovr * s.sncovr;          // SNCOVR ** SNOEXP, SNOEXP = 2.0
        s.t1 = NAM_SNOPAC_TFREEZ * sncov2 + t12 * (1.0f - sncov2);
        s.beta = 1.0f;
        s.ssoil = s.df1 * (s.t1 - s.stc[0]) / dtot;
        if (s.esd - esnow2 <= NAM_SNOPAC_ESDMIN) {
            s.esd = 0.0f;
            ex = 0.0f;
            s.snomlt = 0.0f;
        } else {
            s.esd = s.esd - esnow2;
            const float seh = s.rch * (s.t1 - s.th2);
            float t14 = s.t1 * s.t1;
            t14 = t14 * t14;
            s.flx3 = s.fdown - s.flx1 - s.flx2 - emis_mix * NAM_SNOPAC_SIGMA * t14
                     - s.ssoil - seh - etanrg;
            if (s.flx3 <= 0.0f) s.flx3 = 0.0f;
            ex = s.flx3 * 0.001f / NAM_SNOPAC_LSUBF;
            s.snomlt = ex * s.dt;
            if (s.esd - s.snomlt >= NAM_SNOPAC_ESDMIN) {
                s.esd = s.esd - s.snomlt;
            } else {
                ex = s.esd / s.dt;
                s.flx3 = ex * 1000.0f * NAM_SNOPAC_LSUBF;
                s.snomlt = s.esd;
                s.esd = 0.0f;
            }
        }
    }
    prcp1 = prcp1 + ex;

    if (s.ice != 1) {
        nam_smflx(s.smc, nsoil, s.cmc, s.dt, prcp1, s.zsoil, s.sh2o, s.slope, s.kdt, s.frzfact,
                  s.smcmax, s.bexp, s.smcwlt, s.dksat, s.dwsat, s.shdfac, s.cmcmax,
                  s.runoff1, s.runoff2, s.runoff3, edir1, ec1, et1, s.drip);
    }

    zz1 = 1.0f;
    yy = s.stc[0] - 0.5f * s.ssoil * s.zsoil[0] * zz1 / s.df1;
    t11 = s.t1;
    nam_shflx(ssoil1, s.stc, s.smc, s.smcmax, nsoil, t11, s.dt, yy, zz1, s.zsoil,
              s.tbot, s.zbot, s.smcwlt, s.psisat, s.sh2o, s.bexp, s.f1, s.df1, s.ice,
              s.quartz, s.csoil, s.vegtyp, s.ivegsrc);

    if (s.esd > 0.0f) {
        nam_snowpack(s.esd, s.dt, s.snowh, s.sndens, s.t1, yy);
    } else {
        s.esd = 0.0f;
        s.snowh = 0.0f;
        s.sndens = 0.0f;
        s.sncovr = 0.0f;
    }
    (void)etns1; (void)ett1; (void)ssoil1; (void)t11;
}
