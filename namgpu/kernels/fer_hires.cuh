// fer_hires.cuh -- module constants and table layout of MP_FER_HIRES (Ferrier-Aligo microphysics).
//
// NOTICE: NAM source file phys/module_MP_FER_HIRES.F90:1-207 (NCEP/EMC, US Government work).
// The file has no copyright or licence block. Its changelog (lines 1-105, "Modified fer_hires
// microphysics") is the notice. Ported to CUDA for nam-gpu from phys/module_MP_FER_HIRES.F90:1-207
// (nam.v4.2.11, 83df4e13c1).
//
// Statement map for lines 1-207 of phys/module_MP_FER_HIRES.F90 (Fortran line numbers, checked
// against the file):
//   line 107        MODULE MODULE_MP_FER_HIRES            module header
//   line 110-111    USE ESMF, USE MODULE_INCLUDE          no GPU counterpart (ESMF, kind parameters)
//   line 112-114    USE MODULE_CONSTANTS, ONLY: ...       FER_* module-constant block below; values from
//                                                         nmm/module_CONSTANTS.F90 (md5 identical to share copy)
//   line 116        PUBLIC statement                      no-op on the device; every symbol is a macro or a table
//   line 118-125    PRIVATE SAVE scalars (ABFR .. QAUT0,  runtime scalars and tables set by FERRIER_INIT_HR
//                   CN0r0 .. RQR_DRmin)                   (unit phys-fer-init), held in ctx.tables, not __device__ globals.
//                                                         Init unit scalars: ABFR CBFR CIACW CIACR C_N0r0 C_NR Crain
//                                                         CRACW ARAUT BRAUT ESW0 RFmx1 ARcw RH_NgC RH_NgT RR_DRmin
//                                                         RR_DR1..RR_DR5 RR_DRmax RQhail AVhail BVhail QAUT0 CN0r0
//                                                         CN0r_DMRmin CN0r_DMRmax RFmax RQR_DRmax RQR_DRmin C1XPVS0 C2XPVS0 C1XPVS C2XPVS.
//   line 127-128    MY_T1, MY_T2 PARAMETERs; MY_GROWTH_NMM SAVE table      FER_MY_T1, FER_MY_T2, FER_TBL_NMY
//   line 130-133    DMImin, DMImax, DelDMI, XMImin, XMImax, XMIexp, MDImin, MDImax   FER_DMI_*, FER_XMI_*, FER_MDI_*
//   line 134-136    ACCRI..VENTI2 and MASSI-style MDImin:MDImax arrays, SDENS          table layout, phys-fer-init
//   line 138-143    DMRmin, DMRmax, DelDMR, XMRmin, XMRmax, MDRmin, MDRmax, ACCRR..VENTR2   FER_DMR_*, FER_XMR_*, FER_MDR_*
//   line 145-146    Nrime PARAMETER; VEL_RF(2:9,0:Nrime) SAVE              FER_NRIME, FER_TBL_VEL_RF_*, FER_TBL_VEL_RF_INDEX
//   line 148-151    NX, XMIN, XMAX, TBPVS/TBPVS0(NX), C1XPVS0..C2XPVS      FER_NX, FER_XMIN, FER_XMAX, FER_TBL_NX
//   line 153-166    CVAP, XLF, XLS, EPSQ1 .. XLS3, CLIMIT, CLIMIT1, C1, DMR1..DMR5, XMR1..XMR5, RQRmix, RQSmix, Cdry, Cwet
//                                                          FER_* below
//   line 167-168    MDR1..MDR5 INTEGER PARAMETERs          FER_MDR1..FER_MDR5
//   line 171-173    LOGICAL WARN1, WARN2, WARN3, WARN5 (host-side warning latches, no device output);
//                   REAL SAVE Pwarn, QTwarn (line 172, mutable state, see FER_PWARN_INIT);
//                   INTEGER MAX_ITERATIONS (line 173)      FER_MAX_ITERATIONS
//   line 190-198    T_ICE .. NCW PUBLIC PARAMETERs         FER_T_ICE .. FER_NCW
//   line 200        MASSI (PUBLIC, MDImin:MDImax)          FER_TBL_NMDI
//   line 203        CONTAINS                               routines belong to other units
//
// Changelog: lines 1-105. The WARN4 path (Pwarn/QTwarn update) is at lines 387-394 inside FER_HIRES
// (unit phys-fer-fer-hires).
#pragma once

// ---- MODULE_CONSTANTS (nmm/module_CONSTANTS.F90), single precision ----
#define FER_CP      1004.6f
#define FER_EPSQ    1.0e-12f
#define FER_G       9.8060226f
#define FER_RHOL    1000.0f                    // RHOWATER
#define FER_RD      287.04f                    // R_D
#define FER_RV      461.6f                     // R_V
#define FER_T0C     273.15f                    // TIW
#define FER_EP2     (287.04f / 461.6f)         // EP_2 = r / r_v
#define FER_EP1     (FER_RV / FER_RD - 1.0f)   // EP_1 = r_v / r_d - 1
#define FER_CLIQ    4190.0f
#define FER_CICE    2106.0f
#define FER_PI      3.141592653589793f
#define FER_XLV     2.5e6f                     // xlv, latent heat liquid/vapor

// ---- MP_FER_HIRES PARAMETERs ----
#define FER_DMI_MIN     0.05e-3f
#define FER_DMI_MAX     1.0e-3f
#define FER_DELDMI      1.0e-6f
#define FER_XMI_MIN     (1.0e6f * FER_DMI_MIN)
#define FER_XMI_MAX     (1.0e6f * FER_DMI_MAX)
#define FER_XMI_EXP     0.0536f
#define FER_DMR_MIN     0.05e-3f
#define FER_DMR_MAX     1.0e-3f
#define FER_DELDMR      1.0e-6f
#define FER_XMR_MIN     (1.0e6f * FER_DMR_MIN)
#define FER_XMR_MAX     (1.0e6f * FER_DMR_MAX)
#define FER_XMIN        180.0f
#define FER_XMAX        330.0f
#define FER_CVAP        1846.0f
#define FER_XLF         3.3358e+5f             // FER's own XLF (not the constants-module xlf)
#define FER_XLS         (FER_XLV + FER_XLF)
#define FER_EPSQ1       (1.001f * FER_EPSQ)
#define FER_RCP         (1.0f / FER_CP)
#define FER_RCPRV       (FER_RCP / FER_RV)
#define FER_RGRAV       (1.0f / FER_G)
#define FER_RRHOL       (1.0f / FER_RHOL)
#define FER_XLV1        (FER_XLV / FER_CP)
#define FER_XLF1        (FER_XLF / FER_CP)
#define FER_XLS1        (FER_XLS / FER_CP)
#define FER_XLV2        (FER_XLV * FER_XLV / FER_RV)
#define FER_XLS2        (FER_XLS * FER_XLS / FER_RV)
#define FER_XLV3        (FER_XLV * FER_XLV * FER_RCPRV)
#define FER_XLS3        (FER_XLS * FER_XLS * FER_RCPRV)
#define FER_CLIMIT      (10.0f * FER_EPSQ)
#define FER_CLIMIT1     (-FER_CLIMIT)
#define FER_C1          (1.0f / 3.0f)
#define FER_DMR1        0.1e-3f
#define FER_DMR2        0.2e-3f
#define FER_DMR3        0.32e-3f
#define FER_DMR4        0.45e-3f
#define FER_DMR5        0.67e-3f
#define FER_XMR1        (1.0e6f * FER_DMR1)
#define FER_XMR2        (1.0e6f * FER_DMR2)
#define FER_XMR3        (1.0e6f * FER_DMR3)
#define FER_XMR4        (1.0e6f * FER_DMR4)
#define FER_XMR5        (1.0e6f * FER_DMR5)
#define FER_RQRMIX      0.05e-3f
#define FER_RQSMIX      1.0e-3f
#define FER_CDRY        1.634e13f
#define FER_CWET        (1.0f / 0.224f)
#define FER_T_ICE       (-40.0f)
#define FER_T_ICEK      (FER_T0C + FER_T_ICE)
#define FER_T_ICE_INIT  (-12.0f)
#define FER_NSI_MAX     250.0e3f
#define FER_NLIMIN      1.0e3f
#define FER_N0R0        8.0e6f
#define FER_N0RMIN      1.0e4f
#define FER_NCW         300.0e6f
// Pwarn and QTwarn are mutable SAVE state in Fortran (line 172, initial values 75.E2 and 1.E-3).
// The WARN4 path (lines 389-393) rewrites them, so they are NOT constants. Only the initial values are
// here. The live values are ctx.tables[fer_warn_state] (float32[2]: [0] = Pwarn, [1] = QTwarn),
// written by fer_warn_state_init and updated by the WARN4 path in phys-fer-fer-hires.
#define FER_PWARN_INIT  75.0e2f
#define FER_QTWARN_INIT 1.0e-3f

// INTEGER PARAMETERs (Fortran truncation of the REAL parents; checked in float32):
#define FER_MDI_MIN     50        // INT(XMImin) = 50
#define FER_MDI_MAX     1000      // INT(XMImax), XMImax = 1000.0000610
#define FER_MDR_MIN     50        // INT(XMRmin)
#define FER_MDR_MAX     1000      // INT(XMRmax)
#define FER_MDR1        100
#define FER_MDR2        200
#define FER_MDR3        320
#define FER_MDR4        450
#define FER_MDR5        670
#define FER_NX          7501
#define FER_INDEXRSTRMAX 500
#define FER_MY_T1       1
#define FER_MY_T2       35
#define FER_NRIME       40
#define FER_MAX_ITERATIONS 10

// ---- Table layout (device arrays passed in from ctx.tables, sized here) ----
// MDImin:MDImax arrays (ACCRI, VSNOWI, VENTI1, VENTI2, SDENS, MASSI): index = m - FER_MDI_MIN.
#define FER_TBL_NMDI        (FER_MDI_MAX - FER_MDI_MIN + 1)   // 951
// MDRmin:MDRmax arrays (ACCRR, MASSR, RRATE, VRAIN, VENTR1, VENTR2): index = m - FER_MDR_MIN.
#define FER_TBL_NMDR        (FER_MDR_MAX - FER_MDR_MIN + 1)   // 951
// MY_GROWTH_NMM(MY_T1:MY_T2): index = t - FER_MY_T1.
#define FER_TBL_NMY         (FER_MY_T2 - FER_MY_T1 + 1)       // 35
// VEL_RF(2:9, 0:Nrime): Fortran VEL_RF(IXS, IXRF) maps to C-order [IXRF, IXS-2] (CONTRACT.md section 1),
// i.e. a (41, 8) array with the Fortran row index fastest. Index = IXRF * 8 + (IXS - 2).
#define FER_TBL_VEL_RF_NR   8                                  // IXS rows 2..9
#define FER_TBL_VEL_RF_NC   (FER_NRIME + 1)                    // IXRF cols 0..40
#define FER_TBL_VEL_RF_INDEX(IXS, IXRF) ((IXRF) * FER_TBL_VEL_RF_NR + ((IXS) - 2))
// TBPVS, TBPVS0(NX): index = 0-based table position.
#define FER_TBL_NX          FER_NX

// REAL PARAMETER constants, fixed order. The publish kernel writes them in this order;
// namgpu/physics/fer_hires/phys_fer_module_data.py checks them against the host float32 values.
#define FER_REAL_CONSTANT_LIST(X)                                                                   \
    X(CP) X(EPSQ) X(G) X(RHOL) X(RD) X(RV) X(T0C) X(EP2) X(EP1) X(CLIQ) X(CICE) X(PI) X(XLV)       \
    X(DMI_MIN) X(DMI_MAX) X(DELDMI) X(XMI_MIN) X(XMI_MAX) X(XMI_EXP)                             \
    X(DMR_MIN) X(DMR_MAX) X(DELDMR) X(XMR_MIN) X(XMR_MAX) X(XMIN) X(XMAX)                       \
    X(CVAP) X(XLF) X(XLS) X(EPSQ1) X(RCP) X(RCPRV) X(RGRAV) X(RRHOL)                             \
    X(XLV1) X(XLF1) X(XLS1) X(XLV2) X(XLS2) X(XLV3) X(XLS3)                                       \
    X(CLIMIT) X(CLIMIT1) X(C1) X(DMR1) X(DMR2) X(DMR3) X(DMR4) X(DMR5)                           \
    X(XMR1) X(XMR2) X(XMR3) X(XMR4) X(XMR5) X(RQRMIX) X(RQSMIX) X(CDRY) X(CWET)                  \
    X(T_ICE) X(T_ICEK) X(T_ICE_INIT) X(NSI_MAX) X(NLIMIN) X(N0R0) X(N0RMIN) X(NCW)

// INTEGER PARAMETERs, fixed order. The publish kernel writes them to an int32 vector in this order.
#define FER_INT_CONSTANT_LIST(X)                                                                    \
    X(MDI_MIN) X(MDI_MAX) X(MDR_MIN) X(MDR_MAX) X(MDR1) X(MDR2) X(MDR3) X(MDR4) X(MDR5)             \
    X(NX) X(INDEXRSTRMAX) X(MY_T1) X(MY_T2) X(NRIME) X(MAX_ITERATIONS)
