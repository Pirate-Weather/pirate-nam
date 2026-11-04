// phys_fer_col_processes.cu -- WOOF-NAM unit phys-fer-col-processes (fer_hires column processes).
//
// NOTICE: NOAA/NCEP NAM source phys/module_MP_FER_HIRES.F90 (US government work; no licence file in
// the NAM clone). Header of SUBROUTINE EGCP01COLUMN_hr (lines 650-690), kept as authorship:
//   SUBPROGRAM:  Grid-scale microphysical processes - condensation & precipitation
//   PRGRMMR: Ferrier         ORG: W/NP22     DATE: 08-2001
//   PRGRMMR: Jin  (Modification for WRF structure)
// Ported to CUDA for WOOF-NAM from phys/module_MP_FER_HIRES.F90:1321-1478 (nam.v4.2.11, 83df4e13c1),
// the block "Calculate individual processes" through the opening of the melting/evaporation IF.
// CONDENSE (2160-2225) and DEPOSIT (2231-2289) are called from this range. Their device functions are
// nam_condense (phys_fer_condense.cu, unit phys-fer-condense) and nam_deposit (phys_fer_deposit.cu, unit
// phys-fer-deposit), included below. This file keeps no second copy of either.
//
// Thread map: one thread per (i, j, l) cell (POINT_IJL, grid_2d). The range has no vertical
// recurrence (each term reads only its own level), so the Fortran l-loop is the grid.z dimension.
//
// Inputs (per cell, float [LM, JM, IM] unless noted), set by earlier units of the same level:
//   TK        T_col(L)                                  (line 914)
//   PP        P_col(L)                                  (line 916)
//   WV        Q/(1-Q)                                   (line 918)
//   QW        cloud water mixing ratio                  (line 979)
//   RHO, RRHO air density and reciprocal                (lines 967-968)
//   TK2       1/TK**2                                   (line 1037)
//   QSI       saturation mixing ratio w/r/t ice         (lines 930-933)
//   QSWgrd    RHgrd*QSW, QSIgrd = RHgrd*QSI             (lines 939-940)
//   DIFFUS, DYNVIS, THERM_COND                          (lines 1045-1047)
//   GAMMAS    (line 1052)
//   VEL_INC, NLICE, NSmICE, NInuclei, QLICE, QTICE, PILOSS    (ice-props unit, lines 1029-1320)
//   INDEXS    int32, snow mean-diameter index MDImin..MDImax  (ice-props unit)
//   RCW, DWVI in/out: value on entry. RCW may be set by ice-props; DWVI keeps its previous value
//             when neither ice branch assigns it, as the Fortran local does.
// Tables (device, ctx.tables["fer"]): ACCRI, VENTI1, VENTI2 [951], index M - 50 (MDImin = 50);
//   MY_GROWTH_NMM [35], index T - 1; TBPVS, TBPVS0 [7501] with C1XPVS, C2XPVS, C1XPVS0, C2XPVS0.
// Scalars: DTPH (GSMDT), RHgrd (configfile rhgrd 0.98), CIACW (line 2602), ARAUT, BRAUT, QAUT0, ARcw
// (FERRIER_INIT_hr lines 2658-2673; phys_fer_init scalars).
//
// Outputs (per cell): PRAUT, PIACW, PIACWI, PCOND, PIDEP, PINIT, TCC, QSW0, DWV0, plus in/out RCW, DWVI.
// The Fortran zeroes PCOND, PIDEP, PINIT, PIACW, PIACWI, PRAUT at lines 982-992 before this range and
// nothing in lines 993-1320 assigns them, so the kernel starts those from zero.
//
// Fortran-to-CUDA statement map (every assignment of 1321-1478 has a line here):
//   1327-1335  IF QW>EPSQ and TC>=T_ICE; DUM2=RHO*QW; IF DUM2>QAUT0: DUM2=DUM2**2, DUM, DUM1,
//              PRAUT=MIN(QW, DUM*(1-EXP(-DUM1**2)))                        -> block A
//   1338-1346  IF QLICE>EPSQ: FWS, PIACW=FWS*QW; IF TC<0: PIACWI=PIACW, Rcw=ARcw*DUM2**C1
//              (DUM2 is RHO*QW or its square, as left by 1332)              -> block A
//   1355-1374  ice_only IF (TC<T_ICE and (WV>QSWgrd or QW>EPSQ)): PIACW=QW, PIACWI, Rcw=0,
//              DUM1, DUM2=WV, DUM = limited FPVS, IF DUM2>DUM: PIDEP=DEPOSIT(...), DWVi=0 -> block B
//   1382-1450  ELSE IF TC<0: DENOMI, DWVi=WV-QSIgrd, PIDEP_max, DIDEP (QTICE>0: ABI, DUM, SFACTOR,
//              VENTIS, VENTIL, DIDEP), PINIT (WV>QSWgrd, TC<T_ICE_init, NSmICE<NInuclei: INDEX_MY,
//              DUM, PINIT), PIDEP by the sign of DWVi                        -> block B
//   1456-1461  IF TC>=T_ICE and (QW>EPSQ or WV>QSWgrd): DUM1, DUM2, DUM3, PCOND=CONDENSE(...)  -> block C
//   1466-1472  TCC=TC+XLV1*PCOND+XLS1*PIDEP+XLF1*PIACWI; IF TCC>0: PIACWI=0, Rcw=0, PIDEP=0,
//              TCC=TC+XLV1*PCOND                                             -> block D
//   1475-1477  QSW0=0, DWV0=0
//   1478       IF (TC>0 and TCC>0 and ICE_logical): opens the melting block, not in this unit.
//
// Not carried, on purpose: the WARN1/WARN2/WARN3 WRITE(0,*) warnings (host I/O, once per run, no state
// effect) and the debug WRITEs at 2044-2059 (no state effect). DWVi is an output because line 2044
// reads it after this range. I_index, J_index and L exist only for those WRITEs and are not passed.
//
// Constants (fer_hires.cuh, nmmb_common.cuh): FER_EPSQ, FER_T0C, FER_EP2 (EPS), FER_T_ICE,
// FER_T_ICE_INIT, FER_XLV1, FER_XLS1, FER_XLF1, FER_XLS2, FER_XLS3, FER_XLV3, FER_C1, FER_MDI_MIN,
// FER_MY_T1, FER_MY_T2, FER_MAX_ITERATIONS. The local PARAMETER C2 = 1./6. of EGCP01COLUMN_hr (line 789)
// is FER_C2 below. The REAL(KIND=HIGH_PRES) variables of CONDENSE and DEPOSIT are double here; the
// parameter .001 of those routines is the single literal promoted to double.
#include "nmmb_common.cuh"
#include "fer_hires.cuh"
#include "phys_fer_condense.cu"   // nam_condense (pulls in phys_fer_fpvs.cu once, via #pragma once)
#include "phys_fer_deposit.cu"   // nam_deposit

#define FER_C2 (1.0f / 6.0f)           // EGCP01COLUMN_hr PARAMETER C2 = 1./6. (line 789)

// ---------------------------------------------------------------------------
// Kernel: one thread per (i, j, l). Launch with namgpu.kernels.grid_2d(IM, JM, LM).
// ---------------------------------------------------------------------------
extern "C" __global__ void phys_fer_col_processes(
    const int IM, const int JM, const int LM,
    const float DTPH, const float RHgrd, const float CIACW, const float ARAUT,
    const float BRAUT, const float QAUT0, const float ARcw,
    const float C1XPVS, const float C2XPVS, const float C1XPVS0, const float C2XPVS0,
    const float* __restrict__ TK,
    const float* __restrict__ PP,
    const float* __restrict__ WV,
    const float* __restrict__ QW,
    const float* __restrict__ RHO,
    const float* __restrict__ RRHO,
    const float* __restrict__ TK2,
    const float* __restrict__ QSI,
    const float* __restrict__ QSWgrd,
    const float* __restrict__ QSIgrd,
    const float* __restrict__ DIFFUS,
    const float* __restrict__ DYNVIS,
    const float* __restrict__ THERM_COND,
    const float* __restrict__ GAMMAS,
    const float* __restrict__ VEL_INC,
    const float* __restrict__ NLICE,
    const float* __restrict__ NSmICE,
    const float* __restrict__ NInuclei,
    const float* __restrict__ QLICE,
    const float* __restrict__ QTICE,
    const float* __restrict__ PILOSS,
    const int* __restrict__ INDEXS,
    const float* __restrict__ ACCRI,
    const float* __restrict__ VENTI1,
    const float* __restrict__ VENTI2,
    const float* __restrict__ MY_GROWTH_NMM,
    const float* __restrict__ TBPVS,
    const float* __restrict__ TBPVS0,
    float* __restrict__ RCW,            // in/out
    float* __restrict__ DWVI,           // in/out
    float* __restrict__ PRAUT,
    float* __restrict__ PIACW,
    float* __restrict__ PIACWI,
    float* __restrict__ PCOND,
    float* __restrict__ PIDEP,
    float* __restrict__ PINIT,
    float* __restrict__ TCC,
    float* __restrict__ QSW0,
    float* __restrict__ DWV0,
    int* __restrict__ FLAGS,            // [8] 0: DEPOSIT non-converged, 1: INDEXS out of range, 2: CONDENSE non-converged
    int* __restrict__ WARN2)            // [8] first DEPOSIT non-convergence record (phys_fer_deposit layout)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const size_t n = IDX3(i0, j0, l0, IM, JM);
    const float rhgrd = RHgrd;

    const float tk = TK[n];
    const float pp = PP[n];
    const float wv = WV[n];
    const float qw = QW[n];
    const float rho = RHO[n];
    const float rrho = RRHO[n];
    const float tk2 = TK2[n];
    const float qsi = QSI[n];
    const float qswgrd = QSWgrd[n];
    const float qsigrd = QSIgrd[n];
    const float diffus = DIFFUS[n];
    const float dynvis = DYNVIS[n];
    const float thermcond = THERM_COND[n];
    const float gammas = GAMMAS[n];
    const float vel_inc = VEL_INC[n];
    const float nlice = NLICE[n];
    const float nsmice = NSmICE[n];
    const float ninuclei = NInuclei[n];
    const float qlice = QLICE[n];
    const float qtice = QTICE[n];
    const float pilos = PILOSS[n];
    int idx = INDEXS[n] - FER_MDI_MIN;            // table index for MDImin:MDImax
    if (idx < 0 || idx >= FER_TBL_NMDI) {
        // Fortran line 912 clamps INDEXS to MDImin:MDImax, so an out-of-range value is an upstream fault.
        // It is counted in FLAGS[1] (the done-rule requires zero) and clamped, so no table read goes out of bounds.
        atomicAdd(&FLAGS[1], 1);
        idx = min(max(idx, 0), FER_TBL_NMDI - 1);
    }
    const float tc = tk - FER_T0C;                // line 915

    float rcw = RCW[n];
    float dwvi = DWVI[n];
    // Zeroed at lines 982-992 before this range.
    float pcond = 0.0f;
    float pidep = 0.0f;
    float pinit = 0.0f;
    float piacw = 0.0f;
    float piacwi = 0.0f;
    float praut = 0.0f;

    // ---- Block A: autoconversion (PRAUT) and collection of cloud water by ice (PIACW, PIACWI) ----
    if (qw > FER_EPSQ && tc >= FER_T_ICE) {
        float dum2 = rho * qw;
        if (dum2 > QAUT0) {
            dum2 = dum2 * dum2;
            const float dum = BRAUT * dum2 * qw;
            const float dum1 = ARAUT * dum2;
            praut = fminf(qw, dum * (1.0f - expf(-dum1 * dum1)));
        }
        if (qlice > FER_EPSQ) {
            const float fws = fminf(1.0f, CIACW * vel_inc * nlice * ACCRI[idx]);
            piacw = fws * qw;
            if (tc < 0.0f) {
                piacwi = piacw;
                rcw = ARcw * powf(dum2, FER_C1);
            }
        }
    }

    // ---- Block B: homogeneous freezing and ice deposition/sublimation ----
    if (tc < FER_T_ICE && (wv > qswgrd || qw > FER_EPSQ)) {
        piacw = qw;
        piacwi = piacw;
        rcw = 0.0f;
        float dum1 = tk + FER_XLF1 * piacw;
        float dum2 = wv;
        float dum = fminf(1000.0f * nam_fpvs(dum1, TBPVS, C1XPVS, C2XPVS), 0.99f * pp);
        dum = rhgrd * FER_EP2 * dum / (pp - dum);
        if (dum2 > dum) {
            pidep = nam_deposit(pp, dum1, dum2, rhgrd, TBPVS, C1XPVS, C2XPVS,
                                &FLAGS[0], WARN2, i0 + 1, j0 + 1, l0 + 1);
        }
        dwvi = 0.0f;
    } else if (tc < 0.0f) {
        const float denomi = 1.0f + FER_XLS3 * qsi * tk2;
        dwvi = wv - qsigrd;
        const float pidep_max = fmaxf(pilos, dwvi / denomi);
        float didep = 0.0f;
        if (qtice > 0.0f) {
            const float abi = 1.0f / (rho * FER_XLS2 * qsi * tk2 / thermcond + 1.0f / diffus);
            float dum = powf(rho / (diffus * diffus * dynvis), FER_C2);
            float sfactor = sqrtf(gammas) * dum;
            const float ventis = (VENTI1[0] + sfactor * VENTI2[0]) * nsmice;   // VENTI1(MDImin)
            sfactor = sqrtf(vel_inc) * dum;
            const float ventil = (VENTI1[idx] + sfactor * VENTI2[idx]) * nlice;
            didep = abi * (ventil + ventis) * DTPH;
        }
        if (wv > qswgrd && tc < FER_T_ICE_INIT && nsmice < ninuclei) {
            const int index_my = max(FER_MY_T1, min((int)(0.5f - tc), FER_MY_T2));
            const float dum = fmaxf(0.0f, ninuclei - nsmice);
            pinit = fmaxf(0.0f, dum * MY_GROWTH_NMM[index_my - FER_MY_T1] * rrho);
        }
        if (dwvi > 0.0f) {
            pidep = fminf(dwvi * didep + pinit, pidep_max);
        } else if (dwvi < 0.0f) {
            pidep = fmaxf(dwvi * didep, pidep_max);
        }
    }

    // ---- Block C: cloud water condensation ----
    if (tc >= FER_T_ICE && (qw > FER_EPSQ || wv > qswgrd)) {
        const float dum1 = qw - piacwi;
        const float dum2 = tk + FER_XLS1 * pidep + FER_XLF1 * piacwi;
        const float dum3 = wv - pidep;
        bool cond_ok = true;
        pcond = nam_condense(pp, dum1, dum2, dum3, rhgrd, TBPVS0, C1XPVS0, C2XPVS0, &cond_ok);
        if (!cond_ok) atomicAdd(&FLAGS[2], 1);
    }

    // ---- Block D: limit freezing of accreted rime (Schumann-Ludlam) ----
    float tcc_l = tc + FER_XLV1 * pcond + FER_XLS1 * pidep + FER_XLF1 * piacwi;
    if (tcc_l > 0.0f) {
        piacwi = 0.0f;
        rcw = 0.0f;
        pidep = 0.0f;
        tcc_l = tc + FER_XLV1 * pcond;
    }

    // Lines 1475-1477
    const float qsw0 = 0.0f;
    const float dwv0 = 0.0f;

    PRAUT[n] = praut;
    PIACW[n] = piacw;
    PIACWI[n] = piacwi;
    PCOND[n] = pcond;
    PIDEP[n] = pidep;
    PINIT[n] = pinit;
    RCW[n] = rcw;
    DWVI[n] = dwvi;
    TCC[n] = tcc_l;
    QSW0[n] = qsw0;
    DWV0[n] = dwv0;
    (void)qsigrd;
}
