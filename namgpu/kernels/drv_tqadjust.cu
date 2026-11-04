// WOOF-NAM unit drv-tqadjust: TQADJUST, consistency of T, Q, CW and the condensate fractions after
// the physics. Ferrier-Aligo supersaturation removal, Janjic-style lapse-rate smoothing (theta
// mixing of unstable layers above the PBL), then the second supersaturation pass and the update
// of Q, CWM (or QC with SPEC_ADV), F_ICE and F_RAIN.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Source
// author header: FERRIER, JANJIC, NP22, 5 APR 2016 (nmm/module_SOLVER_GRID_COMP.F90).
// Ported to CUDA for WOOF-NAM from nmm/module_SOLVER_GRID_COMP.F90:11777-12113 (nam.v4.2.11, 83df4e13c1).
//
// Mapping: one thread per (i, j) column (kernels.grid_cols). The column arrays of the Fortran
// (Tcol, Pcol, QVcol, QCcol, EXNcol, THcol, DPcol, DTHcol, Fcol) are per-thread locals. They are
// 1-based (index K = Fortran K) so every index in the translation matches the Fortran text.
// Fortran DO-loop variables that are read after EXIT (ITER in the condensation loops, KTOP, KBOT,
// KMIX) keep their Fortran values here: a loop that runs to completion leaves ITER = 11.
//
// Constants: nmm/module_CONSTANTS.F90 (single precision, kfpt): A2, A4, CP, EPSQ, R, R_V, CLIQ,
// PSAT, TIW, XLV. Derived parameters follow the Fortran expression order exactly.
// TTP = TIW + 0.01; XA and XB appear only in commented-out code and are not ported.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no allocation, no printf.
// LM must be <= TQ_LMAX (the wrapper checks it; NAM 12 km runs LM = 60).

#include "nmmb_common.cuh"

#define TQ_LMAX 64

// Module constants (nmm/module_CONSTANTS.F90:14-56).
#define TQ_A2      17.2693882f
#define TQ_A4      35.86f
#define TQ_CP      1004.6f
#define TQ_EPSQ    1.0e-12f
#define TQ_R       287.04f
#define TQ_R_V     461.6f
#define TQ_CLIQ    4190.0f
#define TQ_PSAT    610.78f
#define TQ_TIW     273.15f
#define TQ_XLV     2.5e6f
#define TQ_CAPPA   (TQ_R / TQ_CP)
#define TQ_EP_2    (TQ_R / TQ_R_V)
#define TQ_CPV     (4.0f * TQ_R_V)
// TQADJUST parameters (module_SOLVER_GRID_COMP.F90:11840-11850).
#define TQ_SUPSAT    0.001f
#define TQ_SUBSAT    (-TQ_SUPSAT)
#define TQ_DTHTHRESH (-0.01f)
#define TQ_TTP       (TQ_TIW + 0.01f)
#define TQ_XLV1      (TQ_XLV / TQ_CP)
#define TQ_XLV2      (TQ_XLV1 * TQ_XLV / TQ_R_V)

// 3-D field element at Fortran level K (1-based) of the column (i0, j0).
#define TQ3(A, K)  A[IDX3(i0, j0, (K) - 1, IM, JM)]
// 1-D vector element at Fortran level K.
#define TQ1(A, K)  A[(K) - 1]

// Saturation mixing ratio over water at (TK, PP), with the Fortran expression order:
//   ESW = PSAT*EXP(A2*(TK-TTP)/(TK-A4)); ESW = MIN(ESW, 0.99*PP); QSW = RHgrd*EP_2*ESW/(PP-ESW).
__device__ inline float tq_esw(float TK, float PP)
{
    return fminf(TQ_PSAT * expf(TQ_A2 * (TK - TQ_TTP) / (TK - TQ_A4)), 0.99f * PP);
}

// One SSrem block of TQADJUST (the two passes are identical).
// Returns 0 when the block is not entered, 1 when it is entered and the ITER<=10 update applies
// (outputs written), 2 when it is entered and the loop ran out (ITER = 11: no update, the flag
// SSFilt is still set by the Fortran).
__device__ inline int tq_ss_block(float TK0, float QV0, float QCW0, float PP, float RHgrd,
                                  float* TKo, float* QVo, float* QCWo)
{
    float TK = TK0, QV = QV0, QCW = QCW0;
    float ESW = tq_esw(TK, PP);
    float QSW = RHgrd * TQ_EP_2 * ESW / (PP - ESW);
    float DQsat = QV - QSW;
    float SSat = DQsat / QSW;
    if (!(SSat > TQ_SUPSAT || (QCW > TQ_EPSQ && SSat < TQ_SUBSAT))) return 0;
    int iter;
    for (iter = 1; iter <= 10; ++iter) {
        float COND = DQsat / (1.0f + TQ_XLV2 * QSW / (TK * TK)); // Asai (1965, J. Japan)
        COND = fmaxf(COND, -QCW);
        TK = TK + TQ_XLV1 * COND;
        QV = QV - COND;
        QCW = QCW + COND;
        ESW = tq_esw(TK, PP);
        QSW = RHgrd * TQ_EP_2 * ESW / (PP - ESW);
        DQsat = QV - QSW;
        SSat = DQsat / QSW;
        if (SSat >= TQ_SUBSAT && SSat <= TQ_SUPSAT) break;
        if (SSat < TQ_SUBSAT && QCW <= TQ_EPSQ) break;
    }
    if (iter <= 10) {
        *TKo = TK; *QVo = QV; *QCWo = QCW;
        return 1;
    }
    return 2;
}

extern "C" __global__ void drv_tqadjust(
    float* __restrict__ T, float* __restrict__ Q, float* __restrict__ QC,
    float* __restrict__ CW, float* __restrict__ F_ICE, float* __restrict__ F_RAIN,
    const float* __restrict__ PD, const float* __restrict__ DSG2,
    const float* __restrict__ PDSG1, const float* __restrict__ PSGML1,
    const float* __restrict__ SGML2,
    const int SPEC_ADV, const float RHgrd,
    const int IM, const int JM, const int LM)
{
    COLUMN_IJ(IM, JM)
    // LM > TQ_LMAX is rejected by the wrapper before launch.

    float Tcol[TQ_LMAX + 2], Pcol[TQ_LMAX + 2], QVcol[TQ_LMAX + 2], QCcol[TQ_LMAX + 2];
    float EXNcol[TQ_LMAX + 2], THcol[TQ_LMAX + 2], DPcol[TQ_LMAX + 2], DTHcol[TQ_LMAX + 2];
    float Fcol[TQ_LMAX + 2];

    const int LM2 = LM - 2;
    const int ITmax = LM / 5;
    int LRFilt = 0;
    int SSFilt = 0;
    const float PDij = PD[IDX2(i0, j0, IM)];

    // Column input (SPEC_ADV selects QC or the CW-based cloud water).
    for (int K = 1; K <= LM; ++K) {
        if (SPEC_ADV) {
            QCcol[K] = TQ3(QC, K);
        } else {
            QCcol[K] = TQ3(CW, K) * (1.0f - TQ3(F_ICE, K)) * (1.0f - TQ3(F_RAIN, K));
        }
        Tcol[K] = TQ3(T, K);
        Pcol[K] = TQ1(SGML2, K) * PDij + TQ1(PSGML1, K);
        QVcol[K] = TQ3(Q, K) / (1.0f - TQ3(Q, K));          // water vapor mixing ratio
        EXNcol[K] = powf(1.0e5f / Pcol[K], TQ_CAPPA);
    }

    // ---- Ferrier-Aligo condensation/evaporation, 1st of 2 times (SSadj1) ----
    for (int K = 1; K <= LM; ++K) {
        float tk, qv, qcw;
        const int r = tq_ss_block(Tcol[K], QVcol[K], QCcol[K], Pcol[K], RHgrd, &tk, &qv, &qcw);
        if (r != 0) SSFilt = 1;
        if (r == 1) { Tcol[K] = tk; QVcol[K] = qv; QCcol[K] = qcw; }
    }

    for (int K = 1; K <= LM; ++K) THcol[K] = Tcol[K] * EXNcol[K];

    DTHcol[1] = 1.0f;
    for (int K = 2; K <= LM; ++K) DTHcol[K] = THcol[K - 1] - THcol[K];

    int KMIX = 0;
    for (int K = LM2; K >= 2; --K) {
        if (DTHcol[K] > 0.0f) { KMIX = K; break; }   // top of the well-mixed layer above the surface
    }

    // ---- Lapse-rate adjustment (LRadjust) ----
    if (KMIX > 2) {
        int KTOP = 0;
        for (int K = 3; K <= KMIX; ++K) {
            if (DTHcol[K] < TQ_DTHTHRESH) { KTOP = K - 1; break; }   // top of highest unstable layer
        }

        if (KTOP > 0) {                                              // Maybe_mix
            int KBOT = 0;
            for (int K = KMIX; K >= 2; --K) {
                if (DTHcol[K] < TQ_DTHTHRESH) { KBOT = K; break; }   // lowest unstable layer
            }
            int ITRmax;
            if (KBOT > 0) {
                LRFilt = 1;
                ITRmax = ITmax;
            } else {
                ITRmax = 0;                                          // do not mix
            }

            for (int K = 1; K <= LM; ++K) {
                DPcol[K] = TQ1(DSG2, K) * PDij + TQ1(PDSG1, K);     // hydrostatic thickness
                Fcol[K] = THcol[K];                                  // modified theta
            }

            for (int ITER = 1; ITER <= ITRmax; ++ITER) {             // Mix_lyrs
                for (int K = KTOP; K <= KBOT; ++K) {
                    if (DTHcol[K] < TQ_DTHTHRESH) {
                        if (DTHcol[K + 1] < TQ_DTHTHRESH) {
                            // mix 3 layers if the current layer and the layers above and below are unstable
                            Fcol[K] = (THcol[K - 1] * DPcol[K - 1] + THcol[K] * DPcol[K]
                                       + THcol[K + 1] * DPcol[K + 1])
                                      / (DPcol[K - 1] + DPcol[K] + DPcol[K + 1]);
                        } else {
                            // mix with the higher layer if the current layer is unstable
                            Fcol[K] = (THcol[K - 1] * DPcol[K - 1] + THcol[K] * DPcol[K])
                                      / (DPcol[K - 1] + DPcol[K]);
                        }
                    } else if (DTHcol[K + 1] < TQ_DTHTHRESH) {
                        // mix with the lower layer if it is unstable
                        Fcol[K] = (THcol[K] * DPcol[K] + THcol[K + 1] * DPcol[K + 1])
                                  / (DPcol[K] + DPcol[K + 1]);
                    }
                    // do nothing if the current layer or the layer below is not unstable
                }

                for (int K = KTOP; K <= KBOT; ++K) THcol[K] = Fcol[K];
                for (int K = KTOP; K <= KBOT; ++K) DTHcol[K] = THcol[K - 1] - THcol[K];

                KTOP = 0;
                for (int K = 3; K <= KMIX; ++K) {
                    if (DTHcol[K] < TQ_DTHTHRESH) { KTOP = K - 1; break; }
                }
                if (KTOP <= 0) break;                                // no unstable layer left

                KBOT = 0;
                for (int K = KMIX; K >= 2; --K) {
                    if (DTHcol[K] < TQ_DTHTHRESH) { KBOT = K; break; }
                }
            }                                                        // Mix_lyrs

            for (int K = 1; K <= LM; ++K) Tcol[K] = THcol[K] / EXNcol[K];   // update T
        }                                                            // Maybe_mix
    }                                                                // LRadjust

    // ---- Ferrier-Aligo condensation/evaporation, 2nd of 2 times (SSadj2) ----
    for (int K = 1; K <= LM; ++K) {
        float tk, qv, qcw;
        const int r = tq_ss_block(Tcol[K], QVcol[K], QCcol[K], Pcol[K], RHgrd, &tk, &qv, &qcw);
        if (r != 0) SSFilt = 1;
        if (r == 1) { Tcol[K] = tk; QVcol[K] = qv; QCcol[K] = qcw; }
    }

    // ---- Update 3-D arrays ----
    if (LRFilt || SSFilt) {                                          // adjust1
        for (int K = 1; K <= LM; ++K) TQ3(T, K) = Tcol[K];
    }
    if (SSFilt) {                                                    // adjust2
        for (int K = 1; K <= LM; ++K) {
            TQ3(Q, K) = QVcol[K] / (1.0f + QVcol[K]);
        }
        if (SPEC_ADV) {
            for (int K = 1; K <= LM; ++K) TQ3(QC, K) = QCcol[K];
        } else {
            for (int K = 1; K <= LM; ++K) {
                const float Qrain = TQ3(CW, K) * (1.0f - TQ3(F_ICE, K)) * TQ3(F_RAIN, K);
                const float Qliq = QCcol[K] + Qrain;
                const float Qice = TQ3(CW, K) * TQ3(F_ICE, K);
                TQ3(CW, K) = Qliq + Qice;
                if (TQ3(CW, K) > TQ_EPSQ) TQ3(F_ICE, K) = Qice / TQ3(CW, K);
                if (Qliq > TQ_EPSQ) TQ3(F_RAIN, K) = Qrain / Qliq;
            }
        }
    }
}
