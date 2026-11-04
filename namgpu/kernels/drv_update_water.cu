// drv_update_water.cu -- nam-gpu unit drv-update-water: UPDATE_WATER, the coupling between the
// total condensate CW (CWM in the Fortran) and the species QC, QR, QS, QI, QG with the fractions
// F_ICE, F_RAIN, F_RIMEF.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Author and
// history block of the Fortran subprogram (nmm/module_SOLVER_GRID_COMP.F90:11069-11099), verbatim:
//   SUBPROGRAM:    UPDATE_WATER          UPDATE WATER ARRAY
//     PRGRMMR: FERRIER         ORG: NP22     DATE: 3 AUG 2009
//   ABSTRACT:
//     UPDATE WATER ARRAY FOR FERRIER MICROPHYSICS
//   PROGRAM HISTORY LOG (with changes to called routines) :
//     2009-08     FERRIER     - Synchronize WATER array with CWM, F_rain, F_ice arrays
// Ported to CUDA for nam-gpu from nmm/module_SOLVER_GRID_COMP.F90:11069-11530 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE UPDATE_WATER(CWM, F_ICE, F_RAIN, F_RIMEF, T, QC, QR, QS, QI, QG,
//                                 MICROPHYSICS, SPEC_ADV, NTIMESTEP, IDS..ITE, LM)
// CLD_INIT = (NTIMESTEP <= 1) (lines 11116-11120).
// Every Fortran loop nest is a pointwise sweep (K, J, I over the full memory domain IMS:IME,
// JMS:JME), and every cell reads and writes only its own element, so one thread per (i, j, l) runs
// the whole case for its cell in the Fortran statement order and is exact.
// The Fortran case selector (TRIM(MICROPHYSICS)) is passed in as `mode`:
//   0 = CASE DEFAULT (lines 11463-11470: nothing is written)
//   1 = fer, fer_hires (lines 11161-11226; the configured scheme, parm/nam_configfile microphysics fer_hires)
//   2 = gfs            (lines 11228-11290)
//   3 = wsm6           (lines 11292-11363)
//   4 = thompson       (lines 11365-11525)
// The WRITE(0,*) diagnostics (GFS spec_adv warning, Thompson comments, default-option message) are
// host print statements and are dropped.
// The CONTRACT default is spec_adv = .false. (parm/nam_configfile spec_adv). The spec_adv = .true.
// branches are ported too, as written.
// Halo / MPI: none (single card, full domain; IMS:IME = 1:IM and JMS:JME = 1:JM).
#include "nmmb_common.cuh"

#define UW_MODE_NONE      0
#define UW_MODE_FER       1
#define UW_MODE_GFS       2
#define UW_MODE_WSM6      3
#define UW_MODE_THOMPSON  4

extern "C" __global__ void drv_update_water(
    const int im, const int jm, const int lm,
    const int mode, const int cld_init, const int spec_adv,
    float* __restrict__ cwm,
    float* __restrict__ f_ice,
    float* __restrict__ f_rain,
    float* __restrict__ f_rimef,
    const float* __restrict__ t,
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ qs,
    float* __restrict__ qi,
    float* __restrict__ qg)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t p = IDX3(i0, j0, l0, im, jm);

    const bool init = cld_init != 0;
    const bool spec = spec_adv != 0;

    float& CWM     = cwm[p];
    float& F_ICE   = f_ice[p];
    float& F_RAIN  = f_rain[p];
    float& F_RIMEF = f_rimef[p];
    const float TT = t[p];
    float& QC = qc[p];
    float& QR = qr[p];
    float& QS = qs[p];
    float& QI = qi[p];
    float& QG = qg[p];

    float LIQW, FRACTION, OLDCWM;
    bool deep_ice;

    switch (mode) {

    // ---------------------------------------------------------------- fer / fer_hires (11161-11226)
    case UW_MODE_FER:
        if (!spec || init) {
            // lines 11166-11180: update WATER arrays when advecting only total condensate
            if (CWM > NMMB_EPSQ) {
                LIQW = (1.f - F_ICE) * CWM;
                QC = (1.f - F_RAIN) * LIQW;
                QR = F_RAIN * LIQW;
                QS = F_ICE * CWM;
            } else {
                QC = 0.f;
                QR = 0.f;
                QS = 0.f;
            }
        } else {
            // lines 11191-11211: update CWM, F_ICE, F_RAIN from separate species advection
            CWM = QC + QR + QS;
            if (QS > NMMB_EPSQ) {
                F_ICE = QS / CWM;
            } else {
                F_ICE = 0.0f;
            }
            if (QR > NMMB_EPSQ) {
                F_RAIN = QR / (QC + QR);
            } else {
                F_RAIN = 0.f;
            }
        }
        break;

    // ---------------------------------------------------------------- gfs (11228-11290)
    case UW_MODE_GFS:
        if (!spec || init) {
            if (init) {
                // lines 11236-11248: initialise F_RAIN, F_RIMEF, F_ICE
                F_RAIN = 0.f;
                F_RIMEF = 1.f;
                if (CWM > NMMB_EPSQ && TT < 233.15f) {
                    F_ICE = 1.f;
                } else {
                    F_ICE = 0.f;
                }
            }
            // lines 11256-11270: QC, QI from CWM and F_ICE
            if (CWM > NMMB_EPSQ) {
                QC = (1.f - F_ICE) * CWM;
                QI = F_ICE * CWM;
            } else {
                QC = 0.f;
                QI = 0.f;
            }
        } else {
            // lines 11277-11286: CWM and F_ICE from separate species advection
            CWM = QC + QI;
            if (CWM > NMMB_EPSQ) {
                F_ICE = QI / CWM;
            } else {
                F_ICE = 0.f;
            }
        }
        break;

    // ---------------------------------------------------------------- wsm6 (11292-11363)
    case UW_MODE_WSM6:
        if (init) {
            // lines 11298-11316: only cloud ice is present at the initial time
            QS = 0.0f;
            QG = 0.0f;
            if (CWM > NMMB_EPSQ) {
                LIQW = (1.f - F_ICE) * CWM;
                QC = (1.f - F_RAIN) * LIQW;
                QR = F_RAIN * LIQW;
                QI = F_ICE * CWM;
            } else {
                QC = 0.f;
                QR = 0.f;
                QI = 0.f;
            }
        } else {
            if (!spec) {
                // lines 11324-11349: fraction of each category unchanged by advection (spec_adv=F)
                OLDCWM = QC + QR + QI + QS + QG;
                if (OLDCWM > NMMB_EPSQ) {
                    FRACTION = CWM / OLDCWM;
                    QC = FRACTION * QC;
                    QR = FRACTION * QR;
                    QI = FRACTION * QI;
                    QS = FRACTION * QS;
                    QG = FRACTION * QG;
                } else {
                    QC = 0.0f;
                    QR = 0.0f;
                    QI = 0.0f;
                    QS = 0.0f;
                    QG = 0.0f;
                    if (TT < 233.15f) {
                        QI = CWM;
                    } else {
                        QC = CWM;
                    }
                }
            }
            // lines 11355-11385: the Fortran couples CWM, F_ICE, F_RAIN, F_RIMEF from the species
            // UNCONDITIONALLY in this (not cld_init) branch, for spec_adv true and false alike.
            CWM = QC + QR + QI + QS + QG;
            if (CWM > NMMB_EPSQ) {
                LIQW = QI + QS + QG;
                F_ICE = LIQW / CWM;
            } else {
                F_ICE = 0.f;
            }
            if (QR > NMMB_EPSQ) {
                F_RAIN = QR / (QC + QR);
            } else {
                F_RAIN = 0.f;
            }
            if (QG > NMMB_EPSQ) {
                // assume 5x higher graupel density (500 kg/m**3) vs snow (100 kg/m**3)
                LIQW = 5.f * QG + QS;
                F_RIMEF = LIQW / (QS + QG);
            } else {
                F_RIMEF = 1.f;
            }
        }
        break;

    // ---------------------------------------------------------------- thompson (11365-11525)
    case UW_MODE_THOMPSON:
        if (init) {
            // lines 11392-11410: translate the initial CWM into species (no snow or graupel)
            QS = 0.0f;
            QG = 0.0f;
            if (CWM > NMMB_EPSQ) {
                LIQW = (1.f - F_ICE) * CWM;
                QC = (1.f - F_RAIN) * LIQW;
                QR = F_RAIN * LIQW;
                QI = F_ICE * CWM;
            } else {
                QC = 0.f;
                QR = 0.f;
                QI = 0.f;
            }
        } else if (spec) {
            // lines 11421-11458: species advected separately; sum into CWM, diagnose fractions
            CWM = QC + QR + QI + QS + QG;
            if (CWM > NMMB_EPSQ) {
                LIQW = fmaxf(0.f, CWM - QI - QS - QG);
                F_ICE = fmaxf(0.f, 1.0f - LIQW / CWM);
                if (QR > NMMB_EPSQ) {
                    F_RAIN = QR / (QC + QR);
                } else {
                    F_RAIN = 0.f;
                }
                if (QG > NMMB_EPSQ) {
                    F_RIMEF = (5.f * QG + QS) / (QS + QG);
                } else {
                    F_RIMEF = 1.f;
                }
            } else {
                F_ICE = 0.f;
                F_RAIN = 0.f;
                F_RIMEF = 1.f;
                CWM = 0.f;
            }
        } else {
            // lines 11466-11523: spec_adv=F (strongly discouraged in the Fortran, ported as written).
            // The Fortran K loop runs downward; no cell reads another level's result, and the
            // deep_ice flag is reset per level, so each level is an independent cell here.
            deep_ice = false;
            if (CWM > NMMB_EPSQ) {
                OLDCWM = QC + QR + QI + QS + QG;
                if (OLDCWM > NMMB_EPSQ) {
                    LIQW = fmaxf(0.f, OLDCWM - QI - QS - QG);
                    F_ICE = fmaxf(0.f, 1.0f - LIQW / OLDCWM);
                    if (QR > NMMB_EPSQ) {
                        F_RAIN = QR / (QC + QR);
                    } else {
                        F_RAIN = 0.f;
                    }
                    if (QG > NMMB_EPSQ) {
                        F_RIMEF = (5.f * QG + QS) / (QS + QG);
                    } else {
                        F_RIMEF = 1.f;
                    }
                    LIQW = fmaxf(0.f, (1.f - F_ICE) * CWM);
                    QR = LIQW * F_RAIN * CWM;
                    QC = LIQW * (1.f - F_RAIN) * CWM;
                    if (QG > NMMB_EPSQ) {
                        FRACTION = fmaxf(0.f, fminf(QG / (QG + QS), 1.f));
                    } else {
                        FRACTION = 0.f;
                    }
                    QG = FRACTION * F_ICE * CWM;
                    QI = 0.1f * (1.f - FRACTION) * F_ICE * CWM;
                    QS = 0.9f * (1.f - FRACTION) * F_ICE * CWM;
                } else {
                    // the condensate is all new here
                    QC = 0.0f;
                    QI = 0.0f;
                    QR = 0.0f;
                    QS = 0.0f;
                    QG = 0.0f;
                    if (TT <= 235.15f) {
                        QI = 0.5f * CWM;
                        QS = 0.5f * CWM;
                    } else if (TT <= 258.15f) {
                        QI = 0.1f * CWM;
                        QS = 0.9f * CWM;
                        deep_ice = true;
                    } else if (TT <= 275.15f) {
                        if (deep_ice && TT < 273.15f) {
                            QS = CWM;
                        } else if (deep_ice && TT < 274.15f) {
                            QS = 0.333f * CWM;
                            QR = 0.667f * CWM;
                        } else if (deep_ice) {
                            QS = 0.1f * CWM;
                            QR = 0.9f * CWM;
                        } else {
                            QC = CWM;
                        }
                    } else {
                        QC = CWM;
                    }
                    LIQW = fmaxf(0.f, CWM - QI - QS - QG);
                    if (CWM > NMMB_EPSQ) {
                        F_ICE = (1.0f - LIQW) / CWM;
                    } else {
                        F_ICE = 0.f;
                    }
                    if (QR > NMMB_EPSQ) {
                        F_RAIN = QR / (QC + QR);
                    } else {
                        F_RAIN = 0.f;
                    }
                    if (QG > NMMB_EPSQ) {
                        F_RIMEF = (5.f * QG + QS) / (QS + QG);
                    } else {
                        F_RIMEF = 1.f;
                    }
                }
            } else {
                QC = 0.0f;
                QR = 0.0f;
                QI = 0.0f;
                QS = 0.0f;
                QG = 0.0f;
                F_ICE = 0.0f;
                F_RAIN = 0.0f;
                F_RIMEF = 1.0f;
            }
        }
        break;

    // ---------------------------------------------------------------- CASE DEFAULT (11463-11470)
    default:
        break;
    }
}
