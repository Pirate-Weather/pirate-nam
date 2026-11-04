// phys_rad_cal_cldfra.cu -- nam-gpu unit phys-rad-cal-cldfra: CAL_CLDFRA, cloud fraction from
// condensate (QC and QI) as a hard threshold.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Module
// header of nmm/module_RADIATION.F90 (the routine carries no author block of its own):
//   MODULE MODULE_RADIATION  ***  THE RADIATION DRIVERS AND PACKAGES
//   Modifications: 2010-04-02 Vasic - Removed WFR driver
// Ported to CUDA for nam-gpu from nmm/module_RADIATION.F90:1045-1134 (nam.v4.2.11, 83df4e13c1).
//
// Fortran SUBROUTINE CAL_CLDFRA(CLDFRA, QC, QI, F_QC, F_QI, IDS..KDE, IMS..KME, ITS..ITE,
//                              JTS..JTE, KTS..KTE)
//   thresh = 1.0e-6 (REAL, single)
//   f_qi .AND. f_qc : CLDFRA = 1 where QC + QI > thresh, else 0   (lines 1074-1087)
//   f_qc only       : CLDFRA = 1 where QC > thresh, else 0        (lines 1095-1108)
//   otherwise       : CLDFRA = 0                                  (lines 1116-1124)
// The Fortran loops run k = KTS..KTE, j = JTS..JTE, i = ITS..ITE, i.e. the interior. The caller
// (RADIATION, line 470) passes ITS..ITE = IQS..IQE = ITS_B1..ITE_B1 and JTS..JTE = JQS..JQE =
// JTS_B1..JTE_B1, which are max(its, ids+1) .. min(ite, ide-1). On the single card that is
// Fortran i = 2 .. IM-1, j = 2 .. JM-1, zero-based i0 = 1 .. IM-2, j0 = 1 .. JM-2. The boundary
// rows and columns are never written by this routine and are not written here.
// Caller note: in the NAM 12 km configuration longwave is rrtm (parm/nam_configfile), which takes
// the CASE (RRTMLWSCHEME) branch at line 464 and does not call CAL_CLDFRA; this kernel is the
// port for the other longwave options that reach CASE DEFAULT (line 470).
//
// One thread per (i, j, l); the Fortran l loop is grid.z. Bounds are passed zero-based and
// inclusive (i_lo..i_hi, j_lo..j_hi). Halo / MPI: none (single card, full domain).
#include "nmmb_common.cuh"

extern "C" __global__ void phys_rad_cal_cldfra(
    const int im, const int jm, const int lm,
    const int i_lo, const int i_hi, const int j_lo, const int j_hi,
    const int f_qc, const int f_qi,
    float* __restrict__ cldfra,
    const float* __restrict__ qc,
    const float* __restrict__ qi)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    // Fortran ITS..ITE / JTS..JTE: outside the interior the routine writes nothing.
    if (i0 < i_lo || i0 > i_hi || j0 < j_lo || j0 > j_hi) return;

    const size_t p = IDX3(i0, j0, l0, im, jm);
    const float thresh = 1.0e-6f;

    if (f_qi != 0 && f_qc != 0) {
        // lines 1074-1087: QC(i,j,k)+QI(i,j,k) .gt. thresh
        cldfra[p] = (qc[p] + qi[p] > thresh) ? 1.0f : 0.0f;
    } else if (f_qc != 0) {
        // lines 1095-1108: QC(i,j,k) .gt. thresh
        cldfra[p] = (qc[p] > thresh) ? 1.0f : 0.0f;
    } else {
        // lines 1116-1124: no active condensate, cloud fraction 0
        cldfra[p] = 0.0f;
    }
}
