// phys_rad_driver.cu -- RADIATION, the outer driver of the NMMB radiation package, for the
// configuration the 12 km NAM parent runs (shortwave rrtm, longwave rrtm, cldfraction default).
//
// Ported to CUDA for nam-gpu from nmm/module_RADIATION.F90:66-995 (NAM nam.v4.2.11, 83df4e13c1).
// NOAA notice (documentation block of RADIATION, module_RADIATION.F90 SUBPROGRAM header):
//   SUBPROGRAM:    RADIATION   RADIATION OUTER DRIVER
//   PRGRMMR: BLACK           ORG: W/NP22     DATE: 2002-06-04
//   ABSTRACT: RADIATION SERVES AS THE INTERFACE BETWEEN THE NMMB PHYSICS COMPONENT
//             AND THE WRF RADIATION DRIVER.
//
// What the Fortran does on every call, and where it lives in nam-gpu:
//   (a) radupdate_nmmb (line 395, every call while either scheme is rrtm): solar declination
//       and constant, a scalar call. Host wrapper (namgpu/physics/driver/phys_rad_driver.py) calls
//       the phys-rrtm-radupdate unit; no kernel here.
//   (b) Zeroing, lines 308-327 (every call, before the radiation-step test): CLDFRA(:,:,1..LM),
//       CFRACH, CFRACL, CFRACM, CZMEAN, SIGT4 over the whole IMS..IME x JMS..JME domain.
//       This is rad_zero_outputs below. It runs every step, as the Fortran does.
//   (c) Radiation step, line 414 (ITIMESTEP == 1 or MOD(ITIMESTEP, NRAD) == 0, NRAD = NRADS,
//       line 247): cloud-input selection (default branch, lines 545-562, see the wrapper: its
//       arrays are read-only aliases of the state, so no copy kernel), then the RRTM call over
//       the rows JDS+1..JDE-1 and all columns (lines 586-625, phys-rrtm-driver-call unit).
//   (d) RETURN at lines 856-860 when SHORTWAVE == rrtm. Everything below it is dead in this
//       configuration (see the list at the end).
//
// Fortran-to-CUDA review (CONTRACT section 6, part 3):
//   CLDFRA(I,J,K)=0 for K=1..LM, I/J over IMS..IME x JMS..JME (line 308)  -> rad_zero_outputs
//   CFRACH/CFRACL/CFRACM/CZMEAN/SIGT4 = 0 over the same domain (lines 319-327) -> rad_zero_outputs
//   Live but in the wrapper, not this kernel: radupdate call (391-397), scheme selection (359-380) and the
//     "unknown scheme" refusal (CASE DEFAULT branches), the cloud
//     default branch (lines 545-562: QTdum=CW, FIdum=F_ICE, FRdum=F_RAIN, QCW=QC, QCI/QSNOW/NCI
//     from QI/QS/NI when their flag is set), the RRTM call (601-625, arguments listed in the
//     wrapper), and the flag table that SOLVER_INTERNAL_STATE sets per microphysics
//     (module_SOLVER_INTERNAL_STATE.F90:868-940).
//   Dead in this configuration, not ported (each one is read only by a branch that never runs):
//     - PHINT, PI3D, PRL, RHO, TSFC, XLAND, QL and THRATEN/THRATENLW/THRATENSW (lines 264-306):
//       read only by GFDL (TSK2D, PI3D, PHINT), by cal_cldfra3 (thompson cloud fraction, not
//       selected: cldfraction = default), and by the post-RRTM block below.
//     - SWDOWN, SWDOWNC, GSW, GLW, CUPPTR zeroing (lines 310-330) and the interior THRATEN and
//       GSW/GLW/SWDOWN reset (lines 441-450): read only by GFDL and the post-RRTM block.
//     - CAL_CLDFRA generic (lines 468-480): longwave is rrtm, so the RRTM branch is taken.
//     - GFDL longwave and shortwave (lines 638-666 and 695-723): shortwave and longwave are both rrtm.
//     - Thompson cloud fraction (lines 493-544, cal_cldfra3 of module_mp_thompson): not
//       selected (cldfraction default).
//     - Post-RRTM block (lines 872-995): the RETURN at 856 is taken for rrtm shortwave, so
//       CZMEAN (ZENITH), RSWIN, RSWINC, RSWOUT, RSWTT, SIGT4, RLWIN, RLWTT are written by RRTM
//       itself, not here.
//   Halo and OpenMP (ENABLE_SMP, chunk_loop_rrtm tiling) are dropped: one card, full domain.
//
// Precision: float (kfpt = single). This kernel only writes zeros; no arithmetic.

#include "nmmb_common.cuh"

extern "C" __global__ void rad_zero_outputs(
    const int IM, const int JM, const int LM,
    float* __restrict__ cldfra,            // [LM, JM, IM]
    float* __restrict__ cfrach,            // [JM, IM]
    float* __restrict__ cfracl,
    float* __restrict__ cfracm,
    float* __restrict__ czmean,
    float* __restrict__ sigt4)
{
    COLUMN_IJ(IM, JM)
    for (int l0 = 0; l0 < LM; ++l0) {
        cldfra[IDX3(i0, j0, l0, IM, JM)] = 0.0f;
    }
    cfrach[IDX2(i0, j0, IM)] = 0.0f;
    cfracl[IDX2(i0, j0, IM)] = 0.0f;
    cfracm[IDX2(i0, j0, IM)] = 0.0f;
    czmean[IDX2(i0, j0, IM)] = 0.0f;
    sigt4[IDX2(i0, j0, IM)] = 0.0f;
}
