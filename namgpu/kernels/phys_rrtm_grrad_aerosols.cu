// phys_rrtm_grrad_aerosols.cu -- call site of SETAER inside GRRAD_NMMB (RRTM group).
//
// NOAA notice: phys/grrad_nmmb.f (nam.v4.2.11, 83df4e13c1), header block kept as the NOAA
// notice (US government work, no licence file; see NOTICE):
// !!!!!  ==========================================================  !!!!!
// !!!!!             'module_radiation_driver' descriptions           !!!!!
// !!!!!  ==========================================================  !!!!!
// !   this is the radiation driver module.  it prepares atmospheric      !
// !   profiles and invokes main radiation calculations.                  !
//
// Ported to CUDA for nam-gpu from phys/grrad_nmmb.f:1540-1576 (nam.v4.2.11, 83df4e13c1).
//
// What the Fortran lines do: one call,
//   call setaer(plvl, plyr, prslk1, tvly, rhly, slmsk, tracer1, xlon, xlat,
//               IM, LMK, LMP, lsswr, lslwr, tlyr, qlyr, faersw, faerlw)
// with no arithmetic of its own. grrad_nmmb passes IM, LMK = LM+LTP, LMP = LMK+1 and the
// lsswr/lslwr flags; setaer (phys-rrtm-setaer, radiation_aerosols_nmmb.f:1845-2371) computes
// every value written to faersw and faerlw. The aerosol optical properties are therefore
// computed entirely by the setaer kernel; this file has no device code of its own.
//
// Its host-side counterpart is namgpu/physics/rrtm/phys_rrtm_grrad_aerosols.py, which binds the
// arrays by CONTRACT names and launches the setaer kernel once per radiation call.
//
// `nday = im` (grrad_nmmb.f, the all-day block just above the call) is a host-side integer that
// the wrapper passes through unchanged; no kernel reads it.
#include "nmmb_common.cuh"
