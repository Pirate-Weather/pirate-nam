// WOOF-NAM unit drv-dfi-tdfi-driver: RUN_DIGITAL_FILTER_NMM, TDFI branch (filter_method 3).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   nmm/module_NMM_GRID_COMP.F90, SUBROUTINE RUN_DIGITAL_FILTER_NMM (lines 3624-5418), and
//   nmm/module_DOMAIN_GRID_COMP.F90, SUBROUTINE NMM_FILTERING (lines 4839-5321). Author headers
//   kept as the notice. Ported to CUDA for WOOF-NAM from nmm/module_NMM_GRID_COMP.F90:3624-5418
//   and nmm/module_DOMAIN_GRID_COMP.F90:4839-5321 (nam.v4.2.11, 83df4e13c1).
//
// This unit owns no per-cell arithmetic. The TDFI routine is a control loop: it sets clocks,
// chooses the integration direction and the HDIFF/MEAN flags, and calls the dependency units
// (drv-dfi-dyn-init, drv-dfi-dyn-sum, drv-dfi-dyn-average, drv-dfi-phy-save, drv-dfi-phy-restore)
// and one model step per filter step. Every field update is in those units, so this file holds
// no kernel. It is compiled as an empty module so the done-rule compile step has a file to load.
//
// Schedule (wrapper namgpu/steps/drv_dfi_tdfi_driver.py, schedule()):
//   backward leg: clock direction Bckward, dt = -filt_dt, HDIFF = 0, MEAN = 0,
//                 2*nsecs_bcktdfi/filt_dt steps (DFITIME = STARTTIME - 2*HALFDFIINTVAL, line 4820)
//   forward leg:  clock direction Forward, dt = +filt_dt, HDIFF = 1, MEAN = 0,
//                 2*nsecs_fwdtdfi/filt_dt steps (line 5151 region, DFITIME = CURRTIME + 2*HALF)

#include "nmmb_common.cuh"
