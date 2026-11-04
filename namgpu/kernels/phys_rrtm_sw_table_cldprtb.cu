// phys_rrtm_sw_table_cldprtb.cu -- unit phys-rrtm-sw-table-cldprtb, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA/NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone), module
//   MODULE_RADSW_CLDPRTB_NMMB (radsw_datatb_nmmb.f, nam.v4.2.11). Author header retained in the NAM clone.
// Ported to CUDA for nam-gpu from phys/radsw_datatb_nmmb.f:52-1064 (nam.v4.2.11, 83df4e13c1).
//
// This module holds no executable code. It is DATA statements and parameters only: the liquid and ice
// cloud optical-property tables (extdat/ssadat/asydat liq1 58 x 14, extdat/ssadat/asydat/fdldat ice3
// 27 x 14), the band vectors b0r c0r b0s b1s c0s, and the scalars a0r a1r a0s a1s absdatliq1.
// The tables are extracted by tools/extract_rrtm_sw_cldprtb_nmmb.py into namgpu/data/rrtm_sw/cldprtb.npz
// and uploaded once to ctx.tables by the wrapper (phys_rrtm_sw_table_cldprtb.py). No device computation
// exists for this unit, so there is no kernel here and nothing to launch per step.
// Every DATA value is kept in single precision: NAM's production build defines -DSINGLE_PREC, so kind_phys =
// real*4 (phys/machine.f:20). A build without that define makes kind_phys real*8 (phys/machine.f:10); this
// unit stores float32 by the kfpt=single contract (recorded declared-precision deviation).
