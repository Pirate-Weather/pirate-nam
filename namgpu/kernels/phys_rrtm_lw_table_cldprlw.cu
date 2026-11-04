// phys_rrtm_lw_table_cldprlw.cu -- unit phys-rrtm-lw-table-cldprlw, WOOF-NAM.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA/NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone), module
//   MODULE_RADLW_CLDPRLW_NMMB (radlw_datatb_nmmb.f, nam.v4.2.11). Author header retained in the NAM clone.
// Ported to CUDA for WOOF-NAM from phys/radlw_datatb_nmmb.f:733-1144 (nam.v4.2.11, 83df4e13c1).
//
// This module holds no executable code. It is DATA statements only: ipat, absice0, absice1,
// absice2 (40 x 16), absliq1, absliq2, absliq3 (58 x 16), absrain, abssnow0, abssnow1.
// The tables are extracted by tools/extract_rrtm_lw_cldprlw_nmmb.py into namgpu/data/rrtm_lw/cldprlw.npz
// and uploaded once to ctx.tables by the wrapper (phys_rrtm_lw_table_cldprlw.py). No device
// computation exists for this unit, so there is no kernel here and nothing to launch per step.
// Every DATA value is kept in single precision, as NAM declares kind_phys = real*4.


