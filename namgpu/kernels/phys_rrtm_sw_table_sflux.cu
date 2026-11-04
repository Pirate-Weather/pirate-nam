// phys_rrtm_sw_table_sflux.cu -- unit phys-rrtm-sw-table-sflux, nam-gpu.
//
// NOTICE: NOAA/NCEP NAM source (US government work, no licence file in the NAM clone), module
//   module_radsw_sflux_nmmb (phys/radsw_datatb_nmmb.f:1070-1378, nam.v4.2.11, 83df4e13c1). The NOAA/NCEP
//   author header is retained in the NAM clone.
//
// This module holds no executable code. It is DATA statements only: sfluxref01/02/03, the band maps,
// strrat, specwt, layreffr, ix1, ix2, ibx and scalekur. The tables are extracted by
// tools/extract_rrtm_sw_sflux_nmmb.py into namgpu/data/rrtm_sw/sflux.npz and uploaded once to
// ctx.tables (keys rrtm_sw_sflux_*) by the wrapper phys_rrtm_sw_table_sflux.py. No device computation
// exists for this unit, so there is no kernel here and nothing to launch per step. The band constants
// that the taumol kernels read are in rrtm_sw_bands.cuh, checked against the same DATA lines.
