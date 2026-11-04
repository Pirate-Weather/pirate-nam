// phys_rrtm_lw_params.cu -- unit phys-rrtm-lw-params: LW control and band parameters, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). Module
//   header of phys/radlw_param_nmmb.f (nam.v4.2.11, 83df4e13c1), kept in the NAM clone; the file
//   describes the rrtm1 LW package: module_radlw_cntr_para_nmmb and module_radlw_parameters_nmmb.
// Ported to CUDA for nam-gpu from phys/radlw_param.f (module_radlw_parameters, used by radlw_main_nmmb.f)
// and phys/physparam.f (control flags), nam.v4.2.11 83df4e13c1. The _nmmb set (radlw_param_nmmb.f) is GFS-family.
//
// Fortran lines covered:
//   57-98     module_radlw_cntr_para_nmmb: six integer control parameters (ilwrate .. iflagice)
//               -> ctrl[0..5] = 2, 1, 1, 1, 3, 1
//   104-175   module_radlw_parameters_nmmb:
//     107     use machine, only : kind_phys            -> float32 (NAM kfpt single)
//     115-135 topflw_type, sfcflw_type, proflw_type   -> type layouts only; no values. The flux fields
//             they name are carried in ctx.st (RLWTOA, RLWIN, ...) and written by phys-rrtm-lw-lwrad.
//     140-146 NBANDS 16, NGPTLW 140, N5000 5000, N200 200, MAXGAS 6, MAXXSEC 4, NPLNK 181 -> dims
//     148     NBDLW = NBANDS = 16                     -> dims[1]
//     153-158 NG01..NG16 = 08 14 16 14 16 08 12 08 12 06 08 08 04 02 02 02  -> ng[0..15]
//     160-164 NS01..NS16 = 00 08 22 38 52 68 76 88 96 108 114 122 130 134 136 138 -> ns[0..15]
//               (0-based begin index of each band's g-points; NS01 = 00 kept as in the source)
//     166-171 wvnlw1(NBANDS) / wvnlw2(NBANDS) data     -> wvn1[0..15] / wvn2[0..15] (cm-1)
//   The tiling (sum NG = 140, NS(b)+NG(b) = NS(b+1)) is checked on the host in the wrapper.
//
// Outputs (device, written once at init by one thread):
//   ctrl  int32[6]   : ilwrate, iaerlw, irgaslw, icfclw, iflagliq, iflagice
//   dims  int32[8]   : NBANDS, NBDLW, NGPTLW, N5000, N200, MAXGAS, MAXXSEC, NPLNK
//   ng    int32[16]  : g-points per band (NG01..NG16)
//   ns    int32[16]  : begin index of each band's g-points (NS01..NS16)
//   wvn1  float32[16]: band lower wavenumber, cm-1 (wvnlw1)
//   wvn2  float32[16]: band upper wavenumber, cm-1 (wvnlw2)
//
// Launch: <<<1, 1>>>, one launch at init, no host sync, no allocation, no print. Off the step path.
// No MPI, halo or OpenMP. The kernel holds the Fortran values as literals; it has no inputs.

extern "C" __global__ void phys_rrtm_lw_params(int* __restrict__ ctrl,
                                               int* __restrict__ dims,
                                               int* __restrict__ ng,
                                               int* __restrict__ ns,
                                               float* __restrict__ wvn1,
                                               float* __restrict__ wvn2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // ---- module_radlw_cntr_para_nmmb (57-98) ----
    ctrl[0] = 2;   // ilwrate  (physparam.f:71): heating rate unit, 2 = K/s
    ctrl[1] = 1;   // iaerlw   (line 71): aerosol optics per band
    ctrl[2] = 1;   // irgaslw  (line 76): rare gases on
    ctrl[3] = 1;   // icfclw   (line 80): halocarbons on
    ctrl[4] = 1;   // iflagliq = ilwcliq (physparam.f:75): liquid cloud optics as the NMMB LW uses it
    ctrl[5] = 3;   // iflagice = ilwcice (physparam.f:79)

    // ---- module_radlw_parameters_nmmb (104-175) ----
    dims[0] = 16;   // NBANDS  (140)
    dims[1] = 16;   // NBDLW   (148) = NBANDS
    dims[2] = 140;  // NGPTLW  (141)
    dims[3] = 10000; // NTBL    (radlw_param.f:96)
    dims[4] = 6;     // NRATES  (radlw_param.f:100)
    dims[5] = 7;     // MAXGAS  (radlw_param.f:99)
    dims[6] = 4;     // MAXXSEC (radlw_param.f:98)
    dims[7] = 181;  // NPLNK   (146)

    // NG01..NG16 (153-158)
    const int ng_v[16] = { 10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2 };
    // NS01..NS16 (160-164)
    const int ns_v[16] = { 0, 10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138 };
    // wvnlw1 (166-168): band lower limits, cm-1
    const float w1_v[16] = { 10.f, 351.f, 501.f, 631.f, 701.f, 821.f, 981.f, 1081.f,
                             1181.f, 1391.f, 1481.f, 1801.f, 2081.f, 2251.f, 2381.f, 2601.f };
    // wvnlw2 (169-171): band upper limits, cm-1
    const float w2_v[16] = { 350.f, 500.f, 630.f, 700.f, 820.f, 980.f, 1080.f, 1180.f,
                             1390.f, 1480.f, 1800.f, 2080.f, 2250.f, 2380.f, 2600.f, 3250.f };

    for (int b = 0; b < 16; ++b) {
        ng[b]   = ng_v[b];
        ns[b]   = ns_v[b];
        wvn1[b] = w1_v[b];
        wvn2[b] = w2_v[b];
    }
}
