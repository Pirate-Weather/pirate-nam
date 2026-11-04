// phys_rrtm_sw_params.cu -- unit phys-rrtm-sw-params: SW control and band parameters, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). Package
//   header of phys/radsw_param_nmmb.f (nam.v4.2.11, 83df4e13c1), reproduced verbatim (Fortran lines 1-46):
// !!!!!  ==========================================================  !!!!!
// !!!!!            sw-rrtm2 radiation package description            !!!!!
// !!!!!  ==========================================================  !!!!!
// !                                                                      !
// !   the sw-rrtm2 package includes these parts:                         !
// !                                                                      !
// !      'radsw_rrtm2_param.f'                                           !
// !      'radsw_rrtm2_datatb.f'                                          !
// !      'radsw_rrtm2_main.f'                                            !
// !                                                                      !
// !   the 'radsw_rrtm2_param.f' contains:                                !
// !                                                                      !
// !      'module_radsw_parameters'  -- band parameters set up            !
// !      'module_radsw_cntr_para'   -- control parameters set up         !
// !                                                                      !
// !   the 'radsw_rrtm2_datatb.f' contains:                               !
// !                                                                      !
// !      'module_radsw_cldprtb'     -- cloud property coefficients table !
// !      'module_radlw_kgbnn'       -- absorption coeffients for 14      !
// !                                    bands, where nn = 16-29           !
// !                                                                      !
// !   the 'radsw_rrtm2_main.f' contains:                                 !
// !                                                                      !
// !      'module_radsw_main'        -- main sw radiation transfer        !
// !                                                                      !
// !   in the main module 'module_radsw_main' there are only two          !
// !   externally callable subroutines:                                   !
// !                                                                      !
// !      'swrad'      -- main rrtm2 sw radiation routine                 !
// !      'rswinit'    -- initialization routine                          !
// !                                                                      !
// !   all the sw radiation subprograms become contained subprograms      !
// !   in module 'module_radsw_main' and many of them are not directly    !
// !   accessable from places outside the module.                         !
// !                                                                      !
// !   compilation sequence is:                                           !
// !                                                                      !
// !      'radsw_rrtm2_param.f'                                           !
// !      'radsw_rrtm2_datatb.f'                                          !
// !      'radsw_rrtm2_main.f'                                            !
// !                                                                      !
// !   and all should be put in front of routines that use sw modules     !
// !                                                                      !
// !!!!!  ==========================================================  !!!!!
// !!!!!                       end descriptions                       !!!!!
// !!!!!  ==========================================================  !!!!!
//
// Ported to CUDA for nam-gpu from phys/radsw_param_nmmb.f:51-189 (nam.v4.2.11, 83df4e13c1).
//
// Fortran lines covered:
//   51-92    module_radsw_cntr_para_nmmb: six integer control parameters (iswrate .. iflagice)
//              -> ctrl[0..5] = 2, 1, 2, 1, 1, 3
//   98-189   module_radsw_parameters_nmmb:
//     101     use machine, only : kind_phys            -> float32 (NAM kfpt single)
//     109-142 topfsw_type, sfcfsw_type, profsw_type, cmpfsw_type -> type layouts only; no values.
//             The flux fields they name are carried in ctx.st (RSWTOA, RSWIN, RSWOUT, ...) and
//             written by phys-rrtm-sw-swrad.
//     147-152 NBLOW 16, NBHGH 29, NBANDS 14 (= NBHGH-NBLOW+1), NGPT 112, NGMAX 16, MAXGAS 6 -> dims
//     154-156 NSWSTR 1, NSWEND = NBANDS (14), NBDSW = NBANDS (14) -> dims[6], dims[7], dims[8]
//     161-163 NG16..NG29 = 06 12 08 08 10 10 02 10 08 06 06 08 06 12 (parameter list)
//     166-167 NG(NBLOW:NBHGH) data statement (same values, same order) -> ng[0..13]
//     172-176 NS16..NS29 parameter chain = 00 06 18 26 34 44 54 56 66 74 80 86 94 100
//             (NSn = NSn-1 + NGn-1, 0-based begin index of each band's g-points) -> ns[0..13]
//     180-182 wvnum1 DATA (cm-1, band lower limit), 14 values in DATA order -> wvn1[0..13]
//     183-185 wvnum2 DATA (cm-1, band upper limit), 14 values in DATA order -> wvn2[0..13]
//
// Slot convention (stated once, applies to ng, ns, wvn1, wvn2): array slot s = band - 16, i.e.
//   slot 0 = band 16 ... slot 12 = band 28, slot 13 = band 29. Slots 14 and 15 are padding (zero).
// The Fortran DATA lists are NBANDS-long in source order, so DATA index k (1-based, k = 1..14)
// is band 15 + k and lands in slot k - 1:
//   wvnum1(1) = 2600 -> band 16 (slot 0)     wvnum1(13) = 38001 -> band 28 (slot 12)
//   wvnum1(14) = 820 -> band 29 (slot 13)    wvnum2(14) = 2600  -> band 29 (slot 13)
// The band-29 entry (820 to 2600 cm-1) is the last DATA element and is the only one stored out of
// wavenumber order, which is physically correct (band 29 is the lowest-wavenumber SW band).
// Within the 112 g-points, NS(b) + NG(b) = NS(b+1) and NS(29) + NG(29) = NGPT; the wrapper checks it.
//
// Outputs (device, written once at init by one thread):
//   ctrl  int32[6]   : iswrate, iaersw, imodsw, irgassw, iflagliq, iflagice
//   dims  int32[9]   : NBANDS, NBLOW, NBHGH, NGPT, NGMAX, MAXGAS, NSWSTR, NSWEND, NBDSW
//   ng    int32[16]  : g-points per band, band b at slot b-16
//   ns    int32[16]  : begin index of each band's g-points, band b at slot b-16
//   wvn1  float32[16]: band lower wavenumber, cm-1 (wvnum1), slot b-16
//   wvn2  float32[16]: band upper wavenumber, cm-1 (wvnum2), slot b-16
//
// Launch: <<<1, 1>>>, one launch at init, no host sync, no allocation, no print. Off the step path.
// No MPI, halo or OpenMP. The kernel holds the Fortran values as literals; it has no inputs.

extern "C" __global__ void phys_rrtm_sw_params(int* __restrict__ ctrl,
                                               int* __restrict__ dims,
                                               int* __restrict__ ng,
                                               int* __restrict__ ns,
                                               float* __restrict__ wvn1,
                                               float* __restrict__ wvn2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // ---- module_radsw_cntr_para_nmmb (51-92) ----
    ctrl[0] = 2;   // iswrate  (line 62): heating rate unit, 2 = K/s
    ctrl[1] = 1;   // iaersw   (line 65): aerosol effect on
    ctrl[2] = 2;   // imodsw   (line 69): pifm two-stream (Zdunkowski et al. 1980)
    ctrl[3] = 1;   // irgassw  (line 74): rare gases on
    ctrl[4] = 1;   // iflagliq (line 78): Stamnes (1993) liquid cloud optics, cwp/rew input
    ctrl[5] = 3;   // iflagice (line 82): Fu (1996) ice cloud optics, cip/rei input

    // ---- module_radsw_parameters_nmmb (98-189) ----
    dims[0] = 14;   // NBANDS  (line 149) = NBHGH - NBLOW + 1
    dims[1] = 16;   // NBLOW   (line 147)
    dims[2] = 29;   // NBHGH   (line 148)
    dims[3] = 112;  // NGPT    (line 150)
    dims[4] = 16;   // NGMAX   (line 151)
    dims[5] = 6;    // MAXGAS  (line 152)
    dims[6] = 1;    // NSWSTR  (line 154)
    dims[7] = 14;   // NSWEND  (line 155) = NBANDS
    dims[8] = 14;   // NBDSW   (line 156) = NBANDS

    // NG16..NG29 (161-163, data 166-167), slot b-16
    const int ng_v[16] = { 6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12, 0, 0 };
    // NS16..NS29 (172-176 parameter chain), slot b-16
    const int ns_v[16] = { 0, 6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 0, 0 };
    // wvnum1 (180-182): band lower limits, cm-1. DATA index k -> band 15+k -> slot k-1.
    const float w1_v[16] = { 2600.f, 3251.f, 4001.f, 4651.f, 5151.f, 6151.f, 7701.f, 8051.f,
                             12851.f, 16001.f, 22651.f, 29001.f, 38001.f, 820.f, 0.f, 0.f };
    // wvnum2 (183-185): band upper limits, cm-1. Same DATA-index-to-slot mapping as wvnum1.
    const float w2_v[16] = { 3250.f, 4000.f, 4650.f, 5150.f, 6150.f, 7700.f, 8050.f, 12850.f,
                             16000.f, 22650.f, 29000.f, 38000.f, 50000.f, 2600.f, 0.f, 0.f };

    for (int b = 0; b < 16; ++b) {
        ng[b]   = ng_v[b];
        ns[b]   = ns_v[b];
        wvn1[b] = w1_v[b];
        wvn2[b] = w2_v[b];
    }
}
