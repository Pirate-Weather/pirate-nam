// rrtm_sw_bands.cuh -- NMMB RRTM shortwave band tables shared by the taumol units (nam-gpu).
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:306-309 (nspa, nspb, data statements),
// phys/radsw_datatb_nmmb.f:1101-1121 (strrat, specwt, layreffr, ix1, ix2, ibx, scalekur) and
// phys/radsw_param_nmmb.f:151-177 (NG16..NG29, NS16..NS29), nam.v4.2.11, 83df4e13c1.
// Notice: sw-rrtm3 radiation package, NCEP modifications of the RRTM-SW code from AER Inc.
//
// Band index b (0-based) is Fortran band jb = b + 16, bands 16..29.
#pragma once

#define SW_NBLOW   16
#define SW_NBANDS  14
#define SW_NGPTSW  112
#define SW_NGMAX   16
#define SW_MAXGAS  6
#define SW_MFS01   1
#define SW_MFS02   5
#define SW_MFS03   9
#define SW_MFB01   7
#define SW_MFB02   2
#define SW_MFB03   5
// oneminus = 1.0 - eps, eps = 1.0e-6 (radsw_main_nmmb.f:291-292), single precision.
#define SW_ONEMINUS 0.999999f

// NG16..NG29 (radsw_param_nmmb.f:157-160): g-points per band.
__constant__ int c_sw_ng[SW_NBANDS] = {6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12};

// NS16..NS29 (radsw_param_nmmb.f:170-177): first g-point of each band, 0-based offset.
__constant__ int c_sw_ngs[SW_NBANDS] = {0, 6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100};

// nspa, nspb (radsw_main_nmmb.f:308-309).
__constant__ int c_sw_nspa[SW_NBANDS] = {9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1};
__constant__ int c_sw_nspb[SW_NBANDS] = {1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1};

// layreffr, ix1, ix2, ibx (radsw_datatb_nmmb.f:1116-1119). ix1/ix2 are 1-based gas indices
// (0 = unused band); ibx is the 1-based band index into the sflux tables.
__constant__ int c_sw_layreffr[SW_NBANDS] = {18, 30, 6, 3, 3, 8, 2, 6, 1, 2, 0, 32, 58, 49};
__constant__ int c_sw_ix1[SW_NBANDS]      = {1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3, 0};
__constant__ int c_sw_ix2[SW_NBANDS]      = {5, 2, 5, 2, 0, 2, 6, 0, 6, 0, 0, 0, 6, 0};
__constant__ int c_sw_ibx[SW_NBANDS]      = {1, 1, 1, 2, 2, 3, 4, 3, 5, 4, 5, 6, 2, 7};

// strrat (radsw_datatb_nmmb.f:1108-1110; strrat(22) already multiplied by o2adj = 1.6).
__constant__ float c_sw_strrat[SW_NBANDS] = {
    2.52131e+2f, 3.64641e-1f, 3.89589e+1f, 5.49281e+0f, 0.0f, 4.53210e-3f, 3.63328e-2f,
    0.0f, 1.24692e-1f, 0.0f, 0.0f, 0.0f, 6.67029e-7f, 0.0f};

// specwt (radsw_datatb_nmmb.f:1112).
__constant__ float c_sw_specwt[SW_NBANDS] = {8.f, 4.f, 8.f, 8.f, 0.f, 8.f, 8.f, 0.f, 8.f, 0.f, 0.f, 0.f, 4.f, 0.f};

// taumol index of band jb (Fortran 16..29) at 1-based layer k1 (radsw_main_nmmb.f:3137-3158).
// Each taumol## unit evaluates this itself: the Fortran id0/id1 were dispatcher locals that are
// host-associated by taumol16..29, and storing them for the full 12 km domain would be 5.4 GB.
//   k1 <= laytrop : id0 = ((jp-1)*5 + (jt-1)) * nspa,  id1 = (jp*5 + (jt1-1)) * nspa
//   k1 >  laytrop : id0 = ((jp-13)*5 + (jt-1)) * nspb, id1 = ((jp-12)*5 + (jt1-1)) * nspb
__device__ inline void rrtm_sw_taumol_index(int jb, int k1, int laytrop, int jp, int jt, int jt1,
                                            int* id0, int* id1)
{
    const int b = jb - SW_NBLOW;
    if (k1 <= laytrop) {
        *id0 = ((jp - 1) * 5 + (jt - 1)) * c_sw_nspa[b];
        *id1 = (jp * 5 + (jt1 - 1)) * c_sw_nspa[b];
    } else {
        *id0 = ((jp - 13) * 5 + (jt - 1)) * c_sw_nspb[b];
        *id1 = ((jp - 12) * 5 + (jt1 - 1)) * c_sw_nspb[b];
    }
}
