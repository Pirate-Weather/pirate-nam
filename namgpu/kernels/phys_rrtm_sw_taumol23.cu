// phys_rrtm_sw_taumol23.cu -- TAUMOL23: SW band 23 (8050-12850 cm-1) gas and Rayleigh optical depths
// of the NMMB RRTM shortwave (RRTM group). One thread per (i, j) column; the layer loop runs inside
// the thread. Every term of the Fortran routine is present.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOAA notice: phys/radsw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The notice is the sw-rrtm3 package description of that file:
//   sw-rrtm3 radiation package description; ncep modifications of the rrtm-sw radiation code
//   from aer inc. The k-table module is phys/radsw_datatb_nmmb.f module_radsw_kgb23_nmmb.
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4206-4272 (SUBROUTINE taumol23, internal to
// taumol). Tables: module_radsw_kgb23_nmmb (phys/radsw_datatb_nmmb.f:15195-15436), uploaded by the
// wrapper from namgpu/data/rrtm_sw/kgb23.npz. Constants: NS23 = 56 (radsw_param_nmmb.f NS table, band
// 23 slot), NG23 = 10, MSA23 = 65, MSF23 = 10, MFR23 = 3, givfac = 1.029 (radsw_datatb_nmmb.f:15225),
// nspa(23) = 1 and nspb(23) = 0 (radsw_main_nmmb.f:308-309).
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   per-layer inputs of the full domain   [nlay, JM, IM]   (flat column c = j*IM + i, layer stride JM*IM)
//   laytrop                               [JM, IM] int32
//   Layer order: the radsw INTERNAL order (NAM iflip = 0, module_SOLVER_GRID_COMP.F90:10714; the Fortran
//   taumol reads the internal array). Plane 0 is the LOWEST layer (surface), as Fortran k = 1, and laytrop
//   counts from the surface. This is NOT the NMMB top-first order. The SW driver (phys-rrtm-sw-swrad) builds the
//   internal-order arrays with IVFLIP = 0, which reads the top-first NMMB input and flips it.
//   colamt                                [MAXGAS, nlay, JM, IM]; only plane 0 (h2o) is read
//   Fortran layer k -> plane k-1; Fortran column (i,j) -> flat column c.
//   Tables are Fortran-natural: absa[(ind-1)*NG23 + j], forref[(n-1)*NG23 + j], selfref[(t-1)*NG23 + j],
//   rayl[j], with j = 0..NG23-1 the band g-interval (Fortran j = 1..NG23).
//   taug, taur are CHUNKED over columns and hold band 23 only: [NG23, nlay, ncol] for the chunk starting at flat column col0.
//   The full-domain taug (112 x 60 x 835 x 954 float32, about 21 GB) does not fit beside the state.
//   Output index for band g-interval j (0..9, g-point 56+j 0-based), layer k0, chunk column cc: (j*nlay + k0)*ncol + cc.
//   The wrapper maps taug[j] to g-point 56 + j of the 112-point spectrum (Fortran taug(:,:,NS23+j+1)).
//
// Translation notes:
//   - taug is computed for layers k <= laytrop(column) and zero above. The Fortran has two WHERE loops:
//     the first (k = 1..MAXVAL(laytrop)) computes k <= laytrop; the second (k = MINVAL(laytrop)+1..NLAY)
//     zeroes k >= laytrop. Net effect: layer k == laytrop keeps the computed value only when k is the
//     MINVAL(laytrop) of the caller's CHK-column tile, because the second loop never reaches it.
//     The Fortran tile is CHNK_RRTM = CHK_T23 = 8 consecutive i of one row (module_RADIATION.F90 chunk_loop_rrtm,
//     tiles start at i = ITS + 8m; configure.nems.* -DCHNK_RRTM=8). laytrop_min holds one int32 per tile,
//     index j * tiles_per_row + i / CHK_T23, built by the wrapper on the device (no host sync). Padded entries
//     carry the wrapper's large sentinel; the Fortran pad repeats an existing tile value, so the minimum over
//     the real columns is the same.
//   - id0 and id1 are formed inline as taumol does for band 23 (nspa = 1): id0 = (jp-1)*5 + (jt-1),
//     id1 = jp*5 + (jt1-1). They are used only for k <= laytrop, as in the Fortran.
//   - ind01 and ind11 are clamped to 1..MSA23, ind02 and ind12 to at most MSA23 (Fortran MAX/MIN).
//     The selfref and forref indices (inds, indf and their +1 partners) are not clamped, as in the Fortran.
//   - taur(k, g) = colmol(k) * rayl(g) for every layer.
//   - Single precision literals (f suffix); fast numerics (fmad on, NVRTC option); no host sync.

#include "nmmb_common.cuh"

#define NS23_T23    56
#define NG23_T23    10
#define NGPT_T23    112
#define CHK_T23     8      // CHNK_RRTM (configure.nems.*)
#define MSA23_T23   65
#define MSF23_T23   10
#define MFR23_T23   3
#define NSPA23_T23  1
#define GIVFAC23    1.029f

extern "C" __global__ void rrtm_sw_taumol23(
    const float* __restrict__ colamt,      // [MAXGAS, nlay, JM, IM]; plane 0 read
    const float* __restrict__ colmol,      // [nlay, JM, IM]
    const float* __restrict__ fac00,       // [nlay, JM, IM]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const int*   __restrict__ jp,          // [nlay, JM, IM] int32
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const int*   __restrict__ laytrop,     // [JM, IM] int32
    const int*   __restrict__ laytrop_min, // [JM * tiles_per_row] int32: MINVAL(laytrop) per CHK-column tile
    const float* __restrict__ forfac,      // [nlay, JM, IM]
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,      // [nlay, JM, IM] int32
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,     // [nlay, JM, IM] int32
    const float* __restrict__ absa,        // [MSA23, NG23] Fortran-natural
    const float* __restrict__ forref,      // [MFR23, NG23]
    const float* __restrict__ selfref,     // [MSF23, NG23]
    const float* __restrict__ rayl,        // [NG23]
    float* __restrict__ taug,              // [NGPT, nlay, ncol] chunk
    float* __restrict__ taur,              // [NGPT, nlay, ncol] chunk
    int nlay, int im, int jm, int maxgas, int col0, int ncol, int tiles_per_row)
{
    (void)maxgas;
    const int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= ncol) return;
    const int col = col0 + cc;
    const int ncol_dom = im * jm;
    if (col >= ncol_dom) return;

    const int lt   = laytrop[col];
    // Fortran MINVAL(laytrop(1:CHK)) is over the caller tile of this column (same row, i in [8m, 8m+7]).
    const int jrow = col / im;
    const int ti   = col - jrow * im;
    const int kmin = laytrop_min[jrow * tiles_per_row + ti / CHK_T23];
    const size_t LS = (size_t)ncol_dom;     // layer stride of the full-domain inputs
    const size_t GS = (size_t)nlay * ncol;  // g-point stride of the chunk outputs

    for (int k = 1; k <= nlay; ++k) {
        const size_t p  = (size_t)(k - 1) * LS + col;    // full-domain input index
        const size_t oo = (size_t)(k - 1) * ncol + cc;   // chunk output index for g = 0

        // taur(1:CHK, k, NS23+j) = colmol(1:CHK, k) * rayl(j)
        const float cm = colmol[p];
        for (int j = 0; j < NG23_T23; ++j) {
            taur[(size_t)j * GS + oo] = cm * rayl[j];
        }

        // Fortran: computed when k <= laytrop; zeroed when k >= laytrop and k >= MINVAL + 1.
        const bool zero_cond = (k >= kmin + 1) && (k >= lt);
        const bool compute   = (k <= lt) && !zero_cond;

        if (!compute) {
            for (int j = 0; j < NG23_T23; ++j) {
                taug[(size_t)j * GS + oo] = 0.0f;
            }
            continue;
        }

        // id0 / id1 (taumol, band 23: nspa = 1) and the clamped absorption indices.
        const int jp_  = jp[p];
        const int jt_  = jt[p];
        const int jt1_ = jt1[p];
        const int id0 = ((jp_ - 1) * 5 + (jt_ - 1)) * NSPA23_T23;
        const int id1 = (jp_ * 5 + (jt1_ - 1)) * NSPA23_T23;
        const int ind01 = max(1, min(MSA23_T23, id0 + 1));
        const int ind02 = min(MSA23_T23, ind01 + 1);
        const int ind11 = max(1, min(MSA23_T23, id1 + 1));
        const int ind12 = min(MSA23_T23, ind11 + 1);

        const int inds  = indself[p];
        const int indf  = indfor[p];
        const int indsp = inds + 1;
        const int indfp = indf + 1;

        const float f00 = fac00[p];
        const float f01 = fac01[p];
        const float f10 = fac10[p];
        const float f11 = fac11[p];
        const float sf  = selffac[p];
        const float sfr = selffrac[p];
        const float ff  = forfac[p];
        const float ffr = forfrac[p];
        const float cam = colamt[p];                     // plane 0 = h2o, colamt(1:CHK,k,1)

        // taug(1:CHK,k,NS23+j) = colamt * ( givfac*(f00*absa(ind01) + f10*absa(ind02)
        //                                    + f01*absa(ind11) + f11*absa(ind12))
        //                                 + selffac*(selfref(inds) + selffrac*(selfref(indsp) - selfref(inds)))
        //                                 + forfac*(forref(indf) + forfrac*(forref(indfp) - forref(indf))) )
        for (int j = 0; j < NG23_T23; ++j) {
            const float a01 = absa[(size_t)(ind01 - 1) * NG23_T23 + j];
            const float a02 = absa[(size_t)(ind02 - 1) * NG23_T23 + j];
            const float a11 = absa[(size_t)(ind11 - 1) * NG23_T23 + j];
            const float a12 = absa[(size_t)(ind12 - 1) * NG23_T23 + j];
            const float s1  = selfref[(size_t)(inds  - 1) * NG23_T23 + j];
            const float s2  = selfref[(size_t)(indsp - 1) * NG23_T23 + j];
            const float r1  = forref[(size_t)(indf  - 1) * NG23_T23 + j];
            const float r2  = forref[(size_t)(indfp - 1) * NG23_T23 + j];

            const float val = cam * (GIVFAC23 * (f00 * a01 + f10 * a02 + f01 * a11 + f11 * a12)
                                     + sf * (s1 + sfr * (s2 - s1))
                                     + ff * (r1 + ffr * (r2 - r1)));
            taug[(size_t)j * GS + oo] = val;
        }
    }
}
