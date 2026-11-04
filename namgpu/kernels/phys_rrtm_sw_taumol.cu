// phys_rrtm_sw_taumol.cu -- NMMB RRTM shortwave band dispatcher (taumol), dispatch work.
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:3107-3383 (subroutine taumol,
// nam.v4.2.11, 83df4e13c1). Notice (source header of radsw_main_nmmb.f): sw-rrtm3 radiation package,
// NCEP modifications of the RRTM-SW radiation code from AER Inc.
//
// What this kernel computes, per column (one thread per (i, j)):
//   sfluxzen(g, col) for all 112 g-points, Fortran lines 3137-3248:
//     case (16,20,23,25,26,29): sfluxref01(j,1,ibd)
//     case (27)               : scalekur * sfluxref01(j,1,ibd)
//     case default, jb 17 & 28: sfluxref02 interpolated in specmult (ks search from laytrop to nlay-1)
//     case default, others    : sfluxref03 interpolated in specmult (ks search from 1 to laytrop-1)
//   Lines 3195-3196 (the padding of columns nday+1..CHK) are dropped: the GPU launch covers the whole
//   domain in one pass, so there is no nday < CHK case.
//
// Not in this kernel (by design, see rrtm_sw_bands.cuh):
//   id0 / id1 (Fortran lines 3137-3158) are per-band, per-layer indices that the taumol16..29 units
//   evaluate with rrtm_sw_taumol_index(). They are not stored here.
//   The calls taumol16 .. taumol29 (Fortran 3376-3390) are separate units, launched by the wrapper
//   after this kernel with the same column layout.
//
// Index conventions (zero-based in CUDA, CONTRACT.md section 1):
//   colamt[n, l, j, i]  float32 [6, NLAY, JM, IM]   Fortran colamt(i,j,l,n), n = gas (1-based in Fortran)
//   jp[l, j, i]         int32   [NLAY, JM, IM]      Fortran jp(i,j,l)
//   laytrop[j, i]       int32   [JM, IM]            Fortran laytrop(i,j), 1-based layer index
//   sf01[ibd, s, g]     float32 [7, 1, 16]          Fortran sfluxref01(g, s, ibd)
//   sf02[ibd, s, g]     float32 [2, 5, 16]          Fortran sfluxref02(g, s, ibd)
//   sf03[ibd, s, g]     float32 [5, 9, 16]          Fortran sfluxref03(g, s, ibd)
//   sfluxzen[g, j, i]   float32 [112, JM, IM]       Fortran sfluxzen(i, g)
// The sflux tables are written by the phys-rrtm-sw-table-sflux unit in exactly these layouts.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

// Layer read at 1-based layer k1 of the current column (int array [NLAY, JM, IM]).
#define SW_LAY_I(arr, k1)  ((arr)[IDX3(i0, j0, (k1) - 1, im, jm)])

// Fortran lines 3186-3190 (colm1, colm2, speccomb, specmult, js, fs) for the column at layer ks.
// Returns js (1-based) and fs = mod(specmult, 1).
__device__ inline void sw_specmult(const float* __restrict__ colamt, int i0, int j0, int im, int jm,
                                   int nlay, int ks, int ix1, int ix2, float strrat, float specwt,
                                   int* js, float* fs)
{
    const float colm1 = colamt[IDX4(i0, j0, ks - 1, ix1 - 1, im, jm, nlay)];
    const float colm2 = colamt[IDX4(i0, j0, ks - 1, ix2 - 1, im, jm, nlay)];
    const float speccomb = colm1 + strrat * colm2;
    const float specmult = specwt * fminf(SW_ONEMINUS, colm1 / speccomb);
    const int jsi = (int)specmult;
    *js = 1 + jsi;
    *fs = specmult - (float)jsi;   // Fortran mod(specmult, f_one), specmult >= 0
}

extern "C" __global__ void phys_rrtm_sw_taumol(
    const float* __restrict__ colamt,
    const int*   __restrict__ jp,
    const int*   __restrict__ laytrop,
    const float* __restrict__ sf01,
    const float* __restrict__ sf02,
    const float* __restrict__ sf03,
    const float scalekur,
    float* __restrict__ sfluxzen,
    const int nlay, const int jm, const int im)
{
    COLUMN_IJ(im, jm)
    const int lt = laytrop[IDX2(i0, j0, im)];

    for (int b = 0; b < SW_NBANDS; ++b) {
        const int jb  = b + SW_NBLOW;        // Fortran band 16..29
        const int ibd = c_sw_ibx[b];         // 1-based sflux band
        const int njb = c_sw_ng[b];
        const int ns  = c_sw_ngs[b];         // first g-point of the band (0-based)
        const int lr  = c_sw_layreffr[b];

        if (jb == 16 || jb == 20 || jb == 23 || jb == 25 || jb == 26 || jb == 29) {
            for (int j = 0; j < njb; ++j)
                sfluxzen[IDX3(i0, j0, ns + j, im, jm)] =
                    sf01[((ibd - 1) * SW_MFS01 + 0) * SW_NGMAX + j];
        } else if (jb == 27) {
            for (int j = 0; j < njb; ++j)
                sfluxzen[IDX3(i0, j0, ns + j, im, jm)] =
                    scalekur * sf01[((ibd - 1) * SW_MFS01 + 0) * SW_NGMAX + j];
        } else if (jb == 17 || jb == 28) {
            // Fortran 3206-3233: ks search k = laytrop .. nlay-1, default ks = nlay.
            int ks = nlay;
            for (int k = lt; k <= nlay - 1; ++k) {
                if (SW_LAY_I(jp, k) < lr && SW_LAY_I(jp, k + 1) >= lr) { ks = k + 1; break; }
            }
            int js; float fs;
            sw_specmult(colamt, i0, j0, im, jm, nlay, ks,
                        c_sw_ix1[b], c_sw_ix2[b], c_sw_strrat[b], c_sw_specwt[b], &js, &fs);
            const int base = ((ibd - 1) * SW_MFS02) * SW_NGMAX;   // [ibd, s, g] with s = js-1
            for (int j = 0; j < njb; ++j) {
                const float a = sf02[base + (js - 1) * SW_NGMAX + j];
                const float c = sf02[base + js * SW_NGMAX + j];
                sfluxzen[IDX3(i0, j0, ns + j, im, jm)] = a + fs * (c - a);
            }
        } else {
            // Fortran 3238-3247: ks search k = 1 .. laytrop-1, default ks = laytrop.
            int ks = lt;
            for (int k = 1; k <= lt - 1; ++k) {
                if (SW_LAY_I(jp, k) < lr && SW_LAY_I(jp, k + 1) >= lr) { ks = k + 1; break; }
            }
            int js; float fs;
            sw_specmult(colamt, i0, j0, im, jm, nlay, ks,
                        c_sw_ix1[b], c_sw_ix2[b], c_sw_strrat[b], c_sw_specwt[b], &js, &fs);
            const int base = ((ibd - 1) * SW_MFS03) * SW_NGMAX;
            for (int j = 0; j < njb; ++j) {
                const float a = sf03[base + (js - 1) * SW_NGMAX + j];
                const float c = sf03[base + js * SW_NGMAX + j];
                sfluxzen[IDX3(i0, j0, ns + j, im, jm)] = a + fs * (c - a);
            }
        }
    }
}
