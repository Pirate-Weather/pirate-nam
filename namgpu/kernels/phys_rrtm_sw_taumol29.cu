// phys_rrtm_sw_taumol29.cu -- taumol29: SW band 29 (820-2600 cm-1, low = H2O, high = CO2) optical depths and
// the Rayleigh optical depth, NMMB RRTM shortwave (physics-rrtm group). One thread per (i, j) column; the
// layer loop k = 1..NLAY runs inside the thread.
//
// NOTICE: NAM source file phys/radsw_main_nmmb.f (NCEP/EMC, US Government work, no licence file; see NOTICE).
// The sw-rrtm3 package header of that file is kept as the notice:
// !!!!!  sw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep's modifications of the rrtm-sw radiation code from aer inc.   !
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:4683-4767 (SUBROUTINE taumol29, nam.v4.2.11,
// 83df4e13c1). Tables: module_radsw_kgb29_nmmb (phys/radsw_datatb_nmmb.f:20065-20973), extracted by
// tools/extract-rrtm-sw-kgb29.py into namgpu/data/rrtm_sw/kgb29.npz (absa, absb, selfref, forref, absh2o,
// absco2 in Fortran column-major order, rayl).
//
// Fortran lines covered:
//   4683-4698  routine header and locals (tauray, ind01/02/11/12, inds/indf/indsp/indfp, j, k)
//   4704-4713  tauray = colmol*rayl; taur(:,k,NS29+j) = tauray for j = 1..NG29 (all k)
//   4715-4742  lower atmosphere (k <= laytrop): ind01/02/11/12 from id0/id1 (absa, no band offset),
//              inds/indf from indself/indfor, the absa interpolation, the water-vapour self-continuum
//              (selfref) and foreign-continuum (forref) terms, and colamt(:,k,2)*absco2(j)
//   4744-4762  upper atmosphere (k >= laytrop): ind01/02/11/12 from absb, colamt(:,k,2)*absb-interp
//              (CO2 band) plus colamt(:,k,1)*absh2o(j)
//   Band 29 uses NG29 = 12 g-points at NS29 = 100 (radsw_param_nmmb.f:159-176; NS29 = sum of NG16..NG28).
//
// Index conventions (CONTRACT.md sections 1 and 3, Fortran A(i,j,l) -> C [l, j, i]):
//   colmol, fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac, id0, id1, indself, indfor
//   and every per-layer field: [NLAY, JM, IM]; colamt: [MAXGAS = 6, NLAY, JM, IM] (Fortran colamt(i,k,n)
//   -> [n-1, k-1, j-1, i-1]; only n = 1 (h2o) and n = 2 (co2) are read); laytrop: [JM, IM].
//   taug29: [NG29 = 12, NLAY, JM, IM], the Fortran taug(:,:,NS29+g) for g = 1..12 at plane g-1. The
//   112-plane taug of the Fortran is NOT allocated (21 GB at 954 x 835 x 60); this band writes only its
//   own twelve planes. taur29 is the Rayleigh depth, the same for all twelve g-points of the band.
//   id0 and id1 are the band-29 index arrays the taumol driver computes (radsw_main_nmmb.f with nspa(29),
//   nspb(29)); they are inputs here, as in the Fortran where they are host-associated.
//
// Translation notes:
//   - Fortran WHERE blocks over CHK columns become per-column branches on k <= laytrop (lower) and
//     k > laytrop (upper).
//   - DEVIATION (k == laytrop). The Fortran runs its upper-atmosphere loop from k = MINVAL(laytrop)+1 with
//     mask k >= laytrop, so at k == laytrop it overwrites the lower-atmosphere value with the absb formula
//     whenever laytrop > MINVAL(laytrop) of its chunk. The result therefore depends on the chunk. This port
//     uses the layer split the routine intends: lower formula for k <= laytrop, upper for k > laytrop.
//     Same declared deviation as taumol22; affects one layer per column (about 95 hPa).
//   - DEVIATION (out-of-table neighbour index). The Fortran reads selfref(inds+1, j) and forref(indf+1, j)
//     without a bound. The CUDA port clamps those two neighbour indices to the table size (MSF29, MFR29),
//     so the read stays inside the table. In-range indices are unchanged.
//   - Fortran INT() truncation and MAX/MIN clamps on ind01..ind12 are kept as written.
//   - Single precision (kind_taum = 4, machine.f) everywhere.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation, no printf.

#include "nmmb_common.cuh"

#define NG29_SW        12
#define NS29_SW        100
#define MSA29_SW       65
#define MSB29_SW       235
#define MSF29_SW       10
#define MFR29_SW       4
#define MAXGAS_SW      6
#define RAYL29_SW      9.30e-11f            // module_radsw_kgb29_nmmb, rayl (radsw_datatb_nmmb.f module parameter)

extern "C" __global__ void rrtm_sw_taumol29(
    const float* __restrict__ colmol,     // [NLAY, JM, IM]
    const float* __restrict__ colamt,     // [MAXGAS, NLAY, JM, IM]
    const int*   __restrict__ laytrop,    // [JM, IM]
    const float* __restrict__ fac00,      // [NLAY, JM, IM]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const int*   __restrict__ id0,        // [NLAY, JM, IM], band-29 index from the taumol driver
    const int*   __restrict__ id1,
    const float* __restrict__ absa,       // Fortran absa(65, 12), column-major: absa[(ms-1) + 65*(g-1)]
    const float* __restrict__ absb,       // Fortran absb(235, 12)
    const float* __restrict__ selfref,    // Fortran selfref(10, 12)
    const float* __restrict__ forref,     // Fortran forref(4, 12)
    const float* __restrict__ absh2o,     // Fortran absh2o(12)
    const float* __restrict__ absco2,     // Fortran absco2(12)
    int NLAY, int IM, int JM,
    float* __restrict__ taug29,           // [NG29, NLAY, JM, IM]
    float* __restrict__ taur29)           // [NLAY, JM, IM]
{
    COLUMN_IJ(IM, JM)

    const size_t cij = IDX2(i0, j0, IM);
    const int lt = laytrop[cij];

    for (int k0 = 0; k0 < NLAY; ++k0) {
        const size_t c = IDX3(i0, j0, k0, IM, JM);
        const int k = k0 + 1;                          // Fortran layer index

        // ---- Rayleigh (4704-4713): every layer, all twelve g-points share it ----
        const float tauray = colmol[c] * RAYL29_SW;
        taur29[c] = tauray;

        const float c1 = colamt[IDX4(i0, j0, k0, 0, IM, JM, NLAY)];   // colamt(:,k,1)  h2o
        const float c2 = colamt[IDX4(i0, j0, k0, 1, IM, JM, NLAY)];   // colamt(:,k,2)  co2

        if (k <= lt) {
            // ---- lower atmosphere: absa table (4715-4742) ----
            const float f00 = fac00[c], f01 = fac01[c], f10 = fac10[c], f11 = fac11[c];

            const int ind01 = max(1, min(MSA29_SW, id0[c] + 1));
            const int ind02 = min(MSA29_SW, ind01 + 1);
            const int ind11 = max(1, min(MSA29_SW, id1[c] + 1));
            const int ind12 = min(MSA29_SW, ind11 + 1);

            const int inds = indself[c];
            const int indf = indfor[c];
            const int indsp = min(inds + 1, MSF29_SW);      // clamp: see header DEVIATION
            const int indfp = min(indf + 1, MFR29_SW);      // clamp: see header DEVIATION

            for (int g = 0; g < NG29_SW; ++g) {
                const float* ag = absa + MSA29_SW * g;      // absa(:, g+1)
                const float* sg = selfref + MSF29_SW * g;   // selfref(:, g+1)
                const float* fg = forref + MFR29_SW * g;    // forref(:, g+1)

                const float taug = c1
                    * ( (f00 * ag[ind01 - 1] + f10 * ag[ind02 - 1]
                       + f01 * ag[ind11 - 1] + f11 * ag[ind12 - 1])
                      + selffac[c] * (sg[inds - 1] + selffrac[c] * (sg[indsp - 1] - sg[inds - 1]))
                      + forfac[c] * (fg[indf - 1] + forfrac[c] * (fg[indfp - 1] - fg[indf - 1])) )
                    + c2 * absco2[g];
                taug29[IDX4(i0, j0, k0, g, IM, JM, NLAY)] = taug;
            }
        } else {
            // ---- upper atmosphere: absb table (4744-4762) ----
            const float f00 = fac00[c], f01 = fac01[c], f10 = fac10[c], f11 = fac11[c];

            const int ind01 = max(1, min(MSB29_SW, id0[c] + 1));
            const int ind02 = min(MSB29_SW, ind01 + 1);
            const int ind11 = max(1, min(MSB29_SW, id1[c] + 1));
            const int ind12 = min(MSB29_SW, ind11 + 1);

            for (int g = 0; g < NG29_SW; ++g) {
                const float* bg = absb + MSB29_SW * g;      // absb(:, g+1)
                const float taug = c2
                    * ( f00 * bg[ind01 - 1] + f10 * bg[ind02 - 1]
                      + f01 * bg[ind11 - 1] + f11 * bg[ind12 - 1] )
                    + c1 * absh2o[g];
                taug29[IDX4(i0, j0, k0, g, IM, JM, NLAY)] = taug;
            }
        }
    }
}
