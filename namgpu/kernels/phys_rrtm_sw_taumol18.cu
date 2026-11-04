// phys_rrtm_sw_taumol18.cu -- NMMB RRTM shortwave band 18 optical depths and Rayleigh depth (taumol18).
//
// Ported to CUDA for nam-gpu from phys/radsw_main_nmmb.f:3632-3738 (subroutine taumol18, nam.v4.2.11,
// 83df4e13c1). Notice (source header of radsw_main_nmmb.f): sw-rrtm3 radiation package, NCEP modifications
// of the RRTM-SW radiation code from AER Inc. Band 18 is 4000-4650 cm-1 (low: H2O, CH4; high: CH4).
// The k-tables are namgpu/data/rrtm_sw/kgb18.npz (module_radsw_kgb18_nmmb, phys/radsw_datatb_nmmb.f:7133-8634,
// identical to module_radsw_kgb18 in phys/radsw_datatb.f:7887-9388, checked by tools/extract-rrtm-sw-kgb18.py).
//
// What this kernel computes, one thread per column (i, j), all layers and all NG18 = 8 g-points:
//   taur(k)          = colmol(k) * rayl                              (Fortran 3640-3645, every layer)
//   taug(g, k), k <= laytrop : water/CH4 lower-atmosphere table     (Fortran 3650-3690)
//   taug(g, k), k >  laytrop : CH4 upper-atmosphere table           (Fortran 3714-3732)
//
// Band-local output: taug[g0, l0, j0, i0] with g0 = 0..7, i.e. the Fortran taug(:,k,NS18+j) with j = g0+1
// (NS18 = 18, global g-points 18..25). taur is the same value for every g-point of the band, so it is
// stored once as taur[l0, j0, i0] (the Fortran writes it NG18 times; storing it once saves ~1.5 GB at 12 km).
//
// Layout (CONTRACT.md, Fortran (i,j,l) -> [l, j, i]; gas index first for colamt):
//   colamt[n, l, j, i]  float32 [6, NLAY, JM, IM]   Fortran colamt(i,j,l,n), n 0-based (H2O = 0, CH4 = 4)
//   colmol[l, j, i]     float32 [NLAY, JM, IM]
//   laytrop[j, i]       int32   [JM, IM]            1-based tropopause layer
//   jp, jt, jt1[l,j,i]  int32   [NLAY, JM, IM]      1-based (setcoef outputs)
//   fac00..fac11[l,j,i] float32 [NLAY, JM, IM]
//   selffac, selffrac, forfac, forfrac [NLAY, JM, IM] float32
//   indself, indfor[l,j,i] int32 [NLAY, JM, IM]     1-based (setcoef outputs)
//   absa[585, 8], absb[235, 8], selfref[10, 8], forref[3, 8]   float32, [ind-1, g-1] (kgb18.npz)
//   taug[8, NLAY, JM, IM], taur[NLAY, JM, IM]       float32 outputs (every element written)
//
// Notes on the Fortran that are kept on purpose (see the report note in the unit wrapper):
//  - DECLARED DIVERGENCE at k == laytrop. The Fortran upper loop (radsw_main_nmmb.f:3719-3732) is WHERE k .GE.
//    laytrop with DO k = MINVAL(laytrop)+1, NLAY, so it overwrites taug at k == laytrop for every column whose
//    laytrop is above the chunk minimum. The value native keeps there depends on CHNK_RRTM (8) and on the other
//    columns in the chunk, so no single Fortran value exists per column. This port takes one branch per layer:
//    k <= laytrop -> lower table, k > laytrop -> upper table. The same choice is made in the other upper-atmosphere
//    bands (phys-rrtm-sw-taumol25, STATUS 2026-11-04T07:28Z). Claim: not a Fortran match at k == laytrop, never bitwise.
//  - Table indices are the exact Fortran indices (no clamp). Only the lower layers (k <= laytrop) read selfref
//    and forref, and setcoef guarantees there indself in 1..9 and indfor in 1..2 (radsw_main_nmmb.f:1989-2139,
//    phys_rrtm_sw_setcoef.py), so indsp <= 10 and indfp <= 3 stay inside the tables. The upper layers never read
//    them (setcoef sets indself = 0, indfor = 3 there). The wrapper checks this range (check_setcoef_ranges).
//  - The id0 / id1 indices come from rrtm_sw_taumol_index() (rrtm_sw_bands.cuh), identical to the Fortran
//    dispatcher at radsw_main_nmmb.f:3250-3257 (WHERE k .LE. laytrop: lower form, .GT.: upper form).
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"
#include "rrtm_sw_bands.cuh"

#define SW18_NG     8       // NG18 (radsw_param_nmmb.f)
#define SW18_MSA    585     // MSA18
#define SW18_MSB    235     // MSB18
#define SW18_MSF    10      // MSF18
#define SW18_MFR    3       // MFR18
#define SW18_JB     18      // Fortran band index
#define SW18_SPECMULT 8.0f  // taumol18: specmult = 8.0 * min(oneminus, ...)

extern "C" __global__ void phys_rrtm_sw_taumol18(
    const float* __restrict__ colamt,
    const float* __restrict__ colmol,
    const int*   __restrict__ laytrop,
    const int*   __restrict__ jp,
    const int*   __restrict__ jt,
    const int*   __restrict__ jt1,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int*   __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int*   __restrict__ indfor,
    const float* __restrict__ absa,
    const float* __restrict__ absb,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const float rayl,
    const float strrat,
    float* __restrict__ taug,
    float* __restrict__ taur,
    const int nlay, const int jm, const int im)
{
    COLUMN_IJ(im, jm)
    const int lt = laytrop[IDX2(i0, j0, im)];

    for (int k1 = 1; k1 <= nlay; ++k1) {
        const int l0 = k1 - 1;
        const size_t lidx = IDX3(i0, j0, l0, im, jm);

        // Fortran 3640-3645: tauray = colmol * rayl, copied to every g-point of the band.
        taur[lidx] = colmol[lidx] * rayl;

        const float fac00_k = fac00[lidx], fac01_k = fac01[lidx];
        const float fac10_k = fac10[lidx], fac11_k = fac11[lidx];
        const float h2o = colamt[IDX4(i0, j0, l0, 0, im, jm, nlay)];   // colamt(:,k,1), H2O
        const float ch4 = colamt[IDX4(i0, j0, l0, 4, im, jm, nlay)];   // colamt(:,k,5), CH4

        int id0, id1;
        rrtm_sw_taumol_index(SW18_JB, k1, lt, jp[lidx], jt[lidx], jt1[lidx], &id0, &id1);

        if (k1 <= lt) {
            // Fortran 3660-3690: lower atmosphere, water vapour + CH4 binary species, 9-point interpolation.
            const float speccomb = h2o + strrat * ch4;
            // speccomb = 0 only when H2O and CH4 are both zero. The Fortran ratio is then 0/0 (NaN). The term
            // speccomb * tab is 0 for any finite fraction and the self/foreign term is h2o * (...) = 0, so the
            // fraction is set to 0 and taug = 0 there. No NaN is produced.
            const float ratio = (speccomb > 0.0f) ? (h2o / speccomb) : 0.0f;
            const float specmult = SW18_SPECMULT * fminf(SW_ONEMINUS, ratio);
            const int   jsi = (int)specmult;                 // int(specmult), specmult >= 0
            const int   js  = 1 + jsi;
            const float fs  = specmult - (float)jsi;         // mod(specmult, f_one)
            const float fs1 = 1.0f - fs;

            const float fac000 = fs1 * fac00_k;
            const float fac010 = fs1 * fac10_k;
            const float fac100 = fs  * fac00_k;
            const float fac110 = fs  * fac10_k;
            const float fac001 = fs1 * fac01_k;
            const float fac011 = fs1 * fac11_k;
            const float fac101 = fs  * fac01_k;
            const float fac111 = fs  * fac11_k;

            const int ind01 = max(1, min(SW18_MSA, id0 + js));
            const int ind02 = min(SW18_MSA, ind01 + 1);
            const int ind03 = min(SW18_MSA, ind01 + 9);
            const int ind04 = min(SW18_MSA, ind01 + 10);
            const int ind11 = max(1, min(SW18_MSA, id1 + js));
            const int ind12 = min(SW18_MSA, ind11 + 1);
            const int ind13 = min(SW18_MSA, ind11 + 9);
            const int ind14 = min(SW18_MSA, ind11 + 10);

            const int inds = indself[lidx];         // exact Fortran index, 1..9 in lower layers (see header)
            const int indf = indfor[lidx];          // exact Fortran index, 1..2 in lower layers (see header)
            const int indsp = inds + 1;
            const int indfp = indf + 1;

            const float selffac_k = selffac[lidx], selffrac_k = selffrac[lidx];
            const float forfac_k = forfac[lidx], forfrac_k = forfrac[lidx];

            for (int g = 0; g < SW18_NG; ++g) {
                const float tab =
                      fac000 * absa[(ind01 - 1) * SW18_NG + g]
                    + fac100 * absa[(ind02 - 1) * SW18_NG + g]
                    + fac010 * absa[(ind03 - 1) * SW18_NG + g]
                    + fac110 * absa[(ind04 - 1) * SW18_NG + g]
                    + fac001 * absa[(ind11 - 1) * SW18_NG + g]
                    + fac101 * absa[(ind12 - 1) * SW18_NG + g]
                    + fac011 * absa[(ind13 - 1) * SW18_NG + g]
                    + fac111 * absa[(ind14 - 1) * SW18_NG + g];
                const float sref_lo = selfref[(inds - 1) * SW18_NG + g];
                const float sref_hi = selfref[(indsp - 1) * SW18_NG + g];
                const float fref_lo = forref[(indf - 1) * SW18_NG + g];
                const float fref_hi = forref[(indfp - 1) * SW18_NG + g];
                const float cont =
                      h2o * (selffac_k * (sref_lo + selffrac_k * (sref_hi - sref_lo))
                           + forfac_k  * (fref_lo + forfrac_k  * (fref_hi - fref_lo)));
                taug[IDX4(i0, j0, l0, g, im, jm, nlay)] = speccomb * tab + cont;
            }
        } else {
            // Fortran 3714-3732: upper atmosphere, CH4 only, 4-point interpolation in the absb table.
            const int ind01 = max(1, min(SW18_MSB, id0 + 1));
            const int ind02 = min(SW18_MSB, ind01 + 1);
            const int ind11 = max(1, min(SW18_MSB, id1 + 1));
            const int ind12 = min(SW18_MSB, ind11 + 1);

            for (int g = 0; g < SW18_NG; ++g) {
                taug[IDX4(i0, j0, l0, g, im, jm, nlay)] = ch4 *
                    (fac00_k * absb[(ind01 - 1) * SW18_NG + g]
                   + fac10_k * absb[(ind02 - 1) * SW18_NG + g]
                   + fac01_k * absb[(ind11 - 1) * SW18_NG + g]
                   + fac11_k * absb[(ind12 - 1) * SW18_NG + g]);
            }
        }
    }
}
