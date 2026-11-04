// phys_rrtm_lw_taugb12.cu -- TAUGB12: LW band 12 (1800-2080 cm-1) optical depths and Planck
// fractions for the NMMB RRTM longwave (RRTM group). One thread per (i, j) column, the layer loop
// k = 1..nlay runs inside the thread, the g-point loop ig = 1..NG12 runs inside the layer loop, and
// every term of the Fortran routine is present.
//
// NOAA notice: phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, US government work
// (no licence file; see NOTICE). The lw-rrtm3 package header of that file is kept as the notice:
// !!!!!  lw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep modifications of the rrtm-lw radiation code from aer inc.     !
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:5482-5662 (SUBROUTINE taugb12).
// The absorption table it reads is module_radlw_kgb12 from phys/radlw_datatb.f:29094-30143 (the
// radlw_datatb.f layout absa(NG12, 585), the one taugb12 indexes as absa(ig, id); extracted by
// tools/extract_rrtm_lw_kgb12.py into namgpu/data/rrtm_lw/kgb12.npz).
// The reference mixing ratio refrat_planck_a = chi_mls(1,10)/chi_mls(2,10) is computed on device
// from the chi_mls table (module_radlw_ref, the setcoef table unit).
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   colamt                                    [MAXGAS=6, nlay, JM, IM]  (Fortran colamt(k,m))
//   rfrate (Fortran rfrate(nlay,NRATES,2))    [2, NRATES=6, nlay, JM, IM]
//   jp, jt, jt1, indself, indfor, laytrop     int32 [nlay, JM, IM] (laytrop [JM, IM])
//   fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac   [nlay, JM, IM]
//   absa   (Fortran absa(NG12,585))           [585, NG12]
//   selfref (Fortran selfref(NG12,10))        [10, NG12]
//   forref  (Fortran forref(NG12,4))          [4, NG12]
//   fracrefa (Fortran fracrefa(NG12,9))       [9, NG12]
//   chi_mls (Fortran chi_mls(7,59))           [59, 7]
//   taug, fracs (Fortran taug(ngptlw,nlay))   [NGPTLW=140, nlay, JM, IM]
//
// Notes on the translation:
//   - Fortran MOD(a, f_one) for a >= 0 is fmodf(a, 1.0f).
//   - Fortran INT() truncates toward zero; the (int) casts do the same for these ranges.
//   - p0**4 is evaluated as (p0*p0)*(p0*p0), the square-then-square form used for an integer power.
//   - oneminus = 1.0 - eps with eps = 1.0e-6 in single precision (radlw_main_nmmb.f:267-268).
//   - The upper-atmosphere loop (k = laytrop+1..nlay) sets taug and fracs to zero.
//   - Per-column laytrop replaces the Fortran host scalar: the lower loop runs k = 1..laytrop of
//     the column.
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation.

#include "nmmb_common.cuh"

#define NG12_TB      8        // NG12 (radlw_param.f:108)
#define NS12_TB      122      // NS12 (radlw_param.f:116)
#define NGPT_LW      140      // NGPTLW (radlw_param.f:95)
#define NSPA12_TB    9        // nspa(12) (radlw_main_nmmb.f:284, data nspa)
#define NRATES_TB    6        // NRATES (radlw_param.f:98)
#define NGAS_REF_TB  7        // chi_mls second index (module_radlw_ref)
#define ONEMINUS_TB  (1.0f - 1.0e-6f)

// Fortran 1-based accessors on the row-major tables.
#define ABSA(id, ig)      absa[((size_t)((id) - 1)) * NG12_TB + ((ig) - 1)]        // absa(ig,id)
#define SELFREF(ii, ig)   selfref[((size_t)((ii) - 1)) * NG12_TB + ((ig) - 1)]     // selfref(ig,ii)
#define FORREF(ii, ig)    forref[((size_t)((ii) - 1)) * NG12_TB + ((ig) - 1)]      // forref(ig,ii)
#define FRACREFA(jj, ig)  fracrefa[((size_t)((jj) - 1)) * NG12_TB + ((ig) - 1)]    // fracrefa(ig,jj)
#define CHI(g, lev)       chi_mls[((size_t)((lev) - 1)) * NGAS_REF_TB + ((g) - 1)]  // chi_mls(g,lev)

// Per-column element of a [nlay, JM, IM] field at layer k (Fortran 1-based).
#define LAY(k)            IDX3(i0, j0, (k) - 1, IM, JM)
// colamt(m, k) Fortran (colamt(k,m)), stored [m, k, j, i].
#define COLAMT_IDX(m, k)  IDX4(i0, j0, (k) - 1, (m) - 1, IM, JM, nlay)
// rfrate(k, m, n) Fortran (rfrate(k,m,n)), stored [n, m, k, j, i].
#define RFRATE_IDX(m, n, k) (((((size_t)((n) - 1) * NRATES_TB + ((m) - 1)) * nlay) + ((k) - 1)) \
                             * ((size_t)JM * IM) + IDX2(i0, j0, IM))
// taug(g, k) / fracs(g, k) Fortran, stored [g, k, j, i] with g 1-based.
#define TAUG_IDX(g, k)    IDX4(i0, j0, (k) - 1, (g) - 1, IM, JM, nlay)

extern "C" __global__ void rrtm_lw_taugb12(
    const float* __restrict__ colamt,
    const float* __restrict__ rfrate,
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
    const int*   __restrict__ laytrop,
    const float* __restrict__ absa,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const float* __restrict__ fracrefa,
    const float* __restrict__ chi_mls,
    int nlay, int IM, int JM,
    float* __restrict__ taug,
    float* __restrict__ fracs)
{
    COLUMN_IJ(IM, JM)

    // ---- reference ratio used for the Planck fraction in the lower atmosphere ----
    // refrat_planck_a = chi_mls(1,10)/chi_mls(2,10)   (P = 174.164 mb)
    const float refrat_planck_a = CHI(1, 10) / CHI(2, 10);

    const int nl = laytrop[IDX2(i0, j0, IM)];

    // ---- lower atmosphere loop: k = 1..laytrop ----
    for (int k = 1; k <= nl; ++k) {
        const size_t lk = LAY(k);
        const float colamt1 = colamt[COLAMT_IDX(1, k)];
        const float colamt2 = colamt[COLAMT_IDX(2, k)];

        const float speccomb = colamt1 + rfrate[RFRATE_IDX(1, 1, k)] * colamt2;
        const float specparm = colamt1 / speccomb;
        const float specmult = 8.0f * fminf(specparm, ONEMINUS_TB);
        const int js = 1 + (int)specmult;
        const float fs = fmodf(specmult, 1.0f);
        const int ind0 = ((jp[lk] - 1) * 5 + (jt[lk] - 1)) * NSPA12_TB + js;

        const float speccomb1 = colamt1 + rfrate[RFRATE_IDX(1, 2, k)] * colamt2;
        const float specparm1 = colamt1 / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, ONEMINUS_TB);
        const int js1 = 1 + (int)specmult1;
        const float fs1 = fmodf(specmult1, 1.0f);
        const int ind1 = (jp[lk] * 5 + (jt1[lk] - 1)) * NSPA12_TB + js1;

        const float speccomb_planck = colamt1 + refrat_planck_a * colamt2;
        float specparm_planck = colamt1 / speccomb_planck;
        if (specparm_planck >= ONEMINUS_TB) specparm_planck = ONEMINUS_TB;
        const float specmult_planck = 8.0f * specparm_planck;
        const int jpl = 1 + (int)specmult_planck;
        const float fpl = fmodf(specmult_planck, 1.0f);

        const int inds = indself[lk];
        const int indf = indfor[lk];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int jplp = jpl + 1;

        float fk00, fk10, fk20, fk01, fk11, fk21;
        int id000, id010, id100, id110, id200, id210;
        int id001, id011, id101, id111, id201, id211;

        if (specparm < 0.125f && specparm1 < 0.125f) {
            const float p0 = fs - 1.0f;
            const float p0sq = p0 * p0;
            const float p40 = p0sq * p0sq;
            fk00 = p40;
            fk10 = 1.0f - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1 = fs1 - 1.0f;
            const float p1sq = p1 * p1;
            const float p41 = p1sq * p1sq;
            fk01 = p41;
            fk11 = 1.0f - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0;
            id010 = ind0 + 9;
            id100 = ind0 + 1;
            id110 = ind0 + 10;
            id200 = ind0 + 2;
            id210 = ind0 + 11;

            id001 = ind1;
            id011 = ind1 + 9;
            id101 = ind1 + 1;
            id111 = ind1 + 10;
            id201 = ind1 + 2;
            id211 = ind1 + 11;
        } else if (specparm > 0.875f && specparm1 > 0.875f) {
            const float p0 = -fs;
            const float p0sq = p0 * p0;
            const float p40 = p0sq * p0sq;
            fk00 = p40;
            fk10 = 1.0f - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1 = -fs1;
            const float p1sq = p1 * p1;
            const float p41 = p1sq * p1sq;
            fk01 = p41;
            fk11 = 1.0f - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0 + 1;
            id010 = ind0 + 10;
            id100 = ind0;
            id110 = ind0 + 9;
            id200 = ind0 - 1;
            id210 = ind0 + 8;

            id001 = ind1 + 1;
            id011 = ind1 + 10;
            id101 = ind1;
            id111 = ind1 + 9;
            id201 = ind1 - 1;
            id211 = ind1 + 8;
        } else {
            fk00 = 1.0f - fs;
            fk10 = fs;
            fk20 = 0.0f;

            fk01 = 1.0f - fs1;
            fk11 = fs1;
            fk21 = 0.0f;

            id000 = ind0;
            id010 = ind0 + 9;
            id100 = ind0 + 1;
            id110 = ind0 + 10;
            id200 = ind0;
            id210 = ind0;

            id001 = ind1;
            id011 = ind1 + 9;
            id101 = ind1 + 1;
            id111 = ind1 + 10;
            id201 = ind1;
            id211 = ind1;
        }

        const float fac000 = fk00 * fac00[lk];
        const float fac100 = fk10 * fac00[lk];
        const float fac200 = fk20 * fac00[lk];
        const float fac010 = fk00 * fac10[lk];
        const float fac110 = fk10 * fac10[lk];
        const float fac210 = fk20 * fac10[lk];

        const float fac001 = fk01 * fac01[lk];
        const float fac101 = fk11 * fac01[lk];
        const float fac201 = fk21 * fac01[lk];
        const float fac011 = fk01 * fac11[lk];
        const float fac111 = fk11 * fac11[lk];
        const float fac211 = fk21 * fac11[lk];

        for (int ig = 1; ig <= NG12_TB; ++ig) {
            const float tauself = selffac[lk] * (SELFREF(inds, ig) + selffrac[lk]
                                  * (SELFREF(indsp, ig) - SELFREF(inds, ig)));
            const float taufor = forfac[lk] * (FORREF(indf, ig) + forfrac[lk]
                                 * (FORREF(indfp, ig) - FORREF(indf, ig)));

            taug[TAUG_IDX(NS12_TB + ig, k)] =
                  speccomb * (fac000 * ABSA(id000, ig) + fac010 * ABSA(id010, ig)
                            + fac100 * ABSA(id100, ig) + fac110 * ABSA(id110, ig)
                            + fac200 * ABSA(id200, ig) + fac210 * ABSA(id210, ig))
                + speccomb1 * (fac001 * ABSA(id001, ig) + fac011 * ABSA(id011, ig)
                            + fac101 * ABSA(id101, ig) + fac111 * ABSA(id111, ig)
                            + fac201 * ABSA(id201, ig) + fac211 * ABSA(id211, ig))
                + tauself + taufor;

            fracs[TAUG_IDX(NS12_TB + ig, k)] = FRACREFA(jpl, ig)
                  + fpl * (FRACREFA(jplp, ig) - FRACREFA(jpl, ig));
        }
    }

    // ---- upper atmosphere loop: k = laytrop+1..nlay ----
    for (int k = nl + 1; k <= nlay; ++k) {
        for (int ig = 1; ig <= NG12_TB; ++ig) {
            taug[TAUG_IDX(NS12_TB + ig, k)] = 0.0f;
            fracs[TAUG_IDX(NS12_TB + ig, k)] = 0.0f;
        }
    }
}
