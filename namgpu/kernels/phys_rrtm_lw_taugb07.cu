// phys_rrtm_lw_taugb07.cu -- unit phys-rrtm-lw-taugb07: LW band 7 optical depths and Planck fractions, nam-gpu.
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). Header of
//   phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), kept verbatim as the notice:
// |  !!!!!  ==============================================================  !!!!!
// |  !!!!!               lw-rrtm3 radiation package description             !!!!!
// |  !!!!!  ==============================================================  !!!!!
// |  !                                                                          !
// |  !   this package includes ncep's modifications of the rrtm-lw radiation    !
// |  !   code from aer inc.                                                     !
// |  !                                                                          !
// |  !    the lw-rrtm3 package includes these parts:                            !
// |  !                                                                          !
// |  !       'radlw_rrtm3_param.f'                                              !
// |  !       'radlw_rrtm3_datatb.f'                                             !
// |  !       'radlw_rrtm3_main.f'                                               !
// |  !                                                                          !
// |  !    the 'radlw_rrtm3_param.f' contains:                                   !
// |  !                                                                          !
// |  !       'module_radlw_parameters'  -- band parameters set up               !
// |  !                                                                          !
// |  !    the 'radlw_rrtm3_datatb.f' contains:                                  !
// |  !                                                                          !
// |  !       'module_radlw_avplank'     -- plank flux data                      !
// |  !       'module_radlw_ref'         -- reference temperature and pressure   !
// |  !       'module_radlw_cldprlw'     -- cloud property coefficients          !
// |  !       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
// |  !                                     bands, where nn = 01-16              !
// |  !                                                                          !
// |  !    the 'radlw_rrtm3_main.f' contains:                                    !
// |  !                                                                          !
// |  !       'module_radlw_main'        -- main lw radiation transfer           !
// |  !                                                                          !
// |  !    in the main module 'module_radlw_main' there are only two             !
// |  !    externally callable subroutines:                                      !
// |  !                                                                          !
// |  !                                                                          !
// |  !       'lwrad'     -- main lw radiation routine                           !
// |  !          inputs:                                                         !
// |  !           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
// |  !            clouds,icseed,aerosols,sfemis,sfgtmp,                         !
// |  !            npts, nlay, nlp1, lprnt,                                      !
// |  !          outputs:                                                        !
// |  !            hlwc,topflx,sfcflx,                                           !
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:4687-4942 (nam.v4.2.11, 83df4e13c1).
// The Fortran reads module_radlw_kgb07 (phys/radlw_datatb.f:21919-24540): namgpu/data/rrtm_lw/kgb07.npz.

// Fortran routine (radlw_main_nmmb.f:4687-4942), band 7 = 980-1080 cm-1 (low key h2o,o3; low minor co2;
// high key o3; high minor co2). Subroutine taugb07 has no arguments; its inputs are the per-column
// taumol state and it writes taug(ns07+ig,k) and fracs(ns07+ig,k) for ig = 1..NG07.
//
// Layout (per-column scratch, see namgpu/physics/rrtm/phys_rrtm_lw_taugb07.py):
//   ncol = IM*JM columns, col = j0*IM + i0. Per-column arrays [ncol, nlay] (layer k: index k0 = k-1).
//   Fortran 1-based integers (jp, jt, jt1, ind*, laytrop) are kept 1-based so every index expression
//   reads like the Fortran; array accesses subtract 1.
//   Band-local outputs: taug7[col, k0, ig0] = taug(ns07+ig, k), ig0 = ig-1, shape [ncol, nlay, 12].
//   The assembler (phys-rrtm-lw-taumol) places band 7 at the Fortran offset NS07 = 76
//   (radlw_param_nmmb.f:161) of the 140 g-point axis. The offset is not applied here, so no
//   140-wide array is needed per column.
//   Tables (C order of the Fortran dims reversed, Fortran A(ig,m) -> [m-1, ig-1]):
//     absa [595,12] (585 Fortran rows + zero pad), absb [236,12] (235 + zero pad), selfref [10,12],
//     forref [4,12], fracrefa [9,12], fracrefb [12],
//     ka_mco2 [19,9,12], kb_mco2 [19,12]; chi_mls [59,7] (module_radlw_ref, chi_mls(m,p) -> [p-1, m-1]).
//
// Fortran-to-CUDA notes:
//   - one thread per column; the layer loops run inside the thread (CONTRACT section 3, column kernels).
//   - MOD(a, f_one) -> fmodf(a, 1.0f): same value for the non-negative specmult (truncation semantics).
//   - (x)**4 is computed as (x*x)*(x*x).
//   - (x)**0.79 -> powf(x, 0.79f).
//   - oneminus = 1.0 - eps with eps = 1.0e-6 (single precision), as the Fortran parameter.
//   - No term is omitted: the 6 upper-layer o3 g-point scalings (taug ns07+6..11) are applied as
//     g0 = 5..10 with 0.92, 0.88, 1.07, 1.1, 0.99, 0.855.

#include "nmmb_common.cuh"

#define NG07        12      // NG07 (radlw_param_nmmb.f:153-158 and module_radlw_parameters)
#define MAF07       9       // fracrefa second dimension, ka_mco2 second dimension
#define NSPA7       9       // nspa(7) (radlw_main_nmmb.f data nspa)
#define NSPB7       1       // nspb(7) (radlw_main_nmmb.f data nspb)
#define NCHI        7       // chi_mls first dimension (module_radlw_ref)
#define ONEMINUS    (1.0f - 1.0e-6f)   // oneminus (radlw_main_nmmb.f:268)

extern "C" __global__ void phys_rrtm_lw_taugb07(
    const float* __restrict__ colamt_h2o,      // [ncol, nlay]  colamt(k,1)
    const float* __restrict__ colamt_co2,      // [ncol, nlay]  colamt(k,2)
    const float* __restrict__ colamt_o3,       // [ncol, nlay]  colamt(k,3)
    const float* __restrict__ coldry,          // [ncol, nlay]
    const float* __restrict__ rfrate_h2oo3,    // [ncol, nlay, 2]  rfrate(k,2,n), n = 1, 2
    const int*   __restrict__ jp,              // [ncol, nlay]  1-based reference pressure index
    const int*   __restrict__ jt,              // [ncol, nlay]  1-based temperature index
    const int*   __restrict__ jt1,             // [ncol, nlay]  1-based temperature index (second)
    const int*   __restrict__ indself,         // [ncol, nlay]
    const int*   __restrict__ indfor,          // [ncol, nlay]
    const int*   __restrict__ indminor,        // [ncol, nlay]
    const float* __restrict__ selffac,         // [ncol, nlay]
    const float* __restrict__ selffrac,        // [ncol, nlay]
    const float* __restrict__ forfac,          // [ncol, nlay]
    const float* __restrict__ forfrac,         // [ncol, nlay]
    const float* __restrict__ minorfrac,       // [ncol, nlay]
    const float* __restrict__ fac00,           // [ncol, nlay]
    const float* __restrict__ fac01,           // [ncol, nlay]
    const float* __restrict__ fac10,           // [ncol, nlay]
    const float* __restrict__ fac11,           // [ncol, nlay]
    const int*   __restrict__ laytrop,         // [ncol]  number of lower-atmosphere layers
    const float* __restrict__ chi_mls,         // [59, 7]
    const float* __restrict__ absa,            // [595, 12]  (585 Fortran rows, zero pad)
    const float* __restrict__ absb,            // [236, 12]  (235 Fortran rows, zero pad)
    const float* __restrict__ selfref,         // [10, 12]
    const float* __restrict__ forref,          // [4, 12]
    const float* __restrict__ fracrefa,        // [9, 12]
    const float* __restrict__ fracrefb,        // [12]
    const float* __restrict__ ka_mco2,         // [19, 9, 12]
    const float* __restrict__ kb_mco2,         // [19, 12]
    float* __restrict__ taug7,                 // [ncol, nlay, 12]  output
    float* __restrict__ fracs7,                // [ncol, nlay, 12]  output
    int ncol,
    int nlay)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    // chi_mls(m, p) with 1-based m, p
#define CHI(m, p)       chi_mls[((size_t)((p) - 1)) * NCHI + ((m) - 1)]
// Fortran first index is the g-interval ig (1..12); second index is the table index (1-based)
#define ABSA(ig, id)    absa[((size_t)((id) - 1)) * NG07 + ((ig) - 1)]
#define ABSB(ig, id)    absb[((size_t)((id) - 1)) * NG07 + ((ig) - 1)]
#define SELFREF(ig, m)  selfref[((size_t)((m) - 1)) * NG07 + ((ig) - 1)]
#define FORREF(ig, m)   forref[((size_t)((m) - 1)) * NG07 + ((ig) - 1)]
#define FRACREFA(ig, m) fracrefa[((size_t)((m) - 1)) * NG07 + ((ig) - 1)]
#define KA_MCO2(ig, m, n) ka_mco2[(((size_t)((n) - 1)) * MAF07 + ((m) - 1)) * NG07 + ((ig) - 1)]
#define KB_MCO2(ig, n)    kb_mco2[((size_t)((n) - 1)) * NG07 + ((ig) - 1)]
#define FRACREFB(ig)      fracrefb[(ig) - 1]

    const size_t cb = (size_t)col * nlay;          // column base into [ncol, nlay] arrays
    const size_t ob = (size_t)col * nlay * NG07;   // column base into [ncol, nlay, 12] outputs
    const int lt = laytrop[col];

    // reference ratios (taugb07 lines 4700-4730): constant per column
    const float refrat_planck_a = CHI(1, 3) / CHI(3, 3);   // P = 706.2620 mb
    const float refrat_m_a      = CHI(1, 3) / CHI(3, 3);   // P = 706.2720 mb

    // ---- lower atmosphere loop: k = 1 .. laytrop ----
    for (int k = 1; k <= lt; ++k) {
        const size_t q = cb + (size_t)(k - 1);
        const int jpk  = jp[q];
        const int jtk  = jt[q];
        const int jt1k = jt1[q];
        const float c1 = colamt_h2o[q];
        const float c2 = colamt_co2[q];
        const float c3 = colamt_o3[q];
        const float r21 = rfrate_h2oo3[2 * q + 0];   // rfrate(k,2,1)
        const float r22 = rfrate_h2oo3[2 * q + 1];   // rfrate(k,2,2)

        // first binary species combination
        const float speccomb = c1 + r21 * c3;
        const float specparm = c1 / speccomb;
        const float specmult = 8.0f * fminf(specparm, ONEMINUS);
        const int   js = 1 + (int)specmult;
        const float fs = fmodf(specmult, 1.0f);
        const int   ind0 = ((jpk - 1) * 5 + (jtk - 1)) * NSPA7 + js;

        // second binary species combination
        const float speccomb1 = c1 + r22 * c3;
        const float specparm1 = c1 / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, ONEMINUS);
        const int   js1 = 1 + (int)specmult1;
        const float fs1 = fmodf(specmult1, 1.0f);
        // ind1 uses jp, not jp-1, as the Fortran taugb07 does. Lower layers have jp <= 12 (wrapper
        // docstring), so the largest id is ((12*5+4)*9 + 8) + 10 = 594: inside absa's 595-row pad.
        const int   ind1 = (jpk * 5 + (jt1k - 1)) * NSPA7 + js1;

        // minor-gas (co2) binary combination
        const float speccomb_mco2 = c1 + refrat_m_a * c3;
        const float specparm_mco2 = c1 / speccomb_mco2;
        const float specmult_mco2 = 8.0f * fminf(specparm_mco2, ONEMINUS);
        const int   jmco2 = 1 + (int)specmult_mco2;
        const float fmco2 = fmodf(specmult_mco2, 1.0f);

        // Planck fraction combination
        const float speccomb_planck = c1 + refrat_planck_a * c3;
        const float specparm_planck = c1 / speccomb_planck;
        const float specmult_planck = 8.0f * fminf(specparm_planck, ONEMINUS);
        const int   jpl = 1 + (int)specmult_planck;
        const float fpl = fmodf(specmult_planck, 1.0f);

        const int inds  = indself[q];
        const int indf  = indfor[q];
        const int indm  = indminor[q];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;
        const int jplp  = jpl + 1;
        const int jmco2p = jmco2 + 1;

        // in atmospheres where CO2 is too great to be a minor species, adjust the CO2 column
        const float temp   = coldry[q] * CHI(2, jpk + 1);
        const float ratco2 = c2 / temp;
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 3.0f + powf(ratco2 - 3.0f, 0.79f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = c2;
        }

        float fk00, fk10, fk20, fk01, fk11, fk21;
        int id000, id010, id100, id110, id200, id210;
        int id001, id011, id101, id111, id201, id211;

        if (specparm < 0.125f && specparm1 < 0.125f) {
            const float p0  = fs - 1.0f;
            const float p40 = (p0 * p0) * (p0 * p0);
            fk00 = p40;
            fk10 = 1.0f - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1  = fs1 - 1.0f;
            const float p41 = (p1 * p1) * (p1 * p1);
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
            const float p0  = -fs;
            const float p40 = (p0 * p0) * (p0 * p0);
            fk00 = p40;
            fk10 = 1.0f - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1  = -fs1;
            const float p41 = (p1 * p1) * (p1 * p1);
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

        const float fac000 = fk00 * fac00[q];
        const float fac100 = fk10 * fac00[q];
        const float fac200 = fk20 * fac00[q];
        const float fac010 = fk00 * fac10[q];
        const float fac110 = fk10 * fac10[q];
        const float fac210 = fk20 * fac10[q];

        const float fac001 = fk01 * fac01[q];
        const float fac101 = fk11 * fac01[q];
        const float fac201 = fk21 * fac01[q];
        const float fac011 = fk01 * fac11[q];
        const float fac111 = fk11 * fac11[q];
        const float fac211 = fk21 * fac11[q];

        const float sf = selffac[q], sfr = selffrac[q];
        const float ff = forfac[q],  ffr = forfrac[q];
        const float mf = minorfrac[q];
        const size_t ok = ob + (size_t)(k - 1) * NG07;   // taug7/fracs7 row for layer k

        for (int ig = 1; ig <= NG07; ++ig) {
            const float tauself = sf * (SELFREF(ig, inds) + sfr * (SELFREF(ig, indsp) - SELFREF(ig, inds)));
            const float taufor  = ff * (FORREF(ig, indf) + ffr * (FORREF(ig, indfp) - FORREF(ig, indf)));
            const float co2m1  = KA_MCO2(ig, jmco2, indm) + fmco2 * (KA_MCO2(ig, jmco2p, indm) - KA_MCO2(ig, jmco2, indm));
            const float co2m2  = KA_MCO2(ig, jmco2, indmp) + fmco2 * (KA_MCO2(ig, jmco2p, indmp) - KA_MCO2(ig, jmco2, indmp));
            const float absco2 = co2m1 + mf * (co2m2 - co2m1);

            taug7[ok + (ig - 1)] = speccomb * (fac000 * ABSA(ig, id000) + fac010 * ABSA(ig, id010)
                                             + fac100 * ABSA(ig, id100) + fac110 * ABSA(ig, id110)
                                             + fac200 * ABSA(ig, id200) + fac210 * ABSA(ig, id210))
                                 + speccomb1 * (fac001 * ABSA(ig, id001) + fac011 * ABSA(ig, id011)
                                             + fac101 * ABSA(ig, id101) + fac111 * ABSA(ig, id111)
                                             + fac201 * ABSA(ig, id201) + fac211 * ABSA(ig, id211))
                                 + tauself + taufor + adjcolco2 * absco2;

            fracs7[ok + (ig - 1)] = FRACREFA(ig, jpl) + fpl * (FRACREFA(ig, jplp) - FRACREFA(ig, jpl));
        }
    }

    // ---- upper atmosphere loop: k = laytrop+1 .. nlay ----
    for (int k = lt + 1; k <= nlay; ++k) {
        const size_t q = cb + (size_t)(k - 1);
        const int jpk  = jp[q];
        const int jtk  = jt[q];
        const int jt1k = jt1[q];
        const float c2 = colamt_co2[q];
        const float c3 = colamt_o3[q];

        // CO2 adjustment, upper-atmosphere form (base 2.0)
        const float temp   = coldry[q] * CHI(2, jpk + 1);
        const float ratco2 = c2 / temp;
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 2.0f + powf(ratco2 - 2.0f, 0.79f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = c2;
        }

        const int ind0 = ((jpk - 13) * 5 + (jtk - 1)) * NSPB7 + 1;
        // Upper layers have jp in 13..58 (setcoef clamp). ind1p reaches 236 at jp = 58, jt1 = 5:
        // inside absb's 236-row pad.
        const int ind1 = ((jpk - 12) * 5 + (jt1k - 1)) * NSPB7 + 1;
        const int indm = indminor[q];
        const int indmp = indm + 1;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;

        const float f00 = fac00[q], f10 = fac10[q], f01 = fac01[q], f11 = fac11[q];
        const float mf = minorfrac[q];
        const size_t ok = ob + (size_t)(k - 1) * NG07;

        for (int ig = 1; ig <= NG07; ++ig) {
            const float absco2 = KB_MCO2(ig, indm) + mf * (KB_MCO2(ig, indmp) - KB_MCO2(ig, indm));

            taug7[ok + (ig - 1)] = c3 * (f00 * ABSB(ig, ind0) + f10 * ABSB(ig, ind0p)
                                       + f01 * ABSB(ig, ind1) + f11 * ABSB(ig, ind1p))
                                 + adjcolco2 * absco2;

            fracs7[ok + (ig - 1)] = FRACREFB(ig);
        }

        // empirical o3 stratospheric-cooling modification (taug ns07+6 .. ns07+11), band-local g0 = 5..10
        taug7[ok + 5]  = taug7[ok + 5]  * 0.92f;
        taug7[ok + 6]  = taug7[ok + 6]  * 0.88f;
        taug7[ok + 7]  = taug7[ok + 7]  * 1.07f;
        taug7[ok + 8]  = taug7[ok + 8]  * 1.1f;
        taug7[ok + 9]  = taug7[ok + 9]  * 0.99f;
        taug7[ok + 10] = taug7[ok + 10] * 0.855f;
    }

#undef CHI
#undef ABSA
#undef ABSB
#undef SELFREF
#undef FORREF
#undef FRACREFA
#undef KA_MCO2
#undef KB_MCO2
#undef FRACREFB
}
