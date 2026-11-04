// phys_rrtm_lw_taugb13.cu -- TAUGB13: LW band 13 (2080-2250 cm-1) optical depths and Planck
// fractions for the NMMB RRTM longwave (RRTM group). One thread per (i, j) column, the layer loops
// run inside the thread, every term of the Fortran routine is present.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOAA notice: phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1), NOAA/NCEP NAM, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE
// (no licence file; see NOTICE). The lw-rrtm3 package header of that file is kept as the notice:
// !!!!!  lw-rrtm3 radiation package description                                              !!!!!
// !   this package includes ncep modifications of the rrtm-lw radiation code from aer inc.     !
//
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:5666-5908 (SUBROUTINE taugb13).
// Tables: module_radlw_kgb13 (phys/radlw_datatb.f, the module taugb13 USEs), extracted to
// namgpu/data/rrtm_lw/kgb13.npz by tools/extract-rrtm-lw-kgb13.py and uploaded once by the wrapper.
// chi_mls (module_radlw_ref) is passed in from the setcoef tables.
//
// Index conventions (CONTRACT.md section 1, Fortran A(i,j,l) -> C [l, j, i]):
//   laytrop                                           [JM, IM] int32
//   jp, jt, jt1, indself, indfor, indminor            [nlay, JM, IM] int32
//   coldry, fac00..fac11, selffac, selffrac, forfac, forfrac, minorfrac   [nlay, JM, IM]
//   colamt (Fortran colamt(nlay,maxgas))              [7, nlay, JM, IM]  (maxgas = 7, radlw_param.f:97, the module radlw_main_nmmb.f uses)
//   rfrate (Fortran rfrate(nlay,NRATES,2))            [2, 6, nlay, JM, IM]
//   chi_mls (Fortran chi_mls(7,59))                   [59, 7], chi_mls[(lev-1)*7 + (gas-1)]
//   taug13, fracs13 (band-local rows ns13+1..ns13+4)  [4, nlay, JM, IM]; the wrapper scatters them
//   k-tables, C shapes from the extract tool (g-point index fastest, as in the Fortran DATA order):
//     absa (585,4): absa(ig,id) = absa[(id-1)*4 + (ig-1)]; selfref (10,4); forref (4,4);
//     fracrefa (9,4); fracrefb (4); ka_mco2 (19,9,4): ka_mco2(ig,j,m) = ka[((m-1)*9+(j-1))*4+(ig-1)];
//     ka_mco (19,9,4); kb_mo3 (19,4).
//
// Notes on the translation:
//   - taug(ns13+ig,k) and fracs(ns13+ig,k) go to band-local row ig-1 (ns13 = 130, radlw_param.f:116).
//     The wrapper scatter_band13 places rows 130..133 into the full gpoint arrays. The lower loop
//     runs k = 1..laytrop and the upper loop k = laytrop+1..nlay, so every layer is written.
//   - colamt(k,7) is the CO column (Fortran term colamt(k,7)*absco). radlw_main_nmmb.f writes it at
//     :711 (ilwrgas > 0: max(0, coldry*gasvmr(co))) or :723 (0), and taugb13 reads it back. It is carried
//     as plane 7 of colamt; no zero substitution.
//   - INT() truncates toward zero (C cast). MOD(a,1) is fmodf(a,1). X**4 is (x*x)*(x*x).
//   - Single precision literals (f suffix) throughout, as kfpt = single.
//   - oneminus = 1.0 - eps with eps = 1.0e-6 (radlw_main_nmmb.f:267-268).
//
// Fast numerics: fmad on (NVRTC option), no host sync, no device allocation.

#include "nmmb_common.cuh"

#define NG13_K      4
#define NSPA13      9
#define NRATES_K    6

extern "C" __global__ void rrtm_lw_taugb13(
    const int* __restrict__ laytrop,
    const float* __restrict__ colamt,
    const float* __restrict__ coldry,
    const int* __restrict__ jp,
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const float* __restrict__ rfrate,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const int* __restrict__ indself,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const int* __restrict__ indfor,
    const float* __restrict__ minorfrac,
    const int* __restrict__ indminor,
    const float* __restrict__ chi_mls,
    const float* __restrict__ absa,
    const float* __restrict__ selfref,
    const float* __restrict__ forref,
    const float* __restrict__ fracrefa,
    const float* __restrict__ fracrefb,
    const float* __restrict__ ka_mco2,
    const float* __restrict__ ka_mco,
    const float* __restrict__ kb_mo3,
    int nlay, int IM, int JM,
    float* __restrict__ taug13,
    float* __restrict__ fracs13)
{
    COLUMN_IJ(IM, JM)

    const size_t plane = (size_t)JM * IM;
    const size_t off = (size_t)j0 * IM + i0;
    // [nlay, JM, IM] element at 0-based layer l0
    #define L3(l0)        ((size_t)(l0) * plane + off)
    // [N, nlay, JM, IM] element at 0-based leading index n0 and layer l0
    #define L4(n0, l0)    ((((size_t)(n0)) * nlay + (size_t)(l0)) * plane + off)
    // rfrate(k, m, n) Fortran, stored [2, 6, nlay, JM, IM]
    #define RF(k, m, n)   (((((size_t)((n) - 1)) * NRATES_K + (size_t)((m) - 1)) * nlay + (size_t)((k) - 1)) * plane + off)
    // colamt(k, m) Fortran, stored [6, nlay, JM, IM]
    #define COL(k, m)     L4((m) - 1, (k) - 1)
    // taug / fracs band-local [4, nlay, JM, IM]: Fortran taug(ns13+ig, k), row ig-1
    #define TG(ig, k)     L4((ig) - 1, (k) - 1)
    // k-table accessors (Fortran 1-based indices, C layouts above)
    #define ABSA(ig, id)     absa[((size_t)((id) - 1)) * NG13_K + ((ig) - 1)]
    #define SELFREF(ig, m)   selfref[((size_t)((m) - 1)) * NG13_K + ((ig) - 1)]
    #define FORREF(ig, m)    forref[((size_t)((m) - 1)) * NG13_K + ((ig) - 1)]
    #define FRACREFA(ig, m)  fracrefa[((size_t)((m) - 1)) * NG13_K + ((ig) - 1)]
    #define FRACREFB(ig)     fracrefb[(ig) - 1]
    #define KA_MCO2(ig, j, m) ka_mco2[(((size_t)((m) - 1)) * 9 + ((j) - 1)) * NG13_K + ((ig) - 1)]
    #define KA_MCO(ig, j, m)  ka_mco[(((size_t)((m) - 1)) * 9 + ((j) - 1)) * NG13_K + ((ig) - 1)]
    #define KB_MO3(ig, m)     kb_mo3[((size_t)((m) - 1)) * NG13_K + ((ig) - 1)]
    // chi_mls(g, lev) Fortran, stored [59, 7]
    #define CHI(g, lev)       chi_mls[((size_t)((lev) - 1)) * 7 + ((g) - 1)]

    const int nl = laytrop[IDX2(i0, j0, IM)];

    const float eps = 1.0e-6f;
    const float oneminus = 1.0f - eps;
    const float f_one = 1.0f;
    const float f_zero = 0.0f;

    // reference ratios for the Planck fraction and the minor-gas mapping levels
    const float refrat_planck_a = CHI(1, 5) / CHI(4, 5);   // P = 473.420 mb (level 5)
    const float refrat_m_a = CHI(1, 1) / CHI(4, 1);        // P = 1053. mb   (level 1)
    const float refrat_m_a3 = CHI(1, 3) / CHI(4, 3);       // P = 706. mb    (level 3)

    // ---- lower atmosphere: k = 1..laytrop ----
    for (int k = 1; k <= nl; ++k) {
        const int kz = k - 1;
        const float colamt1 = colamt[COL(k, 1)];
        const float colamt2 = colamt[COL(k, 2)];
        const float colamt4 = colamt[COL(k, 4)];
        const float colamt7 = colamt[COL(k, 7)];   // CO column, radlw_main_nmmb.f:711/723

        const float speccomb = colamt1 + rfrate[RF(k, 3, 1)] * colamt4;
        const float specparm = colamt1 / speccomb;
        const float specmult = 8.0f * fminf(specparm, oneminus);
        const int js = 1 + (int)specmult;
        const float fs = fmodf(specmult, f_one);
        const int ind0 = ((jp[L3(kz)] - 1) * 5 + (jt[L3(kz)] - 1)) * NSPA13 + js;

        const float speccomb1 = colamt1 + rfrate[RF(k, 3, 2)] * colamt4;
        const float specparm1 = colamt1 / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, oneminus);
        const int js1 = 1 + (int)specmult1;
        const float fs1 = fmodf(specmult1, f_one);
        const int ind1 = (jp[L3(kz)] * 5 + (jt1[L3(kz)] - 1)) * NSPA13 + js1;

        const float speccomb_mco2 = colamt1 + refrat_m_a * colamt4;
        const float specparm_mco2 = colamt1 / speccomb_mco2;
        const float specmult_mco2 = 8.0f * fminf(specparm_mco2, oneminus);
        const int jmco2 = 1 + (int)specmult_mco2;
        const float fmco2 = fmodf(specmult_mco2, f_one);

        // minor gas CO2 adjustment in atmospheres where CO2 is too great to be a minor species
        const float speccomb_mco = colamt1 + refrat_m_a3 * colamt4;
        const float specparm_mco = colamt1 / speccomb_mco;
        const float specmult_mco = 8.0f * fminf(specparm_mco, oneminus);
        const int jmco = 1 + (int)specmult_mco;
        const float fmco = fmodf(specmult_mco, f_one);

        const float speccomb_planck = colamt1 + refrat_planck_a * colamt4;
        const float specparm_planck = colamt1 / speccomb_planck;
        const float specmult_planck = 8.0f * fminf(specparm_planck, oneminus);
        const int jpl = 1 + (int)specmult_planck;
        const float fpl = fmodf(specmult_planck, f_one);

        const int inds = indself[L3(kz)];
        const int indf = indfor[L3(kz)];
        const int indm = indminor[L3(kz)];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;
        const int jplp = jpl + 1;
        const int jmco2p = jmco2 + 1;
        const int jmcop = jmco + 1;

        // CO2 too great to be a minor species: empirical adjustment of the column amount
        const float temp = coldry[L3(kz)] * 3.55e-4f;
        const float ratco2 = colamt2 / temp;
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 2.0f + powf(ratco2 - 2.0f, 0.68f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = colamt2;
        }

        float fk00, fk10, fk20, fk01, fk11, fk21;
        int id000, id010, id100, id110, id200, id210;
        int id001, id011, id101, id111, id201, id211;
        if (specparm < 0.125f && specparm1 < 0.125f) {
            const float p0 = fs - f_one;
            const float p2 = p0 * p0;
            const float p40 = p2 * p2;
            fk00 = p40;
            fk10 = f_one - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1 = fs1 - f_one;
            const float p21 = p1 * p1;
            const float p41 = p21 * p21;
            fk01 = p41;
            fk11 = f_one - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0;  id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10; id200 = ind0 + 2;  id210 = ind0 + 11;
            id001 = ind1;  id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10; id201 = ind1 + 2;  id211 = ind1 + 11;
        } else if (specparm > 0.875f && specparm1 > 0.875f) {
            const float p0 = -fs;
            const float p2 = p0 * p0;
            const float p40 = p2 * p2;
            fk00 = p40;
            fk10 = f_one - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1 = -fs1;
            const float p21 = p1 * p1;
            const float p41 = p21 * p21;
            fk01 = p41;
            fk11 = f_one - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0 + 1;  id010 = ind0 + 10; id100 = ind0;      id110 = ind0 + 9;  id200 = ind0 - 1;  id210 = ind0 + 8;
            id001 = ind1 + 1;  id011 = ind1 + 10; id101 = ind1;      id111 = ind1 + 9;  id201 = ind1 - 1;  id211 = ind1 + 8;
        } else {
            fk00 = f_one - fs;
            fk10 = fs;
            fk20 = f_zero;

            fk01 = f_one - fs1;
            fk11 = fs1;
            fk21 = f_zero;

            id000 = ind0;  id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10; id200 = ind0;  id210 = ind0;
            id001 = ind1;  id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10; id201 = ind1;  id211 = ind1;
        }

        const float f00 = fac00[L3(kz)];
        const float f01 = fac01[L3(kz)];
        const float f10 = fac10[L3(kz)];
        const float f11 = fac11[L3(kz)];
        const float fac000 = fk00 * f00;
        const float fac100 = fk10 * f00;
        const float fac200 = fk20 * f00;
        const float fac010 = fk00 * f10;
        const float fac110 = fk10 * f10;
        const float fac210 = fk20 * f10;

        const float fac001 = fk01 * f01;
        const float fac101 = fk11 * f01;
        const float fac201 = fk21 * f01;
        const float fac011 = fk01 * f11;
        const float fac111 = fk11 * f11;
        const float fac211 = fk21 * f11;

        const float selffac_k = selffac[L3(kz)];
        const float selffrac_k = selffrac[L3(kz)];
        const float forfac_k = forfac[L3(kz)];
        const float forfrac_k = forfrac[L3(kz)];
        const float minorfrac_k = minorfrac[L3(kz)];

        for (int ig = 1; ig <= NG13_K; ++ig) {
            const float tauself = selffac_k * (SELFREF(ig, inds) + selffrac_k
                                  * (SELFREF(ig, indsp) - SELFREF(ig, inds)));
            const float taufor = forfac_k * (FORREF(ig, indf) + forfrac_k
                                 * (FORREF(ig, indfp) - FORREF(ig, indf)));
            const float co2m1 = KA_MCO2(ig, jmco2, indm) + fmco2
                                * (KA_MCO2(ig, jmco2p, indm) - KA_MCO2(ig, jmco2, indm));
            const float co2m2 = KA_MCO2(ig, jmco2, indmp) + fmco2
                                * (KA_MCO2(ig, jmco2p, indmp) - KA_MCO2(ig, jmco2, indmp));
            const float absco2 = co2m1 + minorfrac_k * (co2m2 - co2m1);
            const float com1 = KA_MCO(ig, jmco, indm) + fmco
                               * (KA_MCO(ig, jmcop, indm) - KA_MCO(ig, jmco, indm));
            const float com2 = KA_MCO(ig, jmco, indmp) + fmco
                               * (KA_MCO(ig, jmcop, indmp) - KA_MCO(ig, jmco, indmp));
            const float absco = com1 + minorfrac_k * (com2 - com1);

            taug13[TG(ig, k)] = speccomb
                    * (fac000 * ABSA(ig, id000) + fac010 * ABSA(ig, id010)
                     + fac100 * ABSA(ig, id100) + fac110 * ABSA(ig, id110)
                     + fac200 * ABSA(ig, id200) + fac210 * ABSA(ig, id210))
                    + speccomb1
                    * (fac001 * ABSA(ig, id001) + fac011 * ABSA(ig, id011)
                     + fac101 * ABSA(ig, id101) + fac111 * ABSA(ig, id111)
                     + fac201 * ABSA(ig, id201) + fac211 * ABSA(ig, id211))
                    + tauself + taufor + adjcolco2 * absco2
                    + colamt7 * absco;

            fracs13[TG(ig, k)] = FRACREFA(ig, jpl) + fpl * (FRACREFA(ig, jplp) - FRACREFA(ig, jpl));
        }
    }

    // ---- upper atmosphere: k = laytrop+1 .. nlay ----
    for (int k = nl + 1; k <= nlay; ++k) {
        const int kz = k - 1;
        const int indm = indminor[L3(kz)];
        const int indmp = indm + 1;
        const float colamt3 = colamt[COL(k, 3)];
        const float minorfrac_k = minorfrac[L3(kz)];

        for (int ig = 1; ig <= NG13_K; ++ig) {
            const float abso3 = KB_MO3(ig, indm) + minorfrac_k * (KB_MO3(ig, indmp) - KB_MO3(ig, indm));
            taug13[TG(ig, k)] = colamt3 * abso3;
            fracs13[TG(ig, k)] = FRACREFB(ig);
        }
    }

    #undef L3
    #undef L4
    #undef RF
    #undef COL
    #undef TG
    #undef ABSA
    #undef SELFREF
    #undef FORREF
    #undef FRACREFA
    #undef FRACREFB
    #undef KA_MCO2
    #undef KA_MCO
    #undef KB_MO3
    #undef CHI
}
