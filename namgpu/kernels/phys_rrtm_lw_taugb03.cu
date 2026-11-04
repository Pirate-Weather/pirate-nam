// phys_rrtm_lw_taugb03.cu -- unit phys-rrtm-lw-taugb03: TAUGB03, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA/NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone), module
//   MODULE_RADLW_MAIN and the band-3 k-table module (radlw_main_nmmb.f, radlw_datatb.f, nam.v4.2.11).
//   Author header retained in the NAM clone.
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:3792-4094 (nam.v4.2.11, 83df4e13c1).
//
// What it computes: LW band 3 (500-630 cm-1) optical depths taug and Planck fractions fracs for every
// g-point of the band (NG03 = 16), for every layer of one column. Lower atmosphere (k = 1..laytrop)
// uses the 8-point eta grid and the minor-gas mapping at 706 mb; upper atmosphere (k = laytrop+1..nlay)
// uses the 4-point grid and the 95.58 mb mapping. Both branches are reproduced term by term below.
//
// Layout (CONTRACT section 1/3): one thread per column (i0, j0), a serial l-loop over layers inside the
// thread, as the Fortran column loop. nlay is LMK = LM + 1 (LTP = 1, grrad_nmmb.f:210): the extra top
// layer is computed, so every per-layer array the caller passes has nlay layers, not LM.
// Fortran 1-based indices are kept in the code and converted to
// zero-based C offsets at each access. Outputs are the 16 band-3 g-point rows only:
//   taug3[ig0, l0, j0, i0]  = taug(NS03 + ig, k)   with NS03 = 22 (1-based row NS03+ig; ig0 = ig-1)
//   fracs3[ig0, l0, j0, i0] = fracs(NS03 + ig, k)
// The caller maps the band-3 rows into the 140-row taug/fracs arrays at offset NS03.
//
// Tables (namgpu/data/rrtm_lw/kgb03.npz, extracted by tools/extract_rrtm_lw_kgb03.py from radlw_datatb.f):
//   absa[585,16]   absa(ig,id)            -> absa[(id-1)*16 + (ig-1)]
//   absb[1175,16]  absb(ig,id)            -> absb[(id-1)*16 + (ig-1)]
//   selfref[10,16] selfref(ig,ins)        -> selfref[(ins-1)*16 + (ig-1)]
//   forref[4,16]   forref(ig,indf)        -> forref[(indf-1)*16 + (ig-1)]
//   fracrefa[9,16] fracrefa(ig,jpl)       -> fracrefa[(jpl-1)*16 + (ig-1)]
//   fracrefb[5,16] fracrefb(ig,jpl)       -> fracrefb[(jpl-1)*16 + (ig-1)]
//   ka_mn2o[19,9,16] ka_mn2o(ig,j,m)      -> ka_mn2o[((m-1)*9 + (j-1))*16 + (ig-1)]
//   kb_mn2o[19,5,16] kb_mn2o(ig,j,m)      -> kb_mn2o[((m-1)*5 + (j-1))*16 + (ig-1)]
//   chi_mls[59,7]  chi_mls(m,n)           -> chi_mls[(n-1)*7 + (m-1)]   (module_radlw_ref, radlw_datatb.f:791)
//
// Fortran terms accounted for (line numbers are radlw_main_nmmb.f):
//   constants: eps = 1.0e-6 (:267), oneminus = 1.0-eps (:268); f_zero, f_one (:274-275); nspa(3) = 9 and
//     nspb(3) = 5 (module-level DATA, :284-285); NG03 = 16, NS03 = 22 (radlw_param_nmmb.f:153,160);
//     refrat_planck_a = chi(1,9)/chi(2,9), refrat_planck_b = chi(1,13)/chi(2,13), refrat_m_a = chi(1,3)/chi(2,3),
//     refrat_m_b = chi(1,13)/chi(2,13) (:3812-3815).
//   lower loop (:3819-3955): speccomb/specparm/specmult/js/fs for the two reference gases (8x grid),
//     ind0/ind1, the N2O minor-gas mapping (jmn2o, fmn2o), the Planck mapping (jpl, fpl), the index and
//     fraction table (indself, indfor, indminor and their +1 neighbours), N2O column adjustment
//     (adjfac = 0.5 + (ratn2o-0.5)**0.65 when ratn2o > 1.5), and the three-branch id/fk selection
//     for each of the two reference-gas pairs (specparm < 0.125, > 0.875, else).
//   lower g-loop (:3966-3990): tauself, taufor, n2om1/n2om2/absn2o, tau_major, tau_major1, taug, fracs.
//   upper loop (:4004-4086): same structure with the 4x grid, ind0 = ((jp-13)*5+(jt-1))*nspb(3)+js,
//     ind1 = ((jp-12)*5+(jt1-1))*nspb(3)+js1, id offsets +5/+6, no self continuum, no band-3 fk2 terms.
//   Fortran p**4 is evaluated as (p*p)*(p*p); mod(a,1.0) is fmodf(a,1.0) (a >= 0 in this routine).
//
// Launch: namgpu.kernels.grid_cols(im, jm), one thread per column; no shared memory, no host sync,
// no device allocation inside the kernel. Fast numerics: fmad on (kernels.OPTIONS).

#include "nmmb_common.cuh"

#define NG03        16
#define NSPA3       9
#define NSPB3       5
#define NS03        22
#define ONEMINUS    (1.0f - 1.0e-6f)
#define F_ONE       1.0f
#define F_ZERO      0.0f

#define CHI(m, n)   chi_mls[((n) - 1) * 7 + ((m) - 1)]

#define ABSA(ig, id)     absa[((id) - 1) * NG03 + ((ig) - 1)]
#define ABSB(ig, id)     absb[((id) - 1) * NG03 + ((ig) - 1)]
#define SELFREF(ig, ins) selfref[((ins) - 1) * NG03 + ((ig) - 1)]
#define FORREF(ig, ind)  forref[((ind) - 1) * NG03 + ((ig) - 1)]
#define FRACREFA(ig, j)  fracrefa[((j) - 1) * NG03 + ((ig) - 1)]
#define FRACREFB(ig, j)  fracrefb[((j) - 1) * NG03 + ((ig) - 1)]
#define KA_MN2O(ig, j, m) ka_mn2o[(((m) - 1) * 9 + ((j) - 1)) * NG03 + ((ig) - 1)]
#define KB_MN2O(ig, j, m) kb_mn2o[(((m) - 1) * 5 + ((j) - 1)) * NG03 + ((ig) - 1)]

// Per-layer field accessors. Fortran (k) maps to l0 = k-1 (CONTRACT section 1).
#define L3(arr, k)       arr[IDX3(i0, j0, (k) - 1, im, jm)]
#define COLAMT(k, m)     colamt[IDX4(i0, j0, (k) - 1, (m) - 1, im, jm, nlay)]
#define RFRATE(k, n)     rfrate[IDX4(i0, j0, (k) - 1, (n) - 1, im, jm, nlay)]
#define OUT3(arr, ig, k) arr[IDX4(i0, j0, (k) - 1, (ig) - 1, im, jm, nlay)]

extern "C" __global__ void phys_rrtm_lw_taugb03(
    const int im, const int jm, const int nlay,
    const float* __restrict__ chi_mls,
    const float* __restrict__ absa, const float* __restrict__ absb,
    const float* __restrict__ selfref, const float* __restrict__ forref,
    const float* __restrict__ fracrefa, const float* __restrict__ fracrefb,
    const float* __restrict__ ka_mn2o, const float* __restrict__ kb_mn2o,
    const int* __restrict__ laytrop,
    const int* __restrict__ jp, const int* __restrict__ jt, const int* __restrict__ jt1,
    const float* __restrict__ colamt, const float* __restrict__ rfrate,
    const float* __restrict__ coldry,
    const float* __restrict__ fac00, const float* __restrict__ fac01,
    const float* __restrict__ fac10, const float* __restrict__ fac11,
    const float* __restrict__ selffac, const float* __restrict__ selffrac,
    const float* __restrict__ forfac, const float* __restrict__ forfrac,
    const float* __restrict__ minorfrac,
    const int* __restrict__ indself, const int* __restrict__ indfor, const int* __restrict__ indminor,
    float* __restrict__ taug3, float* __restrict__ fracs3)
{
    COLUMN_IJ(im, jm)

    // Band-constant ratios from chi_mls (radlw_main_nmmb.f:3812-3815).
    const float refrat_planck_a = CHI(1, 9) / CHI(2, 9);     // P = 212.725 mb
    const float refrat_planck_b = CHI(1, 13) / CHI(2, 13);   // P = 95.58 mb
    const float refrat_m_a      = CHI(1, 3) / CHI(2, 3);     // P = 706.270 mb
    const float refrat_m_b      = CHI(1, 13) / CHI(2, 13);   // P = 95.58 mb

    const int lt = laytrop[IDX2(i0, j0, im)];

    // ----- lower atmosphere loop: k = 1 .. laytrop (radlw_main_nmmb.f:3819-3990) -----
    for (int k = 1; k <= lt; ++k) {
        float speccomb = COLAMT(k, 1) + RFRATE(k, 1) * COLAMT(k, 2);
        float specparm = COLAMT(k, 1) / speccomb;
        float specmult = 8.0f * fminf(specparm, ONEMINUS);
        int   js = 1 + (int)specmult;
        float fs = fmodf(specmult, F_ONE);
        int ind0 = ((L3(jp, k) - 1) * 5 + (L3(jt, k) - 1)) * NSPA3 + js;

        float speccomb1 = COLAMT(k, 1) + RFRATE(k, 2) * COLAMT(k, 2);
        float specparm1 = COLAMT(k, 1) / speccomb1;
        float specmult1 = 8.0f * fminf(specparm1, ONEMINUS);
        int   js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, F_ONE);
        int ind1 = (L3(jp, k) * 5 + (L3(jt1, k) - 1)) * NSPA3 + js1;

        float speccomb_mn2o = COLAMT(k, 1) + refrat_m_a * COLAMT(k, 2);
        float specparm_mn2o = COLAMT(k, 1) / speccomb_mn2o;
        float specmult_mn2o = 8.0f * fminf(specparm_mn2o, ONEMINUS);
        int   jmn2o = 1 + (int)specmult_mn2o;
        float fmn2o = fmodf(specmult_mn2o, F_ONE);

        float speccomb_planck = COLAMT(k, 1) + refrat_planck_a * COLAMT(k, 2);
        float specparm_planck = COLAMT(k, 1) / speccomb_planck;
        float specmult_planck = 8.0f * fminf(specparm_planck, ONEMINUS);
        int   jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, F_ONE);

        int inds = L3(indself, k);
        int indf = L3(indfor, k);
        int indm = L3(indminor, k);
        int indsp = inds + 1;
        int indfp = indf + 1;
        int indmp = indm + 1;
        int jmn2op = jmn2o + 1;
        int jplp = jpl + 1;

        // N2O minor-gas adjustment (radlw_main_nmmb.f:3928-3936).
        float p = L3(coldry, k) * CHI(4, L3(jp, k) + 1);
        float ratn2o = COLAMT(k, 4) / p;
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = 0.5f + powf(ratn2o - 0.5f, 0.65f);
            adjcoln2o = adjfac * p;
        } else {
            adjcoln2o = COLAMT(k, 4);
        }

        // Pair 1 (reference gas 1 at ind0): three branches on specparm (:3938-3972).
        float fk0, fk1, fk2;
        int id000, id010, id100, id110, id200, id210;
        if (specparm < 0.125f) {
            float pp = fs - F_ONE;
            float p2 = pp * pp;
            float p4 = p2 * p2;
            fk0 = p4;
            fk1 = F_ONE - pp - 2.0f * p4;
            fk2 = pp + p4;
            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;
            id110 = ind0 + 10; id200 = ind0 + 2;  id210 = ind0 + 11;
        } else if (specparm > 0.875f) {
            float pp = -fs;
            float p2 = pp * pp;
            float p4 = p2 * p2;
            fk0 = p4;
            fk1 = F_ONE - pp - 2.0f * p4;
            fk2 = pp + p4;
            id000 = ind0 + 1;  id010 = ind0 + 10; id100 = ind0;
            id110 = ind0 + 9;  id200 = ind0 - 1;  id210 = ind0 + 8;
        } else {
            fk0 = F_ONE - fs;
            fk1 = fs;
            fk2 = F_ZERO;
            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;
            id110 = ind0 + 10; id200 = ind0;      id210 = ind0;
        }
        float fac000 = fk0 * L3(fac00, k);
        float fac100 = fk1 * L3(fac00, k);
        float fac200 = fk2 * L3(fac00, k);
        float fac010 = fk0 * L3(fac10, k);
        float fac110 = fk1 * L3(fac10, k);
        float fac210 = fk2 * L3(fac10, k);

        // Pair 2 (reference gas at ind1 with the second reference ratio) (:3974-4003).
        int id001, id011, id101, id111, id201, id211;
        if (specparm1 < 0.125f) {
            float pp = fs1 - F_ONE;
            float p2 = pp * pp;
            float p4 = p2 * p2;
            fk0 = p4;
            fk1 = F_ONE - pp - 2.0f * p4;
            fk2 = pp + p4;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;
            id111 = ind1 + 10; id201 = ind1 + 2;  id211 = ind1 + 11;
        } else if (specparm1 > 0.875f) {
            float pp = -fs1;
            float p2 = pp * pp;
            float p4 = p2 * p2;
            fk0 = p4;
            fk1 = F_ONE - pp - 2.0f * p4;
            fk2 = pp + p4;
            id001 = ind1 + 1;  id011 = ind1 + 10; id101 = ind1;
            id111 = ind1 + 9;  id201 = ind1 - 1;  id211 = ind1 + 8;
        } else {
            fk0 = F_ONE - fs1;
            fk1 = fs1;
            fk2 = F_ZERO;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;
            id111 = ind1 + 10; id201 = ind1;      id211 = ind1;
        }
        float fac001 = fk0 * L3(fac01, k);
        float fac101 = fk1 * L3(fac01, k);
        float fac201 = fk2 * L3(fac01, k);
        float fac011 = fk0 * L3(fac11, k);
        float fac111 = fk1 * L3(fac11, k);
        float fac211 = fk2 * L3(fac11, k);

        const float sf = L3(selffac, k), sfr = L3(selffrac, k);
        const float ff = L3(forfac, k), ffr = L3(forfrac, k);
        const float mf = L3(minorfrac, k);
        const float ocm = adjcoln2o;

        // g-point loop (:3966-3990).
        for (int ig = 1; ig <= NG03; ++ig) {
            float tauself = sf * (SELFREF(ig, inds) + sfr * (SELFREF(ig, indsp) - SELFREF(ig, inds)));
            float taufor  = ff * (FORREF(ig, indf) + ffr * (FORREF(ig, indfp) - FORREF(ig, indf)));
            float n2om1 = KA_MN2O(ig, jmn2o, indm) + fmn2o * (KA_MN2O(ig, jmn2op, indm) - KA_MN2O(ig, jmn2o, indm));
            float n2om2 = KA_MN2O(ig, jmn2o, indmp) + fmn2o * (KA_MN2O(ig, jmn2op, indmp) - KA_MN2O(ig, jmn2o, indmp));
            float absn2o = n2om1 + mf * (n2om2 - n2om1);

            float tau_major = speccomb *
                (fac000 * ABSA(ig, id000) + fac010 * ABSA(ig, id010) +
                 fac100 * ABSA(ig, id100) + fac110 * ABSA(ig, id110) +
                 fac200 * ABSA(ig, id200) + fac210 * ABSA(ig, id210));

            float tau_major1 = speccomb1 *
                (fac001 * ABSA(ig, id001) + fac011 * ABSA(ig, id011) +
                 fac101 * ABSA(ig, id101) + fac111 * ABSA(ig, id111) +
                 fac201 * ABSA(ig, id201) + fac211 * ABSA(ig, id211));

            OUT3(taug3, ig, k) = tau_major + tau_major1 + tauself + taufor + ocm * absn2o;
            OUT3(fracs3, ig, k) = FRACREFA(ig, jpl) + fpl * (FRACREFA(ig, jplp) - FRACREFA(ig, jpl));
        }
    }

    // ----- upper atmosphere loop: k = laytrop+1 .. nlay (radlw_main_nmmb.f:4004-4086) -----
    for (int k = lt + 1; k <= nlay; ++k) {
        float speccomb = COLAMT(k, 1) + RFRATE(k, 1) * COLAMT(k, 2);
        float specparm = COLAMT(k, 1) / speccomb;
        float specmult = 4.0f * fminf(specparm, ONEMINUS);
        int   js = 1 + (int)specmult;
        float fs = fmodf(specmult, F_ONE);
        int ind0 = ((L3(jp, k) - 13) * 5 + (L3(jt, k) - 1)) * NSPB3 + js;

        float speccomb1 = COLAMT(k, 1) + RFRATE(k, 2) * COLAMT(k, 2);
        float specparm1 = COLAMT(k, 1) / speccomb1;
        float specmult1 = 4.0f * fminf(specparm1, ONEMINUS);
        int   js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, F_ONE);
        int ind1 = ((L3(jp, k) - 12) * 5 + (L3(jt1, k) - 1)) * NSPB3 + js1;

        float speccomb_mn2o = COLAMT(k, 1) + refrat_m_b * COLAMT(k, 2);
        float specparm_mn2o = COLAMT(k, 1) / speccomb_mn2o;
        float specmult_mn2o = 4.0f * fminf(specparm_mn2o, ONEMINUS);
        int   jmn2o = 1 + (int)specmult_mn2o;
        float fmn2o = fmodf(specmult_mn2o, F_ONE);

        float speccomb_planck = COLAMT(k, 1) + refrat_planck_b * COLAMT(k, 2);
        float specparm_planck = COLAMT(k, 1) / speccomb_planck;
        float specmult_planck = 4.0f * fminf(specparm_planck, ONEMINUS);
        int   jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, F_ONE);

        int indf = L3(indfor, k);
        int indm = L3(indminor, k);
        int indfp = indf + 1;
        int indmp = indm + 1;
        int jmn2op = jmn2o + 1;
        int jplp = jpl + 1;

        int id000 = ind0, id010 = ind0 + 5, id100 = ind0 + 1, id110 = ind0 + 6;
        int id001 = ind1, id011 = ind1 + 5, id101 = ind1 + 1, id111 = ind1 + 6;

        float p = L3(coldry, k) * CHI(4, L3(jp, k) + 1);
        float ratn2o = COLAMT(k, 4) / p;
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = 0.5f + powf(ratn2o - 0.5f, 0.65f);
            adjcoln2o = adjfac * p;
        } else {
            adjcoln2o = COLAMT(k, 4);
        }

        float fk0 = F_ONE - fs;
        float fk1 = fs;
        float fac000 = fk0 * L3(fac00, k);
        float fac010 = fk0 * L3(fac10, k);
        float fac100 = fk1 * L3(fac00, k);
        float fac110 = fk1 * L3(fac10, k);

        fk0 = F_ONE - fs1;
        fk1 = fs1;
        float fac001 = fk0 * L3(fac01, k);
        float fac011 = fk0 * L3(fac11, k);
        float fac101 = fk1 * L3(fac01, k);
        float fac111 = fk1 * L3(fac11, k);

        const float ff = L3(forfac, k), ffr = L3(forfrac, k);
        const float mf = L3(minorfrac, k);
        const float ocm = adjcoln2o;

        for (int ig = 1; ig <= NG03; ++ig) {
            float taufor = ff * (FORREF(ig, indf) + ffr * (FORREF(ig, indfp) - FORREF(ig, indf)));
            float n2om1 = KB_MN2O(ig, jmn2o, indm) + fmn2o * (KB_MN2O(ig, jmn2op, indm) - KB_MN2O(ig, jmn2o, indm));
            float n2om2 = KB_MN2O(ig, jmn2o, indmp) + fmn2o * (KB_MN2O(ig, jmn2op, indmp) - KB_MN2O(ig, jmn2o, indmp));
            float absn2o = n2om1 + mf * (n2om2 - n2om1);

            float tau_major = speccomb *
                (fac000 * ABSB(ig, id000) + fac010 * ABSB(ig, id010) +
                 fac100 * ABSB(ig, id100) + fac110 * ABSB(ig, id110));

            float tau_major1 = speccomb1 *
                (fac001 * ABSB(ig, id001) + fac011 * ABSB(ig, id011) +
                 fac101 * ABSB(ig, id101) + fac111 * ABSB(ig, id111));

            OUT3(taug3, ig, k) = tau_major + tau_major1 + taufor + ocm * absn2o;
            OUT3(fracs3, ig, k) = FRACREFB(ig, jpl) + fpl * (FRACREFB(ig, jplp) - FRACREFB(ig, jpl));
        }
    }
}
