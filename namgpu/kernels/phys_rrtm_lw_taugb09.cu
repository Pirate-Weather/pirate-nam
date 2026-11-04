// phys_rrtm_lw_taugb09.cu -- unit phys-rrtm-lw-taugb09: LW band 9 optical depths and Planck
// fractions, WOOF-NAM.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). The
//   routine is taugb09 of the NMMB rrtm-lw package (phys/radlw_main_nmmb.f, nam.v4.2.11, 83df4e13c1),
//   which carries the NCEP header "lw-rrtm3 radiation package description: this package includes ncep
//   modifications of the rrtm-lw radiation code from aer inc." Its k-tables are module_radlw_kgb09
//   (phys/radlw_datatb.f:25328-27963), the module taugb09 uses (radlw_main_nmmb.f:5077); see
//   namgpu/data/rrtm_lw/kgb09.npz (tools/extract_rrtm_lw_kgb09.py).
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:5069-5313 (nam.v4.2.11, 83df4e13c1).
//
// Fortran coverage (whole subroutine, every statement):
//   reference ratios refrat_planck_a = chi_mls(1,9)/chi_mls(6,9), refrat_m_a = chi_mls(1,3)/chi_mls(6,3)
//   lower loop  do k = 1, laytrop: speccomb, specparm, specmult, js, fs (H2O/CH4 binary species,
//               rfrate(k,4,1)), speccomb1 .. js1 .. fs1 (rfrate(k,4,2)), speccomb_mn2o ..
//               jmn2o, fmn2o (N2O minor), speccomb_planck .. jpl, fpl (Planck), inds/indf/indm and
//               their +1 partners, the N2O column adjustment (temp, ratn2o, adjfac, adjcoln2o), the
//               three branches on specparm/specparm1 (< 0.125, > 0.875, else) with the fk/id tables,
//               fac000..fac211, then the g loop: tauself, taufor, n2om1, n2om2, absn2o,
//               taug(ns09+ig,k), fracs(ns09+ig,k) = fracrefa(ig,jpl) + fpl*(fracrefa(ig,jpl+1)-...)
//   upper loop  do k = laytrop+1, nlay: ind0, ind1 (nspb(9) = 1), ind0p, ind1p, indm, indmp, the N2O
//               column adjustment, the g loop: absn2o from kb_mn2o, taug(ns09+ig,k) from absb and
//               colamt(k,5), fracs = fracrefb(ig).
//   nspa(9) = 9 and nspb(9) = 1 (radlw_main_nmmb.f:284-285), NG09 = 12, NS09 = 96
//   (radlw_param.f:106-115), NGPTLW = 140, oneminus = 1 - 1.0e-6 (radlw_main_nmmb.f:267-268),
//   f_one = 1, f_zero = 0 (radlw_main_nmmb.f:274-275).
//
// Layout. Column c = j*IM + i (flat, 0-based). Layer index l is 0-based in RRTM INTERNAL order,
//   SURFACE FIRST: l = 0 is the LOWEST layer (Fortran k = l+1 of radlw_main_nmmb.f, whose input
//   arrays are reversed to surface-first at the ivflip = 0 branch, 680-690). The NMMB top-first
//   order is NOT used here (CONTRACT section 1, radlw exception). The lower atmosphere is
//   l = 0 .. laytrop-1 (Fortran k = 1 .. laytrop, laytrop counted while plog > 4.56,
//   radlw_main_nmmb.f:2061). Every per-layer input and output below uses this order.
//   Fortran 1-based integer inputs (jp, jt, jt1, indself, indfor, indminor, laytrop) are passed as
//   Fortran values and converted here exactly where the Fortran indexes arrays:
//     per-layer float or int  X[l*NC + c]
//     colamt(k,n)             colamt[(n*LM + l)*NC + c]      n = 0..5 (Fortran gas index n+1)
//     rfrate(k,m,n)           rfrate[((n*6 + m)*LM + l)*NC + c]  n = 0..1 (Fortran 1..2), m = 0..5 (1..6)
//     laytrop                 laytrop[c]
//   Tables (row-major as stored by the extractor): absa[585][12], absb[235][12], selfref[10][12],
//     forref[4][12], ka_mn2o[19][9][12], kb_mn2o[19][12], fracrefa[9][12], fracrefb[12],
//     chi_mls[59][7] with chi_mls(g,lev) = chi[(lev-1)*7 + (g-1)].
//   Outputs are TILE-LOCAL: the kernel handles columns c0 .. c0+ncol-1 and writes
//     taug[g][l][t], fracs[g][l][t] with t = c - c0 and g = 0..139 (NGPTLW). With c0 = 0 and
//     ncol = NC this is the CONTRACT layout [140, LM, JM, IM] reshaped (see the wrapper).
//     Only g = NS09 .. NS09+11 is written; the other bands' g-points are untouched.
//     taug and fracs are in the same surface-first layer order as the inputs.
//
// Launch: one thread per column (kernels.grid_cols), the layer loop and the g loop inside the
//   thread, as the Fortran column loops do. No host sync, no allocation, no print. No MPI, halo,
//   or OpenMP (the Fortran has none in this routine).

#include "nmmb_common.cuh"

#define TGB09_NG    12    // NG09
#define TGB09_NS    96    // NS09: 0-based offset of band 9's g-points (taug index = NS09 + ig - 1)
#define TGB09_NSPA  9     // nspa(9)
#define TGB09_NSPB  1     // nspb(9)
#define TGB09_NRATE 6     // NRATES
#define TGB09_ONEMINUS (1.0f - 1.0e-6f)

extern "C" __global__ void phys_rrtm_lw_taugb09(
    const int*   __restrict__ laytrop,    // [NC]    Fortran laytrop per column
    const int*   __restrict__ jp,         // [LM*NC] Fortran 1-based
    const int*   __restrict__ jt,         // [LM*NC] Fortran 1-based
    const int*   __restrict__ jt1,        // [LM*NC] Fortran 1-based
    const int*   __restrict__ indself,    // [LM*NC] Fortran 1-based
    const int*   __restrict__ indfor,     // [LM*NC] Fortran 1-based
    const int*   __restrict__ indminor,   // [LM*NC] Fortran 1-based
    const float* __restrict__ coldry,     // [LM*NC]
    const float* __restrict__ colamt,     // [6*LM*NC]
    const float* __restrict__ rfrate,     // [2*6*LM*NC]
    const float* __restrict__ selffac,    // [LM*NC]
    const float* __restrict__ selffrac,   // [LM*NC]
    const float* __restrict__ forfac,     // [LM*NC]
    const float* __restrict__ forfrac,    // [LM*NC]
    const float* __restrict__ minorfrac,  // [LM*NC]
    const float* __restrict__ fac00,      // [LM*NC]
    const float* __restrict__ fac10,      // [LM*NC]
    const float* __restrict__ fac01,      // [LM*NC]
    const float* __restrict__ fac11,      // [LM*NC]
    const float* __restrict__ chi_mls,    // [59*7]
    const float* __restrict__ absa,       // [585*12]
    const float* __restrict__ absb,       // [235*12]
    const float* __restrict__ selfref,    // [10*12]
    const float* __restrict__ forref,     // [4*12]
    const float* __restrict__ ka_mn2o,    // [19*9*12]
    const float* __restrict__ kb_mn2o,    // [19*12]
    const float* __restrict__ fracrefa,   // [9*12]
    const float* __restrict__ fracrefb,   // [12]
    float* __restrict__ taug,             // [140*LM*ncol] tile-local
    float* __restrict__ fracs,            // [140*LM*ncol] tile-local
    int nlay, int nc, int c0, int ncol)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncol) return;
    const int c = c0 + t;

    const int lt = laytrop[c];   // Fortran laytrop: layers 1..lt are the lower atmosphere

// output index helper: taug/fracs [g][l][t]
#define TOUT(g, l) (((g) * nlay + (l)) * ncol + t)
// per-layer index helper
#define PIDX(l) ((l) * nc + c)
// colamt(k, n) (n 0-based) and rfrate(k, m, n) (m, n 0-based)
#define CAMT(n, l) colamt[((n) * nlay + (l)) * nc + c]
#define RFR(m, n, l) rfrate[(((n) * TGB09_NRATE + (m)) * nlay + (l)) * nc + c]
// Fortran 1-based chi_mls(g, lev) -> chi_mls[(lev-1)*7 + (g-1)]
#define CHI(g, lev) chi_mls[((lev) - 1) * 7 + ((g) - 1)]
// Fortran 1-based table accessors (ig is 0-based here)
#define ABSA(ig, id)      absa[((id) - 1) * TGB09_NG + (ig)]
#define ABSB(ig, id)      absb[((id) - 1) * TGB09_NG + (ig)]
#define SELFREF(ig, s)    selfref[((s) - 1) * TGB09_NG + (ig)]
#define FORREF(ig, f)     forref[((f) - 1) * TGB09_NG + (ig)]
#define KA_MN2O(ig, j, m) ka_mn2o[(((m) - 1) * 9 + ((j) - 1)) * TGB09_NG + (ig)]
#define KB_MN2O(ig, m)    kb_mn2o[((m) - 1) * TGB09_NG + (ig)]
#define FRACREFA(ig, j)   fracrefa[((j) - 1) * TGB09_NG + (ig)]

    // reference ratios (Fortran, before the loops)
    const float refrat_planck_a = CHI(1, 9) / CHI(6, 9);   // P = 212 mb
    const float refrat_m_a      = CHI(1, 3) / CHI(6, 3);   // P = 706.272 mb

    // ---- lower atmosphere loop: Fortran do k = 1, laytrop ----
    for (int l = 0; l < lt; ++l) {
        const int p = PIDX(l);
        const float cdry = coldry[p];
        const float c1   = CAMT(0, l);     // colamt(k,1)  H2O
        const float c4   = CAMT(3, l);     // colamt(k,4)  N2O
        const float c5   = CAMT(4, l);     // colamt(k,5)  CH4
        const int jpk  = jp[p];
        const int jtk  = jt[p];
        const int jt1k = jt1[p];

        // H2O/CH4 binary species, first reference (rfrate(k,4,1))
        const float speccomb = c1 + RFR(3, 0, l) * c5;
        const float specparm = c1 / speccomb;
        const float specmult = 8.0f * fminf(specparm, TGB09_ONEMINUS);
        const int   js       = 1 + (int)specmult;
        const float fs       = fmodf(specmult, 1.0f);
        const int   ind0     = ((jpk - 1) * 5 + (jtk - 1)) * TGB09_NSPA + js;

        // H2O/CH4 binary species, second reference (rfrate(k,4,2))
        const float speccomb1 = c1 + RFR(3, 1, l) * c5;
        const float specparm1 = c1 / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, TGB09_ONEMINUS);
        const int   js1       = 1 + (int)specmult1;
        const float fs1       = fmodf(specmult1, 1.0f);
        const int   ind1      = (jpk * 5 + (jt1k - 1)) * TGB09_NSPA + js1;

        // N2O minor-species mapping
        const float speccomb_mn2o = c1 + refrat_m_a * c5;
        const float specparm_mn2o = c1 / speccomb_mn2o;
        const float specmult_mn2o = 8.0f * fminf(specparm_mn2o, TGB09_ONEMINUS);
        const int   jmn2o         = 1 + (int)specmult_mn2o;
        const float fmn2o         = fmodf(specmult_mn2o, 1.0f);

        // Planck fraction mapping
        const float speccomb_planck = c1 + refrat_planck_a * c5;
        const float specparm_planck = c1 / speccomb_planck;
        const float specmult_planck = 8.0f * fminf(specparm_planck, TGB09_ONEMINUS);
        const int   jpl             = 1 + (int)specmult_planck;
        const float fpl             = fmodf(specmult_planck, 1.0f);

        const int inds   = indself[p];
        const int indf   = indfor[p];
        const int indm   = indminor[p];
        const int indsp  = inds + 1;
        const int indfp  = indf + 1;
        const int indmp  = indm + 1;
        const int jplp   = jpl + 1;
        const int jmn2op = jmn2o + 1;

        // N2O column amount adjustment (empirical, for N2O too great to be a minor species)
        const float temp   = cdry * CHI(4, jpk + 1);
        const float ratn2o = c4 / temp;
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            const float adjfac = 0.5f + powf(ratn2o - 0.5f, 0.65f);
            adjcoln2o = adjfac * temp;
        } else {
            adjcoln2o = c4;
        }

        // binary-species interpolation coefficients: fk00..fk21
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

        const float f00 = fac00[p];
        const float f10 = fac10[p];
        const float f01 = fac01[p];
        const float f11 = fac11[p];
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

        const float sf  = selffac[p];
        const float sfr = selffrac[p];
        const float ff  = forfac[p];
        const float ffr = forfrac[p];
        const float mf  = minorfrac[p];

        for (int ig = 0; ig < TGB09_NG; ++ig) {
            const float tauself = sf * (SELFREF(ig, inds)
                                        + sfr * (SELFREF(ig, indsp) - SELFREF(ig, inds)));
            const float taufor  = ff * (FORREF(ig, indf)
                                        + ffr * (FORREF(ig, indfp) - FORREF(ig, indf)));
            const float n2om1 = KA_MN2O(ig, jmn2o, indm)
                              + fmn2o * (KA_MN2O(ig, jmn2op, indm) - KA_MN2O(ig, jmn2o, indm));
            const float n2om2 = KA_MN2O(ig, jmn2o, indmp)
                              + fmn2o * (KA_MN2O(ig, jmn2op, indmp) - KA_MN2O(ig, jmn2o, indmp));
            const float absn2o = n2om1 + mf * (n2om2 - n2om1);

            const float g_lower =
                speccomb * (fac000 * ABSA(ig, id000) + fac010 * ABSA(ig, id010)
                            + fac100 * ABSA(ig, id100) + fac110 * ABSA(ig, id110)
                            + fac200 * ABSA(ig, id200) + fac210 * ABSA(ig, id210))
              + speccomb1 * (fac001 * ABSA(ig, id001) + fac011 * ABSA(ig, id011)
                             + fac101 * ABSA(ig, id101) + fac111 * ABSA(ig, id111)
                             + fac201 * ABSA(ig, id201) + fac211 * ABSA(ig, id211))
              + tauself + taufor + adjcoln2o * absn2o;

            taug[TOUT(TGB09_NS + ig, l)] = g_lower;
            fracs[TOUT(TGB09_NS + ig, l)] = FRACREFA(ig, jpl)
                                          + fpl * (FRACREFA(ig, jplp) - FRACREFA(ig, jpl));
        }
    }

    // ---- upper atmosphere loop: Fortran do k = laytrop+1, nlay ----
    for (int l = lt; l < nlay; ++l) {
        const int p = PIDX(l);
        const float cdry = coldry[p];
        const float c4   = CAMT(3, l);     // colamt(k,4)  N2O
        const float c5   = CAMT(4, l);     // colamt(k,5)  CH4
        const int jpk  = jp[p];
        const int jtk  = jt[p];
        const int jt1k = jt1[p];

        // ind0 = ((jp-13)*5 + (jt-1))*nspb(9) + 1 ;  ind1 = ((jp-12)*5 + (jt1-1))*nspb(9) + 1
        const int ind0  = ((jpk - 13) * 5 + (jtk - 1)) * TGB09_NSPB + 1;
        const int ind1  = ((jpk - 12) * 5 + (jt1k - 1)) * TGB09_NSPB + 1;
        const int indm  = indminor[p];
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indmp = indm + 1;

        const float temp   = cdry * CHI(4, jpk + 1);
        const float ratn2o = c4 / temp;
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            const float adjfac = 0.5f + powf(ratn2o - 0.5f, 0.65f);
            adjcoln2o = adjfac * temp;
        } else {
            adjcoln2o = c4;
        }

        const float f00 = fac00[p];
        const float f10 = fac10[p];
        const float f01 = fac01[p];
        const float f11 = fac11[p];
        const float mf  = minorfrac[p];

        for (int ig = 0; ig < TGB09_NG; ++ig) {
            const float absn2o = KB_MN2O(ig, indm)
                               + mf * (KB_MN2O(ig, indmp) - KB_MN2O(ig, indm));

            const float g_upper =
                c5 * (f00 * ABSB(ig, ind0) + f10 * ABSB(ig, ind0p)
                      + f01 * ABSB(ig, ind1) + f11 * ABSB(ig, ind1p))
              + adjcoln2o * absn2o;

            taug[TOUT(TGB09_NS + ig, l)] = g_upper;
            fracs[TOUT(TGB09_NS + ig, l)] = fracrefb[ig];
        }
    }

#undef TOUT
#undef PIDX
#undef CAMT
#undef RFR
#undef CHI
#undef ABSA
#undef ABSB
#undef SELFREF
#undef FORREF
#undef KA_MN2O
#undef KB_MN2O
#undef FRACREFA
}
