// phys_rrtm_lw_taugb08.cu -- unit phys-rrtm-lw-taugb08: LW band 8 optical depths and Planck
// fractions, nam-gpu.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file in the NAM clone). The
//   routine is taugb08 of the NMMB rrtm-lw package (phys/radlw_main_nmmb.f, nam.v4.2.11, 83df4e13c1).
//   Its k-tables are module_radlw_kgb08 (phys/radlw_datatb.f:24545-25324), the module the routine
//   actually uses (radlw_main_nmmb.f:4954); see namgpu/data/rrtm_lw/kgb08.npz.
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:4946-5065 (nam.v4.2.11, 83df4e13c1).
//
// Fortran lines covered (every statement):
//   4957-4965  integer and real locals (ind0 ind0p ind1 ind1p inds indsp indf indfp indm indmp ig,
//              tauself taufor absco2 abso3 absn2o ratco2 adjfac adjcolco2 temp)
//   4973-4979  lower loop header: ind0, ind1 (nspa(8) = 1), inds, indf, indm, indxp
//   4987-4995  temp = coldry*chi_mls(2,jp+1); ratco2 = colamt(2)/temp; adjfac; adjcolco2
//   4997-5005  tauself, taufor, absco2, abso3, absn2o (interpolated in minorfrac)
//   5007-5013  taug(ns08+ig,k) lower-atmosphere sum; fracs(ns08+ig,k) = fracrefa(ig)
//   5020-5036  upper loop: ind0 (jp-13), ind1 (jp-12), absco2/absn2o from kb_mco2/kb_mn2o
//   5038-5049  taug(ns08+ig,k) upper-atmosphere sum; fracs = fracrefb(ig)
//   nspa(8) = nspb(8) = 1 (radlw_main_nmmb.f:284-285), so the index strides are 1.
//   NG08 = 8 (module_radlw_parameters_nmmb), NS08 = 88 (radlw_param_nmmb.f:161). The Fortran
//   taug(ns08+ig, k) with 1-based ig maps to 0-based g = NS08 + (ig-1).
//
// Layout (CONTRACT section 1, Fortran (i,j,l) -> [l,j,i]). Column c = j*IM + i (flat, 0-based);
//   layer l 0-based (l = 0 is the model top, Fortran k = l+1); Fortran 1-based integer inputs
//   (jp, jt, jt1, indself, indfor, indminor, laytrop) are passed as Fortran values and converted
//   here exactly where the Fortran indexes arrays:
//     per-layer float or int  X[l*NC + c]          (NC = JM*IM)
//     colamt(k,n)             colamt[(n*LM + l)*NC + c]   n = 0..5 (Fortran gas index n+1)
//     wx(k,n)                 wx[(n*LM + l)*NC + c]       n = 0..3 (Fortran xsec index n+1)
//     laytrop                 laytrop[c]
//   Tables (row-major as stored by the extractor): absa[65][8], absb[235][8], selfref[10][8],
//     forref[4][8], ka_mco2[19][8], kb_mco2[19][8], ka_mo3[19][8], ka_mn2o[19][8], kb_mn2o[19][8],
//     fracrefa[8], fracrefb[8], cfc12[8], cfc22adj[8], chi_co2[59] = chi_mls(2,m), m = 1..59.
//   Outputs are TILE-LOCAL: the kernel handles columns c0 .. c0+ncol-1 and writes
//     taug[g][l][t], fracs[g][l][t] with t = c - c0 and g = 0..139 (NGPTLW). With c0 = 0 and
//     ncol = NC this is the CONTRACT layout [140, LM, JM, IM]; a tile is a column slice of it.
//     Only g = NS08 .. NS08+7 is written; the other bands' g-points are untouched by this kernel
//     (the wrapper zero-fills the scratch once at allocation, so they read 0, never garbage).
//
// Launch: one thread per column (kernels.grid_cols), the layer loop and the g loop inside the
//   thread, as the Fortran column loops do. No host sync, no allocation, no print. No MPI, halo,
//   or OpenMP (the Fortran has none in this routine).

#include "nmmb_common.cuh"

#define TGB08_NG    8     // NG08
#define TGB08_NS    88    // NS08: 0-based offset of band 8's g-points (taug index = NS08 + ig - 1)
#define TGB08_NAN   __uint_as_float(0x7fc00000u)   // quiet NaN: poisons out-of-contract layers

extern "C" __global__ void phys_rrtm_lw_taugb08(
    const int*   __restrict__ laytrop,    // [NC]    Fortran laytrop per column
    const int*   __restrict__ jp,         // [LM*NC] Fortran 1-based
    const int*   __restrict__ jt,         // [LM*NC] Fortran 1-based
    const int*   __restrict__ jt1,        // [LM*NC] Fortran 1-based
    const int*   __restrict__ indself,    // [LM*NC] Fortran 1-based
    const int*   __restrict__ indfor,     // [LM*NC] Fortran 1-based
    const int*   __restrict__ indminor,   // [LM*NC] Fortran 1-based
    const float* __restrict__ coldry,     // [LM*NC]
    const float* __restrict__ colamt,     // [6*LM*NC]
    const float* __restrict__ wx,         // [4*LM*NC]
    const float* __restrict__ selffac,    // [LM*NC]
    const float* __restrict__ selffrac,   // [LM*NC]
    const float* __restrict__ forfac,     // [LM*NC]
    const float* __restrict__ forfrac,    // [LM*NC]
    const float* __restrict__ minorfrac,  // [LM*NC]
    const float* __restrict__ fac00,      // [LM*NC]
    const float* __restrict__ fac10,      // [LM*NC]
    const float* __restrict__ fac01,      // [LM*NC]
    const float* __restrict__ fac11,      // [LM*NC]
    const float* __restrict__ chi_co2,    // [59]
    const float* __restrict__ absa,       // [65*8]
    const float* __restrict__ absb,       // [235*8]
    const float* __restrict__ selfref,    // [10*8]
    const float* __restrict__ forref,     // [4*8]
    const float* __restrict__ ka_mco2,    // [19*8]
    const float* __restrict__ kb_mco2,    // [19*8]
    const float* __restrict__ ka_mo3,     // [19*8]
    const float* __restrict__ ka_mn2o,    // [19*8]
    const float* __restrict__ kb_mn2o,    // [19*8]
    const float* __restrict__ fracrefa,   // [8]
    const float* __restrict__ fracrefb,   // [8]
    const float* __restrict__ cfc12,      // [8]
    const float* __restrict__ cfc22adj,   // [8]
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
// colamt(k, n) and wx(k, n) helpers, n 0-based
#define CAMT(n, l) colamt[((n) * nlay + (l)) * nc + c]
#define WXV(n, l)  wx[((n) * nlay + (l)) * nc + c]

    // ---- lower atmosphere loop: Fortran do k = 1, laytrop ----
    for (int l = 0; l < lt; ++l) {
        const int p = PIDX(l);
        const int jpk  = jp[p];
        const int jtk  = jt[p];
        const int jt1k = jt1[p];

        // Fortran ind0 = ((jp-1)*5 + (jt-1))*nspa(8) + 1 (1-based); 0-based row = ind0-1
        const int r0 = (jpk - 1) * 5 + (jtk - 1);
        // Fortran ind1 = (jp*5 + (jt1-1))*nspa(8) + 1; 0-based row = ind1-1
        const int r1 = jpk * 5 + (jt1k - 1);
        // Fortran inds, indf, indm are 1-based; 0-based rows are inds-1, indf-1, indm-1.
        // indsp = inds+1, indfp = indf+1, indmp = indm+1 are the next 0-based rows.
        const int s0 = indself[p] - 1;
        const int f0 = indfor[p] - 1;
        const int m0 = indminor[p] - 1;

        // Index contract: lower atmosphere jp 1..12 (setcoef laytrop split, plog > 4.56 gives jp <= 12), so the row jp*5+jt1
        // and row+1 stay inside absa[65]; s0+1 inside selfref[10], f0+1 inside forref[4], m0+1
        // inside ka_*[19]. Out-of-contract input writes NaN for the layer, loudly, never a stale read.
        if (!(jpk >= 1 && jpk <= 12 && jtk >= 1 && jtk <= 5 && jt1k >= 1 && jt1k <= 5 &&
              s0 >= 0 && s0 <= 8 && f0 >= 0 && f0 <= 2 && m0 >= 0 && m0 <= 17)) {
            for (int ig = 0; ig < TGB08_NG; ++ig) {
                taug[TOUT(TGB08_NS + ig, l)]  = TGB08_NAN;
                fracs[TOUT(TGB08_NS + ig, l)] = TGB08_NAN;
            }
            continue;
        }

        // co2 adjustment for minor-species treatment (Fortran lines 4987-4995)
        const float temp   = coldry[p] * chi_co2[jpk];     // chi_mls(2, jp+1): 0-based column jp
        const float ratco2 = CAMT(1, l) / temp;            // colamt(k,2)
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 2.0f + powf(ratco2 - 2.0f, 0.65f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = CAMT(1, l);
        }

        const float sf  = selffac[p];
        const float sfr = selffrac[p];
        const float ff  = forfac[p];
        const float ffr = forfrac[p];
        const float mf  = minorfrac[p];
        const float f00 = fac00[p];
        const float f10 = fac10[p];
        const float f01 = fac01[p];
        const float f11 = fac11[p];
        const float c1  = CAMT(0, l);    // colamt(k,1) h2o
        const float c3  = CAMT(2, l);    // colamt(k,3) o3
        const float c4  = CAMT(3, l);    // colamt(k,4) n2o
        const float w3  = WXV(2, l);     // wx(k,3)
        const float w4  = WXV(3, l);     // wx(k,4)

        for (int ig = 0; ig < TGB08_NG; ++ig) {
            // tauself: selfref(ig,inds) and selfref(ig,inds+1)
            const float tauself = sf * (selfref[s0 * TGB08_NG + ig]
                                        + sfr * (selfref[(s0 + 1) * TGB08_NG + ig]
                                                 - selfref[s0 * TGB08_NG + ig]));
            // taufor: forref(ig,indf) and forref(ig,indf+1)
            const float taufor = ff * (forref[f0 * TGB08_NG + ig]
                                       + ffr * (forref[(f0 + 1) * TGB08_NG + ig]
                                                - forref[f0 * TGB08_NG + ig]));
            // minor-species absorption coefficients interpolated in minorfrac
            const float absco2 = ka_mco2[m0 * TGB08_NG + ig]
                               + mf * (ka_mco2[(m0 + 1) * TGB08_NG + ig] - ka_mco2[m0 * TGB08_NG + ig]);
            const float abso3  = ka_mo3[m0 * TGB08_NG + ig]
                               + mf * (ka_mo3[(m0 + 1) * TGB08_NG + ig] - ka_mo3[m0 * TGB08_NG + ig]);
            const float absn2o = ka_mn2o[m0 * TGB08_NG + ig]
                               + mf * (ka_mn2o[(m0 + 1) * TGB08_NG + ig] - ka_mn2o[m0 * TGB08_NG + ig]);

            const float g_lower =
                c1 * (f00 * absa[r0 * TGB08_NG + ig]
                      + f10 * absa[(r0 + 1) * TGB08_NG + ig]
                      + f01 * absa[r1 * TGB08_NG + ig]
                      + f11 * absa[(r1 + 1) * TGB08_NG + ig])
                + tauself + taufor + adjcolco2 * absco2
                + c3 * abso3 + c4 * absn2o
                + w3 * cfc12[ig] + w4 * cfc22adj[ig];

            taug[TOUT(TGB08_NS + ig, l)]  = g_lower;
            fracs[TOUT(TGB08_NS + ig, l)] = fracrefa[ig];
        }
    }

    // ---- upper atmosphere loop: Fortran do k = laytrop+1, nlay ----
    for (int l = lt; l < nlay; ++l) {
        const int p = PIDX(l);
        const int jpk  = jp[p];
        const int jtk  = jt[p];
        const int jt1k = jt1[p];

        // Fortran ind0 = ((jp-13)*5 + (jt-1))*nspb(8) + 1; ind1 = ((jp-12)*5 + (jt1-1))*nspb(8) + 1
        const int r0 = (jpk - 13) * 5 + (jtk - 1);
        const int r1 = (jpk - 12) * 5 + (jt1k - 1);
        const int m0 = indminor[p] - 1;

        // Index contract: upper atmosphere jp 13..58 (setcoef clamps jp to 1..58; jt1 <= 4 keeps row+1 <= 234 in absb[235]) and
        // m0+1 inside kb_*[19]. Out-of-contract input writes NaN for the layer.
        if (!(jpk >= 13 && jpk <= 58 && jtk >= 1 && jtk <= 5 && jt1k >= 1 && jt1k <= 5 &&
              m0 >= 0 && m0 <= 17)) {
            for (int ig = 0; ig < TGB08_NG; ++ig) {
                taug[TOUT(TGB08_NS + ig, l)]  = TGB08_NAN;
                fracs[TOUT(TGB08_NS + ig, l)] = TGB08_NAN;
            }
            continue;
        }

        const float temp   = coldry[p] * chi_co2[jpk];     // chi_mls(2, jp+1)
        const float ratco2 = CAMT(1, l) / temp;            // colamt(k,2)
        float adjcolco2;
        if (ratco2 > 3.0f) {
            const float adjfac = 2.0f + powf(ratco2 - 2.0f, 0.65f);
            adjcolco2 = adjfac * temp;
        } else {
            adjcolco2 = CAMT(1, l);
        }

        const float mf  = minorfrac[p];
        const float f00 = fac00[p];
        const float f10 = fac10[p];
        const float f01 = fac01[p];
        const float f11 = fac11[p];
        const float c3  = CAMT(2, l);    // colamt(k,3) o3
        const float c4  = CAMT(3, l);    // colamt(k,4) n2o
        const float w3  = WXV(2, l);     // wx(k,3)
        const float w4  = WXV(3, l);     // wx(k,4)

        for (int ig = 0; ig < TGB08_NG; ++ig) {
            const float absco2 = kb_mco2[m0 * TGB08_NG + ig]
                               + mf * (kb_mco2[(m0 + 1) * TGB08_NG + ig] - kb_mco2[m0 * TGB08_NG + ig]);
            const float absn2o = kb_mn2o[m0 * TGB08_NG + ig]
                               + mf * (kb_mn2o[(m0 + 1) * TGB08_NG + ig] - kb_mn2o[m0 * TGB08_NG + ig]);

            const float g_upper =
                c3 * (f00 * absb[r0 * TGB08_NG + ig]
                      + f10 * absb[(r0 + 1) * TGB08_NG + ig]
                      + f01 * absb[r1 * TGB08_NG + ig]
                      + f11 * absb[(r1 + 1) * TGB08_NG + ig])
                + adjcolco2 * absco2 + c4 * absn2o
                + w3 * cfc12[ig] + w4 * cfc22adj[ig];

            taug[TOUT(TGB08_NS + ig, l)]  = g_upper;
            fracs[TOUT(TGB08_NS + ig, l)] = fracrefb[ig];
        }
    }

#undef TOUT
#undef PIDX
#undef CAMT
#undef WXV
}
