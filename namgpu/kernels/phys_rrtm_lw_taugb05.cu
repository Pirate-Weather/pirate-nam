// phys_rrtm_lw_taugb05.cu -- unit phys-rrtm-lw-taugb05: LW band 5 optical depths and Planck fractions, WOOF-NAM.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA / NCEP NAM source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE; no licence file in the NAM clone). The routine is
//   the band-5 subroutine of the NMMB RRTM longwave package, phys/radlw_main_nmmb.f (nam.v4.2.11,
//   83df4e13c1), whose header says "this package includes ncep's modifications of the rrtm-lw radiation".
//   Keep this notice with the code.
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:4341-4598 (nam.v4.2.11, 83df4e13c1), SUBROUTINE taugb05
//   (an internal subroutine of taumol, module_radlw_main_nmmb). Tables: radlw_datatb.f:15209-21676
//   (module_radlw_kgb05, the module taugb05 names in its USE), chi_mls from radlw_datatb.f:733-905.
//
// Fortran lines covered (all of them):
//   4341-4352  header, locals (no values)
//   4362-4364  refrat_planck_a = chi_mls(1,5)/chi_mls(2,5); refrat_planck_b = chi_mls(3,43)/chi_mls(2,43);
//              refrat_m_a = chi_mls(1,7)/chi_mls(2,7)                          -> host scalars (wrapper)
//   4366-4402  lower atmosphere loop, k = 1..laytrop (speccomb, specparm, js, fs, ind0, ind1, the three
//              mixing ratios, jmo3, jpl, the three-branch fk/id tables (specparm < 0.125, > 0.875, else))
//   4404-4438  (lower) the fac* products and the ig loop: taug(ns05+ig,k), fracs(ns05+ig,k) with the
//              self, foreign, minor (O3) and CCL4 terms
//   4440-4598  upper atmosphere loop, k = laytrop+1 .. nlay: ind0/ind1 with nspb(5), jpl, fk/fac*, ig loop
//
// Layout (CONTRACT section 1, zero-based, l0 = 0 is model top, LM = nlay):
//   column inputs  [LM, JM, IM] float32 : wx_ccl4 (= wx(k,1)), fac00, fac01, fac10, fac11, selffac,
//                  selffrac, forfac, forfrac, minorfrac
//   column inputs  [3, LM, JM, IM] float32 : colamt(k,g), g = 0,1,2 for gas 1 (h2o), 2 (co2), 3 (o3)
//                  (taugb05 reads gases 1-3 only; gases 4-7 are not used in band 5)
//   column inputs  [4, LM, JM, IM] float32 : rfrate(k,1,1), rfrate(k,1,2), rfrate(k,6,1), rfrate(k,6,2)
//                  (slot 0..3 respectively)
//   column inputs  [LM, JM, IM] int32   : jp, jt, jt1, indself, indfor, indminor
//   column inputs  [JM, IM] int32       : laytrop
//   tables (device, uploaded once; index order documented per table):
//     absa     [585, 16]  absa(ig, id)       -> absa[(id-1)*16 + (ig-1)]
//     absb     [1175, 16] absb(ig, id)       -> absb[(id-1)*16 + (ig-1)]
//     selfref  [10, 16]   selfref(ig, is)    -> selfref[(is-1)*16 + (ig-1)]
//     forref   [4, 16]    forref(ig, if)     -> forref[(if-1)*16 + (ig-1)]
//     fracrefa [9, 16]    fracrefa(ig, jp)   -> fracrefa[(jp-1)*16 + (ig-1)]
//     fracrefb [5, 16]    fracrefb(ig, jp)   -> fracrefb[(jp-1)*16 + (ig-1)]
//     ka_mo3   [19, 9, 16] ka_mo3(ig, j, m)  -> ka_mo3[((m-1)*9 + (j-1))*16 + (ig-1)]
//     ccl4     [16]       ccl4(ig)           -> ccl4[ig-1]
//   outputs (band-5 slice only; the 16 g-points ns05+1 .. ns05+16 with ns05 = 52, i.e. 0-based g 52..67):
//     taug5    [16, LM, JM, IM] float32 : taug5[ig0, l0, j0, i0]  = taug(NS05 + ig, k)   (ig0 = ig-1)
//     fracs5   [16, LM, JM, IM] float32 : fracs5[ig0, l0, j0, i0] = fracs(NS05 + ig, k)
//   The caller maps band 5 into its taug(NGPTLW, LM) slots 52..67. Every layer 0..LM-1 is written by
//   exactly one of the two loops (lower k0 < laytrop, upper k0 >= laytrop).
//
// DECLARED DIVERGENCE (precision). The NAM Linux.gnu, Linux.intel and wcoss2 configs do not define
//   -DSINGLE_PREC (the only SINGLE_PREC hit is a commented line in a gsm config), so machine.f selects
//   kind_phys=8 and native taugb05 runs in double. This port runs float32 throughout, as the WOOF-NAM rule
//   for kfpt=single. The claim is single precision, not native-identical precision. Not bitwise to native.
// DECLARED DIVERGENCE (tables). Native reads the kind_phys=8 tables; they are rounded once to float32 at
//   load time (wrapper). refrat_planck_a/b and refrat_m_a are computed in float64 and rounded once.
// INDEX GUARD (fail loud, no host sync). Native reads table entries outside the Fortran range in some
//   branches (e.g. id210 = ind0+11 reaches 588 when jp=13, jt=5, js=1, past absa's 585 rows). Every table
//   index used here is range-checked on the device; an out-of-range layer gets NaN for its 16 g-points, and
//   laytrop outside [0, LM] gives a NaN column. The full-size no-NaN check then shows it. Nothing is read
//   out of bounds and no default is filled in.
// nlay: the Fortran upper loop runs to nlay (dummy argument). This kernel runs to LM. The wrapper refuses
//   any nlay other than LM, so the two are the same by construction.
//
// Launch: one thread per column (namgpu.kernels.grid_cols), each thread loops over layers and the 16
// g-points. No shared memory, no atomics, no host sync, no printf. No MPI or halo (single card).
// Fortran `x**4` is evaluated as (x*x)*(x*x), as gfortran expands an integer power of 4.
// Single precision throughout (kfpt = single); the literals 0.125, 0.875 and oneminus are float here.

#include "nmmb_common.cuh"

#define TG_NG05    16      // NG05 (module_radlw_parameters_nmmb)
#define TG_NSPA5   9       // nspa(5) (radlw_main_nmmb.f:284)
#define TG_NSPB5   5       // nspb(5) (radlw_main_nmmb.f:285)
#define TG_EPS     1.0e-6f // eps (radlw_main_nmmb.f:266)
#define TG_ONEMINUS (1.0f - TG_EPS)   // oneminus (radlw_main_nmmb.f:268)
#define TG_F_ONE   1.0f
#define TG_F_ZERO  0.0f

// absa / absb / selfref / forref / fracref / ka_mo3 table access, 1-based Fortran indices.
#define TG_AB(tab, n, ig, ig0)   tab[((size_t)((n) - 1)) * TG_NG05 + (ig0)]
// index guard helpers (see the INDEX GUARD note in the header)
#define TG_NAN          (__int_as_float(0x7fc00000))
#define TG_IN(v, lo, hi) ((v) >= (lo) && (v) <= (hi))
#define TG_NAN_LAYER(k0) \
    for (int ign_ = 0; ign_ < TG_NG05; ++ign_) { \
        taug5[IDX4(i0, j0, k0, ign_, IM, JM, LM)] = TG_NAN; \
        fracs5[IDX4(i0, j0, k0, ign_, IM, JM, LM)] = TG_NAN; \
    }

extern "C" __global__ void phys_rrtm_lw_taugb05(
    const float* __restrict__ colamt,      // [3, LM, JM, IM]
    const float* __restrict__ rfrate,      // [4, LM, JM, IM]
    const float* __restrict__ wx_ccl4,     // [LM, JM, IM]  wx(k,1)
    const float* __restrict__ fac00,       // [LM, JM, IM]
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ minorfrac,
    const int* __restrict__ jp,            // [LM, JM, IM]
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const int* __restrict__ indminor,
    const int* __restrict__ laytrop,       // [JM, IM]
    const float* __restrict__ absa,        // [585, 16]
    const float* __restrict__ absb,        // [1175, 16]
    const float* __restrict__ selfref,     // [10, 16]
    const float* __restrict__ forref,      // [4, 16]
    const float* __restrict__ fracrefa,    // [9, 16]
    const float* __restrict__ fracrefb,    // [5, 16]
    const float* __restrict__ ka_mo3,      // [19, 9, 16]
    const float* __restrict__ ccl4,        // [16]
    const float refrat_planck_a,
    const float refrat_planck_b,
    const float refrat_m_a,
    const int IM,
    const int JM,
    const int LM,
    float* __restrict__ taug5,             // [16, LM, JM, IM]
    float* __restrict__ fracs5)            // [16, LM, JM, IM]
{
    COLUMN_IJ(IM, JM)

    const int lt = laytrop[IDX2(i0, j0, IM)];          // laytrop (Fortran 1-based count of lower layers)
    const float oneminus = TG_ONEMINUS;
    if (lt < 0 || lt > LM) {                           // laytrop outside [0, LM]: NaN column, no loop
        for (int k0 = 0; k0 < LM; ++k0) { TG_NAN_LAYER(k0) }
        return;
    }

    // ------------------------------------------------------------------
    // Lower atmosphere: Fortran k = 1..laytrop  ->  k0 = 0..laytrop-1
    // ------------------------------------------------------------------
    for (int k0 = 0; k0 < lt; ++k0) {
        const int k = k0 + 1;                           // Fortran k (for the k-based table indices)
        const size_t q  = IDX3(i0, j0, k0, IM, JM);     // [l, j, i] layer offset
        const size_t q1 = IDX4(i0, j0, k0, 1, IM, JM, LM);
        const size_t q2 = IDX4(i0, j0, k0, 2, IM, JM, LM);
        const float c1 = colamt[IDX4(i0, j0, k0, 0, IM, JM, LM)];   // colamt(k,1) h2o
        const float c2 = colamt[q1];                                  // colamt(k,2) co2
        const float c3 = colamt[q2];                                  // colamt(k,3) o3
        const float r11 = rfrate[IDX4(i0, j0, k0, 0, IM, JM, LM)];   // rfrate(k,1,1)
        const float r12 = rfrate[IDX4(i0, j0, k0, 1, IM, JM, LM)];   // rfrate(k,1,2)
        const float wxk = wx_ccl4[q];                                 // wx(k,1)

        const int jp_k = jp[q];
        const int jt_k = jt[q];
        const int jt1_k = jt1[q];
        const int inds = indself[q];
        const int indf = indfor[q];
        const int indm = indminor[q];

        // --- binary species parameter, reference ratio index (rfrate(k,1,1), rfrate(k,1,2))
        float speccomb  = c1 + r11 * c2;
        float specparm  = c1 / speccomb;
        float specmult  = 8.0f * fminf(specparm, oneminus);
        int   js        = 1 + (int)specmult;
        float fs        = specmult - (float)(int)specmult;          // mod(specmult, f_one)
        const int ind0  = ((jp_k - 1) * 5 + (jt_k - 1)) * TG_NSPA5 + js;

        float speccomb1 = c1 + r12 * c2;
        float specparm1 = c1 / speccomb1;
        float specmult1 = 8.0f * fminf(specparm1, oneminus);
        int   js1       = 1 + (int)specmult1;
        float fs1       = specmult1 - (float)(int)specmult1;
        const int ind1  = (jp_k * 5 + (jt1_k - 1)) * TG_NSPA5 + js1;

        // --- minor gas O3 reference ratio (refrat_m_a)
        float speccomb_mo3 = c1 + refrat_m_a * c2;
        float specparm_mo3 = c1 / speccomb_mo3;
        float specmult_mo3 = 8.0f * fminf(specparm_mo3, oneminus);
        int   jmo3         = 1 + (int)specmult_mo3;
        float fmo3         = specmult_mo3 - (float)(int)specmult_mo3;

        // --- Planck fraction reference ratio (refrat_planck_a)
        float speccomb_planck = c1 + refrat_planck_a * c2;
        float specparm_planck = c1 / speccomb_planck;
        float specmult_planck = 8.0f * fminf(specparm_planck, oneminus);
        int   jpl             = 1 + (int)specmult_planck;
        float fpl             = specmult_planck - (float)(int)specmult_planck;

        const int indsp  = inds + 1;
        const int indfp  = indf + 1;
        const int indmp  = indm + 1;
        const int jplp   = jpl + 1;
        const int jmo3p  = jmo3 + 1;

        float fk00, fk10, fk20, fk01, fk11, fk21;
        int id000, id010, id100, id110, id200, id210;
        int id001, id011, id101, id111, id201, id211;

        if (specparm < 0.125f && specparm1 < 0.125f) {
            float p0  = fs - TG_F_ONE;
            float q0  = p0 * p0;
            float p40 = q0 * q0;
            fk00 = p40;
            fk10 = TG_F_ONE - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            float p1  = fs1 - TG_F_ONE;
            float q1  = p1 * p1;
            float p41 = q1 * q1;
            fk01 = p41;
            fk11 = TG_F_ONE - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10;
            id200 = ind0 + 2;  id210 = ind0 + 11;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10;
            id201 = ind1 + 2;  id211 = ind1 + 11;
        } else if (specparm > 0.875f && specparm1 > 0.875f) {
            float p0  = -fs;
            float q0  = p0 * p0;
            float p40 = q0 * q0;
            fk00 = p40;
            fk10 = TG_F_ONE - p0 - 2.0f * p40;
            fk20 = p0 + p40;

            float p1  = -fs1;
            float q1  = p1 * p1;
            float p41 = q1 * q1;
            fk01 = p41;
            fk11 = TG_F_ONE - p1 - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0 + 1;  id010 = ind0 + 10; id100 = ind0;      id110 = ind0 + 9;
            id200 = ind0 - 1;  id210 = ind0 + 8;
            id001 = ind1 + 1;  id011 = ind1 + 10; id101 = ind1;      id111 = ind1 + 9;
            id201 = ind1 - 1;  id211 = ind1 + 8;
        } else {
            fk00 = TG_F_ONE - fs;
            fk10 = fs;
            fk20 = TG_F_ZERO;

            fk01 = TG_F_ONE - fs1;
            fk11 = fs1;
            fk21 = TG_F_ZERO;

            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10;
            id200 = ind0;      id210 = ind0;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10;
            id201 = ind1;      id211 = ind1;
        }

        // index guard: every table row this layer reads must lie inside its table (see header)
        const bool bad_idx = !(TG_IN(jp_k, 1, 13) && TG_IN(jt_k, 1, 5) && TG_IN(jt1_k, 1, 5)
            && TG_IN(inds, 1, 10) && TG_IN(indsp, 1, 10)
            && TG_IN(indf, 1, 4) && TG_IN(indfp, 1, 4)
            && TG_IN(indm, 1, 19) && TG_IN(indmp, 1, 19)
            && TG_IN(jmo3, 1, 9) && TG_IN(jmo3p, 1, 9)
            && TG_IN(jpl, 1, 9) && TG_IN(jplp, 1, 9)
            && TG_IN(id000, 1, 585) && TG_IN(id010, 1, 585) && TG_IN(id100, 1, 585)
            && TG_IN(id110, 1, 585) && TG_IN(id200, 1, 585) && TG_IN(id210, 1, 585)
            && TG_IN(id001, 1, 585) && TG_IN(id011, 1, 585) && TG_IN(id101, 1, 585)
            && TG_IN(id111, 1, 585) && TG_IN(id201, 1, 585) && TG_IN(id211, 1, 585));
        if (bad_idx) { TG_NAN_LAYER(k0) continue; }

        const float f00 = fac00[q];
        const float f10 = fac10[q];
        const float f01 = fac01[q];
        const float f11 = fac11[q];

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

        const float sfac  = selffac[q];
        const float sfrac = selffrac[q];
        const float ffac  = forfac[q];
        const float ffrac = forfrac[q];
        const float mfrac = minorfrac[q];
        const float c3_o3 = c3;

        for (int ig0 = 0; ig0 < TG_NG05; ++ig0) {
            // self-continuum (selfref(ig, inds), selfref(ig, indsp))
            const float sr_s  = selfref[(size_t)(inds - 1) * TG_NG05 + ig0];
            const float sr_sp = selfref[(size_t)(indsp - 1) * TG_NG05 + ig0];
            const float tauself = sfac * (sr_s + sfrac * (sr_sp - sr_s));

            // foreign continuum (forref(ig, indf), forref(ig, indfp))
            const float fr_f  = forref[(size_t)(indf - 1) * TG_NG05 + ig0];
            const float fr_fp = forref[(size_t)(indfp - 1) * TG_NG05 + ig0];
            const float taufor = ffac * (fr_f + ffrac * (fr_fp - fr_f));

            // minor gas O3 (ka_mo3(ig, jmo3, indm), ka_mo3(ig, jmo3p, indm), and the indmp row)
            const size_t kmo_a = ((size_t)(indm - 1) * 9 + (jmo3 - 1)) * TG_NG05 + ig0;
            const size_t kmo_b = ((size_t)(indm - 1) * 9 + (jmo3p - 1)) * TG_NG05 + ig0;
            const size_t kmo_c = ((size_t)(indmp - 1) * 9 + (jmo3 - 1)) * TG_NG05 + ig0;
            const size_t kmo_d = ((size_t)(indmp - 1) * 9 + (jmo3p - 1)) * TG_NG05 + ig0;
            const float o3m1 = ka_mo3[kmo_a] + fmo3 * (ka_mo3[kmo_b] - ka_mo3[kmo_a]);
            const float o3m2 = ka_mo3[kmo_c] + fmo3 * (ka_mo3[kmo_d] - ka_mo3[kmo_c]);
            const float abso3 = o3m1 + mfrac * (o3m2 - o3m1);

            const float ab000 = TG_AB(absa, id000, ig0, ig0);
            const float ab010 = TG_AB(absa, id010, ig0, ig0);
            const float ab100 = TG_AB(absa, id100, ig0, ig0);
            const float ab110 = TG_AB(absa, id110, ig0, ig0);
            const float ab200 = TG_AB(absa, id200, ig0, ig0);
            const float ab210 = TG_AB(absa, id210, ig0, ig0);
            const float ab001 = TG_AB(absa, id001, ig0, ig0);
            const float ab011 = TG_AB(absa, id011, ig0, ig0);
            const float ab101 = TG_AB(absa, id101, ig0, ig0);
            const float ab111 = TG_AB(absa, id111, ig0, ig0);
            const float ab201 = TG_AB(absa, id201, ig0, ig0);
            const float ab211 = TG_AB(absa, id211, ig0, ig0);

            const float ccl = ccl4[ig0];

            taug5[IDX4(i0, j0, k0, ig0, IM, JM, LM)] =
                speccomb * (fac000 * ab000 + fac010 * ab010 + fac100 * ab100 + fac110 * ab110
                          + fac200 * ab200 + fac210 * ab210)
              + speccomb1 * (fac001 * ab001 + fac011 * ab011 + fac101 * ab101 + fac111 * ab111
                           + fac201 * ab201 + fac211 * ab211)
              + tauself + taufor + abso3 * c3_o3 + wxk * ccl;

            const float fra = fracrefa[(size_t)(jpl - 1) * TG_NG05 + ig0];
            const float frb = fracrefa[(size_t)(jplp - 1) * TG_NG05 + ig0];
            fracs5[IDX4(i0, j0, k0, ig0, IM, JM, LM)] = fra + fpl * (frb - fra);
        }
    }

    // ------------------------------------------------------------------
    // Upper atmosphere: Fortran k = laytrop+1 .. nlay  ->  k0 = laytrop .. LM-1
    // ------------------------------------------------------------------
    for (int k0 = lt; k0 < LM; ++k0) {
        const int k = k0 + 1;
        const size_t q  = IDX3(i0, j0, k0, IM, JM);
        const size_t q1 = IDX4(i0, j0, k0, 1, IM, JM, LM);
        const size_t q2 = IDX4(i0, j0, k0, 2, IM, JM, LM);
        const float c2 = colamt[q1];                                   // colamt(k,2) co2
        const float c3 = colamt[q2];                                   // colamt(k,3) o3
        const float r61 = rfrate[IDX4(i0, j0, k0, 2, IM, JM, LM)];    // rfrate(k,6,1)
        const float r62 = rfrate[IDX4(i0, j0, k0, 3, IM, JM, LM)];    // rfrate(k,6,2)
        const float wxk = wx_ccl4[q];                                  // wx(k,1)

        const int jp_k = jp[q];
        const int jt_k = jt[q];
        const int jt1_k = jt1[q];

        float speccomb  = c3 + r61 * c2;
        float specparm  = c3 / speccomb;
        float specmult  = 4.0f * fminf(specparm, oneminus);
        int   js        = 1 + (int)specmult;
        float fs        = specmult - (float)(int)specmult;
        const int ind0  = ((jp_k - 13) * 5 + (jt_k - 1)) * TG_NSPB5 + js;

        float speccomb1 = c3 + r62 * c2;
        float specparm1 = c3 / speccomb1;
        float specmult1 = 4.0f * fminf(specparm1, oneminus);
        int   js1       = 1 + (int)specmult1;
        float fs1       = specmult1 - (float)(int)specmult1;
        const int ind1  = ((jp_k - 12) * 5 + (jt1_k - 1)) * TG_NSPB5 + js1;

        float speccomb_planck = c3 + refrat_planck_b * c2;
        float specparm_planck = c3 / speccomb_planck;
        float specmult_planck = 4.0f * fminf(specparm_planck, oneminus);
        int   jpl             = 1 + (int)specmult_planck;
        float fpl             = specmult_planck - (float)(int)specmult_planck;
        const int jplp        = jpl + 1;

        const int id000 = ind0;
        const int id010 = ind0 + 5;
        const int id100 = ind0 + 1;
        const int id110 = ind0 + 6;
        const int id001 = ind1;
        const int id011 = ind1 + 5;
        const int id101 = ind1 + 1;
        const int id111 = ind1 + 6;

        const float fk00 = TG_F_ONE - fs;
        const float fk10 = fs;
        const float fk01 = TG_F_ONE - fs1;
        const float fk11 = fs1;

        // index guard: table rows read by this layer must lie inside absb (1175) and fracrefb (5)
        const bool bad_idx = !(TG_IN(jp_k, 13, 1000) && TG_IN(jt_k, 1, 5) && TG_IN(jt1_k, 1, 5)
            && TG_IN(jpl, 1, 5) && TG_IN(jplp, 1, 5)
            && TG_IN(id000, 1, 1175) && TG_IN(id010, 1, 1175) && TG_IN(id100, 1, 1175)
            && TG_IN(id110, 1, 1175) && TG_IN(id001, 1, 1175) && TG_IN(id011, 1, 1175)
            && TG_IN(id101, 1, 1175) && TG_IN(id111, 1, 1175));
        if (bad_idx) { TG_NAN_LAYER(k0) continue; }

        const float f00 = fac00[q];
        const float f10 = fac10[q];
        const float f01 = fac01[q];
        const float f11 = fac11[q];

        const float fac000 = fk00 * f00;
        const float fac010 = fk00 * f10;
        const float fac100 = fk10 * f00;
        const float fac110 = fk10 * f10;

        const float fac001 = fk01 * f01;
        const float fac011 = fk01 * f11;
        const float fac101 = fk11 * f01;
        const float fac111 = fk11 * f11;

        for (int ig0 = 0; ig0 < TG_NG05; ++ig0) {
            const float ab000 = TG_AB(absb, id000, ig0, ig0);
            const float ab010 = TG_AB(absb, id010, ig0, ig0);
            const float ab100 = TG_AB(absb, id100, ig0, ig0);
            const float ab110 = TG_AB(absb, id110, ig0, ig0);
            const float ab001 = TG_AB(absb, id001, ig0, ig0);
            const float ab011 = TG_AB(absb, id011, ig0, ig0);
            const float ab101 = TG_AB(absb, id101, ig0, ig0);
            const float ab111 = TG_AB(absb, id111, ig0, ig0);

            const float ccl = ccl4[ig0];

            taug5[IDX4(i0, j0, k0, ig0, IM, JM, LM)] =
                speccomb * (fac000 * ab000 + fac010 * ab010 + fac100 * ab100 + fac110 * ab110)
              + speccomb1 * (fac001 * ab001 + fac011 * ab011 + fac101 * ab101 + fac111 * ab111)
              + wxk * ccl;

            const float fra = fracrefb[(size_t)(jpl - 1) * TG_NG05 + ig0];
            const float frb = fracrefb[(size_t)(jplp - 1) * TG_NG05 + ig0];
            fracs5[IDX4(i0, j0, k0, ig0, IM, JM, LM)] = fra + fpl * (frb - fra);
        }
        (void)k;
    }
}
