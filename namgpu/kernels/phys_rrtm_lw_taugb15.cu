// phys_rrtm_lw_taugb15.cu -- unit phys-rrtm-lw-taugb15: RRTM LW band 15 optical depths, nam-gpu.
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). The
//   lw-rrtm3 header of phys/radlw_main_nmmb.f and the module header of phys/radlw_datatb.f
//   (module_radlw_kgb15) are kept in the NAM clone and apply to this port.
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:5980-6184 (subroutine taugb15, internal to
//   taumol; nam.v4.2.11, 83df4e13c1). Its tables are module_radlw_kgb15 from phys/radlw_datatb.f
//   (lines 31225-31611, the module this routine USEs at line 5988), extracted by
//   tools/extract-rrtm-lw-kgb15.py to namgpu/data/rrtm_lw/kgb15.npz.
//
// Band 15: 2380-2600 cm-1 (low key N2 and CO2, low minor N2, high nothing).
//
// Fortran -> CUDA mapping:
//   * Fortran 1-based indices are kept as values; 0-based offsets are taken where the array is read.
//   * ns15 = NS15 = 136 (radlw_param.f:116), ng15 = NG15 = 2 (radlw_param.f:109). The caller passes the
//     2-row band slab, so the kernel writes slab row ig0 = ig-1 (the caller places it at rows 136..137).
//   * nspa(15) = 9 (radlw_main_nmmb.f:284). ind0 = ((jp-1)*5 + (jt-1))*9 + js, ind1 = (jp*5 + (jt1-1))*9 + js1.
//   * oneminus = 1.0 - eps, eps = 1.0e-6 (radlw_main_nmmb.f:267-268), evaluated in single precision.
//   * Fortran MOD(a, f_one) with f_one = 1.0 is fmodf(a, 1.0f) (exact).
//   * p0**4 is evaluated as (p0*p0)*(p0*p0), which is how gfortran expands an integer power of 4.
//   * refrat_planck_a and refrat_m_a are chi_mls(4,1)/chi_mls(2,1) (both equal, N2 over CO2 at the
//     1053 mb reference level); the wrapper computes them once from the module_radlw_ref table and passes
//     them as scalars.
//   * scalen2 = colbrd(k) * scaleminor(k) (setcoef outputs).
//   * rfrate(k,5,1) and rfrate(k,5,2) are the setcoef N2/CO2 ratio rate 5 on the two reference sides.
//   * The lower loop (k = 1..laytrop) and the upper loop (k = laytrop+1..nlay) are both kept; every
//     (ig, k) element of the slab is written by one of them.
//   * Every term of the Fortran assignments is present: tauself, taufor, taun2 (N2 minor with the
//     minorfrac interpolation between the two minor reference levels), the two binary-species sums, and
//     the fracs planck interpolation.
//
// Layouts (device, C order; col = j*IM + i, zero-based):
//   per-layer per-column inputs      [nlay][ncol]                 element (k, col) at k*ncol + col
//   colamt (MAXGAS = 6)              [6][nlay][ncol]              element (m, k, col) at (m*nlay + k)*ncol + col
//   rfrate (setcoef layout)          [2][6][nlay][ncol]           element (n, m, k, col) at ((n*6 + m)*nlay + k)*ncol + col
//   taug15, fracs15 (slab out)       [2][nlay][ncol]              element (ig, k, col) at (ig*nlay + k)*ncol + col
//   absa   [585][2]     absa(ig, m)     at (m-1)*2 + (ig-1)
//   selfref[10][2], forref[4][2]        same convention
//   ka_mn2 [19][9][2]   ka_mn2(ig, j, m) at ((m-1)*9 + (j-1))*2 + (ig-1)
//   fracrefa[9][2]      fracrefa(ig, j)  at (j-1)*2 + (ig-1)
//
// Launch: grid_cols(IM, JM) with BLOCK_COL, one thread per column (the layer loop is the Fortran k-loop).
// No host sync, no allocation, no print. No MPI, halo or OpenMP.

#include "nmmb_common.cuh"

#define NG15_    2
#define NS15_    136    /* radlw_param.f:116, NS15 = 136; slab row = NS15 + ig, caller places the slab */
#define NSPA15_  9      /* radlw_main_nmmb.f:284, nspa(15) = 9 */
#define NRATES15 6      /* radlw_param.f:99, NRATES = 6 (setcoef rfrate layout) */
#define ONEMINUS15 (1.0f - 1.0e-6f)   /* radlw_main_nmmb.f:267-268 */

extern "C" __global__ void phys_rrtm_lw_taugb15(
    const int nlay, const int ncol,
    const float refrat_planck_a,              /* chi_mls(4,1)/chi_mls(2,1)  */
    const float refrat_m_a,                   /* chi_mls(4,1)/chi_mls(2,1)  */
    const int* __restrict__ laytrop,          /* [ncol]          */
    const int* __restrict__ jp,               /* [nlay][ncol]    */
    const int* __restrict__ jt,
    const int* __restrict__ jt1,
    const int* __restrict__ indself,
    const int* __restrict__ indfor,
    const int* __restrict__ indminor,
    const float* __restrict__ colamt,         /* [6][nlay][ncol] */
    const float* __restrict__ rfrate,         /* [2][6][nlay][ncol] */
    const float* __restrict__ colbrd,         /* [nlay][ncol]    */
    const float* __restrict__ scaleminor,     /* [nlay][ncol]    */
    const float* __restrict__ selffac,
    const float* __restrict__ selffrac,
    const float* __restrict__ forfac,
    const float* __restrict__ forfrac,
    const float* __restrict__ minorfrac,
    const float* __restrict__ fac00,
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ absa,           /* [585][2]        */
    const float* __restrict__ selfref,        /* [10][2]         */
    const float* __restrict__ forref,         /* [4][2]          */
    const float* __restrict__ ka_mn2,         /* [19][9][2]      */
    const float* __restrict__ fracrefa,       /* [9][2]          */
    float* __restrict__ taug15,               /* [2][nlay][ncol] */
    float* __restrict__ fracs15)              /* [2][nlay][ncol] */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    const int lt = laytrop[col];              /* Fortran laytrop, a count of lower layers */

    /* ---- lower atmosphere: Fortran do k = 1, laytrop ---- */
    for (int k0 = 0; k0 < lt; ++k0) {
        const size_t kc = (size_t)k0 * ncol + col;     /* [nlay][ncol] element */
        const int jp_k  = jp[kc];
        const int jt_k  = jt[kc];
        const int jt1_k = jt1[kc];

        const float cN2  = colamt[(size_t)(3 * nlay + k0) * ncol + col];  /* colamt(k,4) */
        const float cCO2 = colamt[(size_t)(1 * nlay + k0) * ncol + col];  /* colamt(k,2) */
        /* rfrate(k,5,1) -> n = 0, m = 4 ; rfrate(k,5,2) -> n = 1, m = 4 */
        const float r51 = rfrate[(size_t)((0 * NRATES15 + 4) * nlay + k0) * ncol + col];
        const float r52 = rfrate[(size_t)((1 * NRATES15 + 4) * nlay + k0) * ncol + col];

        /* ---- binary species parameter, reference side 1 ---- */
        const float speccomb = cN2 + r51 * cCO2;
        const float specparm = cN2 / speccomb;
        const float specmult = 8.0f * fminf(specparm, ONEMINUS15);
        const int   js       = 1 + (int)specmult;
        const float fs       = fmodf(specmult, 1.0f);
        const int   ind0     = ((jp_k - 1) * 5 + (jt_k - 1)) * NSPA15_ + js;

        /* ---- binary species parameter, reference side 2 ---- */
        const float speccomb1 = cN2 + r52 * cCO2;
        const float specparm1 = cN2 / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, ONEMINUS15);
        const int   js1       = 1 + (int)specmult1;
        const float fs1       = fmodf(specmult1, 1.0f);
        const int   ind1      = (jp_k * 5 + (jt1_k - 1)) * NSPA15_ + js1;

        /* ---- minor N2 mapping (refrat_m_a) ---- */
        const float speccomb_mn2 = cN2 + refrat_m_a * cCO2;
        const float specparm_mn2 = cN2 / speccomb_mn2;
        const float specmult_mn2 = 8.0f * fminf(specparm_mn2, ONEMINUS15);
        const int   jmn2         = 1 + (int)specmult_mn2;
        const float fmn2         = fmodf(specmult_mn2, 1.0f);

        /* ---- Planck fraction mapping (refrat_planck_a) ---- */
        const float speccomb_planck = cN2 + refrat_planck_a * cCO2;
        const float specparm_planck = cN2 / speccomb_planck;
        const float specmult_planck = 8.0f * fminf(specparm_planck, ONEMINUS15);
        const int   jpl             = 1 + (int)specmult_planck;
        const float fpl             = fmodf(specmult_planck, 1.0f);

        const float scalen2 = colbrd[kc] * scaleminor[kc];

        const int inds  = indself[kc];
        const int indf  = indfor[kc];
        const int indm  = indminor[kc];
        const int indsp = inds + 1;
        const int indfp = indf + 1;
        const int indmp = indm + 1;
        const int jplp  = jpl + 1;
        const int jmn2p = jmn2 + 1;

        const float sfac  = selffac[kc];
        const float sfrac = selffrac[kc];
        const float ffac  = forfac[kc];
        const float ffrac = forfrac[kc];
        const float mfrac = minorfrac[kc];

        /* ---- interpolation weights and table offsets, by the three Fortran branches ---- */
        float fk00, fk10, fk20, fk01, fk11, fk21;
        int id000, id010, id100, id110, id200, id210;
        int id001, id011, id101, id111, id201, id211;

        if (specparm < 0.125f && specparm1 < 0.125f) {
            const float p0  = fs - 1.0f;
            const float p40 = (p0 * p0) * (p0 * p0);
            fk00 = p40;
            fk10 = (1.0f - p0) - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1  = fs1 - 1.0f;
            const float p41 = (p1 * p1) * (p1 * p1);
            fk01 = p41;
            fk11 = (1.0f - p1) - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10;
            id200 = ind0 + 2;  id210 = ind0 + 11;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10;
            id201 = ind1 + 2;  id211 = ind1 + 11;
        } else if (specparm > 0.875f && specparm1 > 0.875f) {
            const float p0  = -fs;
            const float p40 = (p0 * p0) * (p0 * p0);
            fk00 = p40;
            fk10 = (1.0f - p0) - 2.0f * p40;
            fk20 = p0 + p40;

            const float p1  = -fs1;
            const float p41 = (p1 * p1) * (p1 * p1);
            fk01 = p41;
            fk11 = (1.0f - p1) - 2.0f * p41;
            fk21 = p1 + p41;

            id000 = ind0 + 1;  id010 = ind0 + 10; id100 = ind0;      id110 = ind0 + 9;
            id200 = ind0 - 1;  id210 = ind0 + 8;
            id001 = ind1 + 1;  id011 = ind1 + 10; id101 = ind1;      id111 = ind1 + 9;
            id201 = ind1 - 1;  id211 = ind1 + 8;
        } else {
            fk00 = 1.0f - fs;
            fk10 = fs;
            fk20 = 0.0f;

            fk01 = 1.0f - fs1;
            fk11 = fs1;
            fk21 = 0.0f;

            id000 = ind0;      id010 = ind0 + 9;  id100 = ind0 + 1;  id110 = ind0 + 10;
            id200 = ind0;      id210 = ind0;
            id001 = ind1;      id011 = ind1 + 9;  id101 = ind1 + 1;  id111 = ind1 + 10;
            id201 = ind1;      id211 = ind1;
        }

        const float f00 = fac00[kc], f01 = fac01[kc], f10 = fac10[kc], f11 = fac11[kc];
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

        for (int ig0 = 0; ig0 < NG15_; ++ig0) {
            /* Fortran selfref(ig,inds) stored [m-1][ig0] */
            const float selfA = selfref[(size_t)(inds - 1) * NG15_ + ig0];
            const float selfB = selfref[(size_t)(indsp - 1) * NG15_ + ig0];
            const float tauself = sfac * (selfA + sfrac * (selfB - selfA));

            const float forA = forref[(size_t)(indf - 1) * NG15_ + ig0];
            const float forB = forref[(size_t)(indfp - 1) * NG15_ + ig0];
            const float taufor = ffac * (forA + ffrac * (forB - forA));

            /* ka_mn2(ig, j, m) stored [m-1][j-1][ig0] */
            const size_t kmA = ((size_t)(indm - 1) * 9 + (jmn2 - 1)) * NG15_ + ig0;     /* ka_mn2(ig,jmn2,indm)  */
            const size_t kmB = ((size_t)(indm - 1) * 9 + (jmn2p - 1)) * NG15_ + ig0;    /* ka_mn2(ig,jmn2p,indm) */
            const size_t kmC = ((size_t)(indmp - 1) * 9 + (jmn2 - 1)) * NG15_ + ig0;    /* ka_mn2(ig,jmn2,indmp) */
            const size_t kmD = ((size_t)(indmp - 1) * 9 + (jmn2p - 1)) * NG15_ + ig0;   /* ka_mn2(ig,jmn2p,indmp) */
            const float n2m1 = ka_mn2[kmA] + fmn2 * (ka_mn2[kmB] - ka_mn2[kmA]);
            const float n2m2 = ka_mn2[kmC] + fmn2 * (ka_mn2[kmD] - ka_mn2[kmC]);
            const float taun2 = scalen2 * (n2m1 + mfrac * (n2m2 - n2m1));

            /* absa(ig, id) stored [id-1][ig0] */
            const float sumA = fac000 * absa[(size_t)(id000 - 1) * NG15_ + ig0]
                             + fac010 * absa[(size_t)(id010 - 1) * NG15_ + ig0]
                             + fac100 * absa[(size_t)(id100 - 1) * NG15_ + ig0]
                             + fac110 * absa[(size_t)(id110 - 1) * NG15_ + ig0]
                             + fac200 * absa[(size_t)(id200 - 1) * NG15_ + ig0]
                             + fac210 * absa[(size_t)(id210 - 1) * NG15_ + ig0];
            const float sumB = fac001 * absa[(size_t)(id001 - 1) * NG15_ + ig0]
                             + fac011 * absa[(size_t)(id011 - 1) * NG15_ + ig0]
                             + fac101 * absa[(size_t)(id101 - 1) * NG15_ + ig0]
                             + fac111 * absa[(size_t)(id111 - 1) * NG15_ + ig0]
                             + fac201 * absa[(size_t)(id201 - 1) * NG15_ + ig0]
                             + fac211 * absa[(size_t)(id211 - 1) * NG15_ + ig0];

            const float tau = speccomb * sumA + speccomb1 * sumB + tauself + taufor + taun2;

            /* fracrefa(ig, j) stored [j-1][ig0] */
            const float frA = fracrefa[(size_t)(jpl - 1) * NG15_ + ig0];
            const float frB = fracrefa[(size_t)(jplp - 1) * NG15_ + ig0];
            const float frac = frA + fpl * (frB - frA);

            const size_t oc = ((size_t)ig0 * nlay + k0) * ncol + col;   /* taug15(ig, k) */
            taug15[oc]  = tau;
            fracs15[oc] = frac;
        }
    }

    /* ---- upper atmosphere: Fortran do k = laytrop+1, nlay; taug and fracs are zero ---- */
    for (int k0 = lt; k0 < nlay; ++k0) {
        for (int ig0 = 0; ig0 < NG15_; ++ig0) {
            const size_t oc = ((size_t)ig0 * nlay + k0) * ncol + col;
            taug15[oc]  = 0.0f;
            fracs15[oc] = 0.0f;
        }
    }
}
