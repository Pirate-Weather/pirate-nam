// phys_rrtm_lw_taugb01.cu -- unit phys-rrtm-lw-taugb01: TAUGB01, nam-gpu.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), subroutine
//   TAUGB01 inside module_radlw_main_nmmb (radlw_main_nmmb.f, nam.v4.2.11, 83df4e13c1). The author
//   header is retained in the NAM clone: written by Eli J. Mlawer (AER), revised by Michael J. Iacono (AER).
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:3610-3712 (nam.v4.2.11, 83df4e13c1).
//
// Which tables and which NG01: TAUGB01 says `use module_radlw_kgb01`. In the NAM build that module is
// the ORIGINAL one in phys/radlw_datatb.f, and it takes NG01 = 10 from module_radlw_parameters
// (phys/radlw_param.f:107). The _nmmb modules (NG01 = 08) are not used by this routine. The k-tables
// are data/rrtm_lw/kgb01.npz, written by tools/extract_rrtm_lw_kgb01.py.
//
// Index convention. Fortran integer indices (jp, jt, jt1, indself, indfor, indminor, and ind0/ind1
// below) stay 1-based exactly as in TAUGB01. They become memory offsets only at access (T2 macro).
// Fortran X(ig, ind) is stored with ig fastest.
//
// Column layout. One thread per column (TAUGB01 works on one column inside taumol). Layer-by-column
// fields are [NLAY, NCOL] with column fastest. Per-g outputs are [NLAY, NG01, NCOL].
//
// Term-by-term check against TAUGB01 (each statement has a counterpart below):
//  - Lower atmosphere, k = 1..laytrop (:3623-3670):
//      ind0 = ((jp-1)*5 + (jt-1))*nspa(1) + 1 ; ind1 = (jp*5 + (jt1-1))*nspa(1) + 1 ; nspa(1) = 1 (:284)
//      ind0p = ind0+1 ; ind1p = ind1+1 ; inds = indself(k) ; indsp = inds+1 ; indf = indfor(k) ; indfp = indf+1
//      indm = indminor(k) ; indmp = indm+1                                                    (:3626-3639)
//      pp = pavel(k) ; scalen2 = colbrd(k)*scaleminorn2(k)                                     (:3641-3642)
//      corradj = f_one - 0.15*(250-pp)/154.4 when pp < 250, else f_one                         (:3643-3648)
//      per g:
//        tauself = selffac*(selfref(ig,inds) + selffrac*(selfref(ig,indsp)-selfref(ig,inds)))   (:3654-3656)
//        taufor  = forfac*(forref(ig,indf) + forfrac*(forref(ig,indfp)-forref(ig,indf)))        (:3657-3659)
//        taun2   = scalen2*(ka_mn2(ig,indm) + minorfrac*(ka_mn2(ig,indmp)-ka_mn2(ig,indm)))      (:3660-3662)
//        taug(ig,k) = corradj*(colamt(k,1)*(fac00*absa(ig,ind0)+fac10*absa(ig,ind0p)
//                     + fac01*absa(ig,ind1)+fac11*absa(ig,ind1p)) + tauself + taufor + taun2)  (:3663-3667)
//        fracs(ig,k) = fracrefa(ig)                                                             (:3669)
//  - Upper atmosphere, k = laytrop+1..nlay (:3675-3710):
//      ind0 = ((jp-13)*5 + (jt-1))*nspb(1) + 1 ; ind1 = ((jp-12)*5 + (jt1-1))*nspb(1) + 1 ; nspb(1) = 1 (:285)
//      no tauself; taufor as above; taun2 uses kb_mn2 (:3699-3700)
//      corradj = f_one - 0.15*(pavel(k)/95.6)                                                   (:3683)
//      taug(ig,k) = corradj*(colamt(k,1)*(fac00*absb(ig,ind0)+fac10*absb(ig,ind0p)
//                   + fac01*absb(ig,ind1)+fac11*absb(ig,ind1p)) + taufor + taun2)             (:3701-3705)
//      fracs(ig,k) = fracrefb(ig)                                                             (:3707)
//  - TAUGB01 reads only colamt(k,1) (H2O). The wrapper passes that slice as colamt_h2o.
//
// Precision. The Fortran is real(kind_phys) with kind_phys = 8 (machine.f:10). This port is float32
// per the unit instruction (fast numerics, no bitwise claim). The k-table literals are rounded to
// float32 in the npz. This choice is stated in the unit note.
//
// Launch: one thread per column, one kernel launch on ctx.stream, no host sync, no printf.

#include "nmmb_common.cuh"

#define TB1_NG     10   /* NG01 = 10: original module_radlw_parameters (radlw_param.f:107) */
#define TB1_NSPA1  1    /* nspa(1), radlw_main_nmmb.f:284 */
#define TB1_NSPB1  1    /* nspb(1), radlw_main_nmmb.f:285 */

// Fortran-style 1-based access to a table X(ig, ind) stored as [ind][ig] with ig fastest.
#define T2(arr, ind, ig) arr[((size_t)((ind) - 1)) * TB1_NG + ((ig) - 1)]

extern "C" __global__ void taugb01_cols(
    const int*   __restrict__ laytrop,       /* [NCOL]           tropopause layer index per column */
    const int    nlay,
    const int    ncol,
    const float* __restrict__ pavel,         /* [NLAY, NCOL] */
    const float* __restrict__ colamt_h2o,    /* [NLAY, NCOL]     colamt(k,1) */
    const float* __restrict__ colbrd,        /* [NLAY, NCOL] */
    const float* __restrict__ scaleminorn2,  /* [NLAY, NCOL] */
    const float* __restrict__ selffac,       /* [NLAY, NCOL] */
    const float* __restrict__ selffrac,      /* [NLAY, NCOL] */
    const int*   __restrict__ indself,       /* [NLAY, NCOL]     1-based */
    const float* __restrict__ forfac,        /* [NLAY, NCOL] */
    const float* __restrict__ forfrac,       /* [NLAY, NCOL] */
    const int*   __restrict__ indfor,        /* [NLAY, NCOL]     1-based */
    const float* __restrict__ minorfrac,     /* [NLAY, NCOL] */
    const int*   __restrict__ indminor,      /* [NLAY, NCOL]     1-based */
    const float* __restrict__ fac00,         /* [NLAY, NCOL] */
    const float* __restrict__ fac01,         /* [NLAY, NCOL] */
    const float* __restrict__ fac10,         /* [NLAY, NCOL] */
    const float* __restrict__ fac11,         /* [NLAY, NCOL] */
    const int*   __restrict__ jp,            /* [NLAY, NCOL]     1-based */
    const int*   __restrict__ jt,            /* [NLAY, NCOL]     1-based */
    const int*   __restrict__ jt1,           /* [NLAY, NCOL]     1-based */
    const float* __restrict__ absa,          /* [65,  NG01]  absa(ig,ind)  */
    const float* __restrict__ absb,          /* [235, NG01]  absb(ig,ind)  */
    const float* __restrict__ selfref,       /* [10,  NG01]  selfref(ig,ind) */
    const float* __restrict__ forref,        /* [4,   NG01]  forref(ig,ind)  */
    const float* __restrict__ ka_mn2,        /* [19,  NG01]  ka_mn2(ig,ind)  */
    const float* __restrict__ kb_mn2,        /* [19,  NG01]  kb_mn2(ig,ind)  */
    const float* __restrict__ fracrefa,      /* [NG01] */
    const float* __restrict__ fracrefb,      /* [NG01] */
    float*       __restrict__ taug,          /* [NLAY, NG01, NCOL]  taug(ig,k) */
    float*       __restrict__ fracs)         /* [NLAY, NG01, NCOL]  fracs(ig,k) */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int lay = laytrop[col];

    for (int k = 1; k <= nlay; ++k) {
        const size_t kc = (size_t)(k - 1) * ncol + col;           /* [k, col] */
        const size_t ob = (size_t)(k - 1) * TB1_NG * ncol + col;  /* [k, ig=1, col] */

        const float pp        = pavel[kc];
        const float scalen2   = colbrd[kc] * scaleminorn2[kc];
        const float colamtk   = colamt_h2o[kc];
        const float fac00k    = fac00[kc];
        const float fac01k    = fac01[kc];
        const float fac10k    = fac10[kc];
        const float fac11k    = fac11[kc];
        const float forfack   = forfac[kc];
        const float forfrack  = forfrac[kc];
        const float minor     = minorfrac[kc];
        const int   indf      = indfor[kc];
        const int   indfp     = indf + 1;
        const int   indm      = indminor[kc];
        const int   indmp     = indm + 1;
        const int   jpk       = jp[kc];
        const int   jtk       = jt[kc];
        const int   jt1k      = jt1[kc];

        if (k <= lay) {
            // ---- lower atmosphere: TAUGB01 :3623-3670
            const int ind0  = ((jpk - 1) * 5 + (jtk - 1)) * TB1_NSPA1 + 1;
            const int ind1  = (jpk * 5 + (jt1k - 1)) * TB1_NSPA1 + 1;
            const int ind0p = ind0 + 1;
            const int ind1p = ind1 + 1;
            const int inds  = indself[kc];
            const int indsp = inds + 1;
            // corradj (:3643-3648)
            const float corradj = (pp < 250.0f) ? (1.0f - 0.15f * (250.0f - pp) / 154.4f) : 1.0f;

            for (int ig = 1; ig <= TB1_NG; ++ig) {
                const float tauself = selffac[kc] * (T2(selfref, inds, ig)
                                       + selffrac[kc] * (T2(selfref, indsp, ig) - T2(selfref, inds, ig)));
                const float taufor  = forfack * (T2(forref, indf, ig)
                                       + forfrack * (T2(forref, indfp, ig) - T2(forref, indf, ig)));
                const float taun2   = scalen2 * (T2(ka_mn2, indm, ig)
                                       + minor * (T2(ka_mn2, indmp, ig) - T2(ka_mn2, indm, ig)));
                const float tg = corradj * (colamtk
                                   * (fac00k * T2(absa, ind0, ig) + fac10k * T2(absa, ind0p, ig)
                                      + fac01k * T2(absa, ind1, ig) + fac11k * T2(absa, ind1p, ig))
                                   + tauself + taufor + taun2);
                const size_t o = ob + (size_t)(ig - 1) * ncol;
                taug[o]  = tg;
                fracs[o] = fracrefa[ig - 1];
            }
        } else {
            // ---- upper atmosphere: TAUGB01 :3675-3710
            const int ind0  = ((jpk - 13) * 5 + (jtk - 1)) * TB1_NSPB1 + 1;
            const int ind1  = ((jpk - 12) * 5 + (jt1k - 1)) * TB1_NSPB1 + 1;
            const int ind0p = ind0 + 1;
            const int ind1p = ind1 + 1;
            // corradj (:3683)
            const float corradj = 1.0f - 0.15f * (pp / 95.6f);

            for (int ig = 1; ig <= TB1_NG; ++ig) {
                const float taufor = forfack * (T2(forref, indf, ig)
                                      + forfrack * (T2(forref, indfp, ig) - T2(forref, indf, ig)));
                const float taun2  = scalen2 * (T2(kb_mn2, indm, ig)
                                      + minor * (T2(kb_mn2, indmp, ig) - T2(kb_mn2, indm, ig)));
                const float tg = corradj * (colamtk
                                   * (fac00k * T2(absb, ind0, ig) + fac10k * T2(absb, ind0p, ig)
                                      + fac01k * T2(absb, ind1, ig) + fac11k * T2(absb, ind1p, ig))
                                   + taufor + taun2);
                const size_t o = ob + (size_t)(ig - 1) * ncol;
                taug[o]  = tg;
                fracs[o] = fracrefb[ig - 1];
            }
        }
    }
}
