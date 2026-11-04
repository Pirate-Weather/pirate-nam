// phys_rrtm_lw_taugb02.cu -- unit phys-rrtm-lw-taugb02: band 2 of taumol, nam-gpu.
//
// NOTICE: NOAA/NCEP NMMB source (US government work; no licence file in the NAM clone). The package
// header block of phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1) is carried verbatim below as the
// author notice (CONTRACT.md section 3).
/*
!!!!!  ==============================================================  !!!!!
!!!!!               lw-rrtm3 radiation package description             !!!!!
!!!!!  ==============================================================  !!!!!
!                                                                          !
!   this package includes ncep's modifications of the rrtm-lw radiation    !
!   code from aer inc.                                                     !
!                                                                          !
!    the lw-rrtm3 package includes these parts:                            !
!                                                                          !
!       'radlw_rrtm3_param.f'                                              !
!       'radlw_rrtm3_datatb.f'                                             !
!       'radlw_rrtm3_main.f'                                               !
!                                                                          !
!    the 'radlw_rrtm3_param.f' contains:                                   !
!                                                                          !
!       'module_radlw_parameters'  -- band parameters set up               !
!                                                                          !
!    the 'radlw_rrtm3_datatb.f' contains:                                  !
!                                                                          !
!       'module_radlw_avplank'     -- plank flux data                      !
!       'module_radlw_ref'         -- reference temperature and pressure   !
!       'module_radlw_cldprlw'     -- cloud property coefficients          !
!       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
!                                     bands, where nn = 01-16              !
!                                                                          !
!    the 'radlw_rrtm3_main.f' contains:                                    !
!                                                                          !
!       'module_radlw_main'        -- main lw radiation transfer           !
!                                                                          !
!    in the main module 'module_radlw_main' there are only two             !
!    externally callable subroutines:                                      !
!                                                                          !
!                                                                          !
!       'lwrad'     -- main lw radiation routine                           !
!          inputs:                                                         !
!           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
!            clouds,icseed,aerosols,sfemis,sfgtmp,                         !
!            npts, nlay, nlp1, lprnt,                                      !
!          outputs:                                                        !
!            hlwc,topflx,sfcflx,                                           !
!!         optional outputs:                                               !
!            HLW0,HLWB,FLXPRF)                                             !
!                                                                          !
!       'rlwinit'   -- initialization routine                              !
!          inputs:                                                         !
!           ( me )                                                         !
!          outputs:                                                        !
!           (none)                                                         !
!                                                                          !
!    all the lw radiation subprograms become contained subprograms         !
!    in module 'module_radlw_main' and many of them are not directly       !
!    accessable from places outside the module.                            !
!                                                                          !
!    derived data type constructs used:                                    !
!                                                                          !
!     1. radiation flux at toa: (from module 'module_radlw_parameters')    !
!          topflw_type   -  derived data type for toa rad fluxes           !
!            upfxc              total sky upward flux at toa               !
!            upfx0              clear sky upward flux at toa               !
!                                                                          !
!     2. radiation flux at sfc: (from module 'module_radlw_parameters')    !
!          sfcflw_type   -  derived data type for sfc rad fluxes           !
!            upfxc              total sky upward flux at sfc               !
!            upfx0              clear sky upward flux at sfc               !
!            dnfxc              total sky downward flux at sfc             !
!            dnfx0              clear sky downward flux at sfc             !
!                                                                          !
!     3. radiation flux profiles(from module 'module_radlw_parameters')    !
!          proflw_type    -  derived data type for rad vertical prof       !
!            upfxc              level upward flux for total sky            !
!            dnfxc              level downward flux for total sky          !
!            upfx0              level upward flux for clear sky            !
!            dnfx0              level downward flux for clear sky          !
!                                                                          !
!    external modules referenced:                                          !
!                                                                          !
!       'module physparam'                                                 !
!       'module physcons'                                                  !
!       'mersenne_twister'                                                 !
!                                                                          !
!    compilation sequence is:                                              !
!                                                                          !
!       'radlw_rrtm3_param.f'                                              !
!       'radlw_rrtm3_datatb.f'                                             !
!       'radlw_rrtm3_main.f'                                               !
!                                                                          !
!    and all should be put in front of routines that use lw modules        !
!                                                                          !
!==========================================================================!
!                                                                          !
!    the original aer's program declarations:                              !
!                                                                          !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                          |
!  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
!  This software may be used, copied, or redistributed as long as it is    |
!  not sold and this copyright notice is reproduced on each copy made.     |
!  This model is provided as is without any express or implied warranties. |
!                       (http://www.rtweb.aer.com/)                        |
!                                                                          |
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                          !
! ************************************************************************ !
!                                                                          !
!                              rrtmg_lw                                    !
!                                                                          !
!                                                                          !
!                   a rapid radiative transfer model                       !
!                       for the longwave region                            !
!             for application to general circulation models                !
!                                                                          !
!                                                                          !
!            atmospheric and environmental research, inc.                  !
!                        131 hartwell avenue                               !
!                        lexington, ma 02421                               !
!                                                                          !
!                           eli j. mlawer                                  !
!                        jennifer s. delamere                              !
!                         michael j. iacono                                !
!                         shepard a. clough                                !
!                                                                          !
!                                                                          !
!                       email:  miacono@aer.com                            !
!                       email:  emlawer@aer.com                            !
!                       email:  jdelamer@aer.com                           !
!                                                                          !
!        the authors wish to acknowledge the contributions of the          !
!        following people:  steven j. taubman, karen cady-pereira,         !
!        patrick d. brown, ronald e. farren, luke chen, robert bergstrom.  !
!                                                                          !
! ************************************************************************ !
!                                                                          !
!    references:                                                           !
!    (rrtm_lw/rrtmg_lw):                                                   !
!      clough, s.A., m.w. shephard, e.j. mlawer, j.s. delamere,            !
!      m.j. iacono, k. cady-pereira, s. boukabara, and p.d. brown:         !
!      atmospheric radiative transfer modeling: a summary of the aer       !
!      codes, j. quant. spectrosc. radiat. transfer, 91, 233-244, 2005.    !
!                                                                          !
!      mlawer, e.j., s.j. taubman, p.d. brown, m.j. iacono, and s.a.       !
!      clough:  radiative transfer for inhomogeneous atmospheres: rrtm,    !
!      a validated correlated-k model for the longwave.  j. geophys. res., !
!      102, 16663-16682, 1997.                                             !
!                                                                          !
!    (mcica):                                                              !
!      pincus, r., h. w. barker, and j.-j. morcrette: a fast, flexible,    !
!      approximation technique for computing radiative transfer in         !
!      inhomogeneous cloud fields, j. geophys. res., 108(d13), 4376,       !
!      doi:10.1029/2002JD003322, 2003.                                     !
!                                                                          !
! ************************************************************************ !
!                                                                          !
!    aer's revision history:                                               !
!     this version of rrtmg_lw has been modified from rrtm_lw to use a     !
!     reduced set of g-points for application to gcms.                     !
!                                                                          !
! --  original version (derived from rrtm_lw), reduction of g-points,      !
!     other revisions for use with gcms.                                   !
!        1999: m. j. iacono, aer, inc.                                     !
! --  adapted for use with ncar/cam3.                                      !
!        may 2004: m. j. iacono, aer, inc.                                 !
! --  revised to add mcica capability.                                     !
!        nov 2005: m. j. iacono, aer, inc.                                 !
! --  conversion to f90 formatting for consistency with rrtmg_sw.          !
!        feb 2007: m. j. iacono, aer, inc.                                 !
! --  modifications to formatting to use assumed-shape arrays.             !
!        aug 2007: m. j. iacono, aer, inc.                                 !
!                                                                          !
! ************************************************************************ !
!                                                                          !
!    ncep modifications history log:                                       !
!                                                                          !
!       nov 1999,  ken campana       -- received the original code from    !
!                    aer (1998 ncar ccm version), updated to link up with  !
!                    ncep mrf model                                        !
!       jun 2000,  ken campana       -- added option to switch random and  !
!                    maximum/random cloud overlap                          !
!           2001,  shrinivas moorthi -- further updates for mrf model      !
!       may 2001,  yu-tai hou        -- updated on trace gases and cloud   !
!                    property based on rrtm_v3.0 codes.                    !
!       dec 2001,  yu-tai hou        -- rewritten code into fortran 90 std !
!                    set ncep radiation structure standard that contains   !
!                    three plug-in compatable fortran program files:       !
!                    'radlw_param.f', 'radlw_datatb.f', 'radlw_main.f'     !
!                    fixed bugs in subprograms taugb14, taugb2, etc. added !
!                    out-of-bounds protections. (a detailed note of        !
!                    up_to_date modifications/corrections by ncep was sent !
!                    to aer in 2002)                                       !
!       jun 2004,  yu-tai hou        -- added mike iacono's apr 2004       !
!                    modification of variable diffusivity angles.          !
!       apr 2005,  yu-tai hou        -- minor modifications on module      !
!                    structures include rain/snow effect (this version of  !
!                    code was given back to aer in jun 2006)               !
!       mar 2007,  yu-tai hou        -- added aerosol effect for ncep      !
!                    models using the generallized aerosol optical property!
!                    scheme for gfs model.                                 !
!       apr 2007,  yu-tai hou        -- added spectral band heating as an  !
!                    optional output to support the 500 km gfs model's     !
!                    upper stratospheric radiation calculations. and       !
!                    restructure optional outputs for easy access by       !
!                    different models.                                     !
!       oct 2008,  yu-tai hou        -- modified to include new features   !
!                    from aer's newer release v4.4-v4.7, including the     !
!                    mcica sub-grid cloud option. add rain/snow optical    !
!                    properties support to cloudy sky calculations.        !
!                    correct errors in mcica cloud optical properties for  !
!                    ebert & curry scheme (ilwcice=1) that needs band      !
!                    index conversion. simplified and unified sw and lw    !
!                    sub-column cloud subroutines into one module by using !
!                    optional parameters.                                  !
!       mar 2009,  yu-tai hou        -- replaced the original random number!
!                    generator coming from the original code with ncep w3  !
!                    library to simplify the program and moved sub-column  !
!                    cloud subroutines inside the main module. added       !
!                    option of user provided permutation seeds that could  !
!                    be randomly generated from forecast time stamp.       !
!       oct 2009,  yu-tai hou        -- modified subrtines "cldprop" and   !
!                    "rlwinit" according updats from aer's rrtmg_lw v4.8.  !
!       nov 2009,  yu-tai hou        -- modified subrtine "taumol" according
!                    updats from aer's rrtmg_lw version 4.82. notice the   !
!                    cloud ice/liquid are assumed as in-cloud quantities,  !
!                    not as grid averaged quantities.                      !
!       jun 2010,  yu-tai hou        -- optimized code to improve efficiency
!       apr 2012,  b. ferrier and y. hou -- added conversion factor to fu's!
!                    cloud-snow optical property scheme.                   !
!       nov 2012,  yu-tai hou        -- modified control parameters thru   !
!                     module 'physparam'.                                  !
!                                                                          !
!!!!!  ==============================================================  !!!!!
!!!!!                         end descriptions                         !!!!!
!!!!!  ==============================================================  !!!!!
*/
// Ported to CUDA for nam-gpu from phys/radlw_main_nmmb.f:3716-3788 (nam.v4.2.11, 83df4e13c1).
//
// SUBROUTINE taugb02 (MODULE_RADLW_MAIN_NMMB), band 2 (350-500 cm-1), one thread per column. Every
// term of both Fortran loops is here:
//  - lower atmosphere, k = 1..laytrop (Fortran): ind0, ind0p, ind1, ind1p (table rows of absa),
//    inds/indsp (selfref), indf/indfp (forref), corradj = 1 - 0.05*(pavel-100)/900, taug from the
//    four interpolation weights plus the self-continuum and foreign-continuum terms, and
//    fracs = fracrefa.
//  - upper atmosphere, k = laytrop+1..nlay: ind0/ind1 into absb with the jp-13 / jp-12 offsets,
//    foreign continuum only (no self term, no corradj), fracs = fracrefb.
//  - Table widths and nspa(2) = nspb(2) = 1 come from radlw_main_nmmb.f:284-285 (nspa, nspb data).
//    NG02 = 12 and NS02 = 10 come from module_radlw_parameters (radlw_param.f:107, 114).
//    NGPTLW = 140 from radlw_param_nmmb.f:141.
//  - PLACEMENT (the Fortran effect): the Fortran writes taug(ns02+ig,k) and fracs(ns02+ig,k) into the
//    full gpoint arrays of taumol (taug(ngptlw,nlay), local to taumol). The outputs here are those
//    full arrays, [nlay, NGPTLW, ncol] per column, and this kernel writes only rows NS02+1..NS02+NG02
//    (zero-based NS02..NS02+NG02-1 = 10..21). Rows 0..9 and 22..139 are not touched by this unit.
//    The taumol combine tautot(ig,k) = taug(ig,k) + tauaer(ib,k) runs in phys-rrtm-lw-taumol, after
//    all sixteen band units have written their rows.
//  - The k-tables (absa, absb, selfref, forref, fracrefa, fracrefb) are uploaded once by the
//    init_tables step of this wrapper, which phys-rrtm-lw-params must call at init (CONTRACT section
//    3). The kernel never loads or allocates them.
//  - Integer index arrays hold the Fortran (1-based) values produced by setcoef; the kernel does the
//    zero-based shifts exactly where the Fortran index is used.
//  - Single precision (kfpt = single). FMA contraction on (NVRTC --fmad=true). The Fortran
//    expression order is kept so the rounding sequence matches the statement as written.
//  - No host work, no print, no device allocation. Every per-level array is [nlay, ncol] (level
//    major, column fastest) so the column threads read coalesced.

#include "nmmb_common.cuh"

#define NG02_B2     12   /* module_radlw_parameters NG02 (radlw_param.f:107) */
#define NS02_B2     10   /* module_radlw_parameters NS02 (radlw_param.f:114) */
#define NGPTLW_B2  140   /* radlw_param_nmmb.f:141 */
#define NSPA2        1   /* nspa(2) (radlw_main_nmmb.f:284) */
#define NSPB2        1   /* nspb(2) (radlw_main_nmmb.f:285) */

// Table access. absa/absb/selfref/forref are stored [ind, ig] with ig fastest, as written by
// tools/extract_rrtm_lw_kgb02.py. ind is zero-based here.
#define TAB(tab, ind0, ig) ((tab)[(size_t)(ind0) * NG02_B2 + (ig)])

extern "C" __global__ void rrtm_lw_taugb02(
    const int ncol,
    const int nlay,
    const int* __restrict__ laytrop,      /* [ncol]          Fortran laytrop (count, 1-based layers) */
    const int* __restrict__ jp,           /* [nlay, ncol]    Fortran jp(k) */
    const int* __restrict__ jt,           /* [nlay, ncol]    Fortran jt(k) */
    const int* __restrict__ jt1,          /* [nlay, ncol]    Fortran jt1(k) */
    const float* __restrict__ pavel,      /* [nlay, ncol]    layer pressure (mb) */
    const float* __restrict__ colamt,     /* [nlay, ncol]    Fortran colamt(k,1), H2O column */
    const float* __restrict__ fac00,      /* [nlay, ncol] */
    const float* __restrict__ fac01,
    const float* __restrict__ fac10,
    const float* __restrict__ fac11,
    const float* __restrict__ selffac,    /* [nlay, ncol] */
    const float* __restrict__ selffrac,
    const int* __restrict__ indself,      /* [nlay, ncol]    Fortran indself(k) */
    const float* __restrict__ forfac,     /* [nlay, ncol] */
    const float* __restrict__ forfrac,
    const int* __restrict__ indfor,       /* [nlay, ncol]    Fortran indfor(k) */
    const float* __restrict__ absa,       /* [65, NG02]      k-table, lower atmosphere */
    const float* __restrict__ absb,       /* [235, NG02]     k-table, upper atmosphere */
    const float* __restrict__ selfref,    /* [10, NG02] */
    const float* __restrict__ forref,     /* [4, NG02] */
    const float* __restrict__ fracrefa,   /* [NG02] */
    const float* __restrict__ fracrefb,   /* [NG02] */
    float* __restrict__ taug_lw,          /* [nlay, NGPTLW, ncol] out: taug(ngpt,k), band-2 rows written */
    float* __restrict__ fracs_lw)         /* [nlay, NGPTLW, ncol] out: fracs(ngpt,k), band-2 rows written */
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    const int lt = laytrop[col];          /* Fortran k = 1..laytrop is the lower loop */
    const size_t lc = (size_t)col;        /* level k0 of this column is at k0*ncol + col */

    /* ---- lower atmosphere loop: Fortran do k = 1, laytrop ---- */
    for (int k0 = 0; k0 < lt; ++k0) {
        const size_t q = (size_t)k0 * ncol + lc;
        const int jp_k  = jp[q];
        const int jt_k  = jt[q];
        const int jt1_k = jt1[q];
        /* Fortran ind0 = ((jp-1)*5 + (jt-1))*nspa(2) + 1, zero-based here */
        const int ind0  = ((jp_k - 1) * 5 + (jt_k - 1)) * NSPA2;
        /* Fortran ind1 = (jp*5 + (jt1-1))*nspa(2) + 1, zero-based here */
        const int ind1  = (jp_k * 5 + (jt1_k - 1)) * NSPA2;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        /* Fortran inds = indself(k), indsp = inds+1, zero-based here */
        const int inds  = indself[q] - 1;
        const int indsp = inds + 1;
        const int indf  = indfor[q] - 1;
        const int indfp = indf + 1;

        const float corradj = 1.0f - 0.05f * (pavel[q] - 100.0f) / 900.0f;
        const float sf  = selffac[q];
        const float sfr = selffrac[q];
        const float ff  = forfac[q];
        const float ffr = forfrac[q];
        const float f00 = fac00[q];
        const float f01 = fac01[q];
        const float f10 = fac10[q];
        const float f11 = fac11[q];
        const float cam = colamt[q];

        for (int ig = 0; ig < NG02_B2; ++ig) {
            const float tauself = sf * (TAB(selfref, inds, ig) + sfr * (TAB(selfref, indsp, ig) - TAB(selfref, inds, ig)));
            const float taufor  = ff * (TAB(forref, indf, ig) + ffr * (TAB(forref, indfp, ig) - TAB(forref, indf, ig)));

            /* taug(ns02+ig,k) and fracs(ns02+ig,k) at zero-based row NS02_B2 + ig */
            const size_t o = ((size_t)k0 * NGPTLW_B2 + NS02_B2 + ig) * ncol + lc;
            taug_lw[o] = corradj * (cam * (f00 * TAB(absa, ind0, ig) + f10 * TAB(absa, ind0p, ig)
                                         + f01 * TAB(absa, ind1, ig) + f11 * TAB(absa, ind1p, ig))
                                    + tauself + taufor);
            fracs_lw[o] = fracrefa[ig];
        }
    }

    /* ---- upper atmosphere loop: Fortran do k = laytrop+1, nlay ---- */
    for (int k0 = lt; k0 < nlay; ++k0) {
        const size_t q = (size_t)k0 * ncol + lc;
        const int jp_k  = jp[q];
        const int jt_k  = jt[q];
        const int jt1_k = jt1[q];
        /* Fortran ind0 = ((jp-13)*5 + (jt-1))*nspb(2) + 1, zero-based here */
        const int ind0  = ((jp_k - 13) * 5 + (jt_k - 1)) * NSPB2;
        /* Fortran ind1 = ((jp-12)*5 + (jt1-1))*nspb(2) + 1, zero-based here */
        const int ind1  = ((jp_k - 12) * 5 + (jt1_k - 1)) * NSPB2;
        const int ind0p = ind0 + 1;
        const int ind1p = ind1 + 1;
        const int indf  = indfor[q] - 1;
        const int indfp = indf + 1;

        const float ff  = forfac[q];
        const float ffr = forfrac[q];
        const float f00 = fac00[q];
        const float f01 = fac01[q];
        const float f10 = fac10[q];
        const float f11 = fac11[q];
        const float cam = colamt[q];

        for (int ig = 0; ig < NG02_B2; ++ig) {
            const float taufor = ff * (TAB(forref, indf, ig) + ffr * (TAB(forref, indfp, ig) - TAB(forref, indf, ig)));

            const size_t o = ((size_t)k0 * NGPTLW_B2 + NS02_B2 + ig) * ncol + lc;
            taug_lw[o] = cam * (f00 * TAB(absb, ind0, ig) + f10 * TAB(absb, ind0p, ig)
                              + f01 * TAB(absb, ind1, ig) + f11 * TAB(absb, ind1p, ig))
                         + taufor;
            fracs_lw[o] = fracrefb[ig];
        }
    }
}
