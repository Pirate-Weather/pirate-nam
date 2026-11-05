// phys_rrtm_lw_taugb04.cu -- unit phys-rrtm-lw-taugb04: TAUGB04, RRTM LW band 4 optical depths
//   and Planck fractions, WOOF-NAM.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NOAA/NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE; no licence file in the NAM clone). The package
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
!  Copyright (c) 2002-2020, Atmospheric and Environmental                  |
!  Research, Inc. (AER). RRTMG is distributed by AER under the             |
!  BSD-3-Clause licence; the full text is in NOTICE. (This                 |
!  replaces the 2002-2007 terms in NAM's copy.)                            |
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
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:4098-4337 (nam.v4.2.11, 83df4e13c1).
// K-tables: module_radlw_kgb04 of phys/radlw_datatb.f:10030-15204 (the module taugb04 USEs at
//   radlw_main_nmmb.f:4105), extracted to namgpu/data/rrtm_lw/kgb04.npz by tools/extract_rrtm_lw_kgb04.py.
//
// Mapping. Fortran loops over k (layers) and ig (14 g-points) per column. Here one thread per
// (i, j, k): k is blockIdx.z and the lower/upper split comes from laytrop(i,j) (k <= laytrop is the
// lower loop, k > laytrop the upper loop, exactly the two Fortran loops). The ig loop stays inside the
// thread. Each Fortran assignment has a counterpart:
//   speccomb, specparm, specmult, js, fs, ind0                      lower loop and upper loop
//   speccomb1, specparm1, specmult1, js1, fs1, ind1                 same
//   speccomb_planck, specparm_planck, specmult_planck, jpl, fpl     both loops
//   inds, indf, indsp, indfp                                        indself(k), indfor(k) (Fortran values)
//   fk0/fk1/fk2 and id000..id211 three-case selection               lw4_lower_select (lower loop only)
//   upper loop fk = 1-fs, fs and ids ind0, ind0+5, ind0+1, ind0+6 as written in the Fortran
//   fac000..fac211                                                  fk * fac00/fac01/fac10/fac11
//   tauself, taufor (lower loop only: selfref, forref at inds/indsp and indf/indfp)
//   taug(ns04+ig,k) = tau_major + tau_major1 + tauself + taufor     (lower)
//   taug(ns04+ig,k) = tau_major + tau_major1                        (upper)
//   fracs(ns04+ig,k) = fracrefa(ig,jpl) + fpl*(fracrefa(ig,jplp)-fracrefa(ig,jpl))   (lower)
//   fracs(ns04+ig,k) = fracrefb(ig,jpl) + fpl*(fracrefb(ig,jplp)-fracrefb(ig,jpl))   (upper)
//   upper CO2 empirical scaling of taug(ns04+8..14,k) by 0.92 0.88 1.07 1.1 0.99 0.88 0.943
//     (Fortran ig = 8..14, band-local index 7..13 here)
// Constants: NG04 = 14, NSPA(4) = 9, NSPB(4) = 5 (radlw_main_nmmb.f:284-285), oneminus = 1 - 1e-6
//   (radlw_main_nmmb.f:267-268). Single precision; the Fortran 8.0 and 4.0 literals are single.
//   Outputs are the full 140-row LW g-point arrays (NGPTLW = 140, radlw_param_nmmb.f:141). The kernel
//   writes only rows NS04..NS04+13 (NS04 = 38, radlw_param_nmmb.f:160): taug(ns04+ig,k), fracs(ns04+ig,k).
// Single-thread stages: none. No halo, no print, no OpenMP branch.
//
// Layouts (CONTRACT section 1, Fortran (i,j,l) -> [l,j,i]; zero-based in CUDA):
//   laytrop [JM,IM] int32 (Fortran value);  colamt [6,nlay,JM,IM] (colamt(k,n) -> [n-1,k-1,j,i]);
//   rfrate [2,6,nlay,JM,IM] (rfrate(k,n,m) -> [m-1,n-1,k-1,j,i]);  jp, jt, jt1, indself, indfor
//   [nlay,JM,IM] int32 (Fortran 1-based values); fac00, fac01, fac10, fac11, selffac, selffrac,
//   forfac, forfrac [nlay,JM,IM] float32.
//   Tables, C-order [index, ig] with ig fastest (Fortran (ig, index)):
//     absa [585,14], absb [1175,14], selfref [10,14], forref [4,14], fracrefa [9,14], fracrefb [5,14].
//   Outputs: taug04 [nlay,140,JM,IM], fracs04 [nlay,140,JM,IM] (full g-point arrays, rows 38..51 written).
// One launch on ctx.stream, no host sync, no device malloc.

#include "nmmb_common.cuh"

#define LW4_NG       14      /* NG04, radlw_param_nmmb.f:153 */
#define LW4_NSPA     9       /* nspa(4), radlw_main_nmmb.f:284 */
#define LW4_NSPB     5       /* nspb(4), radlw_main_nmmb.f:285 */
#define LW4_NRATES   6
#define LW4_ONEMINUS (1.0f - 1.0e-6f)   /* radlw_main_nmmb.f:267-268 (eps = 1.0e-6) */
#define LW4_NGPT     140     /* NGPTLW, radlw_param_nmmb.f:141 */
#define LW4_NS04     38      /* NS04, radlw_param_nmmb.f:160 (zero-based row offset of band 4) */

// Index structure of one lower-atmosphere branch (the specparm < 0.125 / > 0.875 / otherwise cases).
// id[] = {id000, id010, id100, id110, id200, id210} for ind0; the same pattern gives id001..id211 for ind1.
struct Lw4Sel {
    float fk0, fk1, fk2;
    int   id[6];
};

__device__ inline Lw4Sel lw4_lower_select(float specparm, float fs, int ind)
{
    Lw4Sel s;
    if (specparm < 0.125f) {
        const float p  = fs - 1.0f;          /* p = fs - f_one */
        const float p4 = (p * p) * (p * p);  /* p**4 */
        s.fk0 = p4;
        s.fk1 = 1.0f - p - 2.0f * p4;
        s.fk2 = p + p4;
        s.id[0] = ind;      s.id[1] = ind + 9;  s.id[2] = ind + 1;
        s.id[3] = ind + 10; s.id[4] = ind + 2;  s.id[5] = ind + 11;
    } else if (specparm > 0.875f) {
        const float p  = -fs;
        const float p4 = (p * p) * (p * p);
        s.fk0 = p4;
        s.fk1 = 1.0f - p - 2.0f * p4;
        s.fk2 = p + p4;
        s.id[0] = ind + 1;  s.id[1] = ind + 10; s.id[2] = ind;
        s.id[3] = ind + 9;  s.id[4] = ind - 1;  s.id[5] = ind + 8;
    } else {
        s.fk0 = 1.0f - fs;
        s.fk1 = fs;
        s.fk2 = 0.0f;
        s.id[0] = ind;      s.id[1] = ind + 9;  s.id[2] = ind + 1;
        s.id[3] = ind + 10; s.id[4] = ind;      s.id[5] = ind;
    }
    return s;
}

// Fortran element (1-based index, zero-based g): arr[(idx-1)*NG + ig]
#define TAB14(arr, idx1, ig0) ((arr)[((size_t)((idx1) - 1)) * LW4_NG + (ig0)])

extern "C" __global__ void rrtm_lw_taugb04(
    const int*   __restrict__ laytrop,    /* [JM, IM] */
    const float* __restrict__ colamt,     /* [6, nlay, JM, IM] */
    const float* __restrict__ rfrate,     /* [2, 6, nlay, JM, IM] */
    const int*   __restrict__ jp,         /* [nlay, JM, IM] */
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
    const float* __restrict__ absa,       /* [585, 14] */
    const float* __restrict__ absb,       /* [1175, 14] */
    const float* __restrict__ selfref,    /* [10, 14] */
    const float* __restrict__ forref,     /* [4, 14] */
    const float* __restrict__ fracrefa,   /* [9, 14] */
    const float* __restrict__ fracrefb,   /* [5, 14] */
    const float refrat_planck_a,          /* chi_mls(1,11) / chi_mls(2,11) */
    const float refrat_planck_b,          /* chi_mls(3,13) / chi_mls(2,13) */
    const int IM, const int JM, const int NLAY,
    float* __restrict__ taug04,           /* [nlay, 140, JM, IM] (or the g-first [140, nlay, JM, IM], see ostr_*) */
    float* __restrict__ fracs04,          /* same layout as taug04 */
    const long long ostr_k,               /* output stride of one layer: 140*JM*IM layer first, JM*IM g first */
    const long long ostr_g)               /* output stride of one g-point: JM*IM layer first, nlay*JM*IM g first */
{
    POINT_IJL                              /* i0, j0, l0 = layer (zero-based) */
    if (i0 >= IM || j0 >= JM || l0 >= NLAY) return;

    const int k0 = l0;
    const int kf = k0 + 1;                 /* Fortran k */
    const size_t c2 = IDX2(i0, j0, IM);
    const size_t c3 = IDX3(i0, j0, k0, IM, JM);
#define CA(n0)     colamt[IDX4(i0, j0, k0, (n0), IM, JM, NLAY)]
#define RF(m0, n0) rfrate[IDX4(i0, j0, k0, ((m0) * LW4_NRATES + (n0)), IM, JM, NLAY)]
#define OUTG(ig0)  (((size_t)k0) * (size_t)ostr_k + (size_t)(LW4_NS04 + (ig0)) * (size_t)ostr_g + (size_t)j0 * IM + i0)

    const int lt   = laytrop[c2];
    const int jpk  = jp[c3];
    const int jtk  = jt[c3];
    const int jt1k = jt1[c3];
    const float f00 = fac00[c3], f01 = fac01[c3], f10 = fac10[c3], f11 = fac11[c3];
    const float sf  = selffac[c3], sfr = selffrac[c3];
    const float ff  = forfac[c3],  ffr = forfrac[c3];
    const int inds  = indself[c3];
    const int indf  = indfor[c3];
    const int indsp = inds + 1;
    const int indfp = indf + 1;

    if (kf <= lt) {
        /* ----- lower loop: do k = 1, laytrop ----- */
        const float speccomb = CA(0) + RF(0, 0) * CA(1);          /* rfrate(k,1,1) */
        const float specparm = CA(0) / speccomb;
        const float specmult = 8.0f * fminf(specparm, LW4_ONEMINUS);
        const int   js  = 1 + (int)specmult;
        const float fs  = fmodf(specmult, 1.0f);
        const int   ind0 = ((jpk - 1) * 5 + (jtk - 1)) * LW4_NSPA + js;

        const float speccomb1 = CA(0) + RF(1, 0) * CA(1);         /* rfrate(k,1,2) */
        const float specparm1 = CA(0) / speccomb1;
        const float specmult1 = 8.0f * fminf(specparm1, LW4_ONEMINUS);
        const int   js1  = 1 + (int)specmult1;
        const float fs1  = fmodf(specmult1, 1.0f);
        const int   ind1 = (jpk * 5 + (jt1k - 1)) * LW4_NSPA + js1;

        const float speccomb_planck = CA(0) + refrat_planck_a * CA(1);
        const float specparm_planck = CA(0) / speccomb_planck;
        const float specmult_planck = 8.0f * fminf(specparm_planck, LW4_ONEMINUS);
        const int   jpl  = 1 + (int)specmult_planck;
        const float fpl  = fmodf(specmult_planck, 1.0f);
        const int   jplp = jpl + 1;

        const Lw4Sel s0 = lw4_lower_select(specparm, fs, ind0);
        const Lw4Sel s1 = lw4_lower_select(specparm1, fs1, ind1);

        const float fac000 = s0.fk0 * f00, fac100 = s0.fk1 * f00, fac200 = s0.fk2 * f00;
        const float fac010 = s0.fk0 * f10, fac110 = s0.fk1 * f10, fac210 = s0.fk2 * f10;
        const float fac001 = s1.fk0 * f01, fac101 = s1.fk1 * f01, fac201 = s1.fk2 * f01;
        const float fac011 = s1.fk0 * f11, fac111 = s1.fk1 * f11, fac211 = s1.fk2 * f11;

        for (int ig = 0; ig < LW4_NG; ++ig) {
            const float tauself = sf * (TAB14(selfref, inds, ig)
                                        + sfr * (TAB14(selfref, indsp, ig) - TAB14(selfref, inds, ig)));
            const float taufor  = ff * (TAB14(forref, indf, ig)
                                        + ffr * (TAB14(forref, indfp, ig) - TAB14(forref, indf, ig)));

            const float tau_major = speccomb
                * (fac000 * TAB14(absa, s0.id[0], ig) + fac010 * TAB14(absa, s0.id[1], ig)
                 + fac100 * TAB14(absa, s0.id[2], ig) + fac110 * TAB14(absa, s0.id[3], ig)
                 + fac200 * TAB14(absa, s0.id[4], ig) + fac210 * TAB14(absa, s0.id[5], ig));

            const float tau_major1 = speccomb1
                * (fac001 * TAB14(absa, s1.id[0], ig) + fac011 * TAB14(absa, s1.id[1], ig)
                 + fac101 * TAB14(absa, s1.id[2], ig) + fac111 * TAB14(absa, s1.id[3], ig)
                 + fac201 * TAB14(absa, s1.id[4], ig) + fac211 * TAB14(absa, s1.id[5], ig));

            taug04[OUTG(ig)]  = tau_major + tau_major1 + tauself + taufor;
            fracs04[OUTG(ig)] = TAB14(fracrefa, jpl, ig)
                              + fpl * (TAB14(fracrefa, jplp, ig) - TAB14(fracrefa, jpl, ig));
        }
    } else {
        /* ----- upper loop: do k = laytrop+1, nlay ----- */
        const float speccomb = CA(2) + RF(0, 5) * CA(1);          /* rfrate(k,6,1) */
        const float specparm = CA(2) / speccomb;
        const float specmult = 4.0f * fminf(specparm, LW4_ONEMINUS);
        const int   js  = 1 + (int)specmult;
        const float fs  = fmodf(specmult, 1.0f);
        const int   ind0 = ((jpk - 13) * 5 + (jtk - 1)) * LW4_NSPB + js;

        const float speccomb1 = CA(2) + RF(1, 5) * CA(1);         /* rfrate(k,6,2) */
        const float specparm1 = CA(2) / speccomb1;
        const float specmult1 = 4.0f * fminf(specparm1, LW4_ONEMINUS);
        const int   js1  = 1 + (int)specmult1;
        const float fs1  = fmodf(specmult1, 1.0f);
        const int   ind1 = ((jpk - 12) * 5 + (jt1k - 1)) * LW4_NSPB + js1;

        const float speccomb_planck = CA(2) + refrat_planck_b * CA(1);
        const float specparm_planck = CA(2) / speccomb_planck;
        const float specmult_planck = 4.0f * fminf(specparm_planck, LW4_ONEMINUS);
        const int   jpl  = 1 + (int)specmult_planck;
        const float fpl  = fmodf(specmult_planck, 1.0f);
        const int   jplp = jpl + 1;

        const int id000 = ind0, id010 = ind0 + 5, id100 = ind0 + 1, id110 = ind0 + 6;
        const int id001 = ind1, id011 = ind1 + 5, id101 = ind1 + 1, id111 = ind1 + 6;

        float fk0 = 1.0f - fs, fk1 = fs;
        const float fac000 = fk0 * f00, fac010 = fk0 * f10, fac100 = fk1 * f00, fac110 = fk1 * f10;
        fk0 = 1.0f - fs1; fk1 = fs1;
        const float fac001 = fk0 * f01, fac011 = fk0 * f11, fac101 = fk1 * f01, fac111 = fk1 * f11;

        for (int ig = 0; ig < LW4_NG; ++ig) {
            const float tau_major = speccomb
                * (fac000 * TAB14(absb, id000, ig) + fac010 * TAB14(absb, id010, ig)
                 + fac100 * TAB14(absb, id100, ig) + fac110 * TAB14(absb, id110, ig));
            const float tau_major1 = speccomb1
                * (fac001 * TAB14(absb, id001, ig) + fac011 * TAB14(absb, id011, ig)
                 + fac101 * TAB14(absb, id101, ig) + fac111 * TAB14(absb, id111, ig));

            float tg = tau_major + tau_major1;
            /* empirical CO2 stratospheric-cooling modification; band-local ig 7..13 = Fortran g 8..14 */
            switch (ig) {
                case 7:  tg = tg * 0.92f;  break;
                case 8:  tg = tg * 0.88f;  break;
                case 9:  tg = tg * 1.07f;  break;
                case 10: tg = tg * 1.1f;   break;
                case 11: tg = tg * 0.99f;  break;
                case 12: tg = tg * 0.88f;  break;
                case 13: tg = tg * 0.943f; break;
                default: break;
            }
            taug04[OUTG(ig)]  = tg;
            fracs04[OUTG(ig)] = TAB14(fracrefb, jpl, ig)
                              + fpl * (TAB14(fracrefb, jplp, ig) - TAB14(fracrefb, jpl, ig));
        }
    }
#undef CA
#undef RF
#undef OUTG
}
