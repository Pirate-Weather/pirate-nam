// phys_rrtm_lw_rlwinit.cu -- unit phys-rrtm-lw-rlwinit: RLWINIT, nam-gpu.
//
// NOTICE (carried from the source): the NOAA/NCEP NMMB header of radlw_main_nmmb.f
// (nam.v4.2.11, clone commit 83df4e13c1), lines 1-234, reproduced verbatim below.
// This kernel is a port of SUBROUTINE rlwinit, radlw_main_nmmb.f:1145-1339.
// ----- begin radlw_main_nmmb.f header (lines 1-234) -----
// !!!!!  ==============================================================  !!!!!
// !!!!!               lw-rrtm3 radiation package description             !!!!!
// !!!!!  ==============================================================  !!!!!
// !                                                                          !
// !   this package includes ncep's modifications of the rrtm-lw radiation    !
// !   code from aer inc.                                                     !
// !                                                                          !
// !    the lw-rrtm3 package includes these parts:                            !
// !                                                                          !
// !       'radlw_rrtm3_param.f'                                              !
// !       'radlw_rrtm3_datatb.f'                                             !
// !       'radlw_rrtm3_main.f'                                               !
// !                                                                          !
// !    the 'radlw_rrtm3_param.f' contains:                                   !
// !                                                                          !
// !       'module_radlw_parameters'  -- band parameters set up               !
// !                                                                          !
// !    the 'radlw_rrtm3_datatb.f' contains:                                  !
// !                                                                          !
// !       'module_radlw_avplank'     -- plank flux data                      !
// !       'module_radlw_ref'         -- reference temperature and pressure   !
// !       'module_radlw_cldprlw'     -- cloud property coefficients          !
// !       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
// !                                     bands, where nn = 01-16              !
// !                                                                          !
// !    the 'radlw_rrtm3_main.f' contains:                                    !
// !                                                                          !
// !       'module_radlw_main'        -- main lw radiation transfer           !
// !                                                                          !
// !    in the main module 'module_radlw_main' there are only two             !
// !    externally callable subroutines:                                      !
// !                                                                          !
// !                                                                          !
// !       'lwrad'     -- main lw radiation routine                           !
// !          inputs:                                                         !
// !           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
// !            clouds,icseed,aerosols,sfemis,sfgtmp,                         !
// !            npts, nlay, nlp1, lprnt,                                      !
// !          outputs:                                                        !
// !            hlwc,topflx,sfcflx,                                           !
// !!         optional outputs:                                               !
// !            HLW0,HLWB,FLXPRF)                                             !
// !                                                                          !
// !       'rlwinit'   -- initialization routine                              !
// !          inputs:                                                         !
// !           ( me )                                                         !
// !          outputs:                                                        !
// !           (none)                                                         !
// !                                                                          !
// !    all the lw radiation subprograms become contained subprograms         !
// !    in module 'module_radlw_main' and many of them are not directly       !
// !    accessable from places outside the module.                            !
// !                                                                          !
// !    derived data type constructs used:                                    !
// !                                                                          !
// !     1. radiation flux at toa: (from module 'module_radlw_parameters')    !
// !          topflw_type   -  derived data type for toa rad fluxes           !
// !            upfxc              total sky upward flux at toa               !
// !            upfx0              clear sky upward flux at toa               !
// !                                                                          !
// !     2. radiation flux at sfc: (from module 'module_radlw_parameters')    !
// !          sfcflw_type   -  derived data type for sfc rad fluxes           !
// !            upfxc              total sky upward flux at sfc               !
// !            upfx0              clear sky upward flux at sfc               !
// !            dnfxc              total sky downward flux at sfc             !
// !            dnfx0              clear sky downward flux at sfc             !
// !                                                                          !
// !     3. radiation flux profiles(from module 'module_radlw_parameters')    !
// !          proflw_type    -  derived data type for rad vertical prof       !
// !            upfxc              level upward flux for total sky            !
// !            dnfxc              level downward flux for total sky          !
// !            upfx0              level upward flux for clear sky            !
// !            dnfx0              level downward flux for clear sky          !
// !                                                                          !
// !    external modules referenced:                                          !
// !                                                                          !
// !       'module physparam'                                                 !
// !       'module physcons'                                                  !
// !       'mersenne_twister'                                                 !
// !                                                                          !
// !    compilation sequence is:                                              !
// !                                                                          !
// !       'radlw_rrtm3_param.f'                                              !
// !       'radlw_rrtm3_datatb.f'                                             !
// !       'radlw_rrtm3_main.f'                                               !
// !                                                                          !
// !    and all should be put in front of routines that use lw modules        !
// !                                                                          !
// !==========================================================================!
// !                                                                          !
// !    the original aer's program declarations:                              !
// !                                                                          !
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !                                                                          |
// !  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
// !  This software may be used, copied, or redistributed as long as it is    |
// !  not sold and this copyright notice is reproduced on each copy made.     |
// !  This model is provided as is without any express or implied warranties. |
// !                       (http://www.rtweb.aer.com/)                        |
// !                                                                          |
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !                              rrtmg_lw                                    !
// !                                                                          !
// !                                                                          !
// !                   a rapid radiative transfer model                       !
// !                       for the longwave region                            !
// !             for application to general circulation models                !
// !                                                                          !
// !                                                                          !
// !            atmospheric and environmental research, inc.                  !
// !                        131 hartwell avenue                               !
// !                        lexington, ma 02421                               !
// !                                                                          !
// !                           eli j. mlawer                                  !
// !                        jennifer s. delamere                              !
// !                         michael j. iacono                                !
// !                         shepard a. clough                                !
// !                                                                          !
// !                                                                          !
// !                       email:  miacono@aer.com                            !
// !                       email:  emlawer@aer.com                            !
// !                       email:  jdelamer@aer.com                           !
// !                                                                          !
// !        the authors wish to acknowledge the contributions of the          !
// !        following people:  steven j. taubman, karen cady-pereira,         !
// !        patrick d. brown, ronald e. farren, luke chen, robert bergstrom.  !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !    references:                                                           !
// !    (rrtm_lw/rrtmg_lw):                                                   !
// !      clough, s.A., m.w. shephard, e.j. mlawer, j.s. delamere,            !
// !      m.j. iacono, k. cady-pereira, s. boukabara, and p.d. brown:         !
// !      atmospheric radiative transfer modeling: a summary of the aer       !
// !      codes, j. quant. spectrosc. radiat. transfer, 91, 233-244, 2005.    !
// !                                                                          !
// !      mlawer, e.j., s.j. taubman, p.d. brown, m.j. iacono, and s.a.       !
// !      clough:  radiative transfer for inhomogeneous atmospheres: rrtm,    !
// !      a validated correlated-k model for the longwave.  j. geophys. res., !
// !      102, 16663-16682, 1997.                                             !
// !                                                                          !
// !    (mcica):                                                              !
// !      pincus, r., h. w. barker, and j.-j. morcrette: a fast, flexible,    !
// !      approximation technique for computing radiative transfer in         !
// !      inhomogeneous cloud fields, j. geophys. res., 108(d13), 4376,       !
// !      doi:10.1029/2002JD003322, 2003.                                     !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !    aer's revision history:                                               !
// !     this version of rrtmg_lw has been modified from rrtm_lw to use a     !
// !     reduced set of g-points for application to gcms.                     !
// !                                                                          !
// ! --  original version (derived from rrtm_lw), reduction of g-points,      !
// !     other revisions for use with gcms.                                   !
// !        1999: m. j. iacono, aer, inc.                                     !
// ! --  adapted for use with ncar/cam3.                                      !
// !        may 2004: m. j. iacono, aer, inc.                                 !
// ! --  revised to add mcica capability.                                     !
// !        nov 2005: m. j. iacono, aer, inc.                                 !
// ! --  conversion to f90 formatting for consistency with rrtmg_sw.          !
// !        feb 2007: m. j. iacono, aer, inc.                                 !
// ! --  modifications to formatting to use assumed-shape arrays.             !
// !        aug 2007: m. j. iacono, aer, inc.                                 !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !    ncep modifications history log:                                       !
// !                                                                          !
// !       nov 1999,  ken campana       -- received the original code from    !
// !                    aer (1998 ncar ccm version), updated to link up with  !
// !                    ncep mrf model                                        !
// !       jun 2000,  ken campana       -- added option to switch random and  !
// !                    maximum/random cloud overlap                          !
// !           2001,  shrinivas moorthi -- further updates for mrf model      !
// !       may 2001,  yu-tai hou        -- updated on trace gases and cloud   !
// !                    property based on rrtm_v3.0 codes.                    !
// !       dec 2001,  yu-tai hou        -- rewritten code into fortran 90 std !
// !                    set ncep radiation structure standard that contains   !
// !                    three plug-in compatable fortran program files:       !
// !                    'radlw_param.f', 'radlw_datatb.f', 'radlw_main.f'     !
// !                    fixed bugs in subprograms taugb14, taugb2, etc. added !
// !                    out-of-bounds protections. (a detailed note of        !
// !                    up_to_date modifications/corrections by ncep was sent !
// !                    to aer in 2002)                                       !
// !       jun 2004,  yu-tai hou        -- added mike iacono's apr 2004       !
// !                    modification of variable diffusivity angles.          !
// !       apr 2005,  yu-tai hou        -- minor modifications on module      !
// !                    structures include rain/snow effect (this version of  !
// !                    code was given back to aer in jun 2006)               !
// !       mar 2007,  yu-tai hou        -- added aerosol effect for ncep      !
// !                    models using the generallized aerosol optical property!
// !                    scheme for gfs model.                                 !
// !       apr 2007,  yu-tai hou        -- added spectral band heating as an  !
// !                    optional output to support the 500 km gfs model's     !
// !                    upper stratospheric radiation calculations. and       !
// !                    restructure optional outputs for easy access by       !
// !                    different models.                                     !
// !       oct 2008,  yu-tai hou        -- modified to include new features   !
// !                    from aer's newer release v4.4-v4.7, including the     !
// !                    mcica sub-grid cloud option. add rain/snow optical    !
// !                    properties support to cloudy sky calculations.        !
// !                    correct errors in mcica cloud optical properties for  !
// !                    ebert & curry scheme (ilwcice=1) that needs band      !
// !                    index conversion. simplified and unified sw and lw    !
// !                    sub-column cloud subroutines into one module by using !
// !                    optional parameters.                                  !
// !       mar 2009,  yu-tai hou        -- replaced the original random number!
// !                    generator coming from the original code with ncep w3  !
// !                    library to simplify the program and moved sub-column  !
// !                    cloud subroutines inside the main module. added       !
// !                    option of user provided permutation seeds that could  !
// !                    be randomly generated from forecast time stamp.       !
// !       oct 2009,  yu-tai hou        -- modified subrtines "cldprop" and   !
// !                    "rlwinit" according updats from aer's rrtmg_lw v4.8.  !
// !       nov 2009,  yu-tai hou        -- modified subrtine "taumol" according
// !                    updats from aer's rrtmg_lw version 4.82. notice the   !
// !                    cloud ice/liquid are assumed as in-cloud quantities,  !
// !                    not as grid averaged quantities.                      !
// !       jun 2010,  yu-tai hou        -- optimized code to improve efficiency
// !       apr 2012,  b. ferrier and y. hou -- added conversion factor to fu's!
// !                    cloud-snow optical property scheme.                   !
// !       nov 2012,  yu-tai hou        -- modified control parameters thru   !
// !                     module 'physparam'.                                  !
// !                                                                          !
// !!!!!  ==============================================================  !!!!!
// !!!!!                         end descriptions                         !!!!!
// !!!!!  ==============================================================  !!!!!
//
//
// !========================================!
// ----- end radlw_main_nmmb.f header -----
//
// What the port computes, with the source lines (radlw_main_nmmb.f unless noted):
//  - Parameters: BPADE = 1.0/0.278 (:270, single default-real constant); NTBL = 10000 from
//    radlw_param.f:96 (module_radlw_parameters, the module radlw_main_nmmb.f USEs; radlw_param_nmmb.f
//    defines module_radlw_parameters_nmmb and has no NTBL); expeps = 1.e-20 (rlwinit local, :1220).
//  - semiss0(:) = f_one (:1281), NBANDS = 16 (radlw_param_nmmb.f:140) entries.
//  - pival = 2.0*asin(f_one) (:1286) = 3.14159274 (single). fluxfac = pival * 2.0d4 (:1287): the
//    Fortran multiplies a single by a double literal, so the product is formed in double and rounded
//    to single, which the kernel reproduces.
//  - heatfac (:1290-1296): ilwrate == 1 -> con_g*864.0/con_cp (k/day); otherwise con_g*1.0e-2/con_cp
//    (k/second). physparam.f:71 sets ilwrate = 2 (k/second). con_g = 9.80665, con_cp = 1004.6
//    (physcons.f:46, :59).
//  - Endpoints (:1309-1315): index 0 gives tau = 0, exp = 1, tfn = 0; index NTBL gives tau = 1e10,
//    exp = expeps, tfn = 1.
//  - explimit = aint(-log(tiny(exp_tbl(0)))) (:1317). Precision note (verified in this unit): the
//    value 87.0 is AINT(-LOG(TINY(single))) = AINT(87.33654). It is only correct for kind_phys = 4.
//    machine.f:5-20 sets kind_phys = 4 only under SINGLE_PREC (else 8, where the value would be
//    AINT(708.396) = 708). The port is single by the program precision choice (CONTRACT: kfpt=single).
//    The NMMB build recorded at gate0/logs/build_nmmb_3.log (the radlw_main_nmmb.f compile, line 1356)
//    passes -fdefault-real-8 -fdefault-double-8 and does NOT define SINGLE_PREC, so that build runs
//    kind_phys = 8. Declared divergence from that build until the project lead rules on NAM precision; this kernel
//    does not claim to reproduce the gate0 build's precision.
//  - Table loop (:1319-1336), NTBL-1 entries, one thread per index: tfn = real(i)/real(ntbl-i);
//    tau = bpade*tfn; exp = expeps if tau >= explimit else exp(-tau); tfn_tbl = tau/6 for tau < 0.06
//    (:1330-1331), else the Pade form (:1333-1334).
//  - The Fortran checks on IOVRLW / ISUBCLW / ICLDFLG / ILWCLIQ and the iovrlw = 2 -> 1 adjustment
//    (:1229-1241, with the checks at :1254-1276) are scalar control logic. They run on the host in the
//    wrapper (phys_rrtm_lw_rlwinit.py). The adjusted values reach the consumers only through
//    ctx.tables["rrtm_lw_ctl_host"] (host dict) and ctx.tables["rrtm_lw_ctl"] (device int32[4]).
//    Consumers must read those, never ctx.cfg, because RLWINIT changes iovrlw and owns the module state.
//  - All "print *" statements (:1237-1263 and the banner) are dropped: no print inside kernels.
//  - The module variables semiss0, fluxfac, heatfac, tau_tbl, exp_tbl, tfn_tbl are written to device
//    arrays owned by ctx.tables (same names with the prefix rrtm_lw_). The lwrad unit reads them.
//
// Outputs (device, float32):
//   tau_tbl[NTBL+1], exp_tbl[NTBL+1], tfn_tbl[NTBL+1] : lookup tables
//   semiss0[NBANDS]                                     : default surface emissivity, all 1.0
//   consts[3] = { pival, fluxfac, heatfac }
// One launch on ctx.stream, no host sync, no device malloc.

#include "nmmb_common.cuh"

#define RLW_NTBL        10000               /* NTBL, radlw_param.f:96 */
#define RLW_NBANDS      16                  /* NBANDS, radlw_param.f:94 */
#define RLW_BPADE       (1.0f / 0.278f)     /* radlw_main.f:270, default-real constant */
#define RLW_EXPEPS      1.0e-20f            /* rlwinit local expeps */
#define RLW_EXPLIMIT    87.0f               /* AINT(-LOG(TINY(single))) = AINT(87.33654) */
#define RLW_PIVAL       3.14159274f         /* 2.0*ASIN(1.0) in single */
#define RLW_CON_G       9.80665f            /* physcons.f:46 */
#define RLW_CON_CP      1.0046e+3f          /* physcons.f:59 */

extern "C" __global__ void rlwinit_tables(
    float* __restrict__ tau_tbl,      /* [NTBL+1] */
    float* __restrict__ exp_tbl,      /* [NTBL+1] */
    float* __restrict__ tfn_tbl,      /* [NTBL+1] */
    float* __restrict__ semiss0,      /* [NBANDS] */
    float* __restrict__ consts,       /* [3]: pival, fluxfac, heatfac */
    const int ilwrate)                /* physparam.f:71, 2 (k/second) */
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    // semiss0(:) = f_one  (radlw_main.f:1277)
    if (t < RLW_NBANDS) semiss0[t] = 1.0f;

    if (t == 0) {
        // Scalars (:1281-1291)
        const float pival = RLW_PIVAL;
        const double fluxfac_d = (double)pival * 2.0e4;   // pival * 2.0d4 (double product)
        float heatfac;
        if (ilwrate == 1) {
            heatfac = RLW_CON_G * 864.0f / RLW_CON_CP;    // con_g * 864.0 / con_cp (k/day)
        } else {
            heatfac = RLW_CON_G * 1.0e-2f / RLW_CON_CP;   // con_g * 1.0e-2 / con_cp (k/second)
        }
        consts[0] = pival;
        consts[1] = (float)fluxfac_d;
        consts[2] = heatfac;

        // Endpoint 0 (:1305-1307)
        tau_tbl[0] = 0.0f;
        exp_tbl[0] = 1.0f;
        tfn_tbl[0] = 0.0f;
    }

    if (t == RLW_NTBL) {
        // Endpoint NTBL (:1309-1311)
        tau_tbl[RLW_NTBL] = 1.0e10f;
        exp_tbl[RLW_NTBL] = RLW_EXPEPS;
        tfn_tbl[RLW_NTBL] = 1.0f;
    }

    if (t >= 1 && t <= RLW_NTBL - 1) {
        const int i = t;
        // tfn = real(i)/real(ntbl-i); tau_tbl(i) = bpade*tfn  (:1315-1319)
        const float tfn = (float)i / (float)(RLW_NTBL - i);
        const float tau = RLW_BPADE * tfn;
        tau_tbl[i] = tau;

        // exp_tbl(i) (:1320-1324)
        float ex;
        if (tau >= RLW_EXPLIMIT) {
            ex = RLW_EXPEPS;
        } else {
            ex = expf(-tau);
        }
        exp_tbl[i] = ex;

        // tfn_tbl(i) (:1326-1332)
        if (tau < 0.06f) {
            tfn_tbl[i] = tau / 6.0f;
        } else {
            tfn_tbl[i] = 1.0f - 2.0f * ((1.0f / tau) - (ex / (1.0f - ex)));
        }
    }
}
