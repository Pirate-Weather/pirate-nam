// phys_rrtm_sw_rswinit.cu -- unit phys-rrtm-sw-rswinit: RSWINIT, nam-gpu.
//
// NOTICE: NOAA/NCEP NAM source file sorc/nam_nems_nmmb.fd/src/atmos/phys/radsw_main_nmmb.f (nam.v4.2.11,
//   83df4e13c1), US government work, no licence file in the NAM clone. Its package header (Fortran
//   lines 1-258, including the AER copyright notice) is reproduced verbatim below. Subroutine rswinit
//   is at lines 1244-1395.
// !!!!!  ==============================================================  !!!!!
// !!!!!              sw-rrtm3 radiation package description              !!!!!
// !!!!!  ==============================================================  !!!!!
// !                                                                          !
// !   this package includes ncep's modifications of the rrtm-sw radiation    !
// !   code from aer inc.                                                     !
// !                                                                          !
// !   the sw-rrtm3 package includes these parts:                             !
// !                                                                          !
// !      'radsw_rrtm3_param.f'                                               !
// !      'radsw_rrtm3_datatb.f'                                              !
// !      'radsw_rrtm3_main.f'                                                !
// !                                                                          !
// !   the 'radsw_rrtm3_param.f' contains:                                    !
// !                                                                          !
// !      'module_radsw_parameters'  -- band parameters set up                !
// !                                                                          !
// !   the 'radsw_rrtm3_datatb.f' contains:                                   !
// !                                                                          !
// !      'module_radsw_ref'         -- reference temperature and pressure    !
// !      'module_radsw_cldprtb'     -- cloud property coefficients table     !
// !      'module_radsw_sflux'       -- spectral distribution of solar flux   !
// !      'module_radsw_kgbnn'       -- absorption coeffients for 14          !
// !                                    bands, where nn = 16-29               !
// !                                                                          !
// !   the 'radsw_rrtm3_main.f' contains:                                     !
// !                                                                          !
// !      'module_radsw_main'        -- main sw radiation transfer            !
// !                                                                          !
// !   in the main module 'module_radsw_main' there are only two              !
// !   externally callable subroutines:                                       !
// !                                                                          !
// !      'swrad'      -- main sw radiation routine                           !
// !         inputs:                                                          !
// !           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
// !            clouds,icseed,aerosols,sfcalb,                                !
// !            cosz,solcon,NDAY,idxday,                                      !
// !            npts, nlay, nlp1, lprnt,                                      !
// !         outputs:                                                         !
// !            hswc,topflx,sfcflx,                                           !
// !!        optional outputs:                                                !
// !            HSW0,HSWB,FLXPRF,FDNCMP)                                      !
// !           )                                                              !
// !                                                                          !
// !      'rswinit'    -- initialization routine                              !
// !         inputs:                                                          !
// !           ( me )                                                         !
// !         outputs:                                                         !
// !           (none)                                                         !
// !                                                                          !
// !   all the sw radiation subprograms become contained subprograms          !
// !   in module 'module_radsw_main' and many of them are not directly        !
// !   accessable from places outside the module.                             !
// !                                                                          !
// !    derived data type constructs used:                                    !
// !                                                                          !
// !     1. radiation flux at toa: (from module 'module_radsw_parameters')    !
// !          topfsw_type   -  derived data type for toa rad fluxes           !
// !            upfxc              total sky upward flux at toa               !
// !            dnfxc              total sky downward flux at toa             !
// !            upfx0              clear sky upward flux at toa               !
// !                                                                          !
// !     2. radiation flux at sfc: (from module 'module_radsw_parameters')    !
// !          sfcfsw_type   -  derived data type for sfc rad fluxes           !
// !            upfxc              total sky upward flux at sfc               !
// !            dnfxc              total sky downward flux at sfc             !
// !            upfx0              clear sky upward flux at sfc               !
// !            dnfx0              clear sky downward flux at sfc             !
// !                                                                          !
// !     3. radiation flux profiles(from module 'module_radsw_parameters')    !
// !          profsw_type    -  derived data type for rad vertical prof       !
// !            upfxc              level upward flux for total sky            !
// !            dnfxc              level downward flux for total sky          !
// !            upfx0              level upward flux for clear sky            !
// !            dnfx0              level downward flux for clear sky          !
// !                                                                          !
// !     4. surface component fluxes(from module 'module_radsw_parameters'    !
// !          cmpfsw_type    -  derived data type for component sfc flux      !
// !            uvbfc              total sky downward uv-b flux at sfc        !
// !            uvbf0              clear sky downward uv-b flux at sfc        !
// !            nirbm              surface downward nir direct beam flux      !
// !            nirdf              surface downward nir diffused flux         !
// !            visbm              surface downward uv+vis direct beam flx    !
// !            visdf              surface downward uv+vis diffused flux      !
// !                                                                          !
// !   external modules referenced:                                           !
// !                                                                          !
// !       'module physparam'                                                 !
// !       'module physcons'                                                  !
// !       'mersenne_twister'                                                 !
// !                                                                          !
// !   compilation sequence is:                                               !
// !                                                                          !
// !      'radsw_rrtm3_param.f'                                               !
// !      'radsw_rrtm3_datatb.f'                                              !
// !      'radsw_rrtm3_main.f'                                                !
// !                                                                          !
// !   and all should be put in front of routines that use sw modules         !
// !                                                                          !
// !==========================================================================!
// !                                                                          !
// !   the original program declarations:                                     !
// !                                                                          !
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !                                                                          !
// !  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  !
// !  This software may be used, copied, or redistributed as long as it is    !
// !  not sold and this copyright notice is reproduced on each copy made.     !
// !  This model is provided as is without any express or implied warranties. !
// !                       (http://www.rtweb.aer.com/)                        !
// !                                                                          !
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !                              rrtmg_sw                                    !
// !                                                                          !
// !                                                                          !
// !                   a rapid radiative transfer model                       !
// !                    for the solar spectral region                         !
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
// !        following people:  steven j. taubman, patrick d. brown,           !
// !        ronald e. farren, luke chen, robert bergstrom.                    !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !    references:                                                           !
// !    (rrtm_sw/rrtmg_sw):                                                   !
// !      clough, s.a., m.w. shephard, e.j. mlawer, j.s. delamere,            !
// !      m.j. iacono, k. cady-pereira, s. boukabara, and p.d. brown:         !
// !      atmospheric radiative transfer modeling: a summary of the aer       !
// !      codes, j. quant. spectrosc. radiat. transfer, 91, 233-244, 2005.    !
// !                                                                          !
// !    (mcica):                                                              !
// !      pincus, r., h. w. barker, and j.-j. morcrette: a fast, flexible,    !
// !      approximation technique for computing radiative transfer in         !
// !      inhomogeneous cloud fields, j. geophys. res., 108(d13), 4376,       !
// !      doi:10.1029/2002jd003322, 2003.                                     !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !    aer's revision history:                                               !
// !     this version of rrtmg_sw has been modified from rrtm_sw to use a     !
// !     reduced set of g-point intervals and a two-stream model for          !
// !     application to gcms.                                                 !
// !                                                                          !
// ! --  original version (derived from rrtm_sw)                              !
// !        2002: aer. inc.                                                   !
// ! --  conversion to f90 formatting; addition of 2-stream radiative transfer!
// !        feb 2003: j.-j. morcrette, ecmwf                                  !
// ! --  additional modifications for gcm application                         !
// !        aug 2003: m. j. iacono, aer inc.                                  !
// ! --  total number of g-points reduced from 224 to 112.  original          !
// !     set of 224 can be restored by exchanging code in module parrrsw.f90  !
// !     and in file rrtmg_sw_init.f90.                                       !
// !        apr 2004: m. j. iacono, aer, inc.                                 !
// ! --  modifications to include output for direct and diffuse               !
// !     downward fluxes.  there are output as "true" fluxes without          !
// !     any delta scaling applied.  code can be commented to exclude         !
// !     this calculation in source file rrtmg_sw_spcvrt.f90.                 !
// !        jan 2005: e. j. mlawer, m. j. iacono, aer, inc.                   !
// ! --  revised to add mcica capability.                                     !
// !        nov 2005: m. j. iacono, aer, inc.                                 !
// ! --  reformatted for consistency with rrtmg_lw.                           !
// !        feb 2007: m. j. iacono, aer, inc.                                 !
// ! --  modifications to formatting to use assumed-shape arrays.             !
// !        aug 2007: m. j. iacono, aer, inc.                                 !
// !                                                                          !
// ! ************************************************************************ !
// !                                                                          !
// !   ncep modifications history log:                                        !
// !                                                                          !
// !       sep 2003,  yu-tai hou        -- received aer's rrtm-sw gcm version !
// !                    code (v224)                                           !
// !       nov 2003,  yu-tai hou        -- corrected errors in direct/diffuse !
// !                    surface alabedo components.                           !
// !       jan 2004,  yu-tai hou        -- modified code into standard modular!
// !                    f9x code for ncep models. the original three cloud    !
// !                    control flags are simplified into two: iflagliq and   !
// !                    iflagice. combined the org subr sw_224 and setcoef    !
// !                    into radsw (the main program); put all kgb##together  !
// !                    and reformat into a separated data module; combine    !
// !                    reftra and vrtqdr as swflux; optimized taumol and all !
// !                    taubgs to form a contained subroutines.               !
// !       jun 2004,  yu-tai hou        -- modified code based on aer's faster!
// !                    version rrtmg_sw (v2.0) with 112 g-points.            !
// !       mar 2005,  yu-tai hou        -- modified to aer v2.3, correct cloud!
// !                    scaling error, total sky properties are delta scaled  !
// !                    after combining clear and cloudy parts. the testing   !
// !                    criterion of ssa is saved before scaling. added cloud !
// !                    layer rain and snow contributions. all cloud water    !
// !                    partical contents are treated the same way as other   !
// !                    atmos particles.                                      !
// !       apr 2005,  yu-tai hou        -- modified on module structures (this!
// !                    version of code was given back to aer in jun 2006)    !
// !       nov 2006,  yu-tai hou        -- modified code to include the       !
// !                    generallized aerosol optical property scheme for gcms.!
// !       apr 2007,  yu-tai hou        -- added spectral band heating as an  !
// !                    optional output to support the 500km model's upper    !
// !                    stratospheric radiation calculations. restructure     !
// !                    optional outputs for easy access by different models. !
// !       oct 2008,  yu-tai hou        -- modified to include new features   !
// !                    from aer's newer release v3.5-v3.61, including mcica  !
// !                    sub-grid cloud option and true direct/diffuse fluxes  !
// !                    without delta scaling. added rain/snow opt properties !
// !                    support to cloudy sky calculations. simplified and    !
// !                    unified sw and lw sub-column cloud subroutines into   !
// !                    one module by using optional parameters.              !
// !       mar 2009,  yu-tai hou        -- replaced the original random number!
// !                    generator coming with the original code with ncep w3  !
// !                    library to simplify the program and moved sub-column  !
// !                    cloud subroutines inside the main module. added       !
// !                    option of user provided permutation seeds that could  !
// !                    be randomly generated from forecast time stamp.       !
// !       mar 2009,  yu-tai hou        -- replaced random number generator   !
// !                    programs coming from the original code with the ncep  !
// !                    w3 library to simplify the program and moved sub-col  !
// !                    cloud subroutines inside the main module. added       !
// !                    option of user provided permutation seeds that could  !
// !                    be randomly generated from forecast time stamp.       !
// !       nov 2009,  yu-tai hou        -- updated to aer v3.7-v3.8 version.  !
// !                    notice the input cloud ice/liquid are assumed as      !
// !                    in-cloud quantities, not grid average quantities.     !
// !       aug 2010,  yu-tai hou        -- uptimized code to improve efficiency
// !                    splited subroutine spcvrt into two subs, spcvrc and   !
// !                    spcvrm, to handling non-mcica and mcica type of calls.!
// !       apr 2012,  b. ferrier and y. hou -- added conversion factor to fu's!
// !                    cloud-snow optical property scheme.                   !
// !       jul 2012,  s. moorthi and Y. hou  -- eliminated the pointer array  !
// !                     in subr 'spcvrt' for multi-threading issue running   !
// !                     under intel's fortran compiler.                      !
// !       nov 2012,  yu-tai hou        -- modified control parameters thru   !
// !                     module 'physparam'.                                  !
// !       jun 2013,  yu-tai hou        -- moving band 9 surface treatment    !
// !                     back as in the rrtm2 version, spliting surface flux  !
// !                     into two spectral regions (vis & nir), instead of    !
// !                     designated it in nir region only.                    !
// !                                                                          !
// !!!!!  ==============================================================  !!!!!
// !!!!!                         end descriptions                         !!!!!
// !!!!!  ==============================================================  !!!!!
// 
// 
// !========================================!
//
// Ported to CUDA for nam-gpu from sorc/nam_nems_nmmb.fd/src/atmos/phys/radsw_main_nmmb.f:1244-1395 (nam.v4.2.11, 83df4e13c1).
//
// What this unit computes, every term of the Fortran RSWINIT accounted for:
//  - Parameters: BPADE = 1.0/0.278 (radsw_main.f:293, declared real(kind=kind_phys)); NTBMX = 10000
//    (radsw_param.f:113); expeps = 1.e-20 (:1308); con_g = 9.80665 (physcons.f:46); con_cp = 1.0046e+3
//    (physcons.f:59).
//  - heatfac (Fortran :1369-1375): iswrate == 1 -> con_g*864.0/con_cp (k/day); otherwise
//    con_g*1.0e-2/con_cp (k/second). physparam.f:49 sets iswrate = 2 (k/second).
//  - Transmittance table exp_tbl(0:NTBMX) (:1384-1391): exp_tbl(0) = 1.0; exp_tbl(NTBMX) = expeps;
//    for i = 1..NTBMX-1: tfn = real(i)/real(NTBMX-i), tau = bpade*tfn, exp_tbl(i) = exp(-tau).
//    The SW table has no tau_tbl/tfn_tbl and no tau clamp (unlike the LW rlwinit); exp is taken
//    directly, so large tau underflows to 0 as in the Fortran. Each entry depends only on its own
//    index: one thread per index.
//  - The control checks (:1317-1321 IOVRSW range; :1334-1357 ISUBCSW and the print block; :1361-1366
//    cloud-flag consistency) are scalar logic, run on the host in the wrapper (phys_rrtm_sw_rswinit.py),
//    which raises for the same cases where the Fortran STOPs. The Fortran ISUBCSW STOP sits inside the
//    me == 0 block; the wrapper applies it to every run, which is the single-card case.
//  - All "print *" statements (banner, VTAGSW, option echoes) are dropped: no print inside kernels.
//  - me is used only by those prints; single card, no MPI.
//
// Precision (kind_phys). The Fortran declares every real here kind_phys. In the reference build
// (gate0 GNU strict build of nam-gh, configure.nems.Wcoss_C.intel_ESMF_630r_gsm with CPPFLAGS
// -DENABLE_SMP -DTHREAD_2D -DCHNK_RRTM=8 and no -DSINGLE_PREC) machine.f:5-6 gives kind_phys = 8, and
// radsw_main_nmmb.f is compiled with -fdefault-real-8 -fdefault-double-8. So the arithmetic in this
// kernel (BPADE, tfn, tau, exp, heatfac) is double precision, as the reference is.
// The two tables are STORED as float32 because every consumer kernel reads float32 (spcvrtc
// exp_tbl; swrad heatfac). The only float32 step is that storage cast.
//
// Outputs (device):
//   exp_tbl[NTBMX+1] float32 : transmittance lookup table (ctx.tables["rrtm_sw_exp_tbl"])
//   heatfac[1]       float32 : heating-rate factor (ctx.tables["rrtm_sw_heatfac"])
// One launch on ctx.stream, no host sync, no device malloc.

#include "nmmb_common.cuh"

#define RSW_NTBMX       10000               /* NTBMX, radsw_param.f:113 */
#define RSW_BPADE       (1.0 / 0.278)       /* bpade, radsw_main.f:293, kind_phys */
#define RSW_EXPEPS      1.0e-20             /* rswinit local expeps, :1308, kind_phys */
#define RSW_CON_G       9.80665             /* physcons.f:46, kind_phys */
#define RSW_CON_CP      1.0046e+3           /* physcons.f:59, kind_phys */

extern "C" __global__ void rswinit_tables(
    float* __restrict__ exp_tbl,      /* [NTBMX+1] */
    float* __restrict__ heatfac,      /* [1] */
    const int iswrate)                /* physparam.f:49, 2 (k/second) */
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t == 0) {
        // heatfac (:1369-1375), double arithmetic, stored float32
        double hf;
        if (iswrate == 1) {
            hf = RSW_CON_G * 864.0 / RSW_CON_CP;      // k/day
        } else {
            hf = RSW_CON_G * 1.0e-2 / RSW_CON_CP;     // k/second
        }
        heatfac[0] = (float)hf;
        // exp_tbl(0) = 1.0 (:1384)
        exp_tbl[0] = 1.0f;
    }

    // exp_tbl(NTBMX) = expeps (:1385)
    if (t == RSW_NTBMX) {
        exp_tbl[RSW_NTBMX] = (float)RSW_EXPEPS;
    }

    // do i = 1, NTBMX-1 (:1387-1391): tfn, tau and exp in double, stored float32
    if (t >= 1 && t <= RSW_NTBMX - 1) {
        const int i = t;
        const double tfn = (double)i / (double)(RSW_NTBMX - i);
        const double tau = RSW_BPADE * tfn;
        exp_tbl[i] = (float)exp(-tau);
    }
}
