// phys_rrtm_radupdate.cu -- RADUPDATE_NMMB: calendar-driven radiation update control
// (solar astronomy, monthly aerosols, CO2/gas data). Scalar control logic, one thread.
//
// NOAA notice: header block of phys/grrad_nmmb.f (nam.v4.2.11, 83df4e13c1), kept verbatim:
// !!!!!  ==========================================================  !!!!!
// !!!!!             'module_radiation_driver' descriptions           !!!!!
// !!!!!  ==========================================================  !!!!!
// !                                                                      !
// !   this is the radiation driver module.  it prepares atmospheric      !
// !   profiles and invokes main radiation calculations.                  !
// !                                                                      !
// !   in module 'module_radiation_driver' there are twe externally       !
// !   callable subroutine:                                               !
// !                                                                      !
// !      'radinit'    -- initialization routine                          !
// !         input:                                                       !
// !           ( si, NLAY, iflip, idate, jdate, ICTM, ISOL, ICO2,         !
// !             IAER, IALB, IEMS, ICWP, NP3D, isubcsw, isubclw,          !
// !             iovrsw, iovrlw, me )                                     !
// !         output:                                                      !
// !           ( none )                                                   !
// !                                                                      !
// !      'grrad'      -- setup and invoke main radiation calls           !
// !         input:                                                       !
// !          ( prsi,prsl,prslk,tgrs,qgrs,oz,vvl,slmsk,                   !
// !            xlon,xlat,tsfc,snowd,sncovr,snoalb,zorl,hprim,            !
// !            alvsf,alnsf,alvwf,alnwf,facsf,facwf,fice,tisfc,           !
// !            solcon,coszen,coszdg,k1oz,k2oz,facoz,                     !
// !            cv,cvt,cvb,iovrsw,iovrlw,fcice,frain,rrime,flgmin,        !
// !            icsdsw,icsdlw, np3d,ntcw,ncld,ntoz, NTRAC,NFXR,           !
// !            dtlw,dtsw, lsswr,lslwr,lssav,sashal,norad_precip,         !
// !            crick_proof, ccnorm,                                      !
// !            IX, IM, LM, iflip, me, lprnt, ipt, kdt,                   !
// !         output:                                                      !
// !            htrsw,topfsw,sfcfsw,sfalb,                                !
// !            htrlw,topflw,sfcflw,tsflw,semis,cldcov,cldsa              !
// !         input/output:                                                !
// !            fluxr                                                     !
// !         optional output:                                             !
// !            HTRSWB,HTRLWB)                                            !
// !                                                                      !
// !                                                                      !
// !   external modules referenced:                                       !
// !       'module machine'                    in 'machine.f'             !
// !       'module funcphys'                   in 'funcphys.f'            !
// !       'module physcons'                   in 'physcons.f             !
// !                                                                      !
// !       'module module_radiation_gases'     in 'radiation_gases.f'     !
// !       'module module_radiation_aerosols'  in 'radiation_aerosols.f'  !
// !       'module module_radiation_surface'   in 'radiation_surface.f'   !
// !       'module module_radiation_clouds'    in 'radiation_clouds.f'    !
// !                                                                      !
// !       'module module_radsw_cntr_para'     in 'radsw_xxxx_param.f'    !
// !       'module module_radsw_parameters'    in 'radsw_xxxx_param.f'    !
// !       'module module_radsw_main'          in 'radsw_xxxx_main.f'     !
// !                                                                      !
// !       'module module_radlw_cntr_para'     in 'radlw_xxxx_param.f'    !
// !       'module module_radlw_parameters'    in 'radlw_xxxx_param.f'    !
// !       'module module_radlw_main'          in 'radlw_xxxx_main.f'     !
// !                                                                      !
// !    where xxxx may vary according to different scheme selection       !
// !                                                                      !
// !                                                                      !
// !   program history log:                                               !
// !     mm-dd-yy    ncep         - created program grrad                 !
// !     08-12-03    yu-tai hou   - re-written for modulized radiations   !
// !     11-06-03    yu-tai hou   - modified                              !
// !     01-18-05    s. moorthi   - NOAH/ICE model changes added          !
// !     05-10-05    yu-tai hou   - modified module structure             !
// !     12-xx-05    s. moorthi   - sfc lw flux adj by mean temperature   !
// !     02-20-06    yu-tai hou   - add time variation for co2 data, and  !
// !                                solar const. add sfc emiss change     !
// !     03-21-06    s. Moorthi   - added surface temp over ice           !
// !     07-28-06    yu-tai hou   - add stratospheric vocanic aerosols    !
// !     03-14-07    yu-tai hou   - add generalized spectral band interp  !
// !                                for aerosol optical prop. (sw and lw) !
//
// Ported to CUDA for nam-gpu from phys/grrad_nmmb.f:456-602 (nam.v4.2.11, 83df4e13c1).
//
// radupdate_nmmb has no grid loops: it is the calendar and change-flag logic that decides which
// time-varying radiation data are refreshed. The Fortran-to-CUDA review (CONTRACT section 6,
// part 3) covers every statement of lines 456-602:
//   iyear,imon,iday,ihour = jdate(1,2,3,5)              -> jd_* arguments
//   kyear,kmon,kday,khour = idate or jdate (ictmflg)    -> flags[F_KYEAR..F_KHOUR]
//   lmon_chg = (month0 /= imon); month0 = imon          -> flags[F_LMON], rst[RST_MONTH0]
//   if (lsswr):                                         -> lsswr argument
//     lsol_chg: isolar 0/10 -> false; iyear0/=iyear -> true; else isolar==4 .and. lmon_chg
//     iyear0 = iyear                                    -> rst[RST_IYEAR0]
//     call sol_update_nmmb                              -> flags[F_CALL_SOL] (called by phys-rrtm-sol-update's driver)
//   if (lmon_chg) call aer_update                       -> flags[F_CALL_AER] (called by phys-rrtm-aer-update)
//   lco2_chg: monthd /= kmon -> monthd = kmon, true    -> flags[F_LCO2], rst[RST_MONTHD]
//   call gas_update(kyear,kmon,kday,khour,loz1st,lco2_chg) -> flags[F_CALL_GAS] and loz1st argument
//                                                       (gas_update takes loz1st intent(in))
//   if (loz1st) loz1st = .false.                        -> rst[RST_LOZ1ST] = 0
// aer_update(iyear, imon, me) takes iyear = jdate(1) and imon = jdate(2) (NOT the idate month):
//   iyear,imon -> flags[F_IYEAR], flags[F_IMON]
// The sol/aer/gas bodies, the outputs slag, sdec, cdec, solcon (written by sol_update_nmmb, unit
// phys-rrtm-sol-update, which takes jdate, kyear, deltsw, deltim, lsol_chg, me), and the sfc/cld update
// calls (commented out in the Fortran) are other units or absent, not here.
// No print, MPI, OpenMP or halo code is present in this routine.
//
// Scratch state (rst, int32[4]) is owned by the caller and set once by radinit (phys-rrtm-radinit):
// month0 = iyear0 = monthd = 0, loz1st = (ioznflg == 0). Flags (int32[12]) are written every call.
// Integer-only: no float precision question arises.

#include "nmmb_common.cuh"

#define RST_MONTH0 0
#define RST_IYEAR0 1
#define RST_MONTHD 2
#define RST_LOZ1ST 3

#define F_LSOL      0
#define F_LMON      1
#define F_LCO2      2
#define F_CALL_SOL  3
#define F_CALL_AER  4
#define F_CALL_GAS  5
#define F_KYEAR     6
#define F_KMON      7
#define F_KDAY      8
#define F_KHOUR     9
#define F_LOZ_ARG  10
#define F_LOZ_AFTER 11
#define F_IYEAR    12
#define F_IMON     13

extern "C" __global__ void phys_rrtm_radupdate(
    int* __restrict__ rst, int* __restrict__ flags,
    int jd_yr, int jd_mon, int jd_day, int jd_hr,
    int id_yr, int id_mon, int id_day, int id_hr,
    int isolar, int ictmflg, int lsswr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int iyear = jd_yr, imon = jd_mon;
    int kyear, kmon, kday, khour;
    if (ictmflg == 0 || ictmflg == -2) {
        kyear = id_yr; kmon = id_mon; kday = id_day; khour = id_hr;
    } else {
        kyear = jd_yr; kmon = jd_mon; kday = jd_day;
        khour = jd_hr;
    }

    const int lmon_chg = (rst[RST_MONTH0] != imon) ? 1 : 0;
    if (lmon_chg) rst[RST_MONTH0] = imon;

    int lsol_chg = 0;
    if (lsswr) {
        if (isolar == 0 || isolar == 10) {
            lsol_chg = 0;
        } else if (rst[RST_IYEAR0] != iyear) {
            lsol_chg = 1;
        } else {
            lsol_chg = (isolar == 4 && lmon_chg) ? 1 : 0;
        }
        rst[RST_IYEAR0] = iyear;
    }

    int lco2_chg = 0;
    if (rst[RST_MONTHD] != kmon) {
        rst[RST_MONTHD] = kmon;
        lco2_chg = 1;
    }

    const int loz_arg = rst[RST_LOZ1ST];
    if (loz_arg) rst[RST_LOZ1ST] = 0;

    flags[F_LSOL]      = lsol_chg;
    flags[F_LMON]      = lmon_chg;
    flags[F_LCO2]      = lco2_chg;
    flags[F_CALL_SOL]  = lsswr ? 1 : 0;
    flags[F_CALL_AER]  = lmon_chg;
    flags[F_CALL_GAS]  = 1;
    flags[F_KYEAR]     = kyear;
    flags[F_KMON]      = kmon;
    flags[F_KDAY]      = kday;
    flags[F_KHOUR]     = khour;
    flags[F_LOZ_ARG]   = loz_arg;
    flags[F_LOZ_AFTER] = 0;
    flags[F_IYEAR]     = jd_yr;
    flags[F_IMON]      = jd_mon;
}
