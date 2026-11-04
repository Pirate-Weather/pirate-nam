//!!!!  ==========================================================  !!!!!
//!!!!             'module_radiation_driver' descriptions           !!!!!
//!!!!  ==========================================================  !!!!!
//                                                                      !
//   this is the radiation driver module.  it prepares atmospheric      !
//   profiles and invokes main radiation calculations.                  !
//                                                                      !
//   in module 'module_radiation_driver' there are twe externally       !
//   callable subroutine:                                               !
//                                                                      !
//      'radinit'    -- initialization routine                          !
//         input:                                                       !
//           ( si, NLAY, iflip, idate, jdate, ICTM, ISOL, ICO2,         !
//             IAER, IALB, IEMS, ICWP, NP3D, isubcsw, isubclw,          !
//             iovrsw, iovrlw, me )                                     !
//         output:                                                      !
//           ( none )                                                   !
//                                                                      !
//      'grrad'      -- setup and invoke main radiation calls           !
//         input:                                                       !
//          ( prsi,prsl,prslk,tgrs,qgrs,oz,vvl,slmsk,                   !
//            xlon,xlat,tsfc,snowd,sncovr,snoalb,zorl,hprim,            !
//            alvsf,alnsf,alvwf,alnwf,facsf,facwf,fice,tisfc,           !
//            solcon,coszen,coszdg,k1oz,k2oz,facoz,                     !
//            cv,cvt,cvb,iovrsw,iovrlw,fcice,frain,rrime,flgmin,        !
//            icsdsw,icsdlw, np3d,ntcw,ncld,ntoz, NTRAC,NFXR,           !
//            dtlw,dtsw, lsswr,lslwr,lssav,sashal,norad_precip,         !
//            crick_proof, ccnorm,                                      !
//            IX, IM, LM, iflip, me, lprnt, ipt, kdt,                   !
//         output:                                                      !
//            htrsw,topfsw,sfcfsw,sfalb,                                !
//            htrlw,topflw,sfcflw,tsflw,semis,cldcov,cldsa              !
//         input/output:                                                !
//            fluxr                                                     !
//         optional output:                                             !
//            HTRSWB,HTRLWB)                                            !
//                                                                      !
//                                                                      !
//   external modules referenced:                                       !
//       'module machine'                    in 'machine.f'             !
//       'module funcphys'                   in 'funcphys.f'            !
//       'module physcons'                   in 'physcons.f             !
//                                                                      !
//       'module module_radiation_gases'     in 'radiation_gases.f'     !
//       'module module_radiation_aerosols'  in 'radiation_aerosols.f'  !
//       'module module_radiation_surface'   in 'radiation_surface.f'   !
//       'module module_radiation_clouds'    in 'radiation_clouds.f'    !
//                                                                      !
//       'module module_radsw_cntr_para'     in 'radsw_xxxx_param.f'    !
//       'module module_radsw_parameters'    in 'radsw_xxxx_param.f'    !
//       'module module_radsw_main'          in 'radsw_xxxx_main.f'     !
//                                                                      !
//       'module module_radlw_cntr_para'     in 'radlw_xxxx_param.f'    !
//       'module module_radlw_parameters'    in 'radlw_xxxx_param.f'    !
//       'module module_radlw_main'          in 'radlw_xxxx_main.f'     !
//                                                                      !
//    where xxxx may vary according to different scheme selection       !
//                                                                      !
//                                                                      !
//   program history log:                                               !
//     mm-dd-yy    ncep         - created program grrad                 !
//     08-12-03    yu-tai hou   - re-written for modulized radiations   !
//     11-06-03    yu-tai hou   - modified                              !
//     01-18-05    s. moorthi   - NOAH/ICE model changes added          !
//     05-10-05    yu-tai hou   - modified module structure             !
//     12-xx-05    s. moorthi   - sfc lw flux adj by mean temperature   !
//     02-20-06    yu-tai hou   - add time variation for co2 data, and  !
//                                solar const. add sfc emiss change     !
//     03-21-06    s. Moorthi   - added surface temp over ice           !
//     07-28-06    yu-tai hou   - add stratospheric vocanic aerosols    !
//     03-14-07    yu-tai hou   - add generalized spectral band interp  !
//                                for aerosol optical prop. (sw and lw) !
//     04-10-07    yu-tai hou   - spectral band sw/lw heating rates     !
//     05-04-07    yu-tai hou   - make options for clim based and modis !
//                                based (h. wei and c. marshall) albedo !
//     09-05-08    yu-tai hou   - add the initial date and time 'idate' !
//                    and control param 'ICTM' to the passing param list!
//                    to handel different time/date requirements for    !
//                    external data (co2, aeros, solcon, ...)           !
//     10-10-08    yu-tai hou   - add the ICTM=-2 option for combining  !
//                    initial condition data with seasonal cycle from   !
//                    climatology.                                      !
//     03-12-09    yu-tai hou   - use two time stamps to keep tracking  !
//                    dates for init cond and fcst time. remove volcanic!
//                    aerosols data in climate hindcast (ICTM=-2).      !
//     03-16-09    yu-tai hou   - included sub-column clouds approx.    !
//                    control flags isubcsw/isubclw in initialization   !
//                    subroutine. passed auxiliary cloud control arrays !
//                    icsdsw/icsdlw (if isubcsw/isubclw =2, it will be  !
//                    the user provided permutation seeds) to the sw/lw !
//                    radiation calculation programs. also moved cloud  !
//                    overlapping control flags iovrsw/iovrlw from main !
//                    radiation routines to the initialization routines.!
//     04-02-09    yu-tai hou   - modified surface control flag iems to !
//                    have additional function of if the surface-air    !
//                    interface have the same or different temperature  !
//                    for radiation calculations.                       !
//     04-03-09    yu-tai hou   - modified to add lw surface emissivity !
//                    as output variable. changed the sign of sfcnsw to !
//                    be positive value denote solar flux goes into the !
//                    ground (this is needed to reduce sign confusion   !
//                    in other part of model)                           !
//     04-20-09    carlos perez - prepare driver for nmmb.  added option!
//                    of run the gfs's radiation on nmmb                !
//     09-09-09    fanglin yang (thru s.moorthi) added QME5 QME6 to E-20!
//     01-09-10    sarah lu     - added gocart option, revised grrad for!
//                    gocart coupling. calling argument modifed: ldiag3 !
//                    removed; cldcov/fluxr sequence changed; cldcov is !
//                    changed from accumulative to instant field and    !
//                    from input/output to output field                 !
//     01-24-10    sarah lu     - added aod to fluxr, added prslk and   !
//                    oz to setaer input argument (for gocart coupling),!
//                    added tau_gocart to setaer output argument (for,  !
//                    aerosol diag)                                     !
//     07-08-10    s.moorthi - updated the NEMS version for new physics !
//     07-28-10    yu-tai hou   - changed grrad interface to allow all  !
//                    components of sw/lw toa/sfc instantaneous values  !
//                    being passed to the calling program. moved the    !
//                    computaion of sfc net sw flux (sfcnsw) to the     !
//                    calling program. merged carlos' nmmb modification.!
//     07-30-10    s. moorthi - corrected some errors associated with   !
//                    unit changes                                      !
//     12-02-10    s. moorthi/y. hou - removed the use of aerosol flags !
//                    'iaersw' 'iaerlw' from radiations and replaced    !
//                    them by using the runtime variable iaerflg and    !
//                    laswflg defined in module radiation_aerosols.     !
//                    also replaced param nspc in grrad with the use of !
//                    max_num_gridcomp in module radiation_aerosols.    !
//     01-03-11    y. hou     - added sea/land madk 'slmsk' to the      !
//                    argument list of subrotine setaer call for the    !
//                    newly modified horizontal bi-linear interpolation !
//                    in climatological aerosols schem.                 !
//     09-30-11    H.M. Lin   - rename "grrad" to "grrad_nmmb" for using!
//                    in regional to avoid the confliction with global  !
//                    when compile the nems. Also change names of the   !
//                    related subroutines to "astronomy_nmmb" &         !
//                    "solinit_nmmb"                                    !
//     03-15-12    H.M. Lin   - changed grrad interface to allow 'cldsa'!
//                    (fraction of clouds for low, mid, hi, tot, bl)    !
//                     passed to the calling program.                   !
//     09-25-12    y. hou     - added optional extra top layer 'LTP' on !
//                    top of low ceiling models                         !
//                    (** to cover the gases atop, expecially O3)       !
//                                                                      !
//!!!!  ==========================================================  !!!!!
//!!!!                       end descriptions                       !!!!!
//!!!!  ==========================================================  !!!!!
//
//
//
//=============================================!

// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:1577-1794 (nam.v4.2.11, 83df4e13c1).
//
// Covered (Fortran line ranges in grrad_nmmb.f):
//   1577-1717  cloud dispatch branches of grrad_nmmb (np3d == 4, 3, 5, 8; else diagnostic):
//              the np3d==5 (GFDL) fill and its extra top layer, the np3d==8 (Thompson) input
//              preparation, the diagnostic-scheme vvel/cvt scaling and extra top layer.
//              The calls to progcld1 (np3d 4), progcld2 (np3d 3), progcld8 (np3d 8) and
//              diagcld1 (diagnostic) are other units (phys-rrtm-progcld*, phys-rrtm-diagcld1);
//              the wrapper launches them through its dependency hook, not from this file.
//   1722-1794  IALB == 2 surface albedo (Payne 1972 tables / RRTM 2012 albedo, BSF).
// Not covered: 1795-1817 (ipass and shortwave/longwave pass control, belongs to the sw/lw units).
//
// Index mapping: Fortran local level k1 (1-based) is stored at zero-based index k1-1 in
// [LMK, JM, IM] arrays. kd, lya0 and lyb0 are the zero-based forms of the Fortran kd, lya, lyb
// (ivflip == 0: kd = 1, lya = 2, lyb = 1; ivflip == 1: kd = 0, lya = LM, lyb = LP1), passed in.
//
// Deviation (documented): at zenith angle exactly 0 the Fortran indexes ALBD(:,21), ZA(20) and
// DZA(20), which are out of bounds (undefined). Here jx is clamped to 19 in that case.

#include "nmmb_common.cuh"

#define GRRAD_TWENTY   20.0f
#define GRRAD_HP537    0.537f
#define GRRAD_ONE      1.0f
#define GRRAD_DEGRAD1  (180.0f / 3.14159265f)
#define GRRAD_H74E1    74.0f
#define GRRAD_HAF      0.5f
#define GRRAD_HNINETY  90.0f
#define GRRAD_FIFTY    50.0f
#define GRRAD_QUARTR   0.25f
#define GRRAD_HNINE    9.0f
#define GRRAD_HP1      0.1f
#define GRRAD_H15E1    15.0f

// DATA statements of grrad_nmmb.f (TRN, ZA, DZA, ALBD), copied by script from the NAM source.
__constant__ float c_TRN[21] = {
    0.0f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.45f,
    0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f,
    1.0f
};
__constant__ float c_ZA[20] = {
    90.0f, 88.0f, 86.0f, 84.0f, 82.0f, 80.0f, 78.0f, 76.0f, 74.0f, 70.0f,
    66.0f, 62.0f, 58.0f, 54.0f, 50.0f, 40.0f, 30.0f, 20.0f, 10.0f, 0.0f
};
__constant__ float c_DZA[19] = {
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 4.0f,
    4.0f, 4.0f, 4.0f, 4.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f
};
// Fortran ALBD(21,20), column-major DATA order: ALBD(iq,jx) is at index (iq-1) + 21*(jx-1).
__constant__ float c_ALBD[420] = {
    0.061f, 0.062f, 0.072f, 0.087f, 0.115f, 0.163f, 0.235f, 0.318f, 0.395f, 0.472f,
    0.542f, 0.604f, 0.655f, 0.693f, 0.719f, 0.732f, 0.73f, 0.681f, 0.581f, 0.453f,
    0.425f, 0.061f, 0.062f, 0.07f, 0.083f, 0.108f, 0.145f, 0.198f, 0.263f, 0.336f,
    0.415f, 0.487f, 0.547f, 0.595f, 0.631f, 0.656f, 0.67f, 0.652f, 0.602f, 0.494f,
    0.398f, 0.37f, 0.061f, 0.061f, 0.068f, 0.079f, 0.098f, 0.13f, 0.174f, 0.228f,
    0.29f, 0.357f, 0.424f, 0.498f, 0.556f, 0.588f, 0.603f, 0.592f, 0.556f, 0.488f,
    0.393f, 0.342f, 0.325f, 0.061f, 0.061f, 0.065f, 0.073f, 0.086f, 0.11f, 0.15f,
    0.192f, 0.248f, 0.306f, 0.36f, 0.407f, 0.444f, 0.469f, 0.48f, 0.474f, 0.444f,
    0.386f, 0.333f, 0.301f, 0.29f, 0.061f, 0.061f, 0.065f, 0.07f, 0.082f, 0.101f,
    0.131f, 0.168f, 0.208f, 0.252f, 0.295f, 0.331f, 0.358f, 0.375f, 0.385f, 0.377f,
    0.356f, 0.32f, 0.288f, 0.266f, 0.255f, 0.061f, 0.061f, 0.063f, 0.068f, 0.077f,
    0.092f, 0.114f, 0.143f, 0.176f, 0.21f, 0.242f, 0.272f, 0.288f, 0.296f, 0.3f,
    0.291f, 0.273f, 0.252f, 0.237f, 0.266f, 0.22f, 0.061f, 0.061f, 0.062f, 0.066f,
    0.072f, 0.084f, 0.103f, 0.127f, 0.151f, 0.176f, 0.198f, 0.219f, 0.236f, 0.245f,
    0.25f, 0.246f, 0.235f, 0.222f, 0.211f, 0.205f, 0.2f, 0.061f, 0.061f, 0.061f,
    0.065f, 0.071f, 0.079f, 0.094f, 0.113f, 0.134f, 0.154f, 0.173f, 0.185f, 0.19f,
    0.193f, 0.193f, 0.19f, 0.188f, 0.185f, 0.182f, 0.18f, 0.178f, 0.061f, 0.061f,
    0.061f, 0.064f, 0.067f, 0.072f, 0.083f, 0.099f, 0.117f, 0.135f, 0.15f, 0.16f,
    0.164f, 0.165f, 0.164f, 0.162f, 0.16f, 0.159f, 0.158f, 0.157f, 0.157f, 0.061f,
    0.061f, 0.061f, 0.062f, 0.065f, 0.068f, 0.074f, 0.084f, 0.097f, 0.111f, 0.121f,
    0.127f, 0.13f, 0.131f, 0.131f, 0.13f, 0.129f, 0.127f, 0.126f, 0.125f, 0.122f,
    0.061f, 0.061f, 0.061f, 0.061f, 0.062f, 0.064f, 0.07f, 0.076f, 0.085f, 0.094f,
    0.101f, 0.105f, 0.107f, 0.106f, 0.103f, 0.1f, 0.097f, 0.096f, 0.095f, 0.095f,
    0.095f, 0.061f, 0.061f, 0.061f, 0.06f, 0.061f, 0.062f, 0.065f, 0.07f, 0.075f,
    0.081f, 0.086f, 0.089f, 0.09f, 0.088f, 0.084f, 0.08f, 0.077f, 0.075f, 0.074f,
    0.074f, 0.074f, 0.061f, 0.061f, 0.06f, 0.06f, 0.06f, 0.061f, 0.063f, 0.065f,
    0.068f, 0.072f, 0.076f, 0.077f, 0.076f, 0.074f, 0.071f, 0.067f, 0.064f, 0.062f,
    0.061f, 0.061f, 0.061f, 0.061f, 0.061f, 0.06f, 0.06f, 0.06f, 0.06f, 0.061f,
    0.062f, 0.065f, 0.068f, 0.069f, 0.069f, 0.068f, 0.065f, 0.061f, 0.058f, 0.055f,
    0.054f, 0.053f, 0.052f, 0.052f, 0.061f, 0.061f, 0.06f, 0.06f, 0.06f, 0.06f,
    0.06f, 0.06f, 0.062f, 0.065f, 0.065f, 0.063f, 0.06f, 0.057f, 0.054f, 0.05f,
    0.047f, 0.046f, 0.045f, 0.044f, 0.044f, 0.061f, 0.061f, 0.06f, 0.06f, 0.06f,
    0.059f, 0.059f, 0.059f, 0.059f, 0.059f, 0.058f, 0.055f, 0.051f, 0.047f, 0.043f,
    0.039f, 0.035f, 0.033f, 0.032f, 0.031f, 0.031f, 0.061f, 0.061f, 0.06f, 0.06f,
    0.06f, 0.059f, 0.059f, 0.058f, 0.057f, 0.056f, 0.054f, 0.051f, 0.047f, 0.043f,
    0.039f, 0.036f, 0.033f, 0.03f, 0.028f, 0.027f, 0.026f, 0.061f, 0.061f, 0.06f,
    0.06f, 0.06f, 0.059f, 0.059f, 0.058f, 0.057f, 0.055f, 0.052f, 0.049f, 0.045f,
    0.04f, 0.036f, 0.032f, 0.029f, 0.027f, 0.026f, 0.025f, 0.025f, 0.061f, 0.061f,
    0.06f, 0.06f, 0.06f, 0.059f, 0.059f, 0.058f, 0.056f, 0.053f, 0.05f, 0.046f,
    0.042f, 0.038f, 0.034f, 0.031f, 0.028f, 0.026f, 0.025f, 0.025f, 0.025f, 0.061f,
    0.061f, 0.06f, 0.06f, 0.059f, 0.058f, 0.058f, 0.057f, 0.055f, 0.053f, 0.05f,
    0.046f, 0.042f, 0.038f, 0.034f, 0.03f, 0.028f, 0.029f, 0.025f, 0.025f, 0.025f
};


// ---------------------------------------------------------------------------
// np3d == 5 (GFDL type diagnostic clouds), Fortran 1599-1617:
//   clouds(i,k1,1) = cldf(i,k); clouds(i,k1,2) = tauclouds(i,k);
//   clouds(i,k1,3) = 0.99;      clouds(i,k1,4) = 0.84;      k1 = k + kd
// One thread per (i, j, k); grid.z runs over the LM Fortran levels.
// clouds layout [NF_CLDS, LMK, JM, IM]; cldf/tauclouds layout [LM, JM, IM].
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_clouds_gfdl_fill(
    float* __restrict__ clouds,
    const float* __restrict__ cldf,
    const float* __restrict__ tauclouds,
    const int IM, const int JM, const int LM, const int LMK, const int kd)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const int k1 = l0 + kd;                 // zero-based local index of Fortran k1 = k + kd
    const size_t src = IDX3(i0, j0, l0, IM, JM);
    clouds[IDX4(i0, j0, k1, 0, IM, JM, LMK)] = cldf[src];
    clouds[IDX4(i0, j0, k1, 1, IM, JM, LMK)] = tauclouds[src];
    clouds[IDX4(i0, j0, k1, 2, IM, JM, LMK)] = 0.99f;
    clouds[IDX4(i0, j0, k1, 3, IM, JM, LMK)] = 0.84f;
}

// ---------------------------------------------------------------------------
// Extra top layer: if (lextop) arr(i,lyb,n) = arr(i,lya,n) for n = 1..nfield.
// Used for clouds (nfield = 4, np3d == 5; Fortran 1635-1640) and vvel (nfield = 1;
// Fortran 1697-1700). arr layout [nfield, LMK, JM, IM] (vvel: nfield = 1, same index order).
// Separate launch after the level kernel that writes lya, so the ordering is global.
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_top_copy(
    float* __restrict__ arr,
    const int IM, const int JM, const int LMK,
    const int nfield, const int lya0, const int lyb0)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    for (int n = 0; n < nfield; ++n) {
        arr[IDX4(i0, j0, lyb0, n, IM, JM, LMK)] = arr[IDX4(i0, j0, lya0, n, IM, JM, LMK)];
    }
}

// ---------------------------------------------------------------------------
// np3d == 8 (Thompson) input preparation, Fortran 1650-1674, for all LMK local levels:
//   qc2d = tracer1(.,.,4), qi2d = tracer1(.,.,5), qs2d = tracer1(.,.,6): each set to 0 when
//   <= EPSQ; ni2d = max(1e-8, tracer1(.,.,9)).  Fortran tracer index n (1-based) -> n-1.
// tracer1 layout [NTRAC, LMK, JM, IM]; outputs [LMK, JM, IM].
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_clouds_np8_prep(
    const float* __restrict__ tracer1,
    float* __restrict__ qc2d, float* __restrict__ qi2d,
    float* __restrict__ qs2d, float* __restrict__ ni2d,
    const int IM, const int JM, const int LMK, const float epsq)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LMK) return;
    const size_t o = IDX3(i0, j0, l0, IM, JM);
    float qc = tracer1[IDX4(i0, j0, l0, 3, IM, JM, LMK)];
    float qi = tracer1[IDX4(i0, j0, l0, 4, IM, JM, LMK)];
    float qs = tracer1[IDX4(i0, j0, l0, 5, IM, JM, LMK)];
    float ni = tracer1[IDX4(i0, j0, l0, 8, IM, JM, LMK)];
    if (qc <= epsq) qc = 0.0f;
    if (qi <= epsq) qi = 0.0f;
    if (qs <= epsq) qs = 0.0f;
    qc2d[o] = qc;
    qi2d[o] = qi;
    qs2d[o] = qs;
    ni2d[o] = fmaxf(1.0e-8f, ni);
}

// ---------------------------------------------------------------------------
// Diagnostic cloud scheme (ntcw <= 0), Fortran 1679-1698:
//   vvel(i,k1) = 10.0 * vvl(i,k), k1 = k + kd, for k = 1..LM   (one thread per (i, j, k))
// vvl layout [LM, JM, IM]; vvel layout [LMK, JM, IM].
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_clouds_diag_vvel(
    const float* __restrict__ vvl, float* __restrict__ vvel,
    const int IM, const int JM, const int LM, const int kd)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    vvel[IDX3(i0, j0, l0 + kd, IM, JM)] = 10.0f * vvl[IDX3(i0, j0, l0, IM, JM)];
}

// Fortran 1677-1683: cvt1(i) = 10.0 * cvt(i); cvb1(i) = 10.0 * cvb(i).  Layout [JM, IM].
extern "C" __global__ void grrad_clouds_diag_cvt(
    const float* __restrict__ cvt, const float* __restrict__ cvb,
    float* __restrict__ cvt1, float* __restrict__ cvb1,
    const int IM, const int JM)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const size_t p = IDX2(i0, j0, IM);
    cvt1[p] = 10.0f * cvt[p];
    cvb1[p] = 10.0f * cvb[p];
}

// ---------------------------------------------------------------------------
// IALB == 2 surface albedo, Fortran 1722-1794, one thread per point (i, j). Layout [JM, IM].
//   SMX = SM; SFCALBEDO = SALBEDO; IQ = INT(TWENTY*HP537 + ONE); CZMEAN = coszen;
//   if (CZMEAN > 0 and SMX > 0.5): ZEN = DEGRAD1*ACOS(MAX(CZMEAN,0)), table lookup with
//   JX from ZEN, DZEN, ALB1/ALB2 interpolation in ALBD, SFCALBEDO = ALB1 + TWENTY*(ALB2-ALB1)
//   *(HP537 - TRN(IQ)).  Then ALVB = ALNB = ALVD = ALND = SFCALBEDO (all points), and
//   alvsf = ALVB, alnsf = ALNB, alvwf = ALVD, alnwf = ALND, facsf = facwf = 0.
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_clouds_albedo(
    const float* __restrict__ salbedo, const float* __restrict__ sm,
    const float* __restrict__ coszen,
    float* __restrict__ sfcalbedo, float* __restrict__ smx, float* __restrict__ czmean,
    float* __restrict__ alvsf, float* __restrict__ alnsf,
    float* __restrict__ alvwf, float* __restrict__ alnwf,
    float* __restrict__ facsf, float* __restrict__ facwf,
    const int IM, const int JM)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;
    const size_t p = IDX2(i0, j0, IM);
    smx[p] = sm[p];
    float sfc = salbedo[p];
    const int iq = (int)(GRRAD_TWENTY * GRRAD_HP537 + GRRAD_ONE);   // IQ (1-based), = 11
    const float cz = coszen[p];
    czmean[p] = cz;
    if (cz > 0.0f && smx[p] > 0.5f) {
        const float zen = GRRAD_DEGRAD1 * acosf(fmaxf(cz, 0.0f));
        int jx;
        if (zen >= GRRAD_H74E1) {
            jx = (int)(GRRAD_HAF * (GRRAD_HNINETY - zen) + GRRAD_ONE);
        } else if (zen >= GRRAD_FIFTY) {
            jx = (int)(GRRAD_QUARTR * (GRRAD_H74E1 - zen) + GRRAD_HNINE);
        } else {
            jx = (int)(GRRAD_HP1 * (GRRAD_FIFTY - zen) + GRRAD_H15E1);
        }
        if (jx > 19) jx = 19;                       // see header: zenith 0 exactly, out of bounds
        const float dzen = -(zen - c_ZA[jx - 1]) / c_DZA[jx - 1];
        const int a1 = (iq - 1) + 21 * (jx - 1);    // ALBD(IQ, JX)
        const int a2 = iq + 21 * (jx - 1);          // ALBD(IQ+1, JX)
        const float alb1 = c_ALBD[a1] + dzen * (c_ALBD[a1 + 21] - c_ALBD[a1]);
        const float alb2 = c_ALBD[a2] + dzen * (c_ALBD[a2 + 21] - c_ALBD[a2]);
        sfc = alb1 + GRRAD_TWENTY * (alb2 - alb1) * (GRRAD_HP537 - c_TRN[iq - 1]);
    }
    sfcalbedo[p] = sfc;
    alvsf[p] = sfc;     // ALVB
    alnsf[p] = sfc;     // ALNB
    alvwf[p] = sfc;     // ALVD
    alnwf[p] = sfc;     // ALND
    facsf[p] = 0.0f;
    facwf[p] = 0.0f;
}

// ---------------------------------------------------------------------------
// clw, Fortran 1556-1571: clw(i,k) = sum over j = 1..ncld of tracer1(i,k,ntcw+j-1), then
// clw < EPSQ -> 0. tracer1 layout [NTRAC, LMK, JM, IM]; clw layout [LMK, JM, IM].
// ntcw0 is the zero-based tracer index of ntcw (Fortran 1-based ntcw, so ntcw0 = ntcw - 1).
// ---------------------------------------------------------------------------
extern "C" __global__ void grrad_clouds_clw(
    const float* __restrict__ tracer1, float* __restrict__ clw,
    const int IM, const int JM, const int LMK, const int ntcw0, const int ncld, const float epsq)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LMK) return;
    float s = 0.0f;
    for (int j = 0; j < ncld; ++j) {
        s += tracer1[IDX4(i0, j0, l0, ntcw0 + j, IM, JM, LMK)];
    }
    if (s < epsq) s = 0.0f;
    clw[IDX3(i0, j0, l0, IM, JM)] = s;
}
