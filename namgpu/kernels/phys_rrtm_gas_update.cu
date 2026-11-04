// phys_rrtm_gas_update.cu -- WOOF-NAM unit phys-rrtm-gas-update.
//
// NOTICE: NOAA/NCEP NAM source phys/radiation_gases_nmmb.f (US government work; the NAM clone
// has no licence file). Source header block of the file reproduced as the notice:
// !!!!!  ==========================================================  !!!!!
// !!!!!              'module_radiation_gases'  description           !!!!!
// !!!!!  ==========================================================  !!!!!
// !                                                                      !
// !   set up ozone climatological profiles and other constant gas        !
// !   profiles, such as co2, ch4, n2o, o2, and those of cfc gases.  All  !
// !   data are entered as mixing ratio by volume, except ozone which is  !
// !   mass mixing ratio (g/g).                                           !
// !                                                                      !
// !   in the module, the externally callabe subroutines are :            !
// !                                                                      !
// !      'gas_init'   -- initialization                                  !
// !         input:                                                       !
// !           ( me )                                                     !
// !         output:                                                      !
// !           ( none )                                                   !
// !                                                                      !
// !      'gas_update' -- read in data and update with time               !
// !         input:                                                       !
// !           ( iyear, imon, iday, ihour, loz1st, ldoco2, me )           !
// !         output:                                                      !
// !           ( none )                                                   !
// !                                                                      !
// !      'getozn'     -- setup climatological ozone profile              !
// !         input:                                                       !
// !           ( prslk,xlat,                                              !
// !             IMAX, LM )                                               !
// !         output:                                                      !
// !           ( o3mmr )                                                  !
// !                                                                      !
// !      'getgases'   -- setup constant gas profiles for LW and SW       !
// !         input:                                                       !
// !           ( plvl, xlon, xlat,                                        !
// !             IMAX, LMAX )                                             !
// !         output:                                                      !
// !           ( gasdat )                                                 !
// !                                                                      !
// !   external modules referenced:                                       !
// !       'module machine'                    in 'machine.f'             !
// !       'module funcphys'                   in 'funcphys.f'            !
//
// Ported to CUDA for WOOF-NAM from phys/radiation_gases_nmmb.f:488-838 (nam.v4.2.11, 83df4e13c1).
#include "nmmb_common.cuh"

// gas_update, CO2 2-D monthly field assembly. One thread per (i, j) of the IMXCO2 x JMXCO2 input
// grid; the 12 months are a per-thread loop. Launched once per year change (kyrsav), not per step.
//
//   mode 0  direct monthly save (Fortran 805-812, ico2flg == 2, ictmflg > 0, not -2):
//             co2vmr_sav(i,j,imo) = (co2dat(i,j) + rate) * 1.0e-6
//   mode 1  annual mean plus seasonal cycle (Fortran 763-785, ictmflg == -2, ico2flg == 2):
//             co2ann(i,j) = 0.0, then += co2dat(i,j) for imo = 1..12 in file order
//             co2ann(i,j) = co2ann(i,j) * 1.0e-6 / float(12)
//             co2vmr_sav(i,j,imo) = co2ann(i,j) + co2cyc_sav(i,j,imo)   (rate is NOT added here)
//   mode 2  historical semi-yearly global means spread over the year (Fortran 657-659, the
//           idyr < MINYEAR branch, ico2flg == 2 and a matching year found):
//             co2vmr_sav(i,j,1:6)  = co2g1 * 1.0e-6
//             co2vmr_sav(i,j,7:12) = co2g2 * 1.0e-6
//
// Layout (CONTRACT section 1): dat, cyc and vmr are [12][JMXCO2][IMXCO2], i.e. IDX3(i0, j0, imo),
// zero-based; Fortran co2vmr_sav(IMXCO2,JMXCO2,12) and co2dat(IMXCO2,JMXCO2) are the same memory.
// dat is in ppm (as read from the file); cyc and vmr are volume mixing ratio (cyc is the output of
// the gas_init unit, co2cyc_sav = co2dat * 1.0e-6, Fortran line 468).
// Mode 2 reads co2g1/co2g2 as scalars (not per cell); they are passed as kernel arguments.
// The Fortran mode-2 nested loop reuses the outer variable i; that value is dead after the exit
// (Fortran 660-661), so it is not reproduced.
// In mode 0 and 2 the cyc pointer is not read; the wrapper passes vmr there.

extern "C" __global__ void gas_co2_monthly(
    const int imx, const int jmx, const int mode,
    const float rate, const float g1, const float g2,
    const float* __restrict__ dat,
    const float* __restrict__ cyc,
    float* __restrict__ vmr)
{
    COLUMN_IJ(imx, jmx)

    if (mode == 0) {
        for (int imo = 0; imo < 12; ++imo) {
            const size_t k = IDX3(i0, j0, imo, imx, jmx);
            vmr[k] = (dat[k] + rate) * 1.0e-6f;
        }
    } else if (mode == 1) {
        float co2ann = 0.0f;
        for (int imo = 0; imo < 12; ++imo) {
            co2ann = co2ann + dat[IDX3(i0, j0, imo, imx, jmx)];
        }
        co2ann = co2ann * 1.0e-6f / 12.0f;
        for (int imo = 0; imo < 12; ++imo) {
            const size_t k = IDX3(i0, j0, imo, imx, jmx);
            vmr[k] = co2ann + cyc[k];
        }
    } else if (mode == 2) {
        for (int imo = 0; imo < 12; ++imo) {
            vmr[IDX3(i0, j0, imo, imx, jmx)] = ((imo < 6) ? g1 : g2) * 1.0e-6f;
        }
    }
}
