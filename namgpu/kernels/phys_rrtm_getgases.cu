// phys_rrtm_getgases.cu -- getgases: gas volume-mixing-ratio profiles for the NMMB RRTM call.
//
// NOTICE: NAM source file phys/radiation_gases_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). Routine header kept as the notice (module_radiation_gases_nmmb, getgases):
//   getgases set up global distribution of radiation absorbing gases in volume mixing ratio.
//   currently only co2 has the options from observed values, all other gases are asigned to the
//   climatological values.
//   NCEP RRTM gas profiles (NMMB version, module_radiation_gases_nmmb v5.1, Nov 2012).
// Ported to CUDA for WOOF-NAM from phys/radiation_gases_nmmb.f:843-979 (nam.v4.2.11, 83df4e13c1).
//
// Line-by-line counterpart of the routine's statements in that range:
//   843-849   subroutine getgases(plvl, xlon, xlat, IMAX, LMAX, gasdat)  -> kernel args. The Fortran
//             row (IMAX points) is the full domain here: one thread per (i, j) column, all LMAX
//             levels inside the thread (grid_cols). plvl is [LMAX+1, JM, IM], xlon/xlat [JM, IM].
//   880-890   default loop: gasdat(i,k,1..10) = co2vmr_def, n2ovmr_def, ch4vmr_def, o2vmr_def,
//             covmr_def, f11vmr_def, f12vmr_def, f22vmr_def, cl4vmr_def, f113vmr_def
//             -> defaults below, written for every level and gas before the CO2 branches.
//   ico2flg == 1 (global mean + seasonal cycle):
//   896-901   gasdat(i,k,1) = co2_glb + gco2cyc(kmonsav)            -> co2_base, all levels
//   ico2flg == 2 (monthly 2-D CO2 below prsco2):
//   906       tmp = raddeg / resco2                                  -> GG_TMP
//   907-915   xlon1 = xlon(i); if (xlon1 < 0) xlon1 += con_pi;
//             xlat1 = hfpi - xlat(i)                                  -> xlon1, xlat1
//   917-918   ilon = min(IMXCO2, int(xlon1*tmp+1)); ilat = min(JMXCO2, int(xlat1*tmp+1))
//   920-929   ivflip == 0: k = 1..LMAX: plvl(i,k) >= prsco2 -> co2vmr_sav(ilon,ilat,kmonsav),
//             else co2_glb + gco2cyc(kmonsav)                         -> level loop, l0 = k-1
//   931-940   ivflip /= 0: same test on plvl(i,k+1)                  -> plvl level l0+1
//   945-950   (ico2flg other than 0/1/2: defaults only; ico2flg == 0 is the prescribed global mean
//             in the defaults, nothing else)
//   return    (no output beyond gasdat)
//
// Caveat (Fortran has no clamp here): ilat/ilon are truncated toward zero. For xlat slightly above
// pi/2 (xlat1 slightly negative) int() of a value in (-1, 0] gives 0, a read outside co2vmr_sav in
// the Fortran. This kernel clamps the index to 1 there; every other column is the Fortran index.
//
// Layout (CONTRACT.md sections 1 and 3): gasvmr is gasdat as [NF_VGAS=10, LMAX, JM, IM] float32
// (Fortran gasdat(i,k,n) -> gasvmr[n, k, j, i]); plvl is [LMAX+1, JM, IM] with level 0 the model top
// when ivflip == 0; co2vmr_sav (module, Fortran co2vmr_sav(IMXCO2, JMXCO2, 12)) is [12, JMXCO2, IMXCO2]
// as written by the gas_init unit (layout [imo][j][i]); gco2cyc is [12]; co2_glb is [1].
// One thread per column (i, j); the level loop runs inside the thread. Single precision throughout
// (kfpt = single); no host sync, no allocation, no printf.

#include "nmmb_common.cuh"

// radiation_gases_nmmb.f parameters (kind_phys = single)
#define GG_NF_VGAS     10
#define GG_IMXCO2      24
#define GG_JMXCO2      12
#define GG_RESCO2      15.0f
#define GG_PRSCO2      788.0f
#define GG_CON_PI      3.14159265358979323846f   // physcons.f con_pi (single)
#define GG_HFPI        (0.5f * GG_CON_PI)        // hfpi = 0.5*con_pi
#define GG_RADDEG      (180.0f / GG_CON_PI)      // raddeg = 180/con_pi
#define GG_TMP         (GG_RADDEG / GG_RESCO2)   // tmp = raddeg / resco2

#define GG_CO2VMR_DEF   350.0e-6f
#define GG_N2OVMR_DEF   0.31e-6f
#define GG_CH4VMR_DEF   1.50e-6f
#define GG_O2VMR_DEF    0.209f
#define GG_COVMR_DEF    1.50e-8f
#define GG_F11VMR_DEF   3.520e-10f
#define GG_F12VMR_DEF   6.358e-10f
#define GG_F22VMR_DEF   1.500e-10f
#define GG_CL4VMR_DEF   1.397e-10f
#define GG_F113VMR_DEF  8.2000e-11f

extern "C" __global__ void phys_rrtm_getgases(
    const float* __restrict__ plvl,        // [LMAX+1, JM, IM] level pressure (mb)
    const float* __restrict__ xlon,        // [JM, IM]         longitude, radians
    const float* __restrict__ xlat,        // [JM, IM]         latitude, radians
    const float* __restrict__ co2_glb,     // [1]              module co2_glb
    const float* __restrict__ gco2cyc,     // [12]             module gco2cyc (seasonal cycle)
    const float* __restrict__ co2vmr_sav,  // [12, JMXCO2, IMXCO2] module co2vmr_sav(ilon,ilat,month)
    float* __restrict__ gasvmr,            // [NF_VGAS, LMAX, JM, IM] output gasdat
    int im, int jm, int lmax,
    int ico2flg,                           // module ico2flg (0, 1, 2)
    int kmonsav,                           // module kmonsav (1-based month)
    int ivflip)                            // module ivflip (0: toa->sfc, else sfc->toa)
{
    COLUMN_IJ(im, jm)

    // ---- defaults (Fortran 880-890): every level, every gas
    const float defv[GG_NF_VGAS] = {
        GG_CO2VMR_DEF, GG_N2OVMR_DEF, GG_CH4VMR_DEF, GG_O2VMR_DEF, GG_COVMR_DEF,
        GG_F11VMR_DEF, GG_F12VMR_DEF, GG_F22VMR_DEF, GG_CL4VMR_DEF, GG_F113VMR_DEF
    };
    for (int n0 = 0; n0 < GG_NF_VGAS; ++n0) {
        for (int l0 = 0; l0 < lmax; ++l0) {
            gasvmr[IDX4(i0, j0, l0, n0, im, jm, lmax)] = defv[n0];
        }
    }

    // ---- co2 section (Fortran 892-941)
    if (ico2flg == 1) {
        // 896-901: global annual mean plus the seasonal-cycle month
        const float co2_base = co2_glb[0] + gco2cyc[kmonsav - 1];
        for (int l0 = 0; l0 < lmax; ++l0) {
            gasvmr[IDX4(i0, j0, l0, 0, im, jm, lmax)] = co2_base;
        }
    } else if (ico2flg == 2) {
        // 906-915: column longitude and latitude index into the 24 x 12 monthly 2-D CO2 table
        float xlon1 = xlon[IDX2(i0, j0, im)];
        if (xlon1 < 0.0f) xlon1 = xlon1 + GG_CON_PI;   // -pi..pi -> 0..2pi
        const float xlat1 = GG_HFPI - xlat[IDX2(i0, j0, im)];

        int ilon = min(GG_IMXCO2, (int)(xlon1 * GG_TMP + 1.0f));
        int ilat = min(GG_JMXCO2, (int)(xlat1 * GG_TMP + 1.0f));
        if (ilon < 1) ilon = 1;   // see caveat in the header: Fortran has no lower clamp
        if (ilat < 1) ilat = 1;

        // co2vmr_sav(ilon, ilat, kmonsav), layout [month][ilat][ilon]
        const size_t tab = ((size_t)(kmonsav - 1) * GG_JMXCO2 + (ilat - 1)) * GG_IMXCO2 + (ilon - 1);
        const float co2_tab = co2vmr_sav[tab];
        const float co2_base = co2_glb[0] + gco2cyc[kmonsav - 1];

        if (ivflip == 0) {
            // 920-929: index from toa to sfc, test plvl(i,k), k = 1..LMAX
            for (int l0 = 0; l0 < lmax; ++l0) {
                const float pl = plvl[IDX3(i0, j0, l0, im, jm)];
                gasvmr[IDX4(i0, j0, l0, 0, im, jm, lmax)] = (pl >= GG_PRSCO2) ? co2_tab : co2_base;
            }
        } else {
            // 931-940: index from sfc to toa, test plvl(i,k+1)
            for (int l0 = 0; l0 < lmax; ++l0) {
                const float pl = plvl[IDX3(i0, j0, l0 + 1, im, jm)];
                gasvmr[IDX4(i0, j0, l0, 0, im, jm, lmax)] = (pl >= GG_PRSCO2) ? co2_tab : co2_base;
            }
        }
    }
}
