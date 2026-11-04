// phys_rrtm_grrad_gases.cu -- grrad_nmmb: layer/level thermodynamics that feed the gas, ozone and
// cosine-zenith calls, and the vertical-index dispatch (ivflip) of the NMMB RRTM driver.
//
// NOTICE: NAM source file phys/grrad_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence file).
// Routine header kept as the notice (module_radiation_driver_nmmb, grrad_nmmb):
//   this is the radiation driver module. it prepares atmospheric profiles and invokes main
//   radiation calculations. (NOAA/NCEP radiation driver, NMMB version; nam.v4.2.11, 83df4e13c1)
// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:1397-1539 (nam.v4.2.11, 83df4e13c1).
//
// Line-by-line counterpart of the routine's own statements in that range:
//   1397-1407  call getozn(prslk1, xlat, IM, LMK, olyr)  (ntoz == 0 branch) -> SIBLING CALL, not in this kernel
//              (unit phys-rrtm-getozn; dispatched by the driver plan before this kernel, in Fortran order)
//   1409-1417  call coszmn_nmmb(xlon, sinlat, coslat, solhr, ...)        -> SIBLING CALL, not in this kernel
//              (unit phys-rrtm-coszmn; dispatched by the driver plan before this kernel; coszen_loc, coszdg are not inputs of this block)
//   1419-1427  call getgases(plvl, xlon, xlat, IM, LMK, gasvmr)          -> SIBLING CALL, not in this kernel
//              (unit phys-rrtm-getgases; dispatched by the driver plan before this kernel; gasvmr is not an input of this block)
//   1429-1434  do k=2,LMK: tem2da(i,k)=log(plyr(i,k)); tem2db(i,k)=log(plvl(i,k))   -> log loop, below
//   1436-1488  if (ivflip == 0) ...  (input toa -> sfc, the NAM setting: rad_initialize iflip=0, rad_initialize_nmmb.f:183):
//      tem1d = QME6; tem2da(1)=log(plyr(1)); tem2db(1)= -14 if plvl(1)==0 else log(plvl(1));
//      tsfa = tlyr(LMK); tlvl(1)=tlyr(1); tlvl(LMP)=tskn                        (kd = 1: local = in+1)
//      do k=1,LM: k1=k+kd; qlyr(k1)=max(tem1d,qgrs(k)); tem1d=min(QME5,qlyr(k1));
//                 tvly(k1)=tgrs(k)*(1+con_fvirt*qlyr(k1))
//      lextop: qlyr(lyb)=qlyr(lya); tvly(lyb)=tvly(lya)                        (lya=2, lyb=1)
//      do k=2,LMK: tlvl(k)=tlyr(k)+(tlyr(k-1)-tlyr(k))*(tem2db(k)-tem2da(k))/(tem2da(k-1)-tem2da(k))
//   ivflip /= 0 (input sfc -> toa):
//      tem1d = QME6; tem2da(1)=log(plyr(1)); tem2db(1)=log(plvl(1));
//      tsfa = tlyr(1); tlvl(1)=tskn; tlvl(LMP)=tlyr(LMK)
//      do k=LM,1,-1: qlyr(k)=max(tem1d,qgrs(k)); tem1d=min(QME5,qlyr(k));
//                    tvly(k)=tgrs(k)*(1+con_fvirt*qlyr(k))
//      lextop: qlyr(lyb)=qlyr(lya); tvly(lyb)=tvly(lya)                        (lya=LM, lyb=LP1=LM+1)
//      do k=1,LMK-1: tlvl(k+1)=tlyr(k)+(tlyr(k+1)-tlyr(k))*(tem2db(k+1)-tem2da(k))/(tem2da(k+1)-tem2da(k))
//   1489-1527  else (ivflip /= 0, input sfc -> toa), below
//   1534       nday = im                                                      -> wrapper writes RAD_NDAY (device fill)
//   Fortran print blocks (me == 0 diagnostics) are not device work and are not ported.
//
// Constants: QME5 = QME6 = 1.0e-7 (grrad_nmmb.f:197, parameter); con_fvirt = con_rv/con_rd - 1 with
// con_rv = 4.6150e+2, con_rd = 2.8705e+2 (physcons.f:57-79, single precision). The NMMB_P608 macro in
// nmmb_common.cuh uses 461.6/287.04 and is NOT the value used here; it is not changed (shared header).
// LTP = 1 (grrad_nmmb.f:210), so lextop is true and LMK = LM+1, LMP = LM+2.
//
// Layout (CONTRACT.md section 1): level arrays [LMK, JM, IM] (plyr, tlyr, tvly, qlyr, tem2da),
// [LMP, JM, IM] (plvl, tlvl, tem2db), input model arrays [LM, JM, IM] (tgrs, qgrs), per-column
// [JM, IM] (tskn, tsfa). Index 0 is the model top for the toa->sfc input (ivflip == 0).
// Point mapping: one thread per column (i, j), the level recurrences run in the thread (grid_cols).
// Kernel is sequential in k per column, exactly as the Fortran (tem1d carries across levels).
// No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

#define GRRAD_LTP      1          /* grrad_nmmb.f:210, add an extra top layer */
#define GRRAD_QME5     1.0e-7f    /* grrad_nmmb.f:197 */
#define GRRAD_QME6     1.0e-7f    /* grrad_nmmb.f:197 */
#define GRRAD_CON_FVIRT (4.6150e+2f / 2.8705e+2f - 1.0f)   /* physcons.f:79 */
#define GRRAD_M14      (-14.0f)   /* grrad_nmmb.f, ivflip==0 model-top log(p) */

extern "C" __global__ void phys_rrtm_grrad_gases(
    const float* __restrict__ plyr,   // [LMK, JM, IM] layer pressure (grrad setup output)
    const float* __restrict__ plvl,   // [LMP, JM, IM] level pressure (grrad setup output)
    const float* __restrict__ tlyr,   // [LMK, JM, IM] layer temperature (grrad setup output)
    const float* __restrict__ tskn,   // [JM, IM]      grrad skin temperature (RAD_TSKN = setup tsfc, itsfc == 0)
    const float* __restrict__ tgrs,   // [LM, JM, IM]  model temperature (T)
    const float* __restrict__ qgrs,   // [LM, JM, IM]  model specific humidity (Q)
    float* __restrict__ tlvl,         // [LMP, JM, IM] out: level temperature
    float* __restrict__ tvly,         // [LMK, JM, IM] out: layer virtual temperature
    float* __restrict__ qlyr,         // [LMK, JM, IM] out: layer humidity (floored)
    float* __restrict__ tem2da,       // [LMK, JM, IM] out: log(plyr)
    float* __restrict__ tem2db,       // [LMP, JM, IM] out: log(plvl)
    float* __restrict__ tsfa,         // [JM, IM]      out: surface-layer air temperature
    const int im, const int jm, const int lm, const int ivflip)
{
    COLUMN_IJ(im, jm)

    const int LMK = lm + GRRAD_LTP;      /* num of local layers   (grrad_nmmb.f:1165) */
    const int LMP = LMK + 1;             /* num of local levels   (grrad_nmmb.f:1166) */
    const int LP1 = lm + 1;              /* num of in/out levels  (grrad_nmmb.f:1159) */

    /* lextop block (grrad_nmmb.f:1168-1195), LTP = 1 */
    int kd, lya, lyb;
    if (ivflip == 1) {                   /* vertical from sfc upward */
        kd = 0;  lya = lm;  lyb = LP1;
    } else {                             /* input toa -> sfc */
        kd = 1;  lya = 2;   lyb = 1;
    }

    /* 1427-1431: log loop, k = 2..LMK (both branches) */
    for (int k = 2; k <= LMK; ++k) {
        tem2da[IDX3(i0, j0, k - 1, im, jm)] = logf(plyr[IDX3(i0, j0, k - 1, im, jm)]);
        tem2db[IDX3(i0, j0, k - 1, im, jm)] = logf(plvl[IDX3(i0, j0, k - 1, im, jm)]);
    }

    float tem1d;

    if (ivflip == 0) {                   /* input data from toa to sfc */
        tem1d = GRRAD_QME6;
        tem2da[IDX3(i0, j0, 0, im, jm)] = logf(plyr[IDX3(i0, j0, 0, im, jm)]);
        if (plvl[IDX3(i0, j0, 0, im, jm)] == 0.0f) {
            tem2db[IDX3(i0, j0, 0, im, jm)] = GRRAD_M14;      /* model top */
        } else {
            tem2db[IDX3(i0, j0, 0, im, jm)] = logf(plvl[IDX3(i0, j0, 0, im, jm)]);
        }

        tsfa[IDX2(i0, j0, im)] = tlyr[IDX3(i0, j0, LMK - 1, im, jm)];   /* sfc layer air temp */
        tlvl[IDX3(i0, j0, 0, im, jm)] = tlyr[IDX3(i0, j0, 0, im, jm)];
        tlvl[IDX3(i0, j0, LMP - 1, im, jm)] = tskn[IDX2(i0, j0, im)];

        for (int k = 1; k <= lm; ++k) {
            const int k1 = k + kd;
            const float q = fmaxf(tem1d, qgrs[IDX3(i0, j0, k - 1, im, jm)]);
            qlyr[IDX3(i0, j0, k1 - 1, im, jm)] = q;
            tem1d = fminf(GRRAD_QME5, q);
            tvly[IDX3(i0, j0, k1 - 1, im, jm)] =
                tgrs[IDX3(i0, j0, k - 1, im, jm)] * (1.0f + GRRAD_CON_FVIRT * q);
        }

        /* extra top layer (lextop) */
        qlyr[IDX3(i0, j0, lyb - 1, im, jm)] = qlyr[IDX3(i0, j0, lya - 1, im, jm)];
        tvly[IDX3(i0, j0, lyb - 1, im, jm)] = tvly[IDX3(i0, j0, lya - 1, im, jm)];

        for (int k = 2; k <= LMK; ++k) {
            const float t_k   = tlyr[IDX3(i0, j0, k - 1, im, jm)];
            const float t_km1 = tlyr[IDX3(i0, j0, k - 2, im, jm)];
            const float d2b   = tem2db[IDX3(i0, j0, k - 1, im, jm)];
            const float d2a   = tem2da[IDX3(i0, j0, k - 1, im, jm)];
            const float d2a_m = tem2da[IDX3(i0, j0, k - 2, im, jm)];
            tlvl[IDX3(i0, j0, k - 1, im, jm)] =
                t_k + (t_km1 - t_k) * (d2b - d2a) / (d2a_m - d2a);
        }
    } else {                             /* input data from sfc to toa */
        tem1d = GRRAD_QME6;
        tem2da[IDX3(i0, j0, 0, im, jm)] = logf(plyr[IDX3(i0, j0, 0, im, jm)]);
        tem2db[IDX3(i0, j0, 0, im, jm)] = logf(plvl[IDX3(i0, j0, 0, im, jm)]);
        tsfa[IDX2(i0, j0, im)] = tlyr[IDX3(i0, j0, 0, im, jm)];   /* sfc layer air temp */
        tlvl[IDX3(i0, j0, 0, im, jm)] = tskn[IDX2(i0, j0, im)];
        tlvl[IDX3(i0, j0, LMP - 1, im, jm)] = tlyr[IDX3(i0, j0, LMK - 1, im, jm)];

        for (int k = lm; k >= 1; --k) {
            const float q = fmaxf(tem1d, qgrs[IDX3(i0, j0, k - 1, im, jm)]);
            qlyr[IDX3(i0, j0, k - 1, im, jm)] = q;
            tem1d = fminf(GRRAD_QME5, q);
            tvly[IDX3(i0, j0, k - 1, im, jm)] =
                tgrs[IDX3(i0, j0, k - 1, im, jm)] * (1.0f + GRRAD_CON_FVIRT * q);
        }

        /* extra top layer (lextop) */
        qlyr[IDX3(i0, j0, lyb - 1, im, jm)] = qlyr[IDX3(i0, j0, lya - 1, im, jm)];
        tvly[IDX3(i0, j0, lyb - 1, im, jm)] = tvly[IDX3(i0, j0, lya - 1, im, jm)];

        for (int k = 1; k <= LMK - 1; ++k) {
            const float t_k   = tlyr[IDX3(i0, j0, k - 1, im, jm)];
            const float t_kp1 = tlyr[IDX3(i0, j0, k, im, jm)];
            const float d2b   = tem2db[IDX3(i0, j0, k, im, jm)];
            const float d2a   = tem2da[IDX3(i0, j0, k - 1, im, jm)];
            const float d2a_p = tem2da[IDX3(i0, j0, k, im, jm)];
            tlvl[IDX3(i0, j0, k, im, jm)] =
                t_k + (t_kp1 - t_k) * (d2b - d2a) / (d2a_p - d2a);
        }
    }
    /* nday = im (grrad_nmmb.f:1534) is set by the wrapper (RAD_NDAY) with a device fill, not here. */
}
