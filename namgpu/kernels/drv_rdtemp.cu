// drv_rdtemp.cu -- RDTEMP: radiative temperature change applied every step, with the
// zenith-angle scaling of the shortwave tendency between radiation calls.
//
// Ported to CUDA for nam-gpu from phys/module_RA_GFDL.F90:116-233 (nam.v4.2.11, 83df4e13c1).
// NOAA notice (documentation block of RDTEMP, module_RA_GFDL.F90:121-143):
//   SUBPROGRAM: RDTEMP      RADIATIVE TEMPERATURE CHANGE
//   PRGRMMR: BLACK          ORG: W/NP22     DATE: 93-12-29
//   ABSTRACT: RDTEMP APPLIES THE TEMPERATURE TENDENCIES DUE TO RADIATION AT ALL LAYERS AT
//             EACH ADJUSTMENT TIME STEP
// The two helpers RDTEMP calls are inlined here: ZENITH (module_RA_GFDL.F90:1662-1805) and
// CAL_MON_DAY (module_RA_GFDL.F90:7823-7856). Their scalar solar geometry depends only on the
// step time and calendar, so each thread evaluates it (identical value in every thread) instead
// of a host round trip. The per-cell cosine of the zenith angle, CZEN(I,J), is the ZENITH loop body.
//
// Fortran-to-CUDA review (CONTRACT section 6, part 3):
//   TIMES=NTSD*DT                           -> times
//   CAL_MON_DAY(JULDAY,JULYR)               -> rd_cal_mon_day
//   ZENITH scalar part (DAY..SIDTIM,HRANG)  -> rd_zenith_header (every line, incl. the
//                                              ASIN/ACOS branch and the SIDTIM wrap)
//   ZENITH DAYI reset (after CZEN, lines 1797-1803) -> rd_zenith_header, dayi_o. RDTEMP never reads
//                                              DAYI, so the reset value is computed and dropped.
//   ZENITH cell loop (HRLCL, SINALT, clamp) -> body, written only on the tile ITS..ITE x JTS..JTE
//                                              (the ZENITH MYIS..MYIE loop and RDTEMP's copy loop),
//                                              read from the drv-loop-limits table (lim).
//   FACTR = CZEN/CZMEAN if CZMEAN>0, else 0 -> factr, on the same tile.
//   T += (RSWTT*FACTR + RLWTT)*DT on ITS_B1..ITE_B1 x JTS_B1..JTE_B1 (every level) -> the l loop
//   MONTH(2)=29 persistence: ZENITH and CAL_MON_DAY keep MONTH(2)=29 across calls once a leap
//   year has been seen in the process (DATA-initialised locals, implicit SAVE). The caller passes
//   leap_seen (the sticky flag); a leap year in the current call also sets it. A run has one
//   JULYR, so leap_seen equals JULYR%4==0 for every call, which is the Fortran behaviour.
//
// Precision: float everywhere (kfpt=single). The Fortran literal 3.1415926 is REAL, so the
// constants below are its float value; single-precision sinf/cosf/asinf/acosf.

#include "nmmb_common.cuh"

#define RD_PI      3.1415926f
#define RD_PI2     (2.0f * RD_PI)
#define RD_DEG2RD  (3.1415926f / 180.0f)
#define RD_OBLIQ   (23.440f * RD_DEG2RD)
#define RD_ZEROJD  2451545.0f
#define RD_GSTC1   24110.54841f
#define RD_GSTC2   8640184.812866f
#define RD_GSTC3   9.3104E-2f
#define RD_GSTC4   -6.2E-6f

// MONTH(1..12). MONTH(2)=29 when the current year is leap or a leap year was seen before
// (sticky, as the Fortran DATA-initialised MONTH array persists between calls).
__device__ __forceinline__ void rd_month_days(int year, int leap_seen, int* month) {
    const int base[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (int k = 0; k < 12; k++) month[k] = base[k];
    if (year % 4 == 0 || leap_seen) month[1] = 29;
}

// CAL_MON_DAY(JULDAY, JULYR, Jmonth, Jday): day-of-year to month and day of month.
__device__ __forceinline__ void rd_cal_mon_day(int julday, int julyr, int leap_seen,
                                               int* jmonth, int* jday) {
    int month[12];
    rd_month_days(julyr, leap_seen, month);
    int itmpday = julday;
    *jmonth = 12;
    *jday = itmpday;
    for (int i = 0; i < 12; i++) {
        if (itmpday > month[i]) {
            itmpday -= month[i];
        } else {
            *jday = itmpday;
            *jmonth = i + 1;
            break;
        }
    }
}

// ZENITH scalar part: the solar declination DEC and the hour angle HRANG for this call,
// plus the DAYI reset that ZENITH applies after its cell loop (returned; RDTEMP drops it).
__device__ __forceinline__ void rd_zenith_header(float times, int ihrst,
                                                 int idat_m, int idat_d, int idat_y,
                                                 int leap_seen,
                                                 float* dec_o, float* hrang_o, float* dayi_o) {
    int month[12];
    rd_month_days(idat_y, leap_seen, month);

    float day = 0.0f;
    if (idat_m > 1) {
        for (int knt = 1; knt <= idat_m - 1; knt++) {
            day = day + (float)month[knt - 1];
        }
    }
    day = day + (float)(idat_d - 1) + ((float)ihrst + times / 3600.0f) / 24.0f;
    const float dayi = (float)((int)day + 1);
    const float hour = (day - dayi + 1.0f) * 24.0f;

    const float yfctr = 2000.0f - (float)idat_y;
    const int idifyr = idat_y - 2000;
    float addday;
    if (idifyr < 0) {
        addday = (float)(idifyr / 4);
    } else {
        addday = (float)((idifyr + 3) / 4);
    }
    const float startyr = RD_ZEROJD + (float)idifyr * 365.0f + addday - 0.5f;

    const float datjul = startyr + day;
    const float difjd = datjul - RD_ZEROJD;

    const float slonm = (280.460f + 0.9856474f * difjd) * RD_DEG2RD + yfctr * RD_PI2;
    const float anom = (357.528f + 0.9856003f * difjd) * RD_DEG2RD;
    float slon = slonm + (1.915f * sinf(anom) + 0.020f * sinf(2.0f * anom)) * RD_DEG2RD;
    if (slon > RD_PI2) slon = slon - RD_PI2;

    const float dec = asinf(sinf(slon) * sinf(RD_OBLIQ));
    float ra = acosf(cosf(slon) / cosf(dec));
    if (slon > RD_PI) ra = RD_PI2 - ra;

    const float datj0 = startyr + dayi - 1.0f;
    const float tu = (datj0 - 2451545.0f) / 36525.0f;
    const float stim0 = RD_GSTC1 + tu * (RD_GSTC2 + RD_GSTC3 * tu + RD_GSTC4 * tu * tu);
    float sidtim = stim0 / 3600.0f + yfctr * 24.0f + 1.00273791f * hour;
    sidtim = sidtim * 15.0f * RD_DEG2RD;
    if (sidtim < 0.0f) sidtim = sidtim + RD_PI2;
    if (sidtim > RD_PI2) sidtim = sidtim - RD_PI2;

    // DAYI reset (ZENITH, after the cell loop): LEAP is the current-call flag, not the sticky one.
    const bool leap_now = (idat_y % 4 == 0);
    float dayi_out = dayi;
    if (dayi_out > 365.0f) {
        if (!leap_now) {
            dayi_out = dayi_out - 365.0f;
        } else if (dayi_out > 366.0f) {
            dayi_out = dayi_out - 366.0f;
        }
    }

    *dec_o = dec;
    *hrang_o = sidtim - ra;
    *dayi_o = dayi_out;
}

// RDTEMP body. One thread per column (i, j); the level loop runs inside the thread.
// Arguments (CONTRACT names): NTSD -> ntsd, DT -> dt, JULDAY/JULYR -> julday/julyr,
// IHRST (START_HOUR) -> ihrst, leap_seen (the sticky leap flag of MONTH(2)), GLAT/GLON (radians),
// CZEN (written on the tile), CZMEAN, T (inout), RSWTT, RLWTT, lim (loop-limit table).
// Shapes: GLAT/GLON/CZEN/CZMEAN [JM,IM]; T/RSWTT/RLWTT [LM,JM,IM].
extern "C" __global__ void rdtemp_apply(const int ntsd, const float dt,
                                        const int julday, const int julyr, const int ihrst,
                                        const int leap_seen,
                                        const float* __restrict__ glat,
                                        const float* __restrict__ glon,
                                        float* __restrict__ czen,
                                        const float* __restrict__ czmean,
                                        float* __restrict__ t,
                                        const float* __restrict__ rswtt,
                                        const float* __restrict__ rlwtt,
                                        const int im, const int jm, const int lm,
                                        const int* __restrict__ lim) {
    COLUMN_IJ(im, jm)

    // ZENITH loops MYIS..MYIE x MYJS..MYJE, which RDTEMP passes as ITS..ITE x JTS..JTE; the
    // RDTEMP copy loop writes CZEN and FACTR on the same range. Cells outside keep their value.
    const bool in_tile = (i0 >= lim[DL_ITS] - 1) && (i0 <= lim[DL_ITE] - 1) &&
                         (j0 >= lim[DL_JTS] - 1) && (j0 <= lim[DL_JTE] - 1);
    if (!in_tile) return;

    int jmonth, jday;
    rd_cal_mon_day(julday, julyr, leap_seen, &jmonth, &jday);
    const float times = (float)ntsd * dt;

    float dec, hrang, dayi;
    rd_zenith_header(times, ihrst, jmonth, jday, julyr, leap_seen, &dec, &hrang, &dayi);

    const size_t ij = IDX2(i0, j0, im);
    const float xlat = glat[ij];
    const float xlon = glon[ij];
    const float hrlcl = hrang + xlon + RD_PI2;
    float sinalt = sinf(dec) * sinf(xlat) + cosf(dec) * cosf(hrlcl) * cosf(xlat);
    if (sinalt < 0.0f) sinalt = 0.0f;
    czen[ij] = sinalt;

    float factr;
    if (czmean[ij] > 0.0f) {
        factr = sinalt / czmean[ij];
    } else {
        factr = 0.0f;
    }

    // DO J=JTS_B1,JTE_B1 / DO I=ITS_B1,ITE_B1 (1-based values from the drv-loop-limits table).
    // B1 is inside the tile by construction of the table (its_b1 >= its, ite_b1 <= ite).
    const bool in_b1 = (i0 >= lim[DL_ITS_B1] - 1) && (i0 <= lim[DL_ITE_B1] - 1) &&
                       (j0 >= lim[DL_JTS_B1] - 1) && (j0 <= lim[DL_JTE_B1] - 1);
    if (in_b1) {
        for (int l0 = 0; l0 < lm; l0++) {
            const size_t idx = IDX3(i0, j0, l0, im, jm);
            const float ttndkl = rswtt[idx] * factr + rlwtt[idx];
            t[idx] = t[idx] + ttndkl * dt;
        }
    }
}
