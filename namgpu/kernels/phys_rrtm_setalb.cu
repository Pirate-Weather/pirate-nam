// nam-gpu unit phys-rrtm-setalb: SETALB, surface albedo for the SW radiation call.
//
// NOTICE: NOAA/NCEP NAM (nam.v4.2.11, 83df4e13c1) NMMB physics, phys/radiation_surface_nmmb.f,
// module_radiation_surface, subroutine setalb (lines 407-725). NOAA US government work, no licence
// file in the NAM clone; the NOAA source header is kept as the notice. The module header reads
// "module_radiation_surface description" (NCEP/EMC NMMB physics; the routine comments cite
// Briegleb 1992 and Ferrier 2005).
// Ported to CUDA for nam-gpu from phys/radiation_surface_nmmb.f:407-725 (nam.v4.2.11, 83df4e13c1).
//
// Mapping: one thread per (i, j) column, one launch (kernels.grid_cols). Every Fortran branch of
// ialbflg is kept: 0 = climatological scheme (Briegleb 1992), 1 = MODIS-based land albedo with
// input snow cover, anything else = externally provided albedoes (the NAM regional path, IALB = 2,
// module_SOLVER_GRID_COMP.F90:10676). The Fortran dummy IMAX is the kernel argument im (IM). The
// dummy TAIRF (lowest-layer air temperature) is not an argument: setalb never reads it.
// Arrays are 2-D [JM, IM] (CONTRACT section 1), zero-based c = IDX2(i0, j0, IM); SFCALB is
// [NF_ALBD = 4, JM, IM], component k at IDX3(i0, j0, k, IM, JM).
//
// Units: snowf in mm (water equivalent), zorlf in cm (wrapper scales Z0 m -> cm), hprif in m,
// tisfc and tsknf in K.
//
// Constants (single precision, kfpt):
//   con_t0c  = 2.7315e+2 K (phys/physcons.f:67)
//   con_ttp  = 2.7316e+2 K (phys/physcons.f:68)
//   con_tice = 2.7120e+2 K (phys/physcons.f:69)
// Fortran nint() rounds half away from zero: roundf(). x**2 is x*x; x**1.7 is powf(x, 1.7f).
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

#define SETALB_CON_T0C   273.15f
#define SETALB_CON_TTP   273.16f
#define SETALB_CON_TICE  271.20f

extern "C" __global__ void phys_rrtm_setalb(
    const float* __restrict__ slmsk,    // [JM, IM] mask (0 sea, 1 land, 2 ice)
    const float* __restrict__ snowf,    // [JM, IM] snow depth, water equivalent (mm)
    const float* __restrict__ sncovr,   // [JM, IM] snow cover fraction (ialbflg 1)
    const float* __restrict__ snoalb,   // [JM, IM] max snow albedo over land (ialbflg 1)
    const float* __restrict__ zorlf,    // [JM, IM] surface roughness (cm)
    const float* __restrict__ coszf,    // [JM, IM] cosine of solar zenith angle
    const float* __restrict__ tsknf,    // [JM, IM] ground skin temperature (K)
    const float* __restrict__ hprif,    // [JM, IM] topographic standard deviation (m)
    const float* __restrict__ alvsf,    // [JM, IM] visible albedo, strong cosz dependency
    const float* __restrict__ alnsf,    // [JM, IM] near-IR albedo, strong cosz dependency
    const float* __restrict__ alvwf,    // [JM, IM] visible albedo, weak cosz dependency
    const float* __restrict__ alnwf,    // [JM, IM] near-IR albedo, weak cosz dependency
    const float* __restrict__ facsf,    // [JM, IM] fraction, strong cosz dependency
    const float* __restrict__ facwf,    // [JM, IM] fraction, weak cosz dependency
    const float* __restrict__ fice,     // [JM, IM] sea-ice fraction
    const float* __restrict__ tisfc,    // [JM, IM] sea-ice surface temperature (K)
    const int ialbflg,
    const int im,
    const int jm,
    float* __restrict__ sfcalb)         // [4, JM, IM] output
{
    COLUMN_IJ(im, jm)
    const size_t c = IDX2(i0, j0, im);
#define SA(k) sfcalb[IDX3(i0, j0, (k), im, jm)]

    const int msk = (int)roundf(slmsk[c]);
    const float ts = tsknf[c];
    const float cz = coszf[c];

    float fsno0, fsno1, flnd0, fsea0, fsno, fsea, flnd;
    float asevd, asend, asnvd, asnnd, asevb, asenb, asnvb, asnnb;
    float rfcs, rfcw, csnow, a1, b1, a2, b2, asnow, argh, hrgh, ffw, dtgd, b3;

    if (ialbflg == 0) {
        // ---- climatological albedo scheme (Briegleb 1992): setalb, ialbflg == 0 ----
        asnow = 0.02f * snowf[c];
        argh  = fminf(0.50f, fmaxf(0.025f, 0.01f * zorlf[c]));
        hrgh  = fminf(1.0f, fmaxf(0.20f, 1.0577f - 1.1538e-3f * hprif[c]));
        fsno0 = asnow / (argh + asnow) * hrgh;
        if (msk == 0 && ts > SETALB_CON_TICE) fsno0 = 0.0f;
        fsno1 = 1.0f - fsno0;
        flnd0 = fminf(1.0f, facsf[c] + facwf[c]);
        fsea0 = fmaxf(0.0f, 1.0f - flnd0);
        fsno  = fsno0;
        fsea  = fsea0 * fsno1;
        flnd  = flnd0 * fsno1;

        // diffused sea surface albedo
        if (ts >= 271.5f) {
            asevd = 0.06f;
            asend = 0.06f;
        } else if (ts < 271.1f) {
            asevd = 0.70f;
            asend = 0.65f;
        } else {
            a1 = (ts - 271.1f) * (ts - 271.1f);
            asevd = 0.7f - 4.0f * a1;
            asend = 0.65f - 3.6875f * a1;
        }

        // diffused snow albedo
        if (msk == 2) {
            ffw = 1.0f - fice[c];
            if (ffw < 1.0f) {
                dtgd = fmaxf(0.0f, fminf(5.0f, (SETALB_CON_TTP - tisfc[c])));
                b1 = 0.03f * dtgd;
            } else {
                b1 = 0.0f;
            }
            b3    = 0.06f * ffw;
            asnvd = (0.70f + b1) * fice[c] + b3;
            asnnd = (0.60f + b1) * fice[c] + b3;
            asevd = 0.70f * fice[c] + b3;
            asend = 0.60f * fice[c] + b3;
        } else {
            asnvd = 0.90f;
            asnnd = 0.75f;
        }

        // direct snow albedo
        if (cz < 0.5f) {
            csnow = 0.5f * (3.0f / (1.0f + 4.0f * cz) - 1.0f);
            asnvb = fminf(0.98f, asnvd + (1.0f - asnvd) * csnow);
            asnnb = fminf(0.98f, asnnd + (1.0f - asnnd) * csnow);
        } else {
            asnvb = asnvd;
            asnnb = asnnd;
        }

        // direct sea surface albedo (f. yang's zenith angle treatment; the Briegleb forms are
        // commented out in the Fortran and are not ported)
        if (cz > 0.0001f) {
            rfcs = 2.14f / (1.0f + 1.48f * cz);
            rfcw = rfcs;
            if (ts >= SETALB_CON_T0C) {
                asevb = fmaxf(asevd, 0.026f / (powf(cz, 1.7f) + 0.065f)
                                     + 0.15f * (cz - 0.1f) * (cz - 0.5f) * (cz - 1.0f));
                asenb = asevb;
            } else {
                asevb = asevd;
                asenb = asend;
            }
        } else {
            rfcs  = 1.0f;
            rfcw  = 1.0f;
            asevb = asevd;
            asenb = asend;
        }

        a1 = alvsf[c] * facsf[c];
        b1 = alvwf[c] * facwf[c];
        a2 = alnsf[c] * facsf[c];
        b2 = alnwf[c] * facwf[c];
        SA(0) = (a2 * rfcs + b2 * rfcw) * flnd + asenb * fsea + asnnb * fsno;
        SA(1) = (a2 + b2) * 0.96f * flnd + asend * fsea + asnnd * fsno;
        SA(2) = (a1 * rfcs + b1 * rfcw) * flnd + asevb * fsea + asnvb * fsno;
        SA(3) = (a1 + b1) * 0.96f * flnd + asevd * fsea + asnvd * fsno;

    } else if (ialbflg == 1) {
        // ---- MODIS-based land albedo, snow cover from the land model: setalb, ialbflg == 1 ----
        fsno0 = sncovr[c];
        if (msk == 0 && ts > SETALB_CON_TICE) fsno0 = 0.0f;
        if (msk == 2) {
            asnow = 0.02f * snowf[c];
            argh  = fminf(0.50f, fmaxf(0.025f, 0.01f * zorlf[c]));
            hrgh  = fminf(1.0f, fmaxf(0.20f, 1.0577f - 1.1538e-3f * hprif[c]));
            fsno0 = asnow / (argh + asnow) * hrgh;
        }
        fsno1 = 1.0f - fsno0;
        flnd0 = fminf(1.0f, facsf[c] + facwf[c]);
        fsea0 = fmaxf(0.0f, 1.0f - flnd0);
        fsno  = fsno0;
        fsea  = fsea0 * fsno1;
        flnd  = flnd0 * fsno1;

        // diffused sea surface albedo
        if (ts >= 271.5f) {
            asevd = 0.06f;
            asend = 0.06f;
        } else if (ts < 271.1f) {
            asevd = 0.70f;
            asend = 0.65f;
        } else {
            a1 = (ts - 271.1f) * (ts - 271.1f);
            asevd = 0.7f - 4.0f * a1;
            asend = 0.65f - 3.6875f * a1;
        }

        // diffused snow albedo; land uses the input max snow albedo
        if (msk == 2) {
            ffw = 1.0f - fice[c];
            if (ffw < 1.0f) {
                dtgd = fmaxf(0.0f, fminf(5.0f, (SETALB_CON_TTP - tisfc[c])));
                b1 = 0.03f * dtgd;
            } else {
                b1 = 0.0f;
            }
            b3    = 0.06f * ffw;
            asnvd = (0.70f + b1) * fice[c] + b3;
            asnnd = (0.60f + b1) * fice[c] + b3;
            asevd = 0.70f * fice[c] + b3;
            asend = 0.60f * fice[c] + b3;
        } else {
            asnvd = snoalb[c];
            asnnd = snoalb[c];
        }

        // direct snow albedo
        if (msk == 2) {
            if (cz < 0.5f) {
                csnow = 0.5f * (3.0f / (1.0f + 4.0f * cz) - 1.0f);
                asnvb = fminf(0.98f, asnvd + (1.0f - asnvd) * csnow);
                asnnb = fminf(0.98f, asnnd + (1.0f - asnnd) * csnow);
            } else {
                asnvb = asnvd;
                asnnb = asnnd;
            }
        } else {
            asnvb = snoalb[c];
            asnnb = snoalb[c];
        }

        // direct sea surface albedo (f. yang's zenith angle treatment)
        if (cz > 0.0001f) {
            rfcs = 1.775f / (1.0f + 1.55f * cz);
            if (ts >= SETALB_CON_T0C) {
                asevb = fmaxf(asevd, 0.026f / (powf(cz, 1.7f) + 0.065f)
                                     + 0.15f * (cz - 0.1f) * (cz - 0.5f) * (cz - 1.0f));
                asenb = asevb;
            } else {
                asevb = asevd;
                asenb = asend;
            }
        } else {
            rfcs  = 1.0f;
            asevb = asevd;
            asenb = asend;
        }

        SA(0) = alnsf[c] * rfcs * flnd + asenb * fsea + asnnb * fsno;
        SA(1) = alnwf[c] * flnd + asend * fsea + asnnd * fsno;
        SA(2) = alvsf[c] * rfcs * flnd + asevb * fsea + asnvb * fsno;
        SA(3) = alvwf[c] * flnd + asevd * fsea + asnvd * fsno;

    } else {
        // ---- externally provided albedoes (ialbflg == 2, the NAM regional path) ----
        SA(0) = alnsf[c];
        SA(1) = alnwf[c];
        SA(2) = alvsf[c];
        SA(3) = alvwf[c];
    }
#undef SA
}
