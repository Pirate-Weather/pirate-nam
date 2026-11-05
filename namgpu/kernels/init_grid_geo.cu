// init_grid_geo.cu -- rotated lat-lon geometry of H and V points (GLAT/GLON, VLAT/VLON) and the
// geographic lat/lon of the domain's SW corner, as NMMB grid_consts computes them.
//
// NOTICE: NAM source file nmm/module_CONTROL.F90 (NOAA/NCEP/EMC, US Government work, no licence file).
// The Fortran module carries no licence block; the module header is kept here as the notice:
//   module module_control  (NMMB control, nam.v4.2.11)
// Ported to CUDA for WOOF-NAM from nmm/module_CONTROL.F90:351-1139 (nam.v4.2.11, 83df4e13c1).
//
// Fortran routine GRID_CONSTS(global, ..., tph0d, tlm0d, sbd, wbd, dphd, dlmd, ...).
// This unit owns the geographic-coordinate statements of GRID_CONSTS only. The rest of GRID_CONSTS
// (f, dxh, dxv, dare, rare, wpdar, fah, fcp, fdiv, fad, ddmpu, ddv, hdac*, curv, dyh, dyv) belongs to
// unit init-grid-consts and is not computed here.
//
// Every statement this kernel reproduces (Fortran line -> kernel):
//   tph0 = tph0d*dtr, wb = wbd*dtr, sb = sbd*dtr, dlm = dlmd*dtr, dph = dphd*dtr   (329-335) -> scalars below
//   stph0 = sin(tph0), ctph0 = cos(tph0)                                             (346-347) -> scalars below
//   global H (854-875): tph_base = sb-dph-dph; tlm_base = wb-dlm-dlm; glat = aph; glon = alm
//                       with the global wrap (i<=ite-2, i>=its+2)
//   global V (881-890): tph_base = sb-dph-0.5*dph; tlm_base = wb-dlm-0.5*dlm; vlat, vlon (simple wrap)
//   regional H (1066-1089): tph_base = sb-dph; tlm_base = wb-dlm; rotated -> geographic by asin/atan2
//   regional V (1095-1116): tph_base = sb-0.5*dph; tlm_base = wb-0.5*dlm; rotated -> geographic
//   SW corner (898-907 global, 1124-1133 regional): glat_sw, glon_sw -> sw[0], sw[1]
//   Index limits: ids = jds = 1, ide = IM, jde = JM, its = ids, ite = ide, jts = jds, jte = jde
//   (one card, one domain, no MPI halo: CONTRACT section 1). i_lo/i_hi/j_lo/j_hi = full domain.
//   The MPI-only parts (e_bdy/n_bdy/s_bdy/w_bdy flags, looplimits) are dropped.
//
// Fortran i, j are 1-based; here i = i0+1, j = j0+1, so (j-jds+1) = j and (i-ids+1) = i.
// global_flag selects the global branch. The NAM 12 km parent runs global: false (CONTRACT section 5),
// so the regional branch is the one the forecast uses; the global branch is kept so the statement set is complete.
#include "nmmb_common.cuh"

#define GEO_PI   3.141592653589793f
#define GEO_DTR  (GEO_PI / 180.0f)

// Rotated (tph, tlm) -> geographic (aph, alm) on the regional rotated grid (NMMB regional branch,
// expressions in the order of the Fortran statements at 1066-1089).
__device__ __forceinline__ void geo_rotated_to_geographic(
    const float tph, const float tlm, const float tlm0, const float stph0, const float ctph0,
    float &aph, float &alm)
{
    const float stph = sinf(tph);
    const float ctph = cosf(tph);
    const float stlm = sinf(tlm);
    const float ctlm = cosf(tlm);
    const float sph = ctph0 * stph + stph0 * ctph * ctlm;
    aph = asinf(sph);
    const float anum = ctph * stlm;
    const float denom = (ctlm * ctph - stph0 * sph) / ctph0;
    const float relm = atan2f(anum, denom);
    alm = relm + tlm0;
    if (alm > GEO_PI) alm = alm - GEO_PI - GEO_PI;
    if (alm < -GEO_PI) alm = alm + GEO_PI + GEO_PI;
}

extern "C" __global__ void init_grid_geo(
    const int IM, const int JM, const int global_flag,
    const float tph0d, const float tlm0d, const float sbd, const float wbd,
    const float dlmd, const float dphd, const int ioff, const int joff,
    float* __restrict__ glat, float* __restrict__ glon,
    float* __restrict__ vlat, float* __restrict__ vlon,
    float* __restrict__ sw)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM) return;

    // Derived scalars of GRID_CONSTS, recomputed per thread with the Fortran's float32 operations.
    const float tph0 = tph0d * GEO_DTR;
    const float wb = wbd * GEO_DTR;
    const float sb = sbd * GEO_DTR;
    const float wb_ref = wb, sb_ref = sb;
    const float dlm = dlmd * GEO_DTR;
    const float dph = dphd * GEO_DTR;
    const float stph0 = sinf(tph0);
    const float ctph0 = cosf(tph0);
    const float tlm0 = tlm0d * GEO_DTR;

    // Fortran i, j (ids = jds = 1) of this point on the REFERENCE grid: a moving-nest window passes the reference
    // origin (sbd, wbd) and its whole-cell offset (ioff, joff), so a ground point shared by two windows gets the same
    // bits; 0, 0 for every grid without window keys (the same operations as before)
    const int i = i0 + 1 + ioff;
    const int j = j0 + 1 + joff;
    const size_t idx = IDX2(i0, j0, IM);

    if (global_flag) {
        // ---- global H points (854-875): the rotated coordinates are already geographic ----
        float tph_base = sb - dph - dph;
        float tlm_base = wb - dlm - dlm;
        float aph = tph_base + (float)j * dph;
        float alm = tlm_base + (float)i * dlm;
        if (alm > GEO_PI) {
            if (i <= IM - 2) alm = alm - GEO_PI - GEO_PI;          // i <= ite-2, ite = IM
            else alm = GEO_PI + (float)(i - IM + 1) * dlm;
        }
        if (alm < -GEO_PI) {
            if (i >= 3) alm = alm + GEO_PI + GEO_PI;               // i >= its+2, its = 1
            else alm = -GEO_PI + (float)(i - 2) * dlm;             // (i - its - 1), its = 1
        }
        glat[idx] = aph;
        glon[idx] = alm;

        // ---- global V points (881-890) ----
        tph_base = sb - dph - 0.5f * dph;
        aph = tph_base + (float)j * dph;
        tlm_base = wb - dlm - 0.5f * dlm;
        alm = tlm_base + (float)i * dlm;
        if (alm > GEO_PI) alm = alm - GEO_PI - GEO_PI;
        if (alm < -GEO_PI) alm = alm + GEO_PI + GEO_PI;
        vlat[idx] = aph;
        vlon[idx] = alm;
    } else {
        // ---- regional H points (1066-1089) ----
        float tph = (sb - dph) + (float)j * dph;
        float tlm = (wb - dlm) + (float)i * dlm;
        float aph, alm;
        geo_rotated_to_geographic(tph, tlm, tlm0, stph0, ctph0, aph, alm);
        glat[idx] = aph;
        glon[idx] = alm;

        // ---- regional V points (1095-1116): offset by half a grid step ----
        tph = (sb - 0.5f * dph) + (float)j * dph;
        tlm = (wb - 0.5f * dlm) + (float)i * dlm;
        geo_rotated_to_geographic(tph, tlm, tlm0, stph0, ctph0, aph, alm);
        vlat[idx] = aph;
        vlon[idx] = alm;
    }

    // ---- SW corner of the domain (898-907 global, 1124-1133 regional); one thread writes it ----
    if (i0 == 0 && j0 == 0) {
        // this grid's own SW H point: the reference origin plus the window offset (x + 0.0f = x without a window)
        const float sb = sb_ref + (float)joff * dph;
        const float wb = wb_ref + (float)ioff * dlm;
        float arg = sinf(sb) * ctph0 + cosf(sb) * stph0 * cosf(wb);
        arg = fminf(arg, 1.0f);
        arg = fmaxf(arg, -1.0f);
        const float glat_sw = asinf(arg);

        arg = (cosf(sb) * cosf(wb)) / (cosf(glat_sw) * ctph0)
            - tanf(glat_sw) * tanf(tph0d * GEO_DTR);
        arg = fminf(arg, 1.0f);
        arg = fmaxf(arg, -1.0f);
        sw[0] = glat_sw;
        sw[1] = tlm0d * GEO_DTR + copysignf(1.0f, wb) * acosf(arg);
    }
}
