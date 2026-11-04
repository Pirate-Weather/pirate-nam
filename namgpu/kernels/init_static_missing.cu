// init_static_missing.cu -- fill the static surface fields the analysis does not carry.
//
// NAM unit init-static-missing, mode "new": there is no NAM Fortran counterpart for the fill.
// NAM's own state reset sets the surface fields to the sentinel -1.E6 at allocation
// (module_SOLVER_GRID_COMP.F90:1001 ALBASE, :1031 MXSNAL). NAM's nemsio init reader then
// overwrites ALBASE with 0 before reading the record (module_INIT_READ_NEMSIO.F90:1772, read at
// :1768-1781; a second ALBASE read at :489-498). MXSNAL is read at :519-528 and :2126-2132 with
// no zero default. A missing ALBASE record therefore leaves ALBASE=0 in NAM. This port treats
// ALBASE and MXSNAL as missing when they are not a valid albedo fraction, so that 0 is filled
// here as well (the port's allocation zeros carry no sentinel, so a sentinel test would miss them).
//
// Fill rules (the formulas are the mid-range / fallback branches of the Noah surface
// initialisation as written in ArWen's core/kernels/noah.cu:1140-1143 and core/noah.py:292-305,
// Apache-2.0, notice in namgpu/third_party/arwen/LICENSE):
//   ALBASE = 0.5 * (ALBEDOMIN + ALBEDOMAX)      background albedo, no monthly albedo field
//   MXSNAL = MAXALB[IVGTYP] * 0.01              maximum deep-snow albedo (rdmaxalb = false)
// Out-of-range vegetation (IVGTYP outside 1..LUCATS) is set to the table's BARE category.
// Out-of-range soil (ISLTYP outside 1..SLCATS, including NAM's unfilled default 0 at
// module_INIT_READ_NEMSIO.F90:752-758) is set to the wrapper's soil_default.
//
// Divergences from NAM (declared, not bitwise to any model):
//   - the albedo fill (mid-range ALBASE, MAXALB*0.01 MXSNAL) is ArWen's choice; NAM has no fill;
//   - ISLTYP 0 and IVGTYP 0 (NAM read defaults at module_INIT_READ_NEMSIO.F90:752-758 and
//     :768-774, :1401-1407) are filled to LOAM (6) and BARE; NAM has no such fill;
//   - EMBCK and ISLOPE are not filled here: NAM has no EMBCK (its emissivity is EMISS=1.0 at
//     module_SOLVER_GRID_COMP.F90:10280) and the port has no ISLOPE consumer, so both were removed;
//   - the Noah VEGPARM/SOILPARM tables are WRF v4.6.1 tables (namgpu/data/noah_tables); the NAM
//     NAM source tree carries no Noah table, so the category columns are unverified
//     against NAM.
//
// Not a bitwise match to any model: a documented fill for fields the analysis lacks.
#include "nmmb_common.cuh"

// A valid albedo fraction is finite and in (0, 1]. Zero (NAM's reader default), negative
// values (a GRIB missing value times 0.01), and NaN are missing.
__device__ inline bool albedo_missing(float v)
{
    return !(v > 0.0f && v <= 1.0f);
}

extern "C" __global__ void init_static_missing(
    int* __restrict__ ivgtyp,
    int* __restrict__ isltyp,
    float* __restrict__ albase,
    float* __restrict__ mxsnal,
    const float* __restrict__ veg,      // [lucats][ncol], row-major, category-1 rows
    int lucats, int ncol,
    int c_albmin, int c_albmax, int c_maxalb,
    int slcats, int bare_cat, int soil_default,
    int im, int jm)
{
    POINT_IJL                              // i0, j0 (l0 unused: 2-D field, grid.z = 1)
    if (i0 >= im || j0 >= jm) return;
    const size_t p = IDX2(i0, j0, im);

    int v = ivgtyp[p];
    if (v < 1 || v > lucats) {
        v = bare_cat;
        ivgtyp[p] = v;
    }
    const float* row = veg + (size_t)(v - 1) * (size_t)ncol;

    int s = isltyp[p];
    if (s < 1 || s > slcats) {
        isltyp[p] = soil_default;
    }

    if (albedo_missing(albase[p])) {
        albase[p] = 0.5f * (row[c_albmin] + row[c_albmax]);
    }
    if (albedo_missing(mxsnal[p])) {
        mxsnal[p] = row[c_maxalb] * 0.01f;
    }
}
