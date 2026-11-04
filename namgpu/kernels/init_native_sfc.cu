// nam-gpu unit init-native-sfc: surface and soil initial state from the NAM native analysis
// (bgrd3d00 GRIB2 fields) into the NMMB state arrays of namgpu.state.State.
//
// Mode "new": no Fortran counterpart. Names, units and layouts follow the NAM state tables
// (parm/nam_phy_state.txt, parm/nam_solver_state.txt) and CONTRACT.md section 1.
// Index mapping: 2-D fields [j, i] -> IDX2(i0, j0, IM); soil fields [NSOIL=4, JM, IM] and
// the 3-D fields -> IDX3(i0, j0, l0, IM, JM); l0 = 0 is the top soil layer.
//
// Unit conversions (GRIB -> NMMB state):
//   TSOIL (K)                       -> STC (K)                       layers 1-4 as-is
//   SOILW (volumetric fraction)     -> SMC                           as-is
//   SH2O                            -> liquid part of SMC, below
//   TMP surface (K)                 -> TSKIN (K)                     as-is
//   TMP surface (K) (NAM record tsea) -> SST (K)  all points, no land test (NAM reader)
//   ICEC (fraction)                 -> SICE                          as-is
//   LAND (1 = land)                 -> SM (1 = sea, 0 = land)        SM = 1 - LAND
//   TSOIL at 3 m (K)                -> TG (K)                        as-is
//   SNOD (m)                        -> SI (mm snow depth)            x 1000
//   WEASD (kg m-2 = mm)             -> SNO (mm liquid water)         as-is
//   SNOWC (fraction, when present)  -> SNOWC                         as-is; absent: 0.98 if SNO>0 else 0
//   ALBDO (percent)                 -> ALBEDO (fraction)             / 100 (ALBASE is init-static-missing's)
//   VEG (percent)                   -> VEGFRC (fraction)             / 100
//   SFCR (m)                        -> Z0 and Z0BASE (m)             as-is
//   CNWAT (kg m-2)                  -> CMC (m canopy water)          / 1000
//   VGTYP, SOTYP (int)              -> IVGTYP, ISLTYP                as-is
//
// SI and SNO are millimetres in the NAM solver state (parm/nam_solver_state.txt:178,184):
// SOLVER passes them to physics as SNWDPH = SI and WEASD = SNO in mm (module_SOLVER_GRID_COMP
// .F90:6594-6595), and TURBULENCE forms SNOWH = SI*SNO_FACTR with SNO_FACTR = 0.001 to get metres.
// The phy table's 'm' wording (parm/nam_phy_state.txt:113,119) is the contradicted one.
// Soil liquid water (SH2O) from the frozen fraction: the supercooled liquid content of a
// frozen layer follows the Clapp-Hornberger retention curve with the freezing-point depression
// of Niu and Yang (2006), psi_T = Lf (Tf - T) / (g Tf), theta_liq = theta_s (psi_T / psi_s)^(-1/b),
// bounded by the residual floor SFC_SMC_RES and by SMC. Above Tf, SH2O = SMC. This is written
// from the published equations (no NAM routine matches it), so it is an approximation, not a
// reproduction of a NAM subroutine. Soil class parameters b, psi_s, theta_s come from
// ctx.tables["soil_hydraulics"], which init-static-missing supplies.
//
// Parallelism: one thread per column (IM x JM), each thread loops over the four soil layers
// (kernels.grid_cols). No host sync, no device malloc.

#include "nmmb_common.cuh"

#define SFC_NSOIL     4
#define SFC_NCLASS    30
#define SFC_TFRZ      273.16f
#define SFC_LF        3.336e5f     // latent heat of fusion, J kg-1 (NMMB_LF convention)
#define SFC_GRAV      9.81f        // m s-2
#define SFC_SMC_RES   0.02f        // residual liquid volumetric content floor

extern "C" __global__ void init_native_sfc(
    const float* __restrict__ tsoil,        // [4, JM, IM] K
    const float* __restrict__ soilw,        // [4, JM, IM] volumetric fraction
    const float* __restrict__ tmp_sfc,      // [JM, IM] K
    const float* __restrict__ icec,         // [JM, IM] fraction
    const float* __restrict__ land,         // [JM, IM] 1 = land
    const float* __restrict__ tsoil_3m,     // [JM, IM] K, deep soil temperature
    const float* __restrict__ snod,         // [JM, IM] m
    const float* __restrict__ weasd,        // [JM, IM] kg m-2
    const float* __restrict__ albdo,        // [JM, IM] percent
    const float* __restrict__ veg,          // [JM, IM] percent
    const float* __restrict__ sfcr,         // [JM, IM] m
    const float* __restrict__ snowc,        // [JM, IM] fraction, read only when has_snowc
    const int*   __restrict__ has_snowc_p,  // scalar flag: 1 = snowc record present
    const float* __restrict__ cnwat,        // [JM, IM] kg m-2
    const int*   __restrict__ vgtyp,        // [JM, IM]
    const int*   __restrict__ sotyp,        // [JM, IM]
    const float* __restrict__ bexp_tbl,     // [30] Clapp-Hornberger b
    const float* __restrict__ psisat_tbl,   // [30] m
    const float* __restrict__ smcmax_tbl,   // [30] theta_s
    float* __restrict__ stc,                // [4, JM, IM]
    float* __restrict__ smc,                // [4, JM, IM]
    float* __restrict__ sh2o,               // [4, JM, IM]
    float* __restrict__ tskin,              // [JM, IM]
    float* __restrict__ sst,                // [JM, IM]
    float* __restrict__ sice,               // [JM, IM]
    float* __restrict__ sm,                 // [JM, IM]
    float* __restrict__ tg,                 // [JM, IM]
    float* __restrict__ si,                 // [JM, IM]
    float* __restrict__ sno,                // [JM, IM]
    float* __restrict__ snowc_out,          // [JM, IM]
    float* __restrict__ albedo,             // [JM, IM]
    float* __restrict__ vegfrc,             // [JM, IM]
    float* __restrict__ z0,                 // [JM, IM]
    float* __restrict__ z0base,             // [JM, IM]
    float* __restrict__ cmc,                // [JM, IM]
    int*   __restrict__ ivgtyp,             // [JM, IM]
    int*   __restrict__ isltyp,             // [JM, IM]
    int im, int jm)
{
    COLUMN_IJ(im, jm)

    const size_t p2 = IDX2(i0, j0, im);

    const float lnd   = land[p2];
    const float tsk   = tmp_sfc[p2];
    const int   sotyp_v = sotyp[p2];

    // Soil class parameters (class clamped to the table; class 0 is the table's first row).
    int cls = sotyp_v;
    if (cls < 0) cls = 0;
    if (cls > SFC_NCLASS - 1) cls = SFC_NCLASS - 1;
    const float bexp   = bexp_tbl[cls];
    const float psisat = psisat_tbl[cls];
    const float smax   = smcmax_tbl[cls];

    for (int l0 = 0; l0 < SFC_NSOIL; ++l0) {
        const size_t p3 = IDX3(i0, j0, l0, im, jm);
        const float t = tsoil[p3];
        const float m = soilw[p3];
        stc[p3] = t;
        smc[p3] = m;

        float liq = m;
        if (t < SFC_TFRZ && psisat > 0.0f && bexp > 0.0f) {
            const float psi_t = SFC_LF * (SFC_TFRZ - t) / (SFC_GRAV * SFC_TFRZ);
            const float th = smax * powf(psi_t / psisat, -1.0f / bexp);
            liq = fminf(m, fmaxf(th, SFC_SMC_RES));
        }
        sh2o[p3] = liq;
    }

    tskin[p2] = tsk;
    sst[p2] = tsk;   // NAM reader: SST = tsea on every point
    sice[p2] = fminf(fmaxf(icec[p2], 0.0f), 1.0f);
    sm[p2] = 1.0f - fminf(fmaxf(lnd, 0.0f), 1.0f);
    tg[p2] = tsoil_3m[p2];
    const float sno_mm = weasd[p2];
    si[p2] = snod[p2] * 1000.0f;
    sno[p2] = sno_mm;
    snowc_out[p2] = has_snowc_p[0] ? snowc[p2] : (sno_mm > 0.0f ? 0.98f : 0.0f);
    albedo[p2] = fminf(fmaxf(albdo[p2] * 0.01f, 0.0f), 1.0f);
    vegfrc[p2] = fminf(fmaxf(veg[p2] * 0.01f, 0.0f), 1.0f);
    z0[p2] = sfcr[p2];
    z0base[p2] = sfcr[p2];
    cmc[p2] = fmaxf(cnwat[p2] * 0.001f, 0.0f);
    ivgtyp[p2] = vgtyp[p2];
    isltyp[p2] = sotyp_v;
}
