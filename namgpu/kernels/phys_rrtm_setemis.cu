// phys_rrtm_setemis.cu -- setemis: surface emissivity for the NMMB RRTM longwave call.
//
// NOTICE: NAM source file phys/radiation_surface_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). Module and routine header kept as the notice:
//   module_radiation_surface   VTAGSFC='NCEP-Radiation_surface   v5.1  Nov 2012 '
//   setemis: set up surface emissivity for lw radiation (sea/land/ice type map and snow cover)
// Ported to CUDA for nam-gpu from phys/radiation_surface_nmmb.f:730-897 (nam.v4.2.11, 83df4e13c1).
//
// Line-by-line counterpart of the routine:
//   iemslw = mod(iemsflg,10); iemslw == 0 -> sfcemis = 1.0, return          (iemslw = sfc_flags[2] from phys-rrtm-sfc-init,
//                                                                            which already applied the missing-file fallback)
//   dltg = 360/IMXEMS, hdlt = 0.5*dltg                                       (constants below)
//   slmsk: WHERE (SICE > 0.5) SLMSK = 2, ELSEWHERE SLMSK = 1 - SM          (module_RA_RRTM.F90:913-917, caller side)
//   nint(slmsk)==0 (sea)     -> emsref(1)
//   nint(slmsk)==2 (sea ice) -> emsref(7); reached only through SICE > 0.5
//   land: tmp1 = xlon*rad2dg (+360 if < 0), first i1 with |tmp1-tmp2| <= hdlt -> i2 (default 1)
//         tmp1 = xlat*rad2dg, first j1 with |tmp1-(90-dltg*(j1-1))| <= hdlt -> j2 (default 1)
//         idx = max(2, idxems(i2,j2)); idx >= 7 -> 2; sfcemis = emsref(idx)
//   if ialbflg==1 and land: fsno0 = sncovr, sfcemis = sfcemis*(1-fsno0) + emsref(8)*fsno0
//   else if snowf > 0 (snowf = SI, mm, module_RA_RRTM:908):
//         asnow = 0.02*snowf; argh = min(0.50, max(0.025, 0.01*zorlf)) (zorlf = 100*Z0, :907)
//         hrgh = min(1.0, max(0.20, 1.0577 - 1.1538e-3*hprif)) (hprif = STDH, :911)
//         fsno0 = asnow/(argh+asnow)*hrgh; zero over sea where tsknf > 271.2 (tsknf = TSEA)
//         sfcemis = sfcemis*(1-fsno0) + emsref(8)*fsno0
//   tairf (IMAX-array argument of setemis) is never read in the Fortran body, so it is not passed.
//   Fortran nint() is round-half-away-from-zero: roundf().
//
// Layout (CONTRACT.md section 1): all 2-D fields [JM, IM], index IDX2(i0, j0, IM).
// idxems is Fortran idxems(IMXEMS,JMXEMS) read with format 80i1 (column-major fill, i fastest),
// so it is held as int32 [JMXEMS, IMXEMS] with idxems(i,j) at [j-1, i-1].
// One thread per column (grid_cols), no host sync, no allocation, no printf.
// Fast numerics: float throughout (kfpt = single), FMA contraction allowed.

#include "nmmb_common.cuh"

#define SETEMIS_IMXEMS 360          /* num of lon-pts in glb emis-type map (IMXEMS) */
#define SETEMIS_JMXEMS 180          /* num of lat-pts in glb emis-type map (JMXEMS) */
#define SETEMIS_RAD2DG 57.29578f    /* 180 / con_pi, single precision */

extern "C" __global__ void phys_rrtm_setemis(
    const float* __restrict__ glon,      // [JM, IM] longitude, radians (xlon)
    const float* __restrict__ glat,      // [JM, IM] latitude, radians  (xlat)
    const float* __restrict__ sm,        // [JM, IM] sea mask SM (1 sea, 0 land); slmsk = 1 - SM unless SICE > 0.5
    const float* __restrict__ sice,      // [JM, IM] sea-ice fraction SICE; SICE > 0.5 forces slmsk = 2
    const float* __restrict__ si,        // [JM, IM] snow depth, mm (snowf)
    const float* __restrict__ sncovr,    // [JM, IM] snow cover fraction (ialbflg == 1)
    const float* __restrict__ z0,        // [JM, IM] roughness length, m (zorlf = 100*Z0, cm)
    const float* __restrict__ stdh,      // [JM, IM] topographic standard deviation, m (hprif)
    const float* __restrict__ tskin,     // [JM, IM] surface temperature, K (tsknf)
    const int* __restrict__ idxems,      // [JMXEMS, IMXEMS] emissivity type map, idxems(i,j) at [j-1,i-1];
                                         // read only when iemslw == 1 (sfc_init produces it only then)
    const int* __restrict__ sfc_flags,   // [3] phys-rrtm-sfc-init output: [ialbflg, iemsflg, iemslw]
    float* __restrict__ sfcemis,         // [JM, IM] output surface emissivity
    int im, int jm)
{
    COLUMN_IJ(im, jm)
    const size_t c = IDX2(i0, j0, im);

    // emsref(1..8): open water, grass/shrub, bare soil/tundra, sandy desert, rocky desert, forest,
    // ice, snow (Fortran data statement).
    const float emsref[8] = {0.97f, 0.95f, 0.94f, 0.90f, 0.93f, 0.96f, 0.96f, 0.99f};

    const int ialbflg = sfc_flags[0];               // module ialbflg (sfc_init)
    const int iemslw = sfc_flags[2];                // module iemslw (sfc_init: mod(iemsflg,10), file fallback)

    if (iemslw == 0) {                              // sfc emis default to 1.0; the routine returns here
        sfcemis[c] = 1.0f;
        return;
    }

    const float dltg = 360.0f / (float)SETEMIS_IMXEMS;
    const float hdlt = 0.5f * dltg;

    // module_RA_RRTM.F90:913-917: WHERE (SICE > 0.5) SLMSK = 2.0, ELSEWHERE SLMSK = 1.0 - SM.
    const int nslm = (sice[c] > 0.5f) ? 2 : (int)roundf(1.0f - sm[c]);   // Fortran nint(slmsk)

    float emis;
    if (nslm == 0) {                                // sea point
        emis = emsref[0];
    } else if (nslm == 2) {                         // sea-ice
        emis = emsref[6];
    } else {                                        // land: map grid to the 1-degree emissivity type map
        float tmp1 = glon[c] * SETEMIS_RAD2DG;
        if (tmp1 < 0.0f) tmp1 = tmp1 + 360.0f;
        int i2 = 1;
        for (int i1 = 1; i1 <= SETEMIS_IMXEMS; ++i1) {
            const float tmp2 = dltg * (float)(i1 - 1) + hdlt;
            if (fabsf(tmp1 - tmp2) <= hdlt) {
                i2 = i1;
                break;
            }
        }

        tmp1 = glat[c] * SETEMIS_RAD2DG;
        int j2 = 1;
        for (int j1 = 1; j1 <= SETEMIS_JMXEMS; ++j1) {
            const float tmp2 = 90.0f - dltg * (float)(j1 - 1);
            if (fabsf(tmp1 - tmp2) <= hdlt) {
                j2 = j1;
                break;
            }
        }

        int id = idxems[(size_t)(j2 - 1) * SETEMIS_IMXEMS + (i2 - 1)];
        if (id < 2) id = 2;                         // idx = max(2, idxems(i2,j2))
        if (id >= 7) id = 2;
        emis = emsref[id - 1];
    }

    // Snow cover: input land snow cover when ialbflg == 1, otherwise snow-depth based (Fortran else branch).
    if (ialbflg == 1 && nslm == 1) {
        const float fsno0 = sncovr[c];
        const float fsno1 = 1.0f - fsno0;
        emis = emis * fsno1 + emsref[7] * fsno0;
    } else {
        const float snowf = si[c];
        if (snowf > 0.0f) {
            const float zorlf = 100.0f * z0[c];
            const float hprif = stdh[c];
            const float asnow = 0.02f * snowf;
            float argh = fmaxf(0.025f, 0.01f * zorlf);
            argh = fminf(0.50f, argh);
            float hrgh = fmaxf(0.20f, 1.0577f - 1.1538e-3f * hprif);
            hrgh = fminf(1.0f, hrgh);
            float fsno0 = asnow / (argh + asnow) * hrgh;
            if (nslm == 0 && tskin[c] > 271.2f) fsno0 = 0.0f;
            const float fsno1 = 1.0f - fsno0;
            emis = emis * fsno1 + emsref[7] * fsno0;
        }
    }

    sfcemis[c] = emis;
}
