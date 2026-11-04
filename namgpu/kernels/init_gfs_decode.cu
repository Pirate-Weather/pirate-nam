// init_gfs_decode.cu -- GFS 0.25 deg boundary decode: missing-value sanitize, neighbour fill,
// species clamp, below-ground and surface-unknown flags, for one 3-hourly boundary time.
//
// Unit: init-gfs-decode (group init, mode "new": no NAM Fortran counterpart; written from
// CONTRACT.md section 5 (boundaries: GFS every 3 h) and the GFS 0.25 deg pgrb2 record set
// that init-fetch-case fetches). Consumers: init-gfs-hinterp / init-gfs-vinterp, which read the
// fields as nemsinterp does (GHT_IN in metres, RH_IN as relative humidity, PRES_IN in Pa).
//
// Array layout (CONTRACT.md section 1, GFS native grid after init-grib2-decode, which writes
// rows south to north and columns west to east):
//   pl  [nvar, nlev, nlat, nlon] float32   (lon i fastest, then lat j, then level l, then var v)
//   plev[nlev]                    float32   pressure of each level (Pa), strictly decreasing:
//                                           index 0 = lowest level (metgrid order)
//   sfc [nsfc, nlat, nlon]        float32   PRES (Pa), HGT (gpm)
//   masks uint8, same shape as their field:
//     bit 0 (1) PRESENT  the value came from the file (not missing)
//     bit 1 (2) BELOW    level is below ground (plev > surface pressure, surface known)
//     bit 2 (4) FILLED   missing point replaced by the mean of its present 8-neighbours
//     bit 3 (8) PS_UNKNOWN  surface pressure is missing and could not be filled (BELOW not set)
//
// Missing values: GRIB bitmap-missing points arrive as NaN, Inf, or the NCEP fill 9.999e20. They
// are set to 0.0f in the sanitize pass, then gfs_fill_missing replaces each one with the mean of
// its present neighbours (lon periodic, lat clamped at the poles, same level). A point with no
// present neighbour stays 0.0f with PRESENT and FILLED both clear: consumers must treat it as
// missing. Every stored value is finite. Species and specific humidity/RH (nonneg[v] = 1) are
// clamped to >= 0 (negative values from the spectral-to-grid transform are roundoff).
// Below-ground points keep the decoded value; extrapolation below the surface is the
// interpolation units job, not this decodes.
//
// Launch order (one stream): gfs_decode_sanitize(sfc) -> gfs_fill_missing(sfc) ->
// gfs_decode_pl(pl, uses the filled surface pressure) -> gfs_fill_missing(pl).
// Fill reads only PRESENT points and writes only non-PRESENT points, so it is race-free in place.
// One thread per (i, j, slab); slab = v*nlev + l for pl (contiguous with IDX4), slab = v for sfc.

#include "nmmb_common.cuh"

#define GFS_FILL_MIN 9.0e20f   /* NCEP GRIB missing fill is 9.999e20 */
#define MASK_PRESENT    1
#define MASK_BELOW      2
#define MASK_FILLED     4
#define MASK_PS_UNKNOWN 8

extern "C" __global__ void gfs_decode_sanitize(float* dst, unsigned char* mask,
                                               int nslab, int nlat, int nlon)
{
    POINT_IJL                                   /* l0 is the slab index here */
    if (i0 >= nlon || j0 >= nlat || l0 >= nslab) return;
    const size_t idx = IDX3(i0, j0, l0, nlon, nlat);
    const float v = dst[idx];
    const int ok = (isfinite(v) && fabsf(v) < GFS_FILL_MIN);
    dst[idx] = ok ? v : 0.0f;
    mask[idx] = (unsigned char)(ok ? MASK_PRESENT : 0);
}

// Replace each missing point by the mean of its present 8-neighbours on the same slab.
extern "C" __global__ void gfs_fill_missing(float* f, unsigned char* mask,
                                            int nslab, int nlat, int nlon)
{
    POINT_IJL
    if (i0 >= nlon || j0 >= nlat || l0 >= nslab) return;
    const size_t idx = IDX3(i0, j0, l0, nlon, nlat);
    if (mask[idx] & MASK_PRESENT) return;
    float sum = 0.0f;
    int cnt = 0;
    for (int dj = -1; dj <= 1; ++dj) {
        const int jj = j0 + dj;
        if (jj < 0 || jj >= nlat) continue;
        for (int di = -1; di <= 1; ++di) {
            if (di == 0 && dj == 0) continue;
            const int ii = (i0 + di + nlon) % nlon;          /* longitude is periodic */
            const size_t n = IDX3(ii, jj, l0, nlon, nlat);
            if (mask[n] & MASK_PRESENT) { sum += f[n]; ++cnt; }
        }
    }
    if (cnt > 0) {
        f[idx] = sum / (float)cnt;
        mask[idx] = (unsigned char)(mask[idx] | MASK_FILLED);
    }
}

// Level decode: clamp species, set BELOW from the filled surface pressure, PS_UNKNOWN where the
// surface pressure has no value at all. pl is sanitised here; gfs_fill_missing runs after it.
extern "C" __global__ void gfs_decode_pl(const float* __restrict__ psfc,
                                         const unsigned char* __restrict__ psfc_mask,
                                         const float* __restrict__ plev,
                                         const int* __restrict__ nonneg,
                                         float* pl,
                                         unsigned char* mask,
                                         int nvar, int nlev, int nlat, int nlon)
{
    POINT_IJL                                   /* l0 is the isobaric level */
    if (i0 >= nlon || j0 >= nlat || l0 >= nlev) return;
    const size_t pidx = IDX2(i0, j0, nlon);
    const int ps_known = (psfc_mask[pidx] & (MASK_PRESENT | MASK_FILLED)) != 0;
    const float ps = psfc[pidx];
    const int below = ps_known ? ((plev[l0] > ps) ? MASK_BELOW : 0) : 0;
    const int unknown = ps_known ? 0 : MASK_PS_UNKNOWN;
    for (int v = 0; v < nvar; ++v) {
        const size_t idx = IDX4(i0, j0, l0, v, nlon, nlat, nlev);
        const float x = pl[idx];
        const int ok = (isfinite(x) && fabsf(x) < GFS_FILL_MIN);
        float y = ok ? x : 0.0f;
        if (ok && nonneg[v] && y < 0.0f) y = 0.0f;
        pl[idx] = y;
        mask[idx] = (unsigned char)((ok ? MASK_PRESENT : 0) | below | unknown);
    }
}
