// WOOF-NAM unit init-gfs-hinterp: horizontal interpolation of GFS fields onto the B-grid boundary
// strips (lnsh = lnsv = 5 rows/columns, H and V points), the new-unit replacement for the
// WPS/metgrid step that NAM's boundary input uses.
//
// Rule source: NCEP nam.METGRID.TBL.bocos, interp_option=four_pt+average_4pt for PSFC, RH, TT, UU,
// VV, GHT (lines 137-138, 400-402, 414-416, 422-425, 432-434, 490-491). METGRID semantics as
// implemented here:
//   four_pt:      bilinear on the four nearest source points. If ANY of the four is missing, the
//                 four_pt result is missing and the next option in the list is tried.
//   average_4pt:  the arithmetic mean of the PRESENT points among the four nearest source points.
//                 If none is present, the point is missing.
//   missing:      the output is 0.0 and counted in nmiss (fill_missing=0. in the table for these
//                 fields). The kernel never syncs the host; the caller reads the count.
// A source point is valid when its mask has bit 1 (PRESENT, from the file) or bit 4 (FILLED, the
// neighbour fill of init-gfs-decode). Mask bit 2 (BELOW) does not make a point missing.
//
// NOTICE: no NOAA Fortran routine is ported here (the unit has no Fortran line range). The
// interpolation rule comes from nam.METGRID.TBL.bocos (NCEP NAM boundary tables, US government work,
// no licence file in the NAM clone). Nothing is copied from WRF/WPS code.
//
// Mapping of the rule to the kernel:
//   fx = (lon - lon0) mod 360 / dlon, fy = (lat - lat0) / dlat      fractional GFS index
//   i0 = floor(fx), j0 = floor(fy), wx = fx - i0, wy = fy - j0      four nearest GFS points
//   v  = (1-wx)(1-wy) v(i0,j0) + wx(1-wy) v(i0+1,j0) + (1-wx)wy v(i0,j0+1) + wx wy v(i0+1,j0+1)
//   i0+1 wraps modulo NIS (global GFS, longitude periodic); j0 clamped to [0, NJS-2] (poles and
//   the strip edge reuse the edge row). The source rows are south to north (lat0 + j*dlat).
//   lon_periodic: 0 = plain non-periodic grid (clamped), 1 = global periodic, 2 = a longitude window
//   cut from a global grid (offset from lon0 taken mod 360, then clamped like 0).
//
// Source layout: src and mask are [NSLAB][NJS][NIS], NSLAB = NT * NL, slab = t * NL + l, where t is
// the boundary time level (NT = 2: read_bc reads n3 = 1,2 for 2-D and n4 = 1,2 for 3-D fields) and l
// is the source level (NL = 1 for 2-D fields, the isobaric count for 3-D GFS fields).
//
// Strip layout (CONTRACT.md sections 1 and 4; slab-major, the same slab index as the source):
//   south  out_s[slab][r][i]   r = 0..W-1  -> j = r          (Fortran j = 1..lnsh)
//   north  out_n[slab][r][i]   r = 0..W-1  -> j = JM-W+r     (Fortran j = JM-lnsh+1..JM)
//   west   out_w[slab][j][c]   c = 0..W-1  -> i = c          (Fortran i = 1..lnsv)
//   east   out_e[slab][j][c]   c = 0..W-1  -> i = IM-W+c     (Fortran i = IM-lnsv+1..IM)
// glat/glon are the full H-point or V-point geographic coordinates [JM][IM] (init-grid-geo, GLAT/GLON
// of module_CONTROL); the strip points read the same (i,j) coordinates as the full grid.
//
// Grid: blockIdx.x over the points of one side (ceil(max(W*IM, JM*W)/256)), blockIdx.y = side
// (0 south, 1 north, 2 west, 3 east), blockIdx.z = slab. One thread per strip point; no host sync,
// no allocation.

#include "nmmb_common.cuh"

#define HINTERP_R2D (180.0f / 3.14159265358979323846f)   // init-grid-geo stores radians
#define HINTERP_MASK_PRESENT 1
#define HINTERP_MASK_FILLED  4

// Returns the interpolated value at (lat, lon) in degrees, or 0.0 and increments *nmiss when the
// point is missing under the METGRID rule above.
__device__ inline float hinterp_point(const float* __restrict__ sl, const unsigned char* __restrict__ mk,
                                      int NIS, int NJS, int lon_periodic,
                                      float lat, float lon,
                                      float lat0, float lon0, float dlat, float dlon,
                                      int* __restrict__ nmiss)
{
    float fx = (lon - lon0) / dlon;
    if (lon_periodic == 2) {
        // Longitude window (a source cropped to the columns around the domain, init-gfs-start
        // decode_source): the offset east of the window start is taken modulo 360 so a domain across
        // 0/360 or the dateline lands inside the window; then the non-periodic clamp below.
        float d = lon - lon0;
        d = d - 360.0f * floorf(d / 360.0f);           // into [0, 360)
        fx = d / dlon;
    } else if (lon_periodic) {
        const float nper = 360.0f / dlon;               // source points around the globe
        fx = fx - nper * floorf(fx / nper);             // into [0, nper)
    }
    float fy = (lat - lat0) / dlat;
    fy = fminf(fmaxf(fy, 0.0f), (float)(NJS - 1));     // clamp to the source grid

    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    if (j0 > NJS - 2) j0 = NJS - 2;
    if (j0 < 0) j0 = 0;
    const float wy = fy - (float)j0;
    int i1;
    float wx;
    if (lon_periodic == 1) {
        i0 = i0 % NIS; if (i0 < 0) i0 += NIS;
        i1 = (i0 + 1) % NIS;
        wx = fx - floorf(fx);
    } else {
        fx = fminf(fmaxf(fx, 0.0f), (float)(NIS - 1));
        i0 = (int)floorf(fx);
        if (i0 > NIS - 2) i0 = NIS - 2;
        i1 = i0 + 1;
        wx = fminf(fmaxf(fx - (float)i0, 0.0f), 1.0f);
    }
    const int j1 = j0 + 1;
    const size_t a = IDX2(i0, j0, NIS), b = IDX2(i1, j0, NIS);
    const size_t c = IDX2(i0, j1, NIS), d = IDX2(i1, j1, NIS);
    const int oa = (mk[a] & (HINTERP_MASK_PRESENT | HINTERP_MASK_FILLED)) != 0;
    const int ob = (mk[b] & (HINTERP_MASK_PRESENT | HINTERP_MASK_FILLED)) != 0;
    const int oc = (mk[c] & (HINTERP_MASK_PRESENT | HINTERP_MASK_FILLED)) != 0;
    const int od = (mk[d] & (HINTERP_MASK_PRESENT | HINTERP_MASK_FILLED)) != 0;

    if (oa && ob && oc && od) {                       // four_pt: all four present
        const float v00 = sl[a], v10 = sl[b], v01 = sl[c], v11 = sl[d];
        return (1.0f - wx) * (1.0f - wy) * v00 + wx * (1.0f - wy) * v10
             + (1.0f - wx) * wy * v01 + wx * wy * v11;
    }
    // four_pt is missing (a neighbour is missing): average_4pt over the present neighbours.
    float s = 0.0f;
    int n = 0;
    if (oa) { s += sl[a]; ++n; }
    if (ob) { s += sl[b]; ++n; }
    if (oc) { s += sl[c]; ++n; }
    if (od) { s += sl[d]; ++n; }
    if (n > 0) return s / (float)n;
    atomicAdd(nmiss, 1);                              // no present neighbour: missing
    return 0.0f;
}

extern "C" __global__ void init_gfs_hinterp_strips(
    const float* __restrict__ src,           // [NSLAB][NJS][NIS] GFS field, float32, C order
    const unsigned char* __restrict__ mask,  // [NSLAB][NJS][NIS] source masks (decode bits)
    const float* __restrict__ glat,          // [JM][IM] target latitude (H or V points), radians
    const float* __restrict__ glon,          // [JM][IM] target longitude, radians
    float* __restrict__ out_s,               // [NSLAB][W][IM]
    float* __restrict__ out_n,               // [NSLAB][W][IM]
    float* __restrict__ out_w,               // [NSLAB][JM][W]
    float* __restrict__ out_e,               // [NSLAB][JM][W]
    int* __restrict__ nmiss,                 // [1] count of missing strip points (atomic)
    int IM, int JM, int W, int NSLAB,
    int NIS, int NJS, int lon_periodic,
    float lat0, float lon0, float dlat, float dlon)
{
    const int side = blockIdx.y;
    const int slab = blockIdx.z;
    if (slab >= NSLAB) return;
    const size_t off = (size_t)slab * NJS * NIS;
    const float* sl = src + off;
    const unsigned char* mk = mask + off;

    if (side < 2) {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= W * IM) return;
        const int r = p / IM;
        const int i0 = p % IM;
        const int j0 = (side == 0) ? r : (JM - W + r);
        const size_t k = IDX2(i0, j0, IM);
        const float v = hinterp_point(sl, mk, NIS, NJS, lon_periodic,
                                      glat[k] * HINTERP_R2D, glon[k] * HINTERP_R2D,
                                      lat0, lon0, dlat, dlon, nmiss);
        float* dst = (side == 0) ? out_s : out_n;
        dst[((size_t)slab * W + r) * IM + i0] = v;
    } else {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= JM * W) return;
        const int j0 = p / W;
        const int c = p % W;
        const int i0 = (side == 2) ? c : (IM - W + c);
        const size_t k = IDX2(i0, j0, IM);
        const float v = hinterp_point(sl, mk, NIS, NJS, lon_periodic,
                                      glat[k] * HINTERP_R2D, glon[k] * HINTERP_R2D,
                                      lat0, lon0, dlat, dlon, nmiss);
        float* dst = (side == 2) ? out_w : out_e;
        dst[((size_t)slab * JM + j0) * W + c] = v;
    }
}

// Full-domain variant for init-gfs-vinterp: one thread per (i, j, level) of the target H or V grid.
// Same METGRID rule as init_gfs_hinterp_strips (hinterp_point). src and mask are one variable
// [NLEV][NJS][NIS] for one boundary time; out is [NLEV][JM][IM]. Missing points are 0.0 and counted in nmiss.
extern "C" __global__ void init_gfs_hinterp_full(
    const float* __restrict__ src,
    const unsigned char* __restrict__ mask,
    const float* __restrict__ glat,          // [JM][IM] target latitude, radians
    const float* __restrict__ glon,          // [JM][IM] target longitude, radians
    float* __restrict__ out,                 // [NLEV][JM][IM]
    int IM, int JM, int NLEV, int NIS, int NJS, int lon_periodic,
    float lat0, float lon0, float dlat, float dlon,
    int* __restrict__ nmiss)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z;
    if (i >= IM || j >= JM || l >= NLEV) return;
    const size_t off = (size_t)l * NJS * NIS;
    const size_t k = IDX2(i, j, IM);
    out[((size_t)l * JM + j) * IM + i] = hinterp_point(src + off, mask + off, NIS, NJS, lon_periodic,
                                                       glat[k] * HINTERP_R2D, glon[k] * HINTERP_R2D,
                                                       lat0, lon0, dlat, dlon, nmiss);
}
