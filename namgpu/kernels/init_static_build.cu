// init_static_build.cu -- static surface fields for any rotated lat-lon B grid from WPS_GEOG tiles.
//
// NAM unit init-static-build, mode "new" (the NPS geogrid step for an NMMB B grid, on the GPU; never
// Fortran geogrid). Interpolation options follow NAM's own parm/nam_GEOGRID.TBL.NMB names:
//   gcell_*   : average_gcell / scalar categorical accumulation. Every source pixel is mapped to the
//               model H cell whose centre is nearest (nint of its fractional rotated-grid index) and
//               accumulated. Continuous sums are int64 of the raw integers and counts are int32, so
//               the result is exact and independent of thread order (no float atomics).
//   sample_*  : per-cell interpolation at the H point: four_pt, average_4pt, average_16pt,
//               sixteen_pt (WPS overlapping parabolic, oned), search (nearest valid pixel, square
//               rings), nearest (nint pixel). Codes in METHOD_* below, applied in order; each option
//               fills only the cells still missing.
// Grid: H(i,j) at rotated (wb + i*dlm, sb + j*dph), i, j zero-based, on a grid extended by `halo`
// cells each side (geogrid's processing halo, so terrain smoothing has support past the edge).
// Rotation: NMMB / grid.rotated_to_geographic (rotated equator through (tlm0, tph0)), double precision.
// Source tiles: WPS_GEOG regular_ll, packed interior-only (tile_bdr cropped on the host), one slot per
// tile, slot lookup lut[ty_index * ntx + tx_index] (-1 = tile not loaded = missing).
//
// Declared divergences (not bitwise to NPS geogrid): coordinates in double, not NPS default real;
// `search` takes the closest valid pixel of the first square ring holding one (WPS walks a breadth-
// first frontier; the two differ only inside large missing holes).

#define METHOD_FOUR_PT      1
#define METHOD_AVERAGE_4PT  2
#define METHOD_AVERAGE_16PT 3
#define METHOD_SIXTEEN_PT   4
#define METHOD_SEARCH       5
#define METHOD_NEAREST      6

struct SrcDesc {
    double known_x, known_y, known_lat, known_lon, dx, dy;   // degrees
    int nxg, nyg;          // global source size in pixels
    int tx, ty;            // tile interior size
    int ntx;               // tiles per row of the lut
    int wraps;             // 1 = 360 deg in x
    int has_missing;
    int missing;           // raw missing value
    double scale;
};

// Source descriptor from two small device arrays: sd = known_x, known_y, known_lat, known_lon, dx, dy,
// scale; si = nxg, nyg, tx, ty, ntx, wraps, has_missing, missing.
__device__ inline SrcDesc load_desc(const double* sd, const int* si)
{
    SrcDesc s;
    s.known_x = sd[0]; s.known_y = sd[1]; s.known_lat = sd[2]; s.known_lon = sd[3];
    s.dx = sd[4]; s.dy = sd[5]; s.scale = sd[6];
    s.nxg = si[0]; s.nyg = si[1]; s.tx = si[2]; s.ty = si[3]; s.ntx = si[4]; s.wraps = si[5];
    s.has_missing = si[6]; s.missing = si[7];
    return s;
}

__device__ inline void geo_to_rot(double glat, double glon, double tph0, double tlm0,
                                  double* tlm, double* tph)
{
    const double dl = glon - tlm0;
    const double cphi = cos(glat), sphi = sin(glat);
    const double x = cphi * cos(dl), y = cphi * sin(dl);
    double st = sphi * cos(tph0) - x * sin(tph0);
    st = fmin(1.0, fmax(-1.0, st));
    *tph = asin(st);
    *tlm = atan2(y, x * cos(tph0) + sphi * sin(tph0));
}

__device__ inline void rot_to_geo(double tlm, double tph, double tph0, double tlm0,
                                  double* glat, double* glon)
{
    const double stph0 = sin(tph0), ctph0 = cos(tph0);
    const double stph = sin(tph), ctph = cos(tph);
    const double ctlm = cos(tlm), stlm = sin(tlm);
    double sphi = stph0 * ctph * ctlm + ctph0 * stph;
    sphi = fmin(1.0, fmax(-1.0, sphi));
    *glat = asin(sphi);
    *glon = tlm0 + atan2(ctph * stlm, ctph0 * ctph * ctlm - stph0 * stph);
}

// Raw pixel at global 1-based (xg, yg); returns false when missing / outside / tile absent.
template <typename T>
__device__ inline bool pixel(const T* __restrict__ tiles, const int* __restrict__ lut,
                             const SrcDesc& s, int plane, int nplane, int xg, int yg, double* v)
{
    if (s.wraps) {
        xg = ((xg - 1) % s.nxg + s.nxg) % s.nxg + 1;
    } else if (xg < 1 || xg > s.nxg) {
        return false;
    }
    if (yg < 1 || yg > s.nyg) return false;
    const int ti = (xg - 1) / s.tx, tj = (yg - 1) / s.ty;
    const int slot = lut[tj * s.ntx + ti];
    if (slot < 0) return false;
    const int px = xg - 1 - ti * s.tx, py = yg - 1 - tj * s.ty;
    const size_t off = (((size_t)slot * nplane + plane) * s.ty + py) * s.tx + px;
    const int raw = (int)tiles[off];
    if (s.has_missing && raw == s.missing) return false;
    *v = (double)raw * s.scale;
    return true;
}

__device__ inline double oned(double x, double a, double b, double c, double d)
{
    return (1.0 - x) * (b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b)))
         + x * (c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c)));
}

template <typename T>
__device__ bool interp(const T* __restrict__ tiles, const int* __restrict__ lut, const SrcDesc& s,
                       int plane, int nplane, int method, int search_r, double fx, double fy, double* out)
{
    if (method == METHOD_NEAREST) {
        return pixel(tiles, lut, s, plane, nplane, (int)floor(fx + 0.5), (int)floor(fy + 0.5), out);
    }
    if (method == METHOD_FOUR_PT || method == METHOD_AVERAGE_4PT) {
        const int i0 = (int)floor(fx), j0 = (int)floor(fy);
        const double wx = fx - i0, wy = fy - j0;
        double v[4];
        bool ok[4];
        ok[0] = pixel(tiles, lut, s, plane, nplane, i0, j0, &v[0]);
        ok[1] = pixel(tiles, lut, s, plane, nplane, i0 + 1, j0, &v[1]);
        ok[2] = pixel(tiles, lut, s, plane, nplane, i0, j0 + 1, &v[2]);
        ok[3] = pixel(tiles, lut, s, plane, nplane, i0 + 1, j0 + 1, &v[3]);
        if (method == METHOD_FOUR_PT) {
            if (!(ok[0] && ok[1] && ok[2] && ok[3])) return false;
            *out = (1.0 - wy) * ((1.0 - wx) * v[0] + wx * v[1]) + wy * ((1.0 - wx) * v[2] + wx * v[3]);
            return true;
        }
        double sum = 0.0;
        int n = 0;
        for (int k = 0; k < 4; ++k) if (ok[k]) { sum += v[k]; ++n; }
        if (n == 0) return false;
        *out = sum / n;
        return true;
    }
    if (method == METHOD_AVERAGE_16PT || method == METHOD_SIXTEEN_PT) {
        const int i0 = (int)floor(fx), j0 = (int)floor(fy);
        double v[4][4];
        bool all = true;
        double sum = 0.0;
        int n = 0;
        for (int l = 0; l < 4; ++l) {
            for (int k = 0; k < 4; ++k) {
                double t;
                if (pixel(tiles, lut, s, plane, nplane, i0 + k - 1, j0 + l - 1, &t)) {
                    v[l][k] = t; sum += t; ++n;
                } else {
                    all = false; v[l][k] = 0.0;
                }
            }
        }
        if (method == METHOD_AVERAGE_16PT) {
            if (n == 0) return false;
            *out = sum / n;
            return true;
        }
        if (!all) return false;
        const double x = fx - i0, y = fy - j0;
        double r[4];
        for (int l = 0; l < 4; ++l) r[l] = oned(x, v[l][0], v[l][1], v[l][2], v[l][3]);
        *out = oned(y, r[0], r[1], r[2], r[3]);
        return true;
    }
    if (method == METHOD_SEARCH) {
        const int ic = (int)floor(fx + 0.5), jc = (int)floor(fy + 0.5);
        double t;
        if (pixel(tiles, lut, s, plane, nplane, ic, jc, &t)) { *out = t; return true; }
        for (int r = 1; r <= search_r; ++r) {
            double best = 1e300, bv = 0.0;
            for (int dj = -r; dj <= r; ++dj) {
                const int step = (dj == -r || dj == r) ? 1 : 2 * r;
                for (int di = -r; di <= r; di += step) {
                    if (pixel(tiles, lut, s, plane, nplane, ic + di, jc + dj, &t)) {
                        const double ex = (ic + di) - fx, ey = (jc + dj) - fy;
                        const double d2 = ex * ex + ey * ey;
                        if (d2 < best) { best = d2; bv = t; }
                    }
                }
            }
            if (best < 1e299) { *out = bv; return true; }
        }
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Grid-cell accumulation, one thread per source pixel of the loaded tiles.
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void gcell_body(const T* __restrict__ tiles, const int* __restrict__ slot_xs,
                           const int* __restrict__ slot_ys, int nslot, int plane, int nplane,
                           SrcDesc s, double tph0, double tlm0, double wb, double sb, double dlm,
                           double dph, int halo, int ioff, int joff, int ie, int je, int ncat,
                           long long* __restrict__ sums, int* __restrict__ counts)
{
    const size_t npx = (size_t)s.tx * s.ty;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= npx * nslot) return;
    const int slot = (int)(gid / npx);
    const int p = (int)(gid % npx);
    const int px = p % s.tx, py = p / s.tx;
    const size_t off = (((size_t)slot * nplane + plane) * s.ty + py) * s.tx + px;
    const int raw = (int)tiles[off];
    if (s.has_missing && raw == s.missing) return;
    if (ncat > 0 && (raw < 1 || raw > ncat)) return;
    const int xg = slot_xs[slot] + px, yg = slot_ys[slot] + py;   // 1-based global
    const double r = 0.017453292519943295;
    const double glat = (s.known_lat + (yg - s.known_y) * s.dy) * r;
    const double glon = (s.known_lon + (xg - s.known_x) * s.dx) * r;
    double tlm, tph;
    geo_to_rot(glat, glon, tph0, tlm0, &tlm, &tph);
    // wb/sb: the reference grid's origin; ioff/joff: the moving-nest window's whole-cell offset (0 otherwise)
    const int i = (int)floor((tlm - wb) / dlm + 0.5) - ioff + halo;
    const int j = (int)floor((tph - sb) / dph + 0.5) - joff + halo;
    if (i < 0 || i >= ie || j < 0 || j >= je) return;
    const size_t cell = (size_t)j * ie + i;
    if (ncat > 0) {
        atomicAdd(&counts[(size_t)(raw - 1) * ie * je + cell], 1);
    } else {
        atomicAdd((unsigned long long*)&sums[cell], (unsigned long long)(long long)raw);
        atomicAdd(&counts[cell], 1);
    }
}

// ---------------------------------------------------------------------------------------------
// Per-cell interpolation, one thread per extended cell; fills out[cell] where need[cell] != 0.
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void sample_body(const T* __restrict__ tiles, const int* __restrict__ lut, int plane,
                            int nplane, SrcDesc s, double tph0, double tlm0, double wb, double sb,
                            double dlm, double dph, int halo, int ioff, int joff, int ie, int je,
                            const int* __restrict__ methods, int nmethod, int search_r,
                            const unsigned char* __restrict__ need, double* __restrict__ out,
                            unsigned char* __restrict__ got)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ie || j >= je) return;
    const size_t cell = (size_t)j * ie + i;
    got[cell] = 0;
    if (!need[cell]) return;
    double glat, glon;
    rot_to_geo(wb + (i - halo + ioff) * dlm, sb + (j - halo + joff) * dph, tph0, tlm0, &glat, &glon);
    const double d = 57.29577951308232;
    const double lat = glat * d;
    double dlon = glon * d - s.known_lon;
    dlon = fmod(fmod(dlon, 360.0) + 360.0, 360.0);
    if (!s.wraps && dlon > 180.0) dlon -= 360.0;
    double fx = s.known_x + dlon / s.dx;
    if (s.wraps && fx >= s.nxg + 0.5) fx -= s.nxg;
    const double fy = s.known_y + (lat - s.known_lat) / s.dy;
    for (int m = 0; m < nmethod; ++m) {
        double v;
        if (interp(tiles, lut, s, plane, nplane, methods[m], search_r, fx, fy, &v)) {
            out[cell] = v;
            got[cell] = 1;
            return;
        }
    }
}

#define GCELL_KERNEL(NAME, T)                                                                      \
extern "C" __global__ void NAME(const T* tiles, const int* slot_xs, const int* slot_ys, int nslot,  \
                                int plane, int nplane, const double* sd, const int* si,             \
                                double tph0, double tlm0,                                          \
                                double wb, double sb, double dlm, double dph, int halo, int ioff, int joff, int ie,     \
                                int je, int ncat, long long* sums, int* counts)                    \
{                                                                                                  \
    gcell_body<T>(tiles, slot_xs, slot_ys, nslot, plane, nplane, load_desc(sd, si), tph0, tlm0, wb, sb, dlm, dph, \
                  halo, ioff, joff, ie, je, ncat, sums, counts);                                               \
}

#define SAMPLE_KERNEL(NAME, T)                                                                     \
extern "C" __global__ void NAME(const T* tiles, const int* lut, int plane, int nplane,              \
                                const double* sd, const int* si,                                   \
                                double tph0, double tlm0, double wb, double sb, double dlm,        \
                                double dph, int halo, int ioff, int joff, int ie, int je, const int* methods,          \
                                int nmethod, int search_r, const unsigned char* need, double* out, \
                                unsigned char* got)                                                \
{                                                                                                  \
    sample_body<T>(tiles, lut, plane, nplane, load_desc(sd, si), tph0, tlm0, wb, sb, dlm, dph, halo, ioff, joff, ie, je,      \
                   methods, nmethod, search_r, need, out, got);                                    \
}

GCELL_KERNEL(gcell_u8, unsigned char)
GCELL_KERNEL(gcell_i16, short)
GCELL_KERNEL(gcell_u16, unsigned short)
SAMPLE_KERNEL(sample_u8, unsigned char)
SAMPLE_KERNEL(sample_i16, short)
SAMPLE_KERNEL(sample_u16, unsigned short)
