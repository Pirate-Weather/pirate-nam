// init_gwd_orog.cu -- gravity-wave-drag orographic statistics on the B grid from 30-arcsec terrain.
//
// NOTICE: NAM source file nam_nps.fd/sorc/NMMB_init/NPS/geogrid/src/gridgen_sfc/lsmask_orog.f
// (NOAA/NCEP/EMC, US Government work, no licence file). Kept as notice:
//   module lsmask_orog  (NPS geogrid, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nam_nps.fd/sorc/NMMB_init/NPS/geogrid/src/gridgen_sfc/lsmask_orog.f
//   calc_orog 1236-1973, find_bounds_ll (find_bounds_ll.f) and ll2xy_bgrid_pt_loc 2768-2835
//   (nam.v4.2.11, 83df4e13c1), reached through process_ncep_module.F90 GWD block (do_gwd,
//   ncep_proc_domain_type = bgrid, namelist.nps_input:33).
//
// Output names (fields of the NPS GWD set, output_nemsinit.F90:262-278):
//   HSTDV=orog_stnd_dev  HCNVX=convexity  HASYW/HASYS/HASYSW/HASYNW=oa(1..4)
//   HLENW/HLENS/HLENSW/HLENNW=ol(1..4)  HANGL=theta  HANIS=gamma  HSLOP=sigma  HZMAX=max_height
//
// Statements reproduced (Fortran line -> kernel):
//   source grid constants dlon_src, dlat_src, lon_11_src, lat_11_src (1240-1243)
//       -> GWD_DLON_SRC, GWD_DLAT_SRC, GWD_LON11_SRC, GWD_LAT11_SRC
//   find_bounds_ll (1327, find_bounds_ll.f): the source i/j window the bgrid passes loop over.
//       The window is computed on the host by the wrapper (init_gwd_orog.find_bounds) and the
//       kernels launch only over it; i is wrapped once around the globe as iii (1524-1530).
//   dy_src, dx_src(j) (1297-1307)                                  -> gwd_dy_src, gwd_dx_row
//   bgrid source->model mapping, ll2xy_bgrid_pt_loc (2768-2835, nint = roundf)  -> gwd_bgrid_map
//   ll2xy origin wbd/sbd, first-call block of ll2xy_bgrid_pt_loc (2785-2803)    -> gwd_bgrid_origin
//   max_height init and bgrid update (1396-1397, bgrid branch 1534)  -> gwd_src_accum, gwd_model_topo
//   count, sum_topo (non-ocean), topo = sum/count, max_height=max(max_height-topo,0)
//                                                                  -> gwd_src_accum, gwd_model_topo
//   dhdx/dhdy neighbour derivatives, one-sided at the global source rows (1571-1609)
//                                                                  -> gwd_src_accum
//   xfp2/yfp2/xfpyfp, hk, hl, hlprim, theta, sigma2, sigma, gamma (1617-1641)
//                                                                  -> gwd_model_topo
//   sum_topo / sum_topo4 of (topo-elev)^2 and ^4, ocean counts as topo^2, topo^4 (1647-1667)
//                                                                  -> gwd_src_topo_var
//   orog_stnd_dev and convexity (1670-1679)                        -> gwd_model_std_conv
//   nul/nll/nur/nlr and nul2/nll2/nur2/nlr2 with crit_hgt = 1111.6 - 0.878*stdv (1685-1738)
//                                                                  -> gwd_src_crit
//   oa(1..4), ol(1..4) (1743-1790)                                  -> gwd_model_oaol
// Not reproduced (not in the GWD output set): the orography tiling (TILING block, 1790-1972:
// num_orog_tiles, orog_tiles_elev/prcnt, lsmask-based bins), the lsmask and water remove passes,
// the MPI gather/scatter, the gaussian/latlon/polar/egrid mapping branches (ncep_proc_domain_type
// is bgrid for NAM), and the topo_src deallocation bookkeeping.
//
// Declared divergences (each has a named breakage and is counted in the run, not silent):
//   - ol(*) with a zero quadrant-count denominator is written as 0; Fortran (1754-1786) writes 0/0 = NaN,
//     which would propagate into the GWD tendencies. The wrapper prints how many cells hit this.
//   - the derivative stencil reads the full global DEM (neighbours outside the window are real
//     terrain); the Fortran reads its jstart:jend array, so only the window's edge rows differ.
//   - the asin arguments of the mapping are not clamped: an argument outside [-1,1] gives NaN and the
//     source cell is skipped (Fortran nint of NaN is undefined). The origin argument is checked on the
//     host and refused if it leaves [-1,1].
//
// Float atomics (sums, max) are non-deterministic in order; the result is fast-numerics, not bitwise.
#include "nmmb_common.cuh"

#define GWD_PI        3.14159265f
#define GWD_DTR       (GWD_PI / 180.0f)
#define GWD_RE        6.371e6f
#define GWD_WATER     ((short)-9999)
#define GWD_DLON_SRC  (1.0f / 120.0f)
#define GWD_DLAT_SRC  (-(1.0f / 120.0f))
#define GWD_LON11_SRC (-180.0f + (GWD_DLON_SRC * 0.5f))
#define GWD_LAT11_SRC (90.0f + (GWD_DLAT_SRC * 0.5f))

// wbd, sbd of ll2xy_bgrid_pt_loc (first-call block): wbd = -atan2(anum, denom), sbd = asin(...)
// evaluated at the SW corner (lat11r = sw[0], lon11r = sw[1], radians). Single thread.
// origin[0] = wbd (rad), origin[1] = sbd (rad, unclamped as the Fortran ASIN), origin[2] = the raw
// asin argument, so the host can refuse an argument outside [-1,1] (NaN sbd).
extern "C" __global__ void gwd_bgrid_origin(const float* __restrict__ sw, float tlm0, float stph0,
                                            float ctph0, float* __restrict__ origin)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float lat11r = sw[0];
    const float lon11r = sw[1];
    const float relmi = tlm0 - lon11r;
    const float srlmi = sinf(relmi);
    const float crlmi = cosf(relmi);
    const float sph = sinf(lat11r);
    const float cph = cosf(lat11r);
    const float cc = cph * crlmi;
    const float anum = cph * srlmi;
    const float denom = ctph0 * cc + stph0 * sph;
    const float arg = ctph0 * sph - stph0 * cc;
    origin[0] = -atan2f(anum, denom);
    origin[1] = asinf(arg);
    origin[2] = arg;
}

// Source-cell centre in degrees, 0-based (i0, j0) (Fortran lon_src, lat_src).
__device__ __forceinline__ void gwd_src_latlon(int i0, int j0, float& lat, float& lon)
{
    lat = GWD_LAT11_SRC + (float)j0 * GWD_DLAT_SRC;
    lon = GWD_LON11_SRC + (float)i0 * GWD_DLON_SRC;
}

// Window cell (x, y) -> source cell (i0, j0), 0-based. Fortran i runs i_lo .. i_lo+ni-1 (1-based,
// may be < 1 or > isrc near the dateline) and is wrapped once around the globe (iii, 1524-1530).
__device__ __forceinline__ void gwd_src_cell(int x, int y, int isrc, int i_lo, int j_lo,
                                             int& i0, int& j0)
{
    const int ii = i_lo - 1 + x;
    i0 = ((ii % isrc) + isrc) % isrc;
    j0 = j_lo - 1 + y;
}

// ll2xy_bgrid_pt_loc (2768-2835): source point (deg) -> nearest model H cell (0-based).
__device__ __forceinline__ bool gwd_bgrid_map(float lat, float lon, float stph0, float ctph0,
                                              float tlm0, float dph, float dlm, float wbd,
                                              float sbd, int im, int jm, int& ii0, int& jj0)
{
    const float latr = lat * GWD_DTR;
    const float lonr = lon * GWD_DTR;
    const float relmi = tlm0 - lonr;
    const float srlmi = sinf(relmi);
    const float crlmi = cosf(relmi);
    const float sph = sinf(latr);
    const float cph = cosf(latr);
    const float cc = cph * crlmi;
    const float anum = cph * srlmi;
    const float denom = ctph0 * cc + stph0 * sph;
    const float tlon = -atan2f(anum, denom);
    const float tlat = asinf(ctph0 * sph - stph0 * cc);
    if (!isfinite(tlat)) return false;          // Fortran asin NaN: cell skipped (declared)
    const float y = ((tlat - sbd) / dph) + 1.0f;
    const float x = ((tlon - wbd) / dlm) + 1.0f;
    const int ii = (int)roundf(x);   // Fortran NINT (half away from zero)
    const int jj = (int)roundf(y);
    if (ii < 1 || ii > im || jj < 1 || jj > jm) return false;
    ii0 = ii - 1;
    jj0 = jj - 1;
    return true;
}

// dy_src (1297-1300): pi*rearth/jsrc * sign(1, dlat_src) * 2, with dlat_src < 0.
__device__ __forceinline__ float gwd_dy_src(int jsrc)
{
    float dy = (GWD_PI * GWD_RE) / (float)jsrc;
    dy = dy * (-1.0f) * 2.0f;
    return dy;
}

// dx_src(j) (1302-1307): 2*(2*pi*rearth/isrc)*cos(lat) for the source row j0.
__device__ __forceinline__ float gwd_dx_row(int isrc, int j0)
{
    float lat = GWD_LAT11_SRC + (float)j0 * GWD_DLAT_SRC;
    lat = lat * GWD_PI / 180.0f;
    return 2.0f * (2.0f * GWD_PI * GWD_RE / (float)isrc) * cosf(lat);
}

// Ocean source cells contribute topo 0 to the derivative stencil (1575-1600).
__device__ __forceinline__ float gwd_topo_or_zero(short v)
{
    return (v == GWD_WATER) ? 0.0f : (float)v;
}

// Order-free float atomicMax via the IEEE sign-bit trick.
__device__ __forceinline__ void gwd_atomic_max_f(float* addr, float v)
{
    if (v >= 0.0f) {
        atomicMax((int*)addr, __float_as_int(v));
    } else {
        atomicMin((unsigned int*)addr, __float_as_uint(v));
    }
}

// Pass 1 over the find_bounds_ll window of the 30-arcsec source grid: count, sum_topo (non-ocean),
// max_height (bgrid branch, 1534), and the squared and cross derivative sums (1571-1609).
// One thread per window cell (x, y); the window is (i_lo..i_lo+ni-1) x (j_lo..j_lo+nj-1).
extern "C" __global__ void gwd_src_accum(
    const short* __restrict__ dem, const float* __restrict__ origin,
    int isrc, int jsrc, int i_lo, int j_lo, int ni, int nj, int im, int jm,
    float stph0, float ctph0, float tlm0, float dph, float dlm,
    int* __restrict__ count, float* __restrict__ sum_topo, float* __restrict__ max_h,
    float* __restrict__ sdx2, float* __restrict__ sdy2, float* __restrict__ sdxy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ni || y >= nj) return;
    int i0, j0;
    gwd_src_cell(x, y, isrc, i_lo, j_lo, i0, j0);

    float lat, lon;
    gwd_src_latlon(i0, j0, lat, lon);
    int ii0, jj0;
    if (!gwd_bgrid_map(lat, lon, stph0, ctph0, tlm0, dph, dlm, origin[0], origin[1],
                       im, jm, ii0, jj0)) return;
    const size_t idx = IDX2(ii0, jj0, im);

    const short elev = dem[IDX2(i0, j0, isrc)];
    atomicAdd(&count[idx], 1);
    if (elev != GWD_WATER) {
        atomicAdd(&sum_topo[idx], (float)elev);
        gwd_atomic_max_f(&max_h[idx], (float)elev);
    }

    // derivative stencil: i wraps around the globe; one-sided rows at the first and last source row
    const float dx = gwd_dx_row(isrc, j0);
    float dy = gwd_dy_src(jsrc);
    const int im1 = (i0 - 1 + isrc) % isrc;
    const int ip1 = (i0 + 1) % isrc;
    int jm1 = j0 - 1;
    if (jm1 < 0) { jm1 = 0; dy = gwd_dy_src(jsrc) * 0.5f; }
    int jp1 = j0 + 1;
    if (jp1 > jsrc - 1) { jp1 = jsrc - 1; dy = gwd_dy_src(jsrc) * 0.5f; }

    const float topo_im1 = gwd_topo_or_zero(dem[IDX2(im1, j0, isrc)]);
    const float topo_ip1 = gwd_topo_or_zero(dem[IDX2(ip1, j0, isrc)]);
    const float topo_jm1 = gwd_topo_or_zero(dem[IDX2(i0, jm1, isrc)]);
    const float topo_jp1 = gwd_topo_or_zero(dem[IDX2(i0, jp1, isrc)]);
    const float dhdx = (topo_ip1 - topo_im1) / dx;
    const float dhdy = (topo_jp1 - topo_jm1) / dy;
    atomicAdd(&sdx2[idx], dhdx * dhdx);
    atomicAdd(&sdy2[idx], dhdy * dhdy);
    atomicAdd(&sdxy[idx], dhdx * dhdy);
}

// Per model cell after pass 1: topo (mean), max_height (1557), derivative means and the
// anisotropy/angle/slope terms (1617-1641). count is the total source count of the cell.
extern "C" __global__ void gwd_model_topo(
    int im, int jm,
    const int* __restrict__ count, const float* __restrict__ sum_topo,
    const float* __restrict__ sdx2, const float* __restrict__ sdy2, const float* __restrict__ sdxy,
    float* __restrict__ topo, float* __restrict__ max_h,
    float* __restrict__ theta, float* __restrict__ gamma, float* __restrict__ sigma)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t p = IDX2(i0, j0, im);
    const float n = (float)count[p];
    const float t = sum_topo[p] / n;
    topo[p] = t;
    max_h[p] = fmaxf(max_h[p] - t, 0.0f);

    const float xfp2 = sdx2[p] / n;
    const float yfp2 = sdy2[p] / n;
    const float xfpyfp = sdxy[p] / n;
    const float hk = 0.5f * (xfp2 + yfp2);
    const float hl = 0.5f * (xfp2 - yfp2);
    const float hlprim = sqrtf(hl * hl + xfpyfp * xfpyfp);
    theta[p] = 0.5f * atan2f(xfpyfp, hl);
    const float sigma2 = hk + hlprim;
    sigma[p] = sqrtf(sigma2);
    gamma[p] = 0.0f;
    if (sigma2 != 0.0f && hk >= hlprim) {
        gamma[p] = sqrtf((hk - hlprim) / sigma2);
    }
}

// Pass 2 over the window: sum of (topo - elev)^2 and ^4 per model cell (1647-1667).
// Ocean source cells use topo^2 and topo^4 (no elevation term).
extern "C" __global__ void gwd_src_topo_var(
    const short* __restrict__ dem, const float* __restrict__ origin,
    int isrc, int jsrc, int i_lo, int j_lo, int ni, int nj, int im, int jm,
    float stph0, float ctph0, float tlm0, float dph, float dlm,
    const float* __restrict__ topo,
    float* __restrict__ sum2, float* __restrict__ sum4)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ni || y >= nj) return;
    int i0, j0;
    gwd_src_cell(x, y, isrc, i_lo, j_lo, i0, j0);

    float lat, lon;
    gwd_src_latlon(i0, j0, lat, lon);
    int ii0, jj0;
    if (!gwd_bgrid_map(lat, lon, stph0, ctph0, tlm0, dph, dlm, origin[0], origin[1],
                       im, jm, ii0, jj0)) return;
    const size_t idx = IDX2(ii0, jj0, im);

    const short elev = dem[IDX2(i0, j0, isrc)];
    const float tp = topo[idx];
    const float d = (elev != GWD_WATER) ? (tp - (float)elev) : tp;
    const float d2 = d * d;
    atomicAdd(&sum2[idx], d2);
    atomicAdd(&sum4[idx], d2 * d2);
}

// Per model cell: orog_stnd_dev = sqrt(sum2/count), convexity (1670-1679).
extern "C" __global__ void gwd_model_std_conv(
    int im, int jm, const int* __restrict__ count,
    const float* __restrict__ sum2, const float* __restrict__ sum4,
    float* __restrict__ stdv, float* __restrict__ conv)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t p = IDX2(i0, j0, im);
    const float n = (float)count[p];
    const float s = sqrtf(sum2[p] / n);
    stdv[p] = s;
    conv[p] = 0.0f;
    if (s > 1.0f) {
        const float s2 = s * s;
        conv[p] = (sum4[p] / n) / (s2 * s2);
    }
}

// Pass 3 over the window: quadrant counts of source cells above the model mean topo (nul,
// nll, nur, nlr) and above the critical height crit = 1111.6 - 0.878*stdv (nul2, ...) (1685-1738),
// with the quadrant set by source latitude against lat_mdl and by deltalon > 180.
// Water cells use the raw -9999 value, as the Fortran does (no water test in this pass).
extern "C" __global__ void gwd_src_crit(
    const short* __restrict__ dem, const float* __restrict__ origin,
    int isrc, int jsrc, int i_lo, int j_lo, int ni, int nj, int im, int jm,
    float stph0, float ctph0, float tlm0, float dph, float dlm,
    const float* __restrict__ glat, const float* __restrict__ glon,
    const float* __restrict__ topo, const float* __restrict__ stdv,
    int* __restrict__ nul, int* __restrict__ nll, int* __restrict__ nur, int* __restrict__ nlr,
    int* __restrict__ nul2, int* __restrict__ nll2, int* __restrict__ nur2, int* __restrict__ nlr2,
    int* __restrict__ cnul, int* __restrict__ cnll, int* __restrict__ cnur, int* __restrict__ cnlr)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ni || y >= nj) return;
    int i0, j0;
    gwd_src_cell(x, y, isrc, i_lo, j_lo, i0, j0);

    float lat, lon;
    gwd_src_latlon(i0, j0, lat, lon);
    int ii0, jj0;
    if (!gwd_bgrid_map(lat, lon, stph0, ctph0, tlm0, dph, dlm, origin[0], origin[1],
                       im, jm, ii0, jj0)) return;
    const size_t p = IDX2(ii0, jj0, im);

    const float crit = 1111.6f - 0.878f * stdv[p];
    const float latm = glat[p] * (180.0f / GWD_PI);   // lat_mdl, degrees
    const float lonm = glon[p] * (180.0f / GWD_PI);   // lon_mdl, degrees
    const float deltalon = fmodf((lon - lonm) + 3600.0f, 360.0f);
    const float t = (float)dem[IDX2(i0, j0, isrc)];
    const float tm = topo[p];

    if (lat >= latm) {
        if (deltalon > 180.0f) {
            if (t > tm) atomicAdd(&nul[p], 1);
            if (t > crit) atomicAdd(&nul2[p], 1);
            atomicAdd(&cnul[p], 1);
        } else {
            if (t > tm) atomicAdd(&nur[p], 1);
            if (t > crit) atomicAdd(&nur2[p], 1);
            atomicAdd(&cnur[p], 1);
        }
    } else {
        if (deltalon > 180.0f) {
            if (t > tm) atomicAdd(&nll[p], 1);
            if (t > crit) atomicAdd(&nll2[p], 1);
            atomicAdd(&cnll[p], 1);
        } else {
            if (t > tm) atomicAdd(&nlr[p], 1);
            if (t > crit) atomicAdd(&nlr2[p], 1);
            atomicAdd(&cnlr[p], 1);
        }
    }
}

// Asymmetry oa(1..4) and length ol(1..4) per model cell (1743-1790). ol with zero count -> 0.
__device__ __forceinline__ float gwd_sign_min1(float t)
{
    return copysignf(fminf(fabsf(t), 1.0f), t);
}

extern "C" __global__ void gwd_model_oaol(
    int im, int jm,
    const int* __restrict__ nul, const int* __restrict__ nll,
    const int* __restrict__ nur, const int* __restrict__ nlr,
    const int* __restrict__ nul2, const int* __restrict__ nll2,
    const int* __restrict__ nur2, const int* __restrict__ nlr2,
    const int* __restrict__ cnul, const int* __restrict__ cnll,
    const int* __restrict__ cnur, const int* __restrict__ cnlr,
    float* __restrict__ oa1, float* __restrict__ oa2, float* __restrict__ oa3, float* __restrict__ oa4,
    float* __restrict__ ol1, float* __restrict__ ol2, float* __restrict__ ol3, float* __restrict__ ol4)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= im || j0 >= jm) return;
    const size_t p = IDX2(i0, j0, im);

    const float ful = (float)nul[p], fll = (float)nll[p], ful_r = (float)nur[p], flr = (float)nlr[p];
    const float ful2 = (float)nul2[p], fll2 = (float)nll2[p], fur2 = (float)nur2[p], flr2 = (float)nlr2[p];
    const float cul = (float)cnul[p], cll = (float)cnll[p], cur = (float)cnur[p], clr = (float)cnlr[p];

    float xnpu, xnpd, t;

    // 1: west-east, (nul+nll) against (nur+nlr)
    oa1[p] = 0.0f;
    xnpu = ful + fll;
    xnpd = ful_r + flr;
    if (xnpd != xnpu) oa1[p] = 1.0f - xnpd / fmaxf(xnpu, 1.0f);
    t = oa1[p];
    oa1[p] = gwd_sign_min1(t);
    ol1[p] = (cul + cll) > 0.0f ? (ful2 + fll2) / (cul + cll) : 0.0f;

    // 2: south-north, (nll+nlr) against (nul+nur)
    oa2[p] = 0.0f;
    xnpu = fll + flr;
    xnpd = ful + ful_r;
    if (xnpd != xnpu) oa2[p] = 1.0f - xnpd / fmaxf(xnpu, 1.0f);
    t = oa2[p];
    oa2[p] = gwd_sign_min1(t);
    ol2[p] = (cll + clr) > 0.0f ? (fll2 + flr2) / (cll + clr) : 0.0f;

    // 3: SW-NE diagonal
    oa3[p] = 0.0f;
    xnpu = fll + 0.5f * (ful + flr);
    xnpd = ful_r + 0.5f * (ful + flr);
    if (xnpd != xnpu) oa3[p] = 1.0f - xnpd / fmaxf(xnpu, 1.0f);
    t = oa3[p];
    oa3[p] = gwd_sign_min1(t);
    ol3[p] = (cul + clr) > 0.0f ? (ful2 + flr2) / (cul + clr) : 0.0f;

    // 4: NW-SE diagonal
    oa4[p] = 0.0f;
    xnpu = ful + 0.5f * (ful_r + fll);
    xnpd = flr + 0.5f * (ful_r + fll);
    if (xnpd != xnpu) oa4[p] = 1.0f - xnpd / fmaxf(xnpu, 1.0f);
    t = oa4[p];
    oa4[p] = gwd_sign_min1(t);
    ol4[p] = (cur + cll) > 0.0f ? (fur2 + fll2) / (cur + cll) : 0.0f;
}
