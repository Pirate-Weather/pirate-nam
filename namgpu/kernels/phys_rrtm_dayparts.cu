// phys_rrtm_dayparts.cu -- DAYPARTS sunlit-point selection (+ COSZMN_NMMB), WOOF-NAM.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone), module
//   phys/grrad_nmmb.f and phys/radiation_astronomy_nmmb.f (nam.v4.2.11, 83df4e13c1). The NOAA
//   author headers are kept in the NAM clone and are the notice for this port.
// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:606-656 (nam.v4.2.11, 83df4e13c1), with the
//   COSZMN_NMMB call (grrad_nmmb.f:621-628 -> radiation_astronomy_nmmb.f:744-848).
//
// Fortran DAYPARTS is called once per row J by module_RA_RRTM.F90:879-881 with LENIVEC = IM points
// (XLON = GLON(its:ite,J)). Every row therefore starts its own run list: the first run of each row
// starts at point 1 of that row, and DP_START is row-relative (1-based). Runs never cross a row.
//
// Fortran DAYPARTS, every term accounted for:
//  - inputs XLON, SINLAT, COSLAT (veclen points), SOLHR, MYPE (print control only, dropped),
//    DTSWAV (hours), NRADS, VECLEN.
//  - COSZMN_NMMB produces COSZEN (average of max(0,cosz) over the daytime iterations) and
//    COSZDG (sum of max(0,cosz) times rnstp). Line for line in coszmn_point(): pid12 = 2 asin(1)/12
//    (single precision), f12 = 12, czlimt = 1e-4, rnstp = dtswav/nrads, the it-loop over NRADS
//    iterations with cns = solang + (it-1)*pid12*rnstp + sollag, ss = sinlat*sindec, cc = coslat*cosdec,
//    coszn = ss + cc*cos(cns + xlon), istsun counts coszn > czlimt, final division by istsun when
//    istsun > 0. The module variables sollag, sindec, cosdec are set by the sol_update solar
//    astronomy step and are read on the device from SOL_STATE (sol[0..2]).
//  - Day flag per point: isday = (COSZEN >= 0.0001) (Fortran literal, not czlimt).
//  - Run-length encoding per row (grrad_nmmb.f:636-651): point i=1 of a row always starts a run; a
//    later point starts a run when its day flag differs from point i-1 of the same row. Data-parallel:
//      kernel phys_rrtm_dayparts_cosz    : per point COSZEN, COSZDG, day flag, run-start flag
//                                          (the start flag reads point i-1 of the same row by
//                                          recomputing its COSZEN, so every point is independent);
//      CuPy inclusive cumsum on start    : global scan of the run-start flags (library scan);
//      kernel phys_rrtm_dayparts_scatter : run index inside the row = scan[p] - scan[row start];
//                                          writes DP_START (row-relative, 1-based) and DP_DAY into
//                                          slot row*IM + k;
//      kernel phys_rrtm_dayparts_len     : DP_LEN(k) = next run start - DP_START(k) within the row
//                                          (the last run of a row ends at IM); also NDAYPARTS(row).
//  - MYPE and the veclen < 1 branch are handled by the wrapper. Slots past NDAYPARTS(row) are not
//    written (Fortran also leaves them untouched).
//
// Outputs (device): coszen[n], coszdg[n] float32; day[n], start[n] int32 (scratch);
//   dp_start[n], dp_len[n], dp_day[n] int32 (slot s = row*IM + k, dp_day as 0/1);
//   ndayparts[JM] int32 (runs in each row).

#include "nmmb_common.cuh"

#define DP_PID12   0.261799388f    /* (2*asin(1.0))/12.0, single precision */
#define DP_F12     12.0f
#define DP_CZLIMT  0.0001f         /* radiation_astronomy_nmmb.f:85 */
#define DP_DAYLIM  0.0001f         /* grrad_nmmb.f:641, 0.0001 literal */

// COSZMN_NMMB for one point. Returns COSZEN, writes COSZDG.
__device__ __forceinline__ float coszmn_point(int p, const float* __restrict__ xlon,
                                              const float* __restrict__ sinlat,
                                              const float* __restrict__ coslat,
                                              float solhr, float dtswav, int nrads,
                                              float sollag, float sindec, float cosdec,
                                              float* coszdg_out)
{
    const float solang = DP_PID12 * (solhr - DP_F12);      /* solar angle at present time */
    const float rnstp  = dtswav / (float)nrads;
    const float ss = sinlat[p] * sindec;
    const float cc = coslat[p] * cosdec;
    const float x1 = xlon[p];
    float coszen = 0.0f;
    int istsun = 0;
    for (int it = 1; it <= nrads; ++it) {
        const float cns = solang + (float)(it - 1) * DP_PID12 * rnstp + sollag;
        const float coszn = ss + cc * cosf(cns + x1);
        coszen = coszen + fmaxf(0.0f, coszn);
        if (coszn > DP_CZLIMT) istsun = istsun + 1;
    }
    *coszdg_out = coszen * rnstp;
    if (istsun > 0) coszen = coszen / (float)istsun;
    return coszen;
}

// Kernel 1: one thread per point. Flat p = j*IM + i (row j, point i, both 0-based), n = JM*IM.
extern "C" __global__ void phys_rrtm_dayparts_cosz(
    const float* __restrict__ xlon, const float* __restrict__ sinlat,
    const float* __restrict__ coslat,
    float solhr, float dtswav, int nrads, const float* __restrict__ sol,
    int n, int im,
    float* __restrict__ coszen, float* __restrict__ coszdg,
    int* __restrict__ day, int* __restrict__ start)
{
    const float sollag = sol[0];   /* SOL_STATE float32[4]: sollag, sindec, cosdec (sol_update, kind_phys single) */
    const float sindec = sol[1];
    const float cosdec = sol[2];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float dg;
    const float cz = coszmn_point(p, xlon, sinlat, coslat, solhr, dtswav, nrads, sollag, sindec, cosdec, &dg);
    coszen[p] = cz;
    coszdg[p] = dg;
    const int d = (cz >= DP_DAYLIM) ? 1 : 0;
    day[p] = d;
    if (p % im == 0) {
        start[p] = 1;                       /* grrad_nmmb.f:635-636: each row starts a run at its point 1 */
    } else {
        float dgm;
        const float czm = coszmn_point(p - 1, xlon, sinlat, coslat, solhr, dtswav, nrads, sollag, sindec, cosdec, &dgm);
        const int dm = (czm >= DP_DAYLIM) ? 1 : 0;
        start[p] = (d != dm) ? 1 : 0;
    }
}

// Kernel 2: scatter the run starts. scan = inclusive cumsum of start over the whole domain. The
// row-relative run index is scan[p] - scan[row start]; the row start is always a run start, so it is 0
// for the first run of the row.
extern "C" __global__ void phys_rrtm_dayparts_scatter(
    const int* __restrict__ start, const int* __restrict__ scan, const int* __restrict__ day,
    int n, int im, int* __restrict__ dp_start, int* __restrict__ dp_day)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    if (start[p]) {
        const int row0 = p - (p % im);
        const int k = scan[p] - scan[row0];
        const int slot = row0 + k;          /* slot = row*IM + run index within the row */
        dp_start[slot] = (p - row0) + 1;    /* Fortran dp_start is 1-based and row-relative */
        dp_day[slot]   = day[p];
    }
}

// Kernel 3: run lengths and per-row run counts. One thread per slot (row*IM + k). The run count of a
// row is scan[row end] - scan[row start] + 1 (the row start is a run start, so the count is >= 1).
extern "C" __global__ void phys_rrtm_dayparts_len(
    const int* __restrict__ dp_start, const int* __restrict__ scan, int n, int im,
    int* __restrict__ dp_len, int* __restrict__ ndayparts)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int row0 = s - (s % im);
    const int k = s % im;
    const int nrow = scan[row0 + im - 1] - scan[row0] + 1;
    if (k == 0) ndayparts[s / im] = nrow;
    if (k >= nrow) return;
    const int next = (k + 1 < nrow) ? dp_start[s + 1] : (im + 1);
    dp_len[s] = next - dp_start[s];        /* grrad_nmmb.f:638-650 counts points of the run */
}
