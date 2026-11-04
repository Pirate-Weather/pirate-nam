// phys_rrtm_grrad_sw.cu -- WOOF-NAM unit phys-rrtm-grrad-sw: the SW branch of GRRAD_NMMB
// (setalb call, sfalb, the daytime SWRAD call and the SW heating / flux copies).
//
// NOTICE: NOAA / NCEP NAM source (US government work, no licence file in the NAM clone). Header
// of phys/grrad_nmmb.f (nam.v4.2.11, 83df4e13c1), kept as the notice:
//   !!!!!             'module_radiation_driver' descriptions           !!!!!
//   !   this is the radiation driver module.  it prepares atmospheric      !
//   !   profiles and invokes main radiation calculations.                  !
//   program history log: ncep - created program grrad ... (full block in the source file)
// Ported to CUDA for WOOF-NAM from phys/grrad_nmmb.f:1818-1962 (nam.v4.2.11, 83df4e13c1).
//
// Fortran lines covered (inside the `if (lsswr)` block that starts at 1814):
//   1818-1820  call setalb(...)                       -> phys-rrtm-setalb (separate unit, run first)
//   1822-1824  sfalb(i) = max(0.01, 0.5*(sfcalb(i,2)+sfcalb(i,4)))   -> kernel grrad_sw_sfalb
//   1834       if (isday)                              -> per-column day mask (see below)
//   1841-1867  pad-out to CHK when nday < CHK          -> NOT PORTED: CHK chunk padding is a CPU
//              chunking detail (CONTRACT section 0/3); the full-domain call sees every column once.
//   1872-1897  swrad(..., HSWB=htswb) + htrswb copy     -> phys-rrtm-sw-swrad (separate unit) +
//              kernel grrad_sw_htr (htrswb branch, when the caller asks for band heating)
//   1900-1916  swrad(..., no HSWB)                     -> phys-rrtm-sw-swrad
//   1918-1925  htrsw(i,k) = htswc(i,k1), k1 = k + kd   -> kernel grrad_sw_htr
//   1927-1950  else branch: htrsw, sfcfsw_loc, topfsw_loc, scmpsw zeroed; htrswb zeroed
//                                                      -> kernels grrad_sw_htr and grrad_sw_fluxes
//              (Fortran decides day or night per chunk; grrad_nmmb.f:1530 states that each call
//              is all-day or all-night. Here the decision is per column, as grrad.f does with
//              its idxday list (grrad.f:1308-1312). For a homogeneous chunk the result is the same.)
//   1952-1953  topfsw(1:IM) = topfsw_loc, sfcfsw(1:IM) = sfcfsw_loc  -> kernel grrad_sw_fluxes
//
// Index conventions: the Fortran (i,j,l) arrays are [l, j, i] in CuPy (CONTRACT section 1). The
// multi-component fields are [n, j, i] (sfcalb, topflx, sfcflx) and [n, l, j, i] (htswb, htrswb).
// kd = 1 for this configuration: module_SOLVER_GRID_COMP.F90:10714 sets IFLIP=0 on the NAM NMMB path,
// rad_initialize_nmmb sets ivflip = iflip (rad_initialize_nmmb.f:183), and grrad_nmmb.f:1177-1178
// then gives kd = 1 for lextop (LTP = 1). So htrsw[l] = htswc[l + 1] for l = 0..LM-1 (the extra
// top layer is local level 1). KD is a kernel argument, so the code is the same for either flip.
//
// Device-side day list (no host sync): grrad_sw_daymask writes day[c] = (coszen_loc[c] >= 0.0001)
// (dayparts, grrad_nmmb.f:640-653; coszen_loc is the DAYPARTS COSZMN output, not the instantaneous
// CZEN), the caller runs an inclusive cumsum into csum, and
// grrad_sw_scatter builds idxday[0..nday-1] (zero-based flat column c = j*IM + i) and nday.
// Entries of idxday past nday are stale and must not be read.
//
// Fast numerics: float everywhere (kfpt = single). Halo / MPI / OpenMP: none (single card, full
// domain; iorder ping-pong is OpenMP-only and is dropped, grrad_nmmb.f:1109-1117).
#include "nmmb_common.cuh"

#define NBDSW_SW 14        // radsw_param_nmmb.f:156, NBDSW = NBANDS = NBHGH - NBLOW + 1 = 29 - 16 + 1
#define NF_ALBD_SW 4       // radiation_surface_nmmb.f:94, NF_ALBD = 4
#define NF_TOPFSW 3        // radsw_param_nmmb.f topfsw_type: upfxc, dnfxc, upfx0
#define NF_SFCFSW 4        // radsw_param_nmmb.f sfcfsw_type: upfxc, dnfxc, upfx0, dnfx0
#define NF_SCMP_SW 6       // radsw_param_nmmb.f cmpfsw_type: six components (FDNCMP / scmpsw)
#define SW_DAY_COSZ 0.0001f  // grrad_nmmb.f:640 dayparts threshold (coszen_loc >= 0.0001)

// sfalb(i,j) = max(0.01, 0.5 * (sfcalb(i,2) + sfcalb(i,4)))   (grrad_nmmb.f:1822-1824)
// Fortran sfcalb(IM, NF_ALBD) -> [n, j, i] with n zero-based: component 2 -> n = 1, 4 -> n = 3.
extern "C" __global__ void grrad_sw_sfalb(
    const int im, const int jm,
    const float* __restrict__ sfcalb,      // [NF_ALBD_SW, JM, IM], written by setalb
    float* __restrict__ sfalb)             // [JM, IM]
{
    COLUMN_IJ(im, jm)
    const size_t n_plane = (size_t)jm * (size_t)im;
    const size_t c = (size_t)j0 * (size_t)im + (size_t)i0;
    const float a = sfcalb[1 * n_plane + c];   // sfcalb(i,2)
    const float b = sfcalb[3 * n_plane + c];   // sfcalb(i,4)
    sfalb[c] = fmaxf(0.01f, 0.5f * (a + b));
}

// Day flag per column: dayparts test coszen_loc >= 0.0001 (grrad_nmmb.f:640).
extern "C" __global__ void grrad_sw_daymask(
    const int im, const int jm,
    const float* __restrict__ coszen,      // [JM, IM] (state CZEN)
    int* __restrict__ day)                 // [JM*IM]
{
    COLUMN_IJ(im, jm)
    const size_t c = (size_t)j0 * (size_t)im + (size_t)i0;
    day[c] = (coszen[c] >= SW_DAY_COSZ) ? 1 : 0;
}

// Compact the day columns: idxday[csum[c]-1] = c for day columns; nday = csum[n-1].
extern "C" __global__ void grrad_sw_scatter(
    const int n,
    const int* __restrict__ day,
    const int* __restrict__ csum,          // inclusive cumsum of day
    int* __restrict__ idxday,              // [n], first nday entries valid
    int* __restrict__ nday)                // [1]
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    if (day[c]) idxday[csum[c] - 1] = c;
    if (c == n - 1) nday[0] = csum[c];
}

// htrsw(i,k) = htswc(i,k1) with k1 = k + kd on day columns, 0 on night columns (grrad_nmmb.f:
// 1918-1925 and the else branch 1927-1936). One thread per (i, j, l), l over LM.
// htswc is [LMK, JM, IM] (LMK = LM + LTP = LM + 1). htrswb (optional, grrad_nmmb.f:1887-1897 and
// 1940-1947) is [NBDSW, LM, JM, IM] from htswb [NBDSW, LMK, JM, IM]; it is written only when
// has_htrswb is set.
extern "C" __global__ void grrad_sw_htr(
    const int im, const int jm, const int lm, const int lmk, const int kd,
    const int* __restrict__ day,           // [JM*IM]
    const float* __restrict__ htswc,       // [LMK, JM, IM]
    float* __restrict__ htrsw,             // [LM, JM, IM]
    const int has_htrswb,
    const float* __restrict__ htswb,       // [NBDSW, LMK, JM, IM] (only if has_htrswb)
    float* __restrict__ htrswb)            // [NBDSW, LM, JM, IM] (only if has_htrswb)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    const size_t c = (size_t)j0 * (size_t)im + (size_t)i0;
    const int is_day = day[c];
    const int lsrc = l0 + kd;              // zero-based k1 - 1 = (k + kd) - 1 with k = l0 + 1
    htrsw[IDX3(i0, j0, l0, im, jm)] = is_day ? htswc[IDX3(i0, j0, lsrc, im, jm)] : 0.0f;
    if (has_htrswb) {
        for (int b = 0; b < NBDSW_SW; ++b) {
            htrswb[IDX4(i0, j0, l0, b, im, jm, lm)] =
                is_day ? htswb[IDX4(i0, j0, lsrc, b, im, jm, lmk)] : 0.0f;
        }
    }
}

// topfsw(i) = topfsw_loc(i), sfcfsw(i) = sfcfsw_loc(i) (grrad_nmmb.f:1952-1953) and scmpsw
// (grrad_nmmb.f:1910, 1935). Day columns copy the SWRAD outputs, night columns hold the zero structs
// of grrad_nmmb.f:1933-1935. topflx [NF_TOPFSW, JM, IM], sfcflx [NF_SFCFSW, JM, IM] and
// fdncmp [NF_SCMP, JM, IM] are the swrad outputs (radsw_main_nmmb.f:382-383 and FDNCMP).
extern "C" __global__ void grrad_sw_fluxes(
    const int im, const int jm,
    const int* __restrict__ day,           // [JM*IM]
    const float* __restrict__ topflx,      // [NF_TOPFSW, JM, IM]
    const float* __restrict__ sfcflx,      // [NF_SFCFSW, JM, IM]
    const float* __restrict__ fdncmp,      // [NF_SCMP, JM, IM]
    float* __restrict__ topfsw,            // [NF_TOPFSW, JM, IM]
    float* __restrict__ sfcfsw,            // [NF_SFCFSW, JM, IM]
    float* __restrict__ scmpsw)            // [NF_SCMP, JM, IM]
{
    COLUMN_IJ(im, jm)
    const size_t n_plane = (size_t)jm * (size_t)im;
    const size_t c = (size_t)j0 * (size_t)im + (size_t)i0;
    const int is_day = day[c];
    for (int n = 0; n < NF_TOPFSW; ++n) {
        topfsw[n * n_plane + c] = is_day ? topflx[n * n_plane + c] : 0.0f;
    }
    for (int n = 0; n < NF_SFCFSW; ++n) {
        sfcfsw[n * n_plane + c] = is_day ? sfcflx[n * n_plane + c] : 0.0f;
    }
    for (int n = 0; n < NF_SCMP_SW; ++n) {
        scmpsw[n * n_plane + c] = is_day ? fdncmp[n * n_plane + c] : 0.0f;
    }
}
