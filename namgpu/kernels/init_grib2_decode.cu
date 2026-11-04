// nam-gpu unit init-grib2-decode: GRIB2 record gather into the NMMB [j, i] float32 layout.
//
// What is decoded where:
//   - The gpuwm Rust bridge grib2_dump decodes every packing on the host (simple, complex with
//     spatial differencing (DRT 5.2/5.3), JPEG2000, PNG) and writes one little-endian float64
//     file per record (nx*ny values in the record's stored scan order) plus metadata.tsv.
//     Complex packing is decoded by the bridge; it is not re-implemented here.
//   - This kernel does the one device step left: place each record into NMMB layout, applying
//     the GRIB2 scan-mode flags (WMO Table 3.4), and convert to float32. Non-finite values are
//     replaced by the declared missing value and counted on the device.
//
// NOTICE: new nam-gpu code, not a port of a NAM Fortran routine. The NAM reads GRIB2 through
// NCEPLIBS/NEMS (no NEMS framework code is used here). The scan-mode convention is the GRIB2
// Table 3.4 definition as the gpuwm bridge caller normalises it (gpuwm, Apache-2.0,
// mapped_source.py _regular_latlon_frame: scan 0x40 passes through, scan 0x00 reverses rows).
//
// Scan-mode bits used (Table 3.4):
//   0x80  1 = points run -i in storage (east to west), 0 = +i
//   0x40  1 = points run +j in storage (south to north), 0 = -j
//   0x20  1 = adjacent storage points run along j (column-major), 0 = along i (row-major)
//   0x10  consecutive rows alternate direction (boustrophedon): refused by the wrapper.
// Output: out[j0, i0] with i0 west->east and j0 south->north, i.e. IDX2(i0, j0, nx) (CONTRACT
// section 1, Fortran A(i,j) -> [j, i]). Zero-based: i0 = i-1, j0 = j-1.
//
// Launch: one thread per output point (namgpu.kernels.grid_2d(nx, ny)); no loop over cells.
// Init-only; no per-step use.

#include "nmmb_common.cuh"

extern "C" __global__ void init_grib2_gather(
    const double* __restrict__ raw,     // [ny * nx] float64, stored scan order
    float* __restrict__ out,            // [ny, nx] float32, NMMB layout
    int* __restrict__ n_nonfinite,      // [1] device counter of replaced values
    int nx, int ny, int scan_mode, float missing_value)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;   // west->east column
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;   // south->north row
    if (i0 >= nx || j0 >= ny) return;

    // Storage position of the grid point (i0, j0). Each mapping is its own inverse.
    const int si = (scan_mode & 0x80) ? (nx - 1 - i0) : i0;
    const int sj = (scan_mode & 0x40) ? j0 : (ny - 1 - j0);
    const size_t k = (scan_mode & 0x20)
        ? ((size_t)si * (size_t)ny + (size_t)sj)     // column-major storage
        : ((size_t)sj * (size_t)nx + (size_t)si);   // row-major storage

    const double v = raw[k];
    const size_t o = IDX2(i0, j0, nx);
    if (isfinite(v)) {
        out[o] = (float)v;                           // round-to-nearest, as real(dp) -> real(sp)
    } else {
        out[o] = missing_value;
        atomicAdd(n_nonfinite, 1);
    }
}
