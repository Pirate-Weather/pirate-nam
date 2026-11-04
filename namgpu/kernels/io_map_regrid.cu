// io_map_regrid.cu -- io-map-regrid: bilinear regrid of NMMB H-point history fields from the
// rotated lat-lon B grid onto a Lambert conformal map frame (NAM 218 by default), for maps.
//
// new (no Fortran counterpart; CONTRACT.md section 3, mode new). Layout: source [LM, JM, IM] (H
// points, l=1 model top), destination [LM, NY, NX]. One thread per target point and level
// (grid_2d: x = target i, y = target j, z = level). The index and weight tables are built once
// per frame by io_map_regrid.py (host, vectorised) and passed in as device arrays.
// Outside the source domain (valid == 0), or where any of the four source corners is non-finite,
// the output is `fill`, never NaN.
// No host sync, no allocation, one launch per field.
#include "nmmb_common.cuh"

extern "C" __global__ void io_map_regrid_bilinear(
    const float* __restrict__ src,           // [LM, JM, IM]
    float* __restrict__ dst,                 // [LM, NY, NX]
    const int* __restrict__ wi0,             // [NY, NX] source i of the west column (0..IM-2)
    const int* __restrict__ wj0,             // [NY, NX] source j of the south row   (0..JM-2)
    const float* __restrict__ wx,            // [NY, NX] fraction from i0 toward i0+1
    const float* __restrict__ wy,            // [NY, NX] fraction from j0 toward j0+1
    const unsigned char* __restrict__ valid, // [NY, NX] 1 if the target point lies in the source domain
    const int nx, const int ny, const int im, const int jm, const int lm,
    const float fill)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (ix >= nx || iy >= ny || l0 >= lm) return;

    const size_t p = (size_t)iy * (size_t)nx + (size_t)ix;
    float out = fill;
    if (valid[p]) {
        const int i0 = wi0[p];
        const int j0 = wj0[p];
        const float fx = wx[p];
        const float fy = wy[p];
        const float v00 = src[IDX3(i0,     j0,     l0, im, jm)];
        const float v10 = src[IDX3(i0 + 1, j0,     l0, im, jm)];
        const float v01 = src[IDX3(i0,     j0 + 1, l0, im, jm)];
        const float v11 = src[IDX3(i0 + 1, j0 + 1, l0, im, jm)];
        // A non-finite source corner would otherwise spread NaN/Inf to up to four targets.
        // Such a target is written as fill instead (no renormalised partial blend).
        const bool ok = isfinite(v00) && isfinite(v10) && isfinite(v01) && isfinite(v11);
        if (ok) {
            const float bot = v00 + fx * (v10 - v00);
            const float top = v01 + fx * (v11 - v01);
            out = bot + fy * (top - bot);
        }
    }
    dst[((size_t)l0 * (size_t)ny + (size_t)iy) * (size_t)nx + (size_t)ix] = out;
}
