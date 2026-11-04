// io_history_rust.cu -- device-side staging for the Rust NetCDF history writer (unit io-history-rust).
//
// No Fortran counterpart (UNITS.json fortran "-", mode "new"). The history file is written by
// gpuwm's Rust classic-NetCDF writer (libnetcdf_writer.so, ABI 1; see namgpu/io/io_history_rust.py).
// That writer takes host bytes in WRF order (bottom-up), so the one kernel here does the one device
// step the write needs: it copies a 3-D NMMB field [LM, JM, IM] (l = 1 is the model top, CONTRACT
// section 1) into a staging buffer with the level axis reversed (WRF bottom_top, bottom-up), so the
// single device->host copy per field is contiguous and already in file order. Same effect as the
// host-side h[::-1] in io/history.py, without a host pass over the field.
//
// In:  im, jm, nl (scalars), src [NL, JM, IM] float32 (NMMB layout, device), im, jm, nl.
// Out: dst [NL, JM, IM] float32 (WRF bottom-up), device staging buffer, allocated once per shape.
// Launch: grid_2d(im, jm, nl) on ctx.stream, no host sync, no device malloc, no printf.
#include "nmmb_common.cuh"

extern "C" __global__ void io_history_flip_levels(
    const int im, const int jm, const int nl,
    const float* __restrict__ src, float* __restrict__ dst)
{
    POINT_IJL
    if (i0 >= im || j0 >= jm || l0 >= nl) return;
    dst[IDX3(i0, j0, nl - 1 - l0, im, jm)] = src[IDX3(i0, j0, l0, im, jm)];
}
