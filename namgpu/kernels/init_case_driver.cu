// nam-gpu unit init-case-driver: device-side non-finite census for the init chain.
//
// Mode "new": there is no NAM Fortran counterpart. The unit is the init-chain driver (CONTRACT.md
// section 0 and 6). The only device work it needs is the done-rule check that no array written by
// the init chain holds a NaN or Inf. No NOAA source block is copied here.
//
// Kernel init_case_nonfinite_count: every thread walks a grid-stride range of one float array and
// adds one to count[0] for each element that is NaN or Inf (exponent bits all ones). The counter
// is unsigned int, updated with atomicAdd, so the host reads one integer per array, once, at the
// end of the init chain. No printf, no device malloc, no host sync inside the kernel.
// Launch: 1-D grid, block 256, n = number of float elements, count = one device uint32 that the
// caller zeroes before the launch.

#include "nmmb_common.cuh"

extern "C" __global__ void init_case_nonfinite_count(const float* __restrict__ a,
                                                     long long n,
                                                     unsigned int* __restrict__ count)
{
    long long stride = (long long)gridDim.x * (long long)blockDim.x;
    unsigned int bad = 0u;
    for (long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
         k < n; k += stride) {
        unsigned int bits = __float_as_uint(a[k]);
        if ((bits & 0x7F800000u) == 0x7F800000u) bad += 1u;   // NaN or Inf
    }
    if (bad) atomicAdd(count, bad);
}
