// WOOF-NAM unit drv-step-plan: device-side finiteness scan used by the step-plan checker.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// The unit has no Fortran numerics of its own. The plan it describes is SOLVER_RUN call order and
// cadence (nmm/module_SOLVER_GRID_COMP.F90:2453-7168), held as data in namgpu/driver.py.
// Ported to CUDA for WOOF-NAM from nmm/module_SOLVER_GRID_COMP.F90:2453-7168 (nam.v4.2.11, 83df4e13c1).
//
// Kernel drv_step_plan_nonfinite_count: counts NaN and Inf values in a float32 array.
// Grid-stride loop, one block-level reduction, one atomicAdd per block into a device counter.
// The counter stays on the device; the caller decides when to read it (no host sync here).
// Used by the done rule (CONTRACT.md section 6) and by debug runs that check a state after a unit.
// Launch: grid (ceil(n/256), 1, 1), block (256, 1, 1), any grid size works (grid-stride).

#include "nmmb_common.cuh"

extern "C" __global__ void drv_step_plan_nonfinite_count(
    const float* __restrict__ a, long long n, unsigned int* __restrict__ count)
{
    __shared__ unsigned int partial[256];
    unsigned int local = 0u;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        const float v = a[k];
        // NaN fails v == v; Inf fails the magnitude test.
        if (!(v == v) || v > 3.402823466e38f || v < -3.402823466e38f) local += 1u;
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && partial[0] != 0u) atomicAdd(count, partial[0]);
}
