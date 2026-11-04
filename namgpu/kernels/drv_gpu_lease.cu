// drv_gpu_lease.cu -- device probe for the shared-card pool lease (unit drv-gpu-lease).
//
// No Fortran counterpart (UNITS.json mode "new", fortran "-"). The lease itself is host
// work (namgpu/steps/drv_gpu_lease.py takes and releases the pool-v3 card lease around a run).
// This file holds the one device-side action the lease unit needs: a one-thread probe that
// writes 1.0 into a one-element scratch flag on the leased card, so the wrapper can prove the
// card the lease handed out is usable by this build (NVRTC compile + a launch). No host sync
// happens here; the host reads the flag only in drv_gpu_lease.verify(), off the step path.
//
// Written for WOOF-NAM (private tree). Float32, single precision like every WOOF-NAM kernel.
#include "nmmb_common.cuh"

extern "C" __global__ void lease_probe(float* __restrict__ flag, const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        flag[k] = 1.0f;
    }
}
