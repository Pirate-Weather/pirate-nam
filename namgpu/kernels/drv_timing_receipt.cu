// WOOF-NAM unit drv-timing-receipt: device-side wall-clock stamp for the run receipt.
//
// Mode "new": there is no NAM Fortran counterpart. The unit is driver glue that the run receipt
// needs (CONTRACT.md section 0 and 6; the receipt stage table in namgpu/steps/drv_timing_receipt.py).
// No NOAA source block is copied here.
//
// Kernel timing_stamp: writes the SM global nanosecond timer (%globaltimer) into buf[slot].
// One thread, one launch, enqueued on the step stream, so the value is the GPU's own clock at the
// point where the stream reaches the stamp. Nothing here synchronises the host or the device, and
// no printf or device malloc is used. The host reads the buffer once, when the receipt is written.
// Launch: grid (1, 1, 1), block (1, 1, 1).

#include "nmmb_common.cuh"

extern "C" __global__ void timing_stamp(unsigned long long* __restrict__ buf, int slot)
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    buf[slot] = t;
}
