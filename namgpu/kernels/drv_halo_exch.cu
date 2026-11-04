// drv_halo_exch.cu -- WOOF-NAM unit drv-halo-exch: pack and unpack the JHALO-row halo of a row-slab field, and the
// per-row owned-row hash used by NAMGPU_UNIT_HASH (MULTICARD-PLAN.md sections 3 and 10).
//
// Replaces the NMMB MPI halo exchange (nmm/module_EXCHANGE.F90 HALO_EXCH, called from module_SOLVER_GRID_COMP.F90 at
// the sites listed in namgpu/steps/drv_halo_exch.py). Layout: every field is C-contiguous with its last two axes
// [JM, IM] (local memory rows); the leading axes (levels, tracers) fold into P planes. Words are 4 bytes (float32 or
// int32), copied as raw 32-bit words, so the exchange is exact for either type.
//
// No arithmetic on field values: pack/unpack are copies. The hash is integer only (wraparound uint64 sums are
// associative and commutative, so the atomics give the same value in any order).

extern "C" __global__ void halo_pack(
    const unsigned int* __restrict__ src,   // field [P, JM, IM]
    unsigned int* __restrict__ dst,         // packed [P, NR, IM] at its offset in the exchange buffer
    const long long P, const int JM, const int IM,
    const int r0,                           // first local row (0-based) to pack
    const int NR)                           // rows to pack
{
    const long long n = P * (long long)NR * IM;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x) {
        const long long p = k / ((long long)NR * IM);
        const int rem = (int)(k - p * NR * IM);
        const int r = rem / IM;
        const int i = rem - r * IM;
        dst[k] = src[(p * JM + (r0 + r)) * (long long)IM + i];
    }
}

extern "C" __global__ void halo_unpack(
    const unsigned int* __restrict__ src,   // packed [P, NR, IM]
    unsigned int* __restrict__ dst,         // field [P, JM, IM]
    const long long P, const int JM, const int IM,
    const int r0, const int NR)
{
    const long long n = P * (long long)NR * IM;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x) {
        const long long p = k / ((long long)NR * IM);
        const int rem = (int)(k - p * NR * IM);
        const int r = rem / IM;
        const int i = rem - r * IM;
        dst[(p * JM + (r0 + r)) * (long long)IM + i] = src[k];
    }
}

// splitmix64 finaliser
__device__ __forceinline__ unsigned long long mix64(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Per-row hash of rows r0..r0+NR-1: h[r] = sum over (p, i) of mix64(word ^ (position(p, i) << 32)). The position key
// depends on (p, i) only, not on j, so a row hashes the same on any card that holds it.
extern "C" __global__ void row_hash(
    const unsigned int* __restrict__ src, const long long P, const int JM, const int IM,
    const int r0, const int NR, unsigned long long* __restrict__ h)
{
    const long long n = P * (long long)NR * IM;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x) {
        const long long p = k / ((long long)NR * IM);
        const int rem = (int)(k - p * NR * IM);
        const int r = rem / IM;
        const int i = rem - r * IM;
        const unsigned long long pos = (unsigned long long)(p * IM + i);
        const unsigned long long w = src[(p * JM + (r0 + r)) * (long long)IM + i];
        atomicAdd(&h[r], mix64(w ^ (pos << 32)));
    }
}
