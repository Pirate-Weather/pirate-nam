// Row-band copies for namgpu/physics/band.py (column physics in row bands): one launch moves every field a band
// uses, in or out, instead of one CuPy copy per field per band.
//
// A field [.., JM, IM] (C order, 4-byte words: float32 or int32) is `outer` planes of JM*IM words; its band buffer
// [.., W, IM] is `outer` planes of W*IM words. One segment = one plane of one field:
//   seg_parent[s] = byte address of plane s of the parent field (row 0, column 0),
//   seg_band[s]   = byte address of plane s of the band buffer.
// band_copy_in : band[s][t] = parent[s][off + t], t < n   (off = w0*IM, n = W*IM)
// band_copy_out: parent[s][poff + t] = band[s][boff + t], t < n   (poff = a*IM, boff = (a-w0)*IM, n = (b-a)*IM)
// Bit copies only (no arithmetic), so the band result is bitwise the whole-domain result.

extern "C" __global__ void band_copy_in(
    const unsigned long long* __restrict__ seg_parent,
    const unsigned long long* __restrict__ seg_band,
    const int nseg, const long long off, const long long n)
{
    for (int s = blockIdx.y; s < nseg; s += gridDim.y) {
        const unsigned int* __restrict__ src = reinterpret_cast<const unsigned int*>(seg_parent[s]) + off;
        unsigned int* __restrict__ dst = reinterpret_cast<unsigned int*>(seg_band[s]);
        for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
             t += (long long)gridDim.x * blockDim.x)
            dst[t] = src[t];
    }
}

extern "C" __global__ void band_copy_out(
    const unsigned long long* __restrict__ seg_parent,
    const unsigned long long* __restrict__ seg_band,
    const int nseg, const long long poff, const long long boff, const long long n)
{
    for (int s = blockIdx.y; s < nseg; s += gridDim.y) {
        unsigned int* __restrict__ dst = reinterpret_cast<unsigned int*>(seg_parent[s]) + poff;
        const unsigned int* __restrict__ src = reinterpret_cast<const unsigned int*>(seg_band[s]) + boff;
        for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
             t += (long long)gridDim.x * blockDim.x)
            dst[t] = src[t];
    }
}
