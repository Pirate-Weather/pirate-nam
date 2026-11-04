// phys_rrtm_lw_mcica_subcol.cu -- McICA sub-column cloud generator for the NMMB RRTM longwave.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:1702-1858 (subroutine mcica_subcol,
// nam.v4.2.11, 83df4e13c1). Notice (source header of radlw_main_nmmb.f): lw-rrtm3 radiation
// package, NCEP modifications of the RRTM-LW radiation code from AER Inc. The random-number calls
// of the source use the Mersenne-twister module phys/mersenne_twister.f (LGPL). That file is NOT
// copied: the generator below is written from the published MT19937 algorithm (Matsumoto and
// Nishimura 1998) with the seeding, tempering and real-conversion arithmetic that the module
// implements, so the random stream is the same.
//
// What the routine computes, per column (one thread per (i, j)):
//   seed the generator with ipseed (0 -> 4357), draw the stream in the order the Fortran does,
//   and produce lcloudy(g, k) = .true. for every g-point (g = 1..ngptlw = 140) and layer k.
//   iovrlw 0: random overlap, one new draw per (g, k).
//   iovrlw 1: max-random overlap, first draw per g at k = 1, then walk up the column (k = 2..nlay).
//   iovrlw 2: maximum overlap, one draw per g shared by all layers.
//   lcloudy(g, k) = cdfunc(g, k) >= 1 - cldf(k).
//
// Index conventions (zero-based in CUDA):
//   cldf[k0, j0, i0]      float32 [NLAY, JM, IM]; k0 = 0 is the first Fortran layer of the radlw
//                         internal order (the Fortran k = 1, the lowest layer for ivflip = 1, which
//                         is NAM's setting). cldf is the array after the cldmin threshold of cldprop.
//   ipseed[j0, i0]        int32 [JM, IM] permutation seed of the column.
//   lcloudy (output)      uint32 [NWORD = 5, NLAY, JM, IM], bit (g & 31) of word (g >> 5) holds
//                         lcloudy(g+1, k0+1). Bit-packed because the unpacked logical array
//                         (140 x 60 per column) would need about 6.7 GB at 954 x 835 x 60.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"

#define MCICA_NGPTLW     140           // radlw_param_nmmb.f:141 NGPTLW
#define MCICA_NWORD      5             // ceil(140 / 32) packed words per layer
#define MCICA_MT_N       624           // MT19937 state words
#define MCICA_MT_M       397
#define MCICA_MATA       0x9908B0DFu   // MT19937 matrix A (mata in mersenne_twister.f)
#define MCICA_UMASK      0x80000000u   // upper bit
#define MCICA_LMASK      0x7FFFFFFFu   // lower 31 bits
#define MCICA_TMASKB     0x9D2C5680u   // tempering mask b (tmaskb)
#define MCICA_TMASKC     0xEFC60000u   // tempering mask c (tmaskc)
#define MCICA_ISEED      4357          // default seed when ipseed == 0 (iseed)
#define MCICA_2POW32     4294967296.0f // 2**32 in single precision; (2**32 - 1) also rounds to this

struct McicaRng {
    unsigned int mt[MCICA_MT_N];
    int mti;
};

// random_setseed_t: mt(0) = seed (0 -> iseed), mt(i) = 69069 * mt(i-1) with 32-bit wrap; mti = n.
__device__ inline void mcica_setseed(McicaRng& r, int inseed)
{
    unsigned int ii = (inseed == 0) ? (unsigned int)MCICA_ISEED : (unsigned int)inseed;
    r.mt[0] = ii;
    for (int i = 1; i < MCICA_MT_N; ++i)
        r.mt[i] = 69069u * r.mt[i - 1];
    r.mti = MCICA_MT_N;
}

// MT19937 twist of the whole state (random_number_t, the mti >= n branch), 0-based form.
__device__ inline void mcica_twist(McicaRng& r)
{
    const unsigned int mag01[2] = {0u, MCICA_MATA};
    unsigned int y;
    int kk;
    for (kk = 0; kk < MCICA_MT_N - MCICA_MT_M; ++kk) {
        y = (r.mt[kk] & MCICA_UMASK) | (r.mt[kk + 1] & MCICA_LMASK);
        r.mt[kk] = r.mt[kk + MCICA_MT_M] ^ (y >> 1) ^ mag01[y & 1u];
    }
    for (kk = MCICA_MT_N - MCICA_MT_M; kk < MCICA_MT_N - 1; ++kk) {
        y = (r.mt[kk] & MCICA_UMASK) | (r.mt[kk + 1] & MCICA_LMASK);
        r.mt[kk] = r.mt[kk + MCICA_MT_M - MCICA_MT_N] ^ (y >> 1) ^ mag01[y & 1u];
    }
    y = (r.mt[MCICA_MT_N - 1] & MCICA_UMASK) | (r.mt[0] & MCICA_LMASK);
    r.mt[MCICA_MT_N - 1] = r.mt[MCICA_MT_M - 1] ^ (y >> 1) ^ mag01[y & 1u];
    r.mti = 0;
}

// random_number_t for one value: tempering, then the real conversion of the module:
//   y >= 0 : real(y) / (2**32 - 1)          (denominator is 2**32 in single precision)
//   y <  0 : (real(y) + 2**32) / (2**32 - 1)
// Single-precision edge kept as in the source: for y in [-128, -1] the numerator rounds to 2**32,
// so the draw is exactly 1.0 and a clear layer (cldf = 0, tem1 = 1) still passes cdfunc >= tem1.
__device__ inline float mcica_next(McicaRng& r)
{
    if (r.mti >= MCICA_MT_N)
        mcica_twist(r);
    unsigned int y = r.mt[r.mti];
    r.mti = r.mti + 1;
    y ^= (y >> 11);
    y ^= (y << 7) & MCICA_TMASKB;
    y ^= (y << 15) & MCICA_TMASKC;
    y ^= (y >> 18);
    const int ys = (int)y;
    if (ys < 0) {
        const float fy = (float)ys;
        return (fy + MCICA_2POW32) / MCICA_2POW32;
    }
    return (float)ys / MCICA_2POW32;
}

extern "C" __global__ void phys_rrtm_lw_mcica_subcol(
    const float* __restrict__ cldf,
    const int* __restrict__ ipseed,
    unsigned int* __restrict__ lcloudy,
    const int nlay, const int jm, const int im, const int iovrlw)
{
    COLUMN_IJ(im, jm)
    // Packed output for this column is written in place: each thread owns its column, so the words are
    // zeroed first and the bits are OR-ed straight into global memory (no per-thread scratch array; the
    // old 128-layer scratch was about 2.5 KB of local memory per thread on top of the MT state).
#define MCICA_OUT(w0, k0) lcloudy[(((size_t)(w0) * nlay + (k0)) * jm + j0) * im + i0]
    for (int w = 0; w < MCICA_NWORD; ++w)
        for (int k0 = 0; k0 < nlay; ++k0)
            MCICA_OUT(w, k0) = 0u;

    McicaRng rng;
    mcica_setseed(rng, ipseed[IDX2(i0, j0, im)]);

    if (iovrlw == 0) {
        // Fortran case(0): rand2d(k1) in order n outer, k inner -> cdfunc(n,k) = rand2d((n-1)*nlay + k).
        for (int g = 0; g < MCICA_NGPTLW; ++g) {
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float r = mcica_next(rng);
                const float tem1 = 1.0f - cldf[IDX3(i0, j0, k0, im, jm)];   // f_one - cldf(k)
                if (r >= tem1)
                    MCICA_OUT(g >> 5, k0) |= (1u << (g & 31));
            }
        }
    } else if (iovrlw == 1) {
        // Fortran case(1): raw draws as case 0; then for k = 2..nlay:
        //   if cdfunc(n,k-1) > 1 - cldf(k-1) then cdfunc(n,k) = cdfunc(n,k-1)
        //   else cdfunc(n,k) = raw(n,k) * (1 - cldf(k-1)).
        for (int g = 0; g < MCICA_NGPTLW; ++g) {
            float prev = 0.0f;
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float r = mcica_next(rng);
                float cd;
                if (k0 == 0) {
                    cd = r;
                } else {
                    const float tem1 = 1.0f - cldf[IDX3(i0, j0, k0 - 1, im, jm)];
                    cd = (prev > tem1) ? prev : r * tem1;
                }
                prev = cd;
                const float tem2 = 1.0f - cldf[IDX3(i0, j0, k0, im, jm)];
                if (cd >= tem2)
                    MCICA_OUT(g >> 5, k0) |= (1u << (g & 31));
            }
        }
    } else if (iovrlw == 2) {
        // Fortran case(2): rand1d(n) for n = 1..ngptlw, the same value at every level.
        for (int g = 0; g < MCICA_NGPTLW; ++g) {
            const float r = mcica_next(rng);
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float tem1 = 1.0f - cldf[IDX3(i0, j0, k0, im, jm)];
                if (r >= tem1)
                    MCICA_OUT(g >> 5, k0) |= (1u << (g & 31));
            }
        }
    } else {
        return;   // RLWINIT stops for iovrlw outside 0..2; the Fortran cdfunc is undefined here
    }
}

#undef MCICA_OUT

// lcloudy -> cldfmc: the step of cldprop that Fortran does at radlw_main_nmmb.f:1684-1691,
// cldfmc(ig,k) = 1 if lcloudy(ig,k) else 0. Written by the same unit because the packed bits are
// produced here. words: uint32 [NWORD, NLAY, NCOL] (C-contiguous chunk, c = flat column index j*IM+i);
// cldfmc: float32 [NLAY, NGPTLW, NCOL], the layout phys-rrtm-lw-rtrnmc reads (index ((k*NGPTLW+g)*NCOL+c)).
// Grid-stride over the output elements: no per-column loop, no host sync.
extern "C" __global__ void phys_rrtm_lw_mcica_unpack(
    const unsigned int* __restrict__ words,
    float* __restrict__ cldfmc,
    const int nlay, const int ncol)
{
    const size_t n = (size_t)nlay * MCICA_NGPTLW * (size_t)ncol;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (size_t)gridDim.x * blockDim.x) {
        const int c = (int)(t % (size_t)ncol);
        const size_t r = t / (size_t)ncol;
        const int g = (int)(r % MCICA_NGPTLW);
        const int k = (int)(r / MCICA_NGPTLW);
        const unsigned int w = words[((size_t)(g >> 5) * nlay + k) * (size_t)ncol + c];
        cldfmc[t] = ((w >> (g & 31)) & 1u) ? 1.0f : 0.0f;
    }
}
