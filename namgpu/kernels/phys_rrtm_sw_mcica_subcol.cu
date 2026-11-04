// phys_rrtm_sw_mcica_subcol.cu -- McICA sub-column cloud generator for the NMMB RRTM shortwave.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// Ported to CUDA for WOOF-NAM from phys/radsw_main_nmmb.f:1827-1984 (subroutine mcica_subcol,
// nam.v4.2.11, 83df4e13c1). Notice (source header of radsw_main_nmmb.f): sw-rrtm3 radiation
// package, NCEP modifications of the RRTM-SW radiation code from AER Inc. The random-number calls
// of the source use the Mersenne-twister module phys/mersenne_twister.f (LGPL). That file is NOT
// copied and was NOT read for this port: the generator below is written from the published MT19937
// algorithm (Matsumoto and Nishimura 1998) with the tempering and real-conversion arithmetic of the
// published algorithm. The seeding (mt(0) = seed, mt(i) = 69069 * mt(i-1)) is the one the port was
// written against; it is not yet checked against the module (open item in the wrapper docstring).
//
// What the routine computes, per column (one thread per (i, j)):
//   seed the generator with ipseed (0 -> 4357), draw the stream in the order the Fortran does,
//   and produce lcloudy(g, k) = .true. for every g-point (g = 1..ngptsw = 112) and layer k.
//   iovrsw 0: random overlap, one new draw per (g, k).
//   iovrsw 1: max-random overlap, first draw per g at the lowest layer, then walk up the column.
//   iovrsw 2: maximum overlap, one draw per g shared by all layers.
//   lcloudy(g, k) = cdfunc(g, k) >= 1 - cf(k), with cf = cldf after the ftiny threshold.
//
// Layer order (verified 2026-11-04 from the NAM driver, not assumed):
//   The NMMB driver calls rad_initialize_nmmb with IFLIP = 0 (module_SOLVER_GRID_COMP.F90:10714), so
//   radsw_main_nmmb.f reverses the NMMB top-first input to its internal array (lines 732-944, ivflip
//   = 0 branch). The internal array runs surface-first: k0 = 0 is the LOWEST layer. cldf (produced by
//   phys-rrtm-sw-cldprop) and lcloudy use that same internal order. This is the radiation exception in
//   CONTRACT.md section 1, not the NMMB array order.
//
// Index conventions (zero-based in CUDA):
//   cldf[k0, j0, i0]      float32 [NLAY, JM, IM]; k0 = 0 is the lowest layer (radsw internal order).
//   ipseed[j0, i0]        int32 [JM, IM] permutation seed of the column.
//   lcloudy (output)      uint32 [NWORD = 4, NLAY, JM, IM], bit (g & 31) of word (g >> 5) holds
//                         lcloudy(g+1, k0+1). Bit-packed because the unpacked logical array
//                         (112 x 60 per column) would need about 6.7 GB at 954 x 835 x 60.
//
// Threshold: radsw_main_nmmb.f:1787-1792 sets cldf = cfrac and cldf = 0 where cldf < ftiny
// (ftiny = 1.0e-12, radsw_main_nmmb.f:295) before the call. The kernel applies that step itself
// (cf below), because it is the consumer and the step sits immediately before mcica_subcol.
//
// Guards: a column that the wrapper would refuse (nlay > 128, iovrsw outside 0..2) writes all-zero
// lcloudy for that column. It never leaves stale bits.
//
// Launch: grid_cols(IM, JM), BLOCK_COL. No host sync, no device allocation, no printf.

#include "nmmb_common.cuh"

#define MCICA_NGPTSW     112           // radsw_param.f:110 NGPTSW
#define MCICA_NWORD      4             // ceil(112 / 32) packed words per layer
#define MCICA_NLAY_MAX   128           // local scratch per thread; wrapper checks nlay <= 128
#define MCICA_MT_N       624           // MT19937 state words
#define MCICA_MT_M       397
#define MCICA_MATA       0x9908B0DFu   // MT19937 matrix A
#define MCICA_UMASK      0x80000000u   // upper bit
#define MCICA_LMASK      0x7FFFFFFFu   // lower 31 bits
#define MCICA_TMASKB     0x9D2C5680u   // tempering mask b
#define MCICA_TMASKC     0xEFC60000u   // tempering mask c
#define MCICA_ISEED      4357          // default seed when ipseed == 0 (iseed)
#define MCICA_FTINY      1.0e-12f      // radsw_main_nmmb.f:295 ftiny (single precision, kind_phys = 4)
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

// MT19937 twist of the whole state, 0-based form.
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

// One draw: tempering, then the real conversion:
//   y >= 0 : real(y) / 2**32
//   y <  0 : (real(y) + 2**32) / 2**32
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

// Zero every packed word of one column (the refused-input guard).
__device__ inline void mcica_zero_column(unsigned int* lcloudy, const int nlay, const int jm,
                                         const int im, const int i0, const int j0)
{
    for (int w = 0; w < MCICA_NWORD; ++w)
        for (int k0 = 0; k0 < nlay; ++k0)
            lcloudy[(((size_t)w * nlay + k0) * jm + j0) * im + i0] = 0u;
}

extern "C" __global__ void phys_rrtm_sw_mcica_subcol(
    const float* __restrict__ cldf,
    const int* __restrict__ ipseed,
    unsigned int* __restrict__ lcloudy,
    const int nlay, const int jm, const int im, const int iovrsw)
{
    COLUMN_IJ(im, jm)
    if (nlay > MCICA_NLAY_MAX || iovrsw < 0 || iovrsw > 2) {
        mcica_zero_column(lcloudy, nlay, jm, im, i0, j0);   // the wrapper refuses these first
        return;
    }

    // radsw_main_nmmb.f:1787-1792: cldf = cfrac, then cldf = 0 where cldf < ftiny.
    float cf[MCICA_NLAY_MAX];
    for (int k0 = 0; k0 < nlay; ++k0) {
        const float c = cldf[IDX3(i0, j0, k0, im, jm)];
        cf[k0] = (c < MCICA_FTINY) ? 0.0f : c;
    }

    McicaRng rng;
    mcica_setseed(rng, ipseed[IDX2(i0, j0, im)]);

    // Packed result for this column, one word-set per layer (k0 = 0 lowest).
    unsigned int bits[MCICA_NLAY_MAX][MCICA_NWORD];
    for (int k0 = 0; k0 < nlay; ++k0)
        for (int w = 0; w < MCICA_NWORD; ++w)
            bits[k0][w] = 0u;

    if (iovrsw == 0) {
        // Fortran case(0): rand2d(k1) in order n outer, k inner -> cdfunc(n,k) = rand2d((n-1)*nlay + k).
        for (int g = 0; g < MCICA_NGPTSW; ++g) {
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float r = mcica_next(rng);
                const float tem1 = 1.0f - cf[k0];                  // f_one - cldf(k)
                if (r >= tem1)
                    bits[k0][g >> 5] |= (1u << (g & 31));
            }
        }
    } else if (iovrsw == 1) {
        // Fortran case(1): raw draws as case 0; then for k = 2..nlay (k0 = 1..nlay-1, upward):
        //   if cdfunc(n,k-1) > 1 - cldf(k-1) then cdfunc(n,k) = cdfunc(n,k-1)
        //   else cdfunc(n,k) = raw(n,k) * (1 - cldf(k-1)).
        for (int g = 0; g < MCICA_NGPTSW; ++g) {
            float prev = 0.0f;
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float r = mcica_next(rng);
                float cd;
                if (k0 == 0) {
                    cd = r;
                } else {
                    const float tem1 = 1.0f - cf[k0 - 1];
                    cd = (prev > tem1) ? prev : r * tem1;
                }
                prev = cd;
                const float tem2 = 1.0f - cf[k0];
                if (cd >= tem2)
                    bits[k0][g >> 5] |= (1u << (g & 31));
            }
        }
    } else {
        // Fortran case(2): rand1d(n) for n = 1..ngptsw, the same value at every level.
        for (int g = 0; g < MCICA_NGPTSW; ++g) {
            const float r = mcica_next(rng);
            for (int k0 = 0; k0 < nlay; ++k0) {
                const float tem1 = 1.0f - cf[k0];
                if (r >= tem1)
                    bits[k0][g >> 5] |= (1u << (g & 31));
            }
        }
    }

    for (int w = 0; w < MCICA_NWORD; ++w)
        for (int k0 = 0; k0 < nlay; ++k0)
            lcloudy[(((size_t)w * nlay + k0) * jm + j0) * im + i0] = bits[k0][w];
}
