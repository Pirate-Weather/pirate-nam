// dyn_mono_fix.cu -- WOOF-NAM unit dyn-mono-fix: MONO setup and per-species monotonisation
// (local min/max limiter, per-species sums of positive and negative limited changes).
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). The Fortran
// module module_DYNAMICS_ROUTINES.F90 has no author/history block of its own; the NOAA notice is
// in the NOTICE file at the repo root.
// Ported to CUDA for WOOF-NAM from nmm/module_DYNAMICS_ROUTINES.F90:4768-5075 (nam.v4.2.11, 83df4e13c1),
// the allreduce branch of MONO (use_allreduce = true, parm/nam_configfile line 159).
//
// Fortran subroutine MONO(idtadt, kss, kse, lm, dsg2, pdsg1, epsq2, dare, pd, indx_q2, s, inpes,
//                         jnpes, use_allreduce, s1, tcs)
// Lines mapped here:
//   4768-4893  argument list and declarations. Only s, s1, tcs, pd, dare, dsg2, pdsg1, idtadt,
//              kss, kse, lm and use_allreduce are used by this unit. epsq2 and indx_q2 belong to
//              the Q2 back-interpolation unit dyn-mono-q2-back.
//   4894       steep = 1. - 0.040*idtadt              -> steep in the kernel (float)
//   4900-4907  dvol(i,j,l) = (dsg2(l)*pd(i,j)+pdsg1(l))*dare(j), rdvol = 1/dvol.
//              dvol is recomputed inline per cell (a pure function of pd, dare, dsg2, pdsg1).
//              rdvol is not stored here: the rescale loop at 5160-5200 (owned by dyn-mono-global-sum)
//              recomputes it from the same inputs.
//   4916-4996  use_allreduce branch: for each species, for each interior (i, j, l):
//                s1p = (s1 + tcs)**2 ; tcs = s1p - s
//                sminh/smaxh = min/max of the 3x3 horizontal neighbourhood of s at level l
//                sminv/smaxv = min/max of s over the vertical neighbours (l-1, l, l+1), with the
//                              top level (l=1) using l, l+1 and the bottom level (l=lm) using l-1, l
//                smin = min(sminh, sminv), smax = max(smaxh, smaxv)
//                sn = s1p, then clipped above at steep*smax and below at smin (Fortran order)
//                dsp = (sn - s1p)*dvol ; added to the positive sum if dsp > 0, else to the other sum
//              These are the local sums that dyn-mono-global-sum reduces (a device reduction replaces
//              MPI_Allreduce; on one card the local sums are the global sums).
//              The allreduce branch does NOT write s1 (the Fortran rescale then reads the raw s1; that is
//              the literal quirk the dyn-mono-global-sum rescale keeps).
//   4997-5072  send/recv branch (use_allreduce = false): s1(i,j,l,ks) = dsp (stored to kfpt) for the
//              interior cells, same limiter as above. The per-column sums s1l_sum are formed by
//              dyn-mono-global-sum (mono_global_sum_reduce) from that s1, as the Fortran sums them after
//              the send/recv. No xsums are written on this branch.
//   5073-5078  endif. The gsums/xsums zeroing (Fortran 4925-4928, the allreduce branch's do-ks loop) is
//              done by the wrapper: the xsums buffer is zeroed on the device before the launch.
//
// Layout (CONTRACT.md sections 1 and 3): Fortran s(i,j,l,ks) -> s[ks-1][l-1][j-1][i-1] via
// IDX4(i0, j0, l0, ks0, IM, JM, LM); the same for s1 and tcs. s1 is the wrapper's S1 scratch
// (sqrt-transformed tracers, TRACERS_SQRT in the Fortran), tcs is TRACERS_TEND. pd is [JM, IM]
// (IDX2). dare, dsg2 and pdsg1 are per row or per level. xsums is double [2*NTR]: entry 2*(ks-1)
// is the positive sum of dsp for species ks, entry 2*(ks-1)+1 the non-positive sum (Fortran
// gsums(2*ks-1) and gsums(2*ks) before the global reduce).
//
// Loop limits: NMMB _b1 limits on the single full domain (module_DM_PARALLEL.F90 looplimits,
// ids = 1, ide = IM): its_b1 = 2, ite_b1 = IM-1, jts_b1 = 2, jte_b1 = JM-1 (1-based). Points
// outside the interior keep their tcs unchanged and contribute nothing, as in the Fortran.
//
// Thread mapping: one thread per (i, j) with the level in blockIdx.z (grid_2d, BLOCK_2D = 32 x 8).
// Each thread loops over the species ks in kss..kse for its (i, j, l) point. No thread returns
// early. On the allreduce branch every warp runs the shuffle reduction: the per-species dsp
// contributions are reduced across the 32 lanes of a warp (a warp is one row of 32 threads in x),
// and lane 0 stores the warp's partial per sign into wpart. mono_row_partials then sums each row in
// a fixed order and mono_rows_total sums the rows 1..JMG in order (allgathered over the cards first),
// so xsums is the same bits run to run and on any number of cards (2026-11-04: the old float64
// atomicAdd made the order, and the answer, vary run to run). Accumulation is in double, as the
// Fortran kdbl sums are. On the send/recv branch the warp reduction is skipped and s1 is written.
// The order differs from the Fortran serial order, so the sums are not bit-identical to native.
// That is not required (project rule, 2026-11-04). Each term is computed in the same order as the Fortran.
//
// Fast numerics: fmad contraction is allowed by the loader. The per-term float operations keep the
// Fortran order; the product (sn - s1p)*dvol is formed in float and then promoted to double.

#include "nmmb_common.cuh"

extern "C" __global__ void dyn_mono_fix(
    const int im,                          // IM
    const int jm,                          // JM
    const int lm,                          // LM
    const int idtadt,                      // IDTADT (configfile idtadt = 2 for the tracer call)
    const int kss,                         // first tracer slot, 1-based
    const int kse,                         // last tracer slot, 1-based
    const int ntr,                         // NTR (first dimension of s, s1, tcs)
    const int use_allreduce,               // 1: local sums to xsums; 0: send/recv branch, s1 = dsp
    const float* __restrict__ s,           // TRACERS [NTR, LM, JM, IM], read only
    float* __restrict__ s1,                // TRACERS_SQRT (sqrt-transformed tracers) [NTR, LM, JM, IM];
                                           // read as s1 in the allreduce branch, written as dsp in send/recv
    float* __restrict__ tcs,               // TRACERS_TEND [NTR, LM, JM, IM], interior overwritten
    const float* __restrict__ pd,          // PD [JM, IM]
    const float* __restrict__ dare,        // DARE [JM], grid box area per row
    const float* __restrict__ dsg2,        // DSG2 [LM]
    const float* __restrict__ pdsg1,       // PDSG1 [LM]
    const int joff,                        // global row of local row 0, minus 1 (0 on one card; namgpu/slab.py)
    const int jmg,                         // global JM (= jm on one card)
    const int nwx,                         // warps along x = gridDim.x (blockDim.x is 32)
    double* __restrict__ wpart)            // [kse-kss+1, 2, LM, JM, NWX] per-warp partial sums (allreduce branch)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;

    // NMMB interior limits (1-based i, j in [2, IM-1], [2, JM-1]) -> zero-based [1, IM-2], [1, JM-2].
    // The row test is on the GLOBAL row jg = j0 + joff; j0 in [1, jm-2] is the memory guard for the 3x3 stencil
    // (on one card both are the same test). Out-of-range threads still take part in the warp reduction.
    const int jg = j0 + joff;
    const bool inside = (i0 >= 1) && (i0 <= im - 2) && (jg >= 1) && (jg <= jmg - 2) && (j0 >= 1) && (j0 <= jm - 2)
                        && (l0 < lm);

    // steep = 1. - 0.040*idtadt, evaluated in single precision as the Fortran does (line 4894).
    const float steep = 1.0f - 0.040f * (float)idtadt;

    // dvol(i,j,l) = (dsg2(l)*pd(i,j) + pdsg1(l)) * dare(j)   (lines 4900-4902).
    float dvol = 0.0f;
    if (inside) {
        dvol = (dsg2[l0] * pd[IDX2(i0, j0, im)] + pdsg1[l0]) * dare[j0];
    }

    // Vertical neighbour levels used by the l-branches below (only read when inside).
    const int lm1 = (l0 > 0) ? (l0 - 1) : l0;        // bottom level uses l-1
    const int lp1 = (l0 < lm - 1) ? (l0 + 1) : l0;   // top level uses l+1

    for (int ks = kss; ks <= kse; ++ks) {
        const int k0 = ks - 1;
        double pos = 0.0;                              // contribution to xsums(2*ks-1)
        double neg = 0.0;                              // contribution to xsums(2*ks)

        if (inside && k0 >= 0 && k0 < ntr) {
            const size_t ix = IDX4(i0, j0, l0, k0, im, jm, lm);

            // s1p = (s1 + tcs)**2 ; tcs = s1p - s   (lines 4937-4939)
            const float t = s1[ix] + tcs[ix];
            const float s1p = t * t;
            tcs[ix] = s1p - s[ix];

            // Horizontal 3x3 min and max of s at level l (lines 4941-4958).
            float sminh = s[IDX4(i0 - 1, j0 - 1, l0, k0, im, jm, lm)];
            float smaxh = sminh;
#define MONO_H(ii, jj)                                                        \
            {                                                                 \
                const float v_ = s[IDX4((ii), (jj), l0, k0, im, jm, lm)];     \
                sminh = fminf(sminh, v_);                                     \
                smaxh = fmaxf(smaxh, v_);                                     \
            }
            MONO_H(i0,     j0 - 1)
            MONO_H(i0 + 1, j0 - 1)
            MONO_H(i0 - 1, j0    )
            MONO_H(i0,     j0    )
            MONO_H(i0 + 1, j0    )
            MONO_H(i0 - 1, j0 + 1)
            MONO_H(i0,     j0 + 1)
            MONO_H(i0 + 1, j0 + 1)
#undef MONO_H

            // Vertical min and max (lines 4960-4972).
            const float sc = s[ix];
            float sminv, smaxv;
            if (l0 > 0 && l0 < lm - 1) {
                const float sa = s[IDX4(i0, j0, l0 - 1, k0, im, jm, lm)];
                const float sb = s[IDX4(i0, j0, l0 + 1, k0, im, jm, lm)];
                sminv = fminf(fminf(sa, sc), sb);
                smaxv = fmaxf(fmaxf(sa, sc), sb);
            } else if (l0 == 0) {
                const float sb = s[IDX4(i0, j0, lp1, k0, im, jm, lm)];
                sminv = fminf(sc, sb);
                smaxv = fmaxf(sc, sb);
            } else {
                const float sa = s[IDX4(i0, j0, lm1, k0, im, jm, lm)];
                sminv = fminf(sa, sc);
                smaxv = fmaxf(sa, sc);
            }

            // smin = min(sminh, sminv), smax = max(smaxh, smaxv)   (lines 4974-4975)
            const float smin = fminf(sminh, sminv);
            const float smax = fmaxf(smaxh, smaxv);

            // Limiter (lines 4977-4980): sn = s1p, clipped above at steep*smax, then below at smin.
            float sn = s1p;
            if (sn > steep * smax) sn = smax;
            if (sn < smin) sn = smin;

            // dsp = (sn - s1p)*dvol (line 4982). The product is formed in float, then held in double.
            const double dsp = (double)((sn - s1p) * dvol);

            if (use_allreduce) {
                // Positive and non-positive sums (lines 4984-4990).
                if (dsp > 0.0) pos = dsp;
                else           neg = dsp;
            } else {
                // send/recv branch of MONO: s1 = dsp, kfpt store.
                s1[ix] = (float)dsp;
            }
        }

        if (use_allreduce) {
            // Warp reduction along x. Every lane of the warp reaches this point, so the full-mask
            // shuffles are safe. The branch is uniform across the grid (use_allreduce is a scalar).
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                pos += __shfl_down_sync(0xffffffffu, pos, off);
                neg += __shfl_down_sync(0xffffffffu, neg, off);
            }
            // Deterministic: lane 0 stores the warp's partials (no atomics; E1 2026-11-04 defect fix, the old
            // float64 atomicAdd made the sum order, and so the answer, vary run to run). Every warp of the grid
            // stores, zeros included, so the buffer needs no zeroing.
            if ((threadIdx.x & 31) == 0 && k0 >= 0 && k0 < ntr && j0 < jm) {
                const int kk = ks - kss;
                const size_t base = (((size_t)(kk * 2) * lm + l0) * jm + j0) * nwx + blockIdx.x;
                wpart[base] = pos;
                wpart[base + (size_t)lm * jm * nwx] = neg;
            }
        }
    }
}

// One block per local row j: the row's partial sums P[j][2*k0 + sign] over (l, warp) in a fixed strided order and a
// fixed shared-memory tree. A row's value depends only on that row's points, so it is the same on any card that owns
// the row. Entries outside kss..kse are written 0.
extern "C" __global__ void mono_row_partials(
    const int jm, const int lm, const int nwx, const int kss, const int kse, const int ntr,
    const double* __restrict__ wpart,      // [kse-kss+1, 2, LM, JM, NWX]
    double* __restrict__ rowp)             // [JM, 2*NTR]
{
    const int j0 = blockIdx.x;
    const int tid = threadIdx.x;           // blockDim.x == 256
    __shared__ double sh[256];
    const int n = lm * nwx;
    for (int e = 0; e < 2 * ntr; ++e) {
        const int k0 = e >> 1, sgn = e & 1;
        const int ks = k0 + 1;
        if (ks < kss || ks > kse) {
            if (tid == 0) rowp[(size_t)j0 * 2 * ntr + e] = 0.0;
            continue;
        }
        const int kk = ks - kss;
        const double* src = wpart + ((size_t)(kk * 2 + sgn) * lm) * jm * nwx;
        double acc = 0.0;
        for (int t = tid; t < n; t += 256) {
            const int l0 = t / nwx, w = t - l0 * nwx;
            acc += src[((size_t)l0 * jm + j0) * nwx + w];
        }
        sh[tid] = acc;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        if (tid == 0) rowp[(size_t)j0 * 2 * ntr + e] = sh[0];
        __syncthreads();
    }
}

// xsums[e] = sum over the GLOBAL rows in order j = 1..JMG of P[j][e]. growp holds the ranks' owned-row blocks
// [nranks][nmax][2*NTR] (rank r's first nown[r] rows valid), ranks in row order, so this visits rows 1..JMG in order on
// any number of cards. One card: nranks 1, nmax = nown[0] = JM, growp = rowp.
extern "C" __global__ void mono_rows_total(
    const int nranks, const int nmax, const int* __restrict__ nown, const int ntr, const int kss, const int kse,
    const double* __restrict__ growp, double* __restrict__ xsums)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= 2 * ntr) return;
    const int ks = (e >> 1) + 1;
    double acc = 0.0;
    if (ks >= kss && ks <= kse) {
        for (int r = 0; r < nranks; ++r) {
            const double* blk = growp + (size_t)r * nmax * 2 * ntr;
            for (int q = 0; q < nown[r]; ++q) acc += blk[(size_t)q * 2 * ntr + e];
        }
    }
    xsums[e] = acc;
}
