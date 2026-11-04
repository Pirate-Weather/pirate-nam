// nam-gpu unit dyn-mono-global-sum: global mass sums per species and the conservation rescale.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   module module_dynamics_routines  (NMMB dynamics, nam.v4.2.11)
// Ported to CUDA for nam-gpu from nmm/module_DYNAMICS_ROUTINES.F90:5076-5203 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (subroutine mono, active branch use_allreduce = .true. in nam_configfile):
//   [G1] gsums(2ks-1) = xsums(2ks-1), gsums(2ks) = xsums(2ks)        (mpi_allreduce, identity on one card)
//   [F]  per species: sumps = gsums(2ks-1), sumns = gsums(2ks);
//        if sumps*(-sumns) > 1: sfacs = -sumns/sumps, rfacs = 1/sfacs, else sfacs = rfacs = 0
//   [R]  per species, l=1..lm, b1 range:
//        dvol  = (dsg2(l)*pd(i,j)+pdsg1(l))*dare(j)          (kfpt, recomputed as in the top of MONO)
//        rdvol = 1/dvol                                       (kfpt)
//        dsp   = s1(i,j,l,ks)*rdvol                           (dsp is kdbl in the routine)
//        if sfacs < 1: if dsp > 0: dsp = dsp*sfacs; else: if dsp < 0: dsp = dsp*rfacs
//        tcs(i,j,l,ks) = tcs(i,j,l,ks) + dsp                 (stored to kfpt)
//   use_allreduce=.false. (send/recv, Fortran 5076-5203 global_reduce else arm): S1L_SUM(i,j,ks) is the sum over l
//   of dsp = s1 written by dyn-mono-fix, and gsums takes the columns by sign. On one card the send/recv
//   reduction is the same serial sum over the interior columns, formed here in one block per species.
//
// Index ranges (module_DM_PARALLEL.F90 looplimits, one card, regional): its_b1 = 2 .. IM-1 and
// jts_b1 = 2 .. JM-1 (1-based)  ->  i0 in [1, IM-2], j0 in [1, JM-2].  The MPI/OpenMP split is dropped.
//
// Two launches, both on the stream, no host sync:
//   mono_global_sum_reduce  grid = nspec blocks (one block per species). Thread 0 of each block copies the
//                           allreduced sums into gsums (double), or on the send/recv branch reduces the column
//                           sums from s1 into gsums, then forms sfacs/rfacs.
//   mono_global_sum_rescale grid_2d over (i, j, l), species loop inside the thread.
// Species index k0 = ks-1 (zero-based). Layout: s1/tcs [NTR,LM,JM,IM] (IDX4), xsums/gsums/sfac double [2*NTR]
// with the positive sum at 2*k0 and the negative sum at 2*k0+1. xsums is written by dyn-mono-fix (MONO_XSUMS,
// the same buffer); s1 is TRACERS_SQRT from dyn-mono-fix; dare [JM] is the grid row constant of init-grid-consts.

#include "nmmb_common.cuh"

extern "C" __global__ void mono_global_sum_reduce(
    const int kss, const int kse, const int IM, const int JM, const int LM,
    const int use_allreduce,              // 1: gsums = xsums (identity on one card); 0: send/recv sums
    const double* __restrict__ xsums,     // [2*NTR]   Fortran xsums (allreduce branch only)
    const float* __restrict__ s1,         // [NTR,LM,JM,IM] dsp from dyn-mono-fix (send/recv branch only)
    double* __restrict__ gsums,           // [2*NTR]   Fortran gsums(1:2*kse), output
    double* __restrict__ sfac)            // [2*NTR]   sfac[2k] = sfacs, sfac[2k+1] = rfacs
{
    const int k0 = (kss - 1) + (int)blockIdx.x;      // zero-based species, ks = k0+1
    const int tid = threadIdx.x;
    const int nwarp = blockDim.x >> 5;

    __shared__ double sh_pos[32];
    __shared__ double sh_neg[32];

    if (use_allreduce) {
        // [G1] the allreduce is the identity on one card: gsums = xsums
        if (tid == 0) {
            gsums[2 * k0]     = xsums[2 * k0];
            gsums[2 * k0 + 1] = xsums[2 * k0 + 1];
        }
    } else {
        // [S] send/recv branch, one card (Fortran 4997-5072 and the mpi_bcast of gsums):
        // s1l_sum(i,j) = sum over l of dsp (kfpt accumulation, as the Fortran s1l_sum is kfpt),
        // then gsums(2ks-1) takes the columns with s1l_sum > 0 and gsums(2ks) the rest (Fortran 5160-5172).
        // Interior columns only: i in [1, IM-2], j in [1, JM-2] (zero-based).
        const int ncol = (IM - 2) * (JM - 2);
        double pos = 0.0;
        double neg = 0.0;
        for (int c = tid; c < ncol; c += blockDim.x) {
            const int i0 = 1 + c % (IM - 2);
            const int j0 = 1 + c / (IM - 2);
            float cs = 0.0f;
            for (int l0 = 0; l0 < LM; ++l0) {
                cs = (float)((double)cs + (double)s1[IDX4(i0, j0, l0, k0, IM, JM, LM)]);
            }
            if (cs > 0.0f) pos += (double)cs;
            else           neg += (double)cs;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            pos += __shfl_down_sync(0xffffffffu, pos, off);
            neg += __shfl_down_sync(0xffffffffu, neg, off);
        }
        if ((tid & 31) == 0) {
            sh_pos[tid >> 5] = pos;
            sh_neg[tid >> 5] = neg;
        }
        __syncthreads();
        if (tid == 0) {
            double gp = 0.0;
            double gn = 0.0;
            for (int w = 0; w < nwarp; ++w) {
                gp += sh_pos[w];
                gn += sh_neg[w];
            }
            gsums[2 * k0]     = gp;
            gsums[2 * k0 + 1] = gn;
        }
    }
    __syncthreads();

    // [F] scale factors for this species (thread 0 only; the sums are already on the device)
    if (tid == 0) {
        const double sumps = gsums[2 * k0];
        const double sumns = gsums[2 * k0 + 1];
        double sfacs = 0.0, rfacs = 0.0;
        if (sumps * (-sumns) > 1.0) {
            sfacs = -sumns / sumps;
            rfacs = 1.0 / sfacs;
        }
        sfac[2 * k0]     = sfacs;
        sfac[2 * k0 + 1] = rfacs;
    }
}

extern "C" __global__ void mono_global_sum_rescale(
    const int kss, const int kse, const int IM, const int JM, const int LM,
    const float* __restrict__ pd,         // [JM,IM]   PD
    const float* __restrict__ dsg2,       // [LM]      DSG2
    const float* __restrict__ pdsg1,      // [LM]      PDSG1
    const float* __restrict__ dare,       // [JM]      DARE (grid box area, row constant)
    const float* __restrict__ s1,         // [NTR,LM,JM,IM] S1 (TRACERS_SQRT) from dyn-mono-fix
    const double* __restrict__ sfac,      // [2*NTR]   from mono_global_sum_reduce
    float* __restrict__ tcs,              // [NTR,LM,JM,IM] TRACERS_TEND, inout
    const int joff,                       // global row of local row 0, minus 1 (0 on one card; namgpu/slab.py)
    const int JMG)                        // global JM (= JM on one card)
{
    POINT_IJL                              // l0 = blockIdx.z
    // b1 range on the GLOBAL row; j0 < JM is the memory guard (one card: the same test as before)
    const int jg = j0 + joff;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JMG - 2 || j0 >= JM) return;

    // [R] rdvol as the top of MONO computes it (all kfpt)
    const float dvol = (dsg2[l0] * pd[IDX2(i0, j0, IM)] + pdsg1[l0]) * dare[j0];
    const float rdvol = 1.0f / dvol;

    for (int k0 = kss - 1; k0 <= kse - 1; ++k0) {
        const double sfacs = sfac[2 * k0];
        const double rfacs = sfac[2 * k0 + 1];
        const size_t n4 = IDX4(i0, j0, l0, k0, IM, JM, LM);
        double dsp = (double)(s1[n4] * rdvol);
        if (sfacs < 1.0) {
            if (dsp > 0.0) dsp = dsp * sfacs;
        } else {
            if (dsp < 0.0) dsp = dsp * rfacs;
        }
        tcs[n4] = (float)((double)tcs[n4] + dsp);
    }
}
