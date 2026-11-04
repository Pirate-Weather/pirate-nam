// WOOF-NAM unit drv-dfi-dyn-sum: DIGITAL_FILTER_DYN_SUM, add the Dolph-weighted state each step.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
// Module header kept as the notice:
//   module module_DIGITAL_FILTER_NMM  (NMMB digital filter, nam.v4.2.11)
//   a generic digital filter for any model under ESMF
//   March 2007 Hann-Ming Henry Juang; Feb 2008 - Jul 2012 Weiyu Yang, T Black (NCEP/EMC)
// Ported to CUDA for WOOF-NAM from nmm/module_DIGITAL_FILTER_NMM.F90:208-383 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (digital_filter_dyn_sum_nmm), dolph branch (dolph = .true. in the routine):
//   kstep = kstep + 1                                  (host, in the wrapper)
//   digfil = dolph_wgts(kstep)                         (device table, zero-based idx = kstep + nstep)
//   totalsum = totalsum + digfil                       [dfi_dyn_sum_total]
//   2-D:  array_save_2d(i,j,n) += digfil * hold_2d(i,j)                 over i=its..ite, j=jts..jte
//   3-D:  array_save_3d(i,j,l,n) += digfil * hold_3d(i,j,l)             l = 1..size(hold_3d,3)
//         (size = LM for T,U,V,Q2,DWDT; LM+1 for PINT; the save array holds LM+1 levels)
//   4-D:  array_save_4d(i,j,l,p,n) += digfil * hold_4d(i,j,l,p)         l=1..lm, p=1..num_spec
//         num_spec = num_tracers for 'TRACERS', num_water for 'WATER' (wrapper resolves it)
// The non-dolph branch (sin/sinc weights, mean_on) is dead code in the routine (dolph is hard-set
// .true.); the wrapper computes its digfil on the host and writes it into the same table slot.
// Every field is a full-domain loop (its = 1..IM, jts = 1..JM for one card): no b1 limits.
//
// Launches: one kernel per field (the field pointers differ, so the wrapper passes one field at a
// time); one 1-thread kernel for totalsum. No host sync. The digfil value is read on the device
// from the table, so the step never waits for the host.
//
// Layout (CONTRACT.md section 1): 2-D [JM,IM]; 3-D [LEV,JM,IM]; 4-D [NSPEC,LM,JM,IM];
// save arrays [n,JM,IM], [n,LM+1,JM,IM], [n,SPEC_MAX,LM,JM,IM].

#include "nmmb_common.cuh"

extern "C" __global__ void dfi_dyn_sum_2d(
    const float* __restrict__ src,          // [JM,IM]  hold_2d
    float* __restrict__ sum,                // [JM,IM]  array_save_2d(:,:,n)
    const float* __restrict__ digfil_tab,   // digfil table (dolph_wgts or the host-filled slot)
    const int idx,                          // zero-based index of digfil for this step
    const int IM, const int JM)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const float digfil = digfil_tab[idx];
    const size_t k = IDX2(i0, j0, IM);
    sum[k] = sum[k] + digfil * src[k];
}

extern "C" __global__ void dfi_dyn_sum_3d(
    const float* __restrict__ src,          // [LDIM,JM,IM]  hold_3d, LDIM = size(hold_3d,3)
    float* __restrict__ sum,                // [LSUM,JM,IM]  array_save_3d(:,:,:,n), LSUM >= LDIM
    const float* __restrict__ digfil_tab,
    const int idx,
    const int IM, const int JM, const int LDIM)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || j0 >= JM || l0 >= LDIM) return;
    const float digfil = digfil_tab[idx];
    const size_t k = IDX3(i0, j0, l0, IM, JM);
    sum[k] = sum[k] + digfil * src[k];
}

extern "C" __global__ void dfi_dyn_sum_4d(
    const float* __restrict__ src,          // [NSPEC,LM,JM,IM]  hold_4d
    float* __restrict__ sum,                // [SPEC_MAX,LM,JM,IM]  array_save_4d(:,:,:,:,n)
    const float* __restrict__ digfil_tab,
    const int idx,
    const int IM, const int JM, const int LM, const int NSPEC)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const float digfil = digfil_tab[idx];
    for (int p0 = 0; p0 < NSPEC; ++p0) {
        const size_t k = IDX4(i0, j0, l0, p0, IM, JM, LM);
        sum[k] = sum[k] + digfil * src[k];
    }
}

// totalsum = totalsum + digfil  (kfpt scalar, one thread, device-resident)
extern "C" __global__ void dfi_dyn_sum_total(
    const float* __restrict__ digfil_tab,
    const int idx,
    float* __restrict__ totalsum)           // [1]
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        totalsum[0] = totalsum[0] + digfil_tab[idx];
    }
}

// Non-dolph digfil (Fortran 272-285, dead while dolph is hard-set .true.), computed on the device so
// the step never copies from the host. Single precision. kstep and nstep are host ints passed as args.
//   digfil = sin(wx)/wx*sin(sx)/sx, wx = pi*kstep/(nstep+1), sx = pi*kstep/nstep, digfil = 1 if kstep==0
//   digfil = 1 if mean_on > 0
extern "C" __global__ void dfi_digfil_nondolph(
    float* __restrict__ digfil_out,         // [1]
    const int kstep,
    const int nstep,
    const int mean_on)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float pi = acosf(-1.0f);
        float digfil = 1.0f;
        if (kstep != 0) {
            const float sx = pi * (float)kstep / (float)nstep;
            const float wx = pi * (float)kstep / (float)(nstep + 1);
            digfil = sinf(wx) / wx * sinf(sx) / sx;
        }
        if (mean_on > 0) digfil = 1.0f;
        digfil_out[0] = digfil;
    }
}
