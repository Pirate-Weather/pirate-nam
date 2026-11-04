// nam-gpu unit drv-dfi-phy-save: save the physics ('R' restore) state before the digital filter.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   module_DIGITAL_FILTER_NMM.F90, module header (Hann-Ming Henry Juang, Weiyu Yang, T Black;
//   author headers kept as the notice). Routine DIGITAL_FILTER_PHY_SAVE_NMM.
// Ported to CUDA for nam-gpu from nmm/module_DIGITAL_FILTER_NMM.F90:741-824 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (for every 2-D field n: array_save_2d_phys(i,j,n) = hold_2d(i,j), and for every 3-D field n:
//   array_save_3d_phys(i,j,l,n) = hold_3d(i,j,l) for l = 1..size(hold_3d,3)), with ITS..ITE x JTS..JTE.
// On one card the tile is the whole domain (CONTRACT.md section 1). The ESMF bundle lookup
// (ESMF_FieldBundleGet / ESMF_FieldGet farrayPtr) becomes the device pointer table built by the
// wrapper: src_tab[f] is field f (2-D [JM,IM] or 3-D [NL,JM,IM] with NL = its leading extent),
// dst_tab[f] is its save array of the same shape. Both are contract names (ctx.st).
// The copy is one kernel launch for all fields: blockIdx.z = f * lmax + l0, lmax = largest NL.
// Fields with NL < lmax (2-D: NL = 1) stop at their own level count (nlev[f]).
// No Fortran term is dropped: the bundle loop over n (2-D then 3-D), the j/i loops and the l loop
// of the 3-D branch all have their counterpart here. MPI/halo/ESMF handles and the rc checks are
// dropped (single process, no halo exchange, no error path on the device).

#include "nmmb_common.cuh"

extern "C" __global__ void drv_dfi_phy_save(const int im, const int jm, const int nf, const int lmax,
                                            const unsigned long long* __restrict__ src_tab,
                                            const unsigned long long* __restrict__ dst_tab,
                                            const int* __restrict__ nlev)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int f = blockIdx.z / lmax;
    const int l0 = blockIdx.z - f * lmax;
    if (f >= nf || i0 >= im || j0 >= jm || l0 >= nlev[f]) return;

    const float* __restrict__ src = (const float*)src_tab[f];
    float* __restrict__ dst = (float*)dst_tab[f];
    const size_t n = IDX3(i0, j0, l0, im, jm);
    dst[n] = src[n];
}
