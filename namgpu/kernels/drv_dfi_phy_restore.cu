// nam-gpu unit drv-dfi-phy-restore: DIGITAL_FILTER_PHY_RESTORE_NMM, copy the physics state that was
// saved before the digital filter (drv-dfi-phy-save) back into the physics fields after DFI.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone). Author headers
// of MODULE module_DIGITAL_FILTER_NMM (nmm/module_DIGITAL_FILTER_NMM.F90, NCEP/NOAA, nam.v4.2.11), kept as
// the notice:
//   a generic digital filter for any model under ESMF
//   March 2007 Hann-Ming Henry Juang; Feb 2008-Sep 2011 Weiyu Yang; July 2012 T Black
// Ported to CUDA for nam-gpu from nmm/module_DIGITAL_FILTER_NMM.F90:829-921 (nam.v4.2.11, 83df4e13c1).
//
// Fortran (DIGITAL_FILTER_PHY_RESTORE_NMM):
//   DO N=1,TOT_RANK_2D_PHYS: HOLD_2D(I,J) = ARRAY_SAVE_2D_PHYS(I,J,N)  for J=JTS..JTE, I=ITS..ITE
//   DO N=1,TOT_RANK_3D_PHYS: LDIM = SIZE(HOLD_3D,3); HOLD_3D(I,J,L) = ARRAY_SAVE_3D_PHYS(I,J,L,N)
//                             for L=1..LDIM, J, I
// HOLD_2D/HOLD_3D are the ESMF field arrays of the N-th physics field of FILT_BUNDLE (named by
// NAME_SAVE_2D/3D_PHYS(N)); in nam-gpu they are the device arrays of ctx.st, passed here as a table of
// device pointers (one entry per field, built once by the wrapper).
// The Fortran HALO_EXCH calls are dropped (single card, the whole domain is one tile, CONTRACT.md 1).
// Loop limits ITS..ITE, JTS..JTE are the full domain 1..IM, 1..JM (zero-based 0..IM-1, 0..JM-1).
// The 3-D field length LDIM is the first dimension of each device array (soil fields carry NSOIL=4
// layers, CW carries LM), exactly as the Fortran SIZE(HOLD_3D,3).
//
// Layouts (CONTRACT.md 1): save2d [N2][JM][IM]; save3d [N3][LM][JM][IM]; field n is [LDIM][JM][IM]
// (2-D fields [JM][IM]).  One launch per section: all 2-D fields, then all 3-D fields.

#include "nmmb_common.cuh"

// grid: (IM/bx, JM/by, N2); blockIdx.z = field index n.
extern "C" __global__ void dfi_phy_restore_2d(const float* __restrict__ save2d,
                                              float* const* dst,
                                              const int im, const int jm, const int n2)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int n  = blockIdx.z;
    if (n >= n2 || i0 >= im || j0 >= jm) return;
    float* d = dst[n];
    d[IDX2(i0, j0, im)] = save2d[((size_t)n * jm * im) + IDX2(i0, j0, im)];
}

// grid: (IM/bx, JM/by, N3*LM); blockIdx.z = n*LM + l.  ldim[n] is the device-field level count (LDIM).
extern "C" __global__ void dfi_phy_restore_3d(const float* __restrict__ save3d,
                                              float* const* dst,
                                              const int* __restrict__ ldim,
                                              const int im, const int jm, const int lm, const int n3)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int z  = blockIdx.z;
    const int n  = z / lm;
    const int l0 = z - n * lm;
    if (n >= n3 || i0 >= im || j0 >= jm) return;
    if (l0 >= ldim[n]) return;
    float* d = dst[n];
    d[IDX3(i0, j0, l0, im, jm)] = save3d[IDX4(i0, j0, l0, n, im, jm, lm)];
}
