// drv_dfi_dyn_average.cu -- DIGITAL_FILTER_DYN_AVERAGE: filtered state from the accumulated sums.
//
// NOTICE: NOAA/NCEP NMMB source, nmm/module_DIGITAL_FILTER_NMM.F90 (US Government work, no licence
// file in the NAM clone; see NOTICE at the repo root). Routine kept as the notice:
//   SUBROUTINE DIGITAL_FILTER_DYN_AVERAGE_NMM (module_DIGITAL_FILTER_NMM, nam.v4.2.11).
// Ported to CUDA for nam-gpu from nmm/module_DIGITAL_FILTER_NMM.F90:388-646 (nam.v4.2.11, 83df4e13c1).
//
// What the routine does (Fortran order kept):
//   1. TOTALSUMI = 1.0 / TOTALSUM                                      (line 439)
//   2. scale every summed field by TOTALSUMI in place:
//        2-D  ARRAY_SAVE_2D(:,:,N)                  N = 1..TOT_RANK_2D           (441-449)
//        3-D  ARRAY_SAVE_3D(:,:,L,N)  L = 1..LDIM, LDIM = LM+1 for PINT, else LM (451-469)
//        4-D  ARRAY_SAVE_4D(:,:,L,P,N) P = 1..NUM_SPEC, L = 1..LM                (471-489)
//        NUM_SPEC = NUM_TRACERS for TRACERS, NUM_WATER for WATER.
//   3. copy the averaged sums into the filter-bundle fields, looked up by name:
//        2-D (491-529), 3-D with LDIM = level count of the bundle field (531-575),
//        4-D with NUM_SPEC species and all LM levels (577-625).
//   4. deallocate the sum arrays and reset TOT_RANK_* / KSTEP / NSTEP to 0 (627-644).
//      Those are Fortran allocation and bookkeeping. The device sum arrays are CONTRACT scratch
//      owned by drv-dfi-dyn-init, so they are kept and reused, not freed here.
//
// Mapping (CONTRACT section 1): Fortran A(i,j,l,n) -> A[n, l, j, i] float32 with IDX2/IDX3/IDX4
// from nmmb_common.cuh. Zero-based: i0 = i-1, j0 = j-1, l0 = l-1, p0 = p-1, n0 = n-1.
// Scratch layout the wrapper expects (dyn-init and dyn-sum must match):
//   DFI_SAVE_2D [N2, JM, IM], DFI_SAVE_3D [N3, LM+1, JM, IM], DFI_SAVE_4D [N4, SPEC_MAX, LM, JM, IM].
//
// Halo exchanges (HALO_EXCH in the 2-D and 3-D copy sections; the 4-D one is commented out in the
// source) are dropped: single card, whole domain on one device (CONTRACT section 1). The loop
// limits ITS..ITE, JTS..JTE are the full domain 1..IM, 1..JM; the routine has no _b1 trimming.
// The ESMF calls (ESMF_FieldBundleGet, ESMF_FieldGet, ESMF_LogWrite, ERR_MSG) only locate the
// field pointer in the bundle. The wrapper does that lookup by name, so they have no kernel term.
//
// Single precision throughout (KFPT = single). Each scaled element is one multiply, the same
// rounding as the Fortran TOTALSUMI*ARRAY_SAVE(...). FMA contraction cannot change a single
// multiply, so fmad stays on.
//
// Launch convention: one kernel per routine section. A 2-D section is one launch over all fields
// (z = field index). The 3-D and 4-D sections are launched once per field, because LDIM and
// NUM_SPEC depend on the field name. Thread mapping is grid_2d: one thread per (i, j), with the
// level (or field) index in blockIdx.z. No host sync, no allocation, no printf.

#include "nmmb_common.cuh"

// Section 2, 2-D: scale all TOT_RANK_2D summed fields (blockIdx.z = field index N0).
extern "C" __global__ void dfi_scale2d(
    const int im,                               // IM
    const int jm,                               // JM
    const float totalsumi,                      // 1 / TOTALSUM
    float* __restrict__ save2d)                 // [N2, JM, IM]
{
    POINT_IJL                                   // l0 holds the field index here
    if (i0 >= im || j0 >= jm) return;
    const size_t ix = IDX3(i0, j0, l0, im, jm); // IDX3 with the field index as its third index
    save2d[ix] = totalsumi * save2d[ix];        // ARRAY_SAVE_2D(I,J,N) = TOTALSUMI*ARRAY_SAVE_2D(I,J,N)
}

// Section 2, 3-D: scale one field over its LDIM levels. Launched once per field.
extern "C" __global__ void dfi_scale3d(
    const int im,                               // IM
    const int jm,                               // JM
    const int ldim,                             // LM+1 for PINT, LM otherwise
    const float totalsumi,                      // 1 / TOTALSUM
    float* __restrict__ save3d_n)               // [LM+1, JM, IM], the field's slice of [N3, LM+1, JM, IM]
{
    POINT_IJL                                   // l0 = level
    if (i0 >= im || j0 >= jm || l0 >= ldim) return;
    const size_t ix = IDX3(i0, j0, l0, im, jm);
    save3d_n[ix] = totalsumi * save3d_n[ix];    // ARRAY_SAVE_3D(I,J,L,N) = TOTALSUMI*...
}

// Section 2, 4-D: scale one field over NUM_SPEC species and all LM levels. Launched once per field.
extern "C" __global__ void dfi_scale4d(
    const int im,                               // IM
    const int jm,                               // JM
    const int lm,                               // LM
    const int nspec,                            // NUM_SPEC (NUM_TRACERS or NUM_WATER)
    const float totalsumi,                      // 1 / TOTALSUM
    float* __restrict__ save4d_n)               // [SPEC_MAX, LM, JM, IM], the field's slice
{
    POINT_IJL                                   // l0 = level
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    for (int p0 = 0; p0 < nspec; ++p0) {        // P = 1..NUM_SPEC
        const size_t ix = IDX4(i0, j0, l0, p0, im, jm, lm);
        save4d_n[ix] = totalsumi * save4d_n[ix]; // ARRAY_SAVE_4D(I,J,L,P,N) = TOTALSUMI*...
    }
}

// Section 3, 2-D copy into the bundle field (one launch per field).
extern "C" __global__ void dfi_copy2d(
    const int im,                               // IM
    const int jm,                               // JM
    const float* __restrict__ src,              // [JM, IM] one slice of the averaged sums
    float* __restrict__ dst)                    // [JM, IM] filter-bundle field
{
    POINT_IJL
    if (i0 >= im || j0 >= jm) return;
    dst[IDX2(i0, j0, im)] = src[IDX2(i0, j0, im)];   // HOLD_2D(I,J) = ARRAY_SAVE_2D(I,J,N)
}

// Section 3, 3-D copy. ldim = destination level count (size(HOLD_3D,3)); src has LM+1 levels.
extern "C" __global__ void dfi_copy3d(
    const int im,                               // IM
    const int jm,                               // JM
    const int ldim,                             // destination level count (LM or LM+1)
    const float* __restrict__ src,              // [LM+1, JM, IM] one slice of the averaged sums
    float* __restrict__ dst)                    // [ldim, JM, IM] filter-bundle field
{
    POINT_IJL                                   // l0 = level
    if (i0 >= im || j0 >= jm || l0 >= ldim) return;
    const size_t ix = IDX3(i0, j0, l0, im, jm);
    dst[ix] = src[ix];                          // HOLD_3D(I,J,L) = ARRAY_SAVE_3D(I,J,L,N)
}

// Section 3, 4-D copy: NUM_SPEC species, all LM levels.
extern "C" __global__ void dfi_copy4d(
    const int im,                               // IM
    const int jm,                               // JM
    const int lm,                               // LM
    const int nspec,                            // NUM_SPEC
    const float* __restrict__ src,              // [SPEC_MAX, LM, JM, IM] one slice of the sums
    float* __restrict__ dst)                    // [NT, LM, JM, IM] filter-bundle field
{
    POINT_IJL                                   // l0 = level
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    for (int p0 = 0; p0 < nspec; ++p0) {        // P = 1..NUM_SPEC
        const size_t ix = IDX4(i0, j0, l0, p0, im, jm, lm);
        dst[ix] = src[ix];                      // HOLD_4D(I,J,L,P) = ARRAY_SAVE_4D(I,J,L,P,N)
    }
}
