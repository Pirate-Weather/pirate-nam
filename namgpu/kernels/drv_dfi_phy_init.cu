// nam-gpu unit drv-dfi-phy-init: DIGITAL_FILTER_PHY_INIT_NMM, allocate and zero the physics save
// arrays of the digital filter (array_save_2d_phys, array_save_3d_phys) once, on the first filter call.
//
// NOTICE: NOAA/NCEP NMMB source (US government work, no licence file in the NAM clone).
//   nmm/module_DIGITAL_FILTER_NMM.F90, SUBROUTINE DIGITAL_FILTER_PHY_INIT_NMM (module header authors
//   Weiyu Yang, 2008-2011, kept as the notice).
// Ported to CUDA for nam-gpu from nmm/module_DIGITAL_FILTER_NMM.F90:650-736 (nam.v4.2.11, 83df4e13c1).
//
// Fortran semantics (650-736) and where the port does each part:
//   - Bundle walk and rank counting (FieldBundleGet, FieldGet dimCount, TOT_RANK_2D/3D_PHYS): the wrapper
//     namgpu/steps/drv_dfi_phy_init.py takes the restore bundle (drv-dfi-phy-restore.restore_names) and
//     each field's rank from the state. The counts are the array extents (n2, n3) passed to the kernels.
//   - NAME_SAVE_2D/3D_PHYS recording (Fortran 685-707): the wrapper records the slot-to-name lists in
//     ctx.tables["dfi_phy_init"]; drv-dfi-phy-restore checks its own slot names against that record.
//     The ESMF field-name read (Fortran 697-698, PHY_NAME) is replaced by the table name, because the
//     bundle is built from that same table.
//   - ALLOCATE array_save_2d_phys(ITS:ITE,JTS:JTE,n2) and array_save_3d_phys(ITS:ITE,JTS:JTE,LM,n3),
//     then `=0.`, per array and only if that array is absent from the state and its count is positive
//     (Fortran 712 and 722, each gated separately). The wrapper launches these kernels for that case.
//   - Single card, full domain: ITS..ITE x JTS..JTE is the whole 1..IM x 1..JM (CONTRACT section 1).
// Dropped without a GPU counterpart (stated, not silent):
//   - Fortran 714-718 and 724-728 allocation-failure branches (write, ESMF_FINALIZE abort). CuPy raises
//     on out-of-memory at the allocation, so the abort branch has no separate device path.
//   - Fortran 732 deallocate(phy_name): host bookkeeping for the name buffer; there is no name buffer.
// Layout (CONTRACT section 1): save2d[n, j, i] shape [n2, JM, IM]; save3d[n, l, j, i] shape [n3, LM, JM, IM].
// The zeroing is one launch per array: zero_save_2d over (i, j, field), zero_save_3d over (i, j, level)
// with the fixed field loop inside the thread. No host sync, no device malloc, no printf.

#include "nmmb_common.cuh"

extern "C" __global__ void zero_save_2d(const int im, const int jm, const int n2,
                                        float* __restrict__ save2d)
{
    // grid_2d(im, jm, n2): blockIdx.z runs over the 2-D save field index (not a level here).
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    const int n0 = blockIdx.z;
    if (i0 >= im || j0 >= jm || n0 >= n2) return;
    save2d[IDX3(i0, j0, n0, im, jm)] = 0.0f;   /* array_save_2d_phys = 0. */
}

extern "C" __global__ void zero_save_3d(const int im, const int jm, const int lm, const int n3,
                                        float* __restrict__ save3d)
{
    POINT_IJL                                  /* grid_2d(im, jm, lm): l0 = model level */
    if (i0 >= im || j0 >= jm || l0 >= lm) return;
    for (int n = 0; n < n3; ++n) {
        save3d[IDX4(i0, j0, l0, n, im, jm, lm)] = 0.0f;   /* array_save_3d_phys = 0. */
    }
}
