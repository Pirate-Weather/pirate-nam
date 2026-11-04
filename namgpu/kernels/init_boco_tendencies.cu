// init_boco_tendencies.cu -- WOOF-NAM unit init-boco-tendencies (CONTRACT.md sections 3 and 6).
//
// Ported to CUDA for WOOF-NAM from nmm/module_SOLVER_GRID_COMP.F90:8155-9298 (UPDATE_BC_TENDS,
// nam.v4.2.11, 83df4e13c1). NOAA/NCEP NMMB source: US government work, the NOAA author headers
// are kept as notices (NOTICE).
//
// UPDATE_BC_TENDS, for every boundary strip (south, north, west, east), every H and V class and
// every variable/level/species, writes the time tendency for the interval that starts now:
//     BND(slot 2)_Fortran = (BND_DATA_future - BND(slot 1)_Fortran) * RECIP
//     RECIP = 1 / (DT * PARENT_CHILD_TIME_RATIO)   (= 1 / TBOCO)
// In zero-based slots: buf[0] = value_now, the carried Fortran slot 1 (advanced every step by
// bnd-bocoh-update, module_FLTBNDS.F90:5176-5180; not refreshed from the state edges here),
// buf[1] = tendency. The kernel only reads slot 0 and writes slot 1, so no element is read after
// it is written. The Fortran's per-side, per-class extraction loops are collapsed into one pointer
// table: each entry is one boundary strip of one variable, and one launch covers every entry.
//
// MOVE_NOW (UPDATE_BC_TENDS move_now_* blocks, SOLVER 8200-9298) is not implemented, and that is
// correct for NAM: the move_now branches run only when the nest moved at the start of this step.
// parm/nam_configfile sets my_domain_moves: false, and the NAM 12 km parent is not a nest.
// Nothing here can reach them. A moving-nest config must add those blocks before it is run.
//
// Loop limits: H strips run over every along index (I2_H = IDE, J2_H = JDE). V strips stop one
// point short of the domain edge (I2_V = IDE-1, J2_V = JDE-1, Fortran UPDATE_BC_TENDS), so the
// last along index of a V entry is not part of the Fortran KOUNT loops and its tendency slot is
// left unchanged (vmask[e] = 1 for V entries).
//
// Buffer layout (namgpu/init/init_boco_tendencies.py, matching namgpu/boundary/bnd_write_bc.py):
//     buf[e] holds 2 * n_e floats: slot 0 = carried value_now, slot 1 = tendency,
//     each slot laid out [nspec, LM, lines, along] in C order, n_e = nspec * LM * lines * along.
//     fut[e] holds n_e floats: the boundary value at the END of the interval (the next 3-h
//     boundary file), same layout as one slot.
//
// No host sync, no allocation. One launch: grid.y = entry, grid.x = element blocks (grid-stride).
#include "nmmb_common.cuh"

extern "C" __global__ void boco_tend_all(
    const unsigned long long* __restrict__ buf_ptr,   // [nent] device address of buf[e] (2 slots)
    const unsigned long long* __restrict__ fut_ptr,   // [nent] device address of fut[e] (1 slot)
    const long long* __restrict__ n_elem,             // [nent] elements per slot
    const int* __restrict__ along_e,                  // [nent] along-strip length of entry e
    const int* __restrict__ vmask,                    // [nent] 1 = V entry (skip last along index)
    int nent,
    float recip)                                      // 1 / TBOCO (Fortran RECIP)
{
    const int e = blockIdx.y;
    if (e >= nent) return;

    float* buf = reinterpret_cast<float*>(buf_ptr[e]);
    const float* fut = reinterpret_cast<const float*>(fut_ptr[e]);
    const long long n = n_elem[e];
    const int along = along_e[e];
    const int skip_last = vmask[e];

    // slot 0 = buf[0 .. n), slot 1 = buf[n .. 2n)
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x) {
        if (skip_last && (int)(k % along) == along - 1) continue;
        const float now = buf[k];
        buf[n + k] = (fut[k] - now) * recip;
    }
}
