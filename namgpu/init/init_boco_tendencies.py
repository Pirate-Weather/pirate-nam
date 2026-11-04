"""init-boco-tendencies: boundary time tendencies for one parent/child interval (NMMB UPDATE_BC_TENDS).

Ported from nmm/module_SOLVER_GRID_COMP.F90:8155-9298 (UPDATE_BC_TENDS, nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/init_boco_tendencies.cu (boco_tend_all, one launch for every boundary
entry of the interval).

Path selection (Fortran SOLVER nest_or_parent, module_SOLVER_GRID_COMP.F90:3820). UPDATE_BC_TENDS is
called only when I_AM_A_NEST (the call at line 3849, gated by COMPUTE_BC). The uppermost domain (the
NAM 12 km CONUS parent, I_AM_A_NEST false) never calls it: its boundary value and tendency come from
the boco.NNNN record (both slots, READ_BC module_FLTBNDS.F90:3579-3925), placed by namgpu.boundary.
bnd_read_bc. So on the uppermost path this unit must not write slot 1; staging a future interval
there raises, because that would overwrite the file tendency.

What it computes (nest path). For every boundary strip (N, S, W, E) of every boundary variable, the
tendency that carries the carried value to the interval-end value:
    tend = (value_end - value_now) * RECIP,   RECIP = 1./(DT*PARENT_CHILD_TIME_RATIO) in REAL,
    TBOCO = PARENT_CHILD_TIME_RATIO*DT (Fortran line 3828)
which is the Fortran `BND_VARS%...(...,2) = (BND_DATA(KOUNT) - BND_VARS%...(...,1)) * RECIP`.
value_now is slot 1 of the Fortran boundary object as carried across intervals: bnd-bocoh-update
advances it every step (module_FLTBNDS.F90:5176-5180, `south(ib,jb,1) += south(ib,jb,2)*dt`). This
unit reads that carried value and does not refresh it from the state edges. The carried buffer must
already exist (bnd-boundary-init, or the nest digital-filter WRITE_BC at module_SOLVER_GRID_COMP.F90:3829);
a missing buffer raises.
V-point entries (cls v2d / v3d) skip the last along index: the Fortran V loops stop at IDE-1 and
JDE-1, so that tendency slot is left unchanged.

Inputs (nest path):
    ctx.i_am_a_nest              bool, required (Fortran I_AM_A_NEST). No default.
    ctx.dt                       float, required (the domain's fundamental timestep, Fortran DT).
    ctx.parent_child_time_ratio  int, required (Fortran PARENT_CHILD_TIME_RATIO).
    ctx.bc[(cls, name, side)]    float32 [2, nspec, LM, lines, along]; slot 0 = carried value_now.
    ctx.boco_future[(cls, name, side)]  float32 [nspec, LM, lines, along]: the boundary values at the
                 END of the interval, in the full-size boco record layout (the record is dimensioned
                 ids:ide and jds:jde, READ_BC module_FLTBNDS.F90:3600+). For V entries the last along
                 index is padding that the Fortran UPDATE_BC_TENDS loops never read (I2_V = IDE-1);
                 the kernel skips it. The shape is therefore the same for H and V entries.
Output:
    ctx.bc[(cls, name, side)][1]  the tendency; slot 0 is not written.
    Layout as bnd-write-bc: [2, nspec, LM, lines, along], float32, side 0 N, 1 S, 2 W, 3 E.

Behaviour. If ctx.boco_future is None (no interval staged) the wrapper does nothing and returns None,
as bnd-read-bc does when no interval is pending. No host sync; the pointer tables are small host
arrays copied once per call.

Not this unit: the WRITE_BC "recomp" form (module_FLTBNDS.F90:4734-5010) and the nest write at
module_SOLVER_GRID_COMP.F90:3829 (bnd-write-bc). Moving-nest (MOVE_NOW) tendencies are not formed here;
see the kernel header.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "init_boco_tendencies"
KERNEL = "boco_tend_all"
BLOCK = 256              # threads per block (tuning knob, kept here)
MAX_GRID_X = 65535       # grid-stride loop covers the rest


def recip_and_tboco(dt, parent_child_time_ratio):
    """Fortran RECIP = 1./(DT*PARENT_CHILD_TIME_RATIO) and TBOCO = PARENT_CHILD_TIME_RATIO*DT, both REAL
    (single precision), in the Fortran evaluation order."""
    if dt is None or parent_child_time_ratio is None:
        raise ValueError("init-boco-tendencies: ctx.dt and ctx.parent_child_time_ratio are required on "
                         "the nest path (Fortran RECIP and TBOCO have no default)")
    dt32 = np.float32(dt)
    pctr32 = np.float32(int(parent_child_time_ratio))
    tboco = np.float32(pctr32 * dt32)
    recip = np.float32(1.0) / np.float32(dt32 * pctr32)
    return recip, float(tboco)


def run(ctx):
    fut = getattr(ctx, "boco_future", None)
    if fut is None or len(fut) == 0:
        return None

    if getattr(ctx, "i_am_a_nest", None) is None:
        raise ValueError("init-boco-tendencies: ctx.i_am_a_nest is required (Fortran I_AM_A_NEST selects "
                         "the UPDATE_BC_TENDS path)")
    if not ctx.i_am_a_nest:
        raise ValueError("init-boco-tendencies: uppermost domain: its boundary tendency is the boco record "
                         "slot 2 placed by bnd-read-bc; UPDATE_BC_TENDS is not called here (Fortran "
                         "SOLVER line 3849 is inside I_AM_A_NEST). Staging a future interval would overwrite "
                         "the file tendency.")

    import cupy as cp

    recip, _tboco = recip_and_tboco(getattr(ctx, "dt", None), getattr(ctx, "parent_child_time_ratio", None))

    # One pointer table over every staged entry. The carried value (slot 0) must already exist.
    keys = list(fut.keys())
    buf_ptr, fut_ptr, n_elem, along_e, vmask = [], [], [], [], []
    for key in keys:
        if ctx.bc is None or key not in ctx.bc:
            raise RuntimeError(f"init-boco-tendencies: no carried boundary buffer for {key}; the Fortran "
                               "value slot 1 comes from bnd-boundary-init / the nest WRITE_BC and is "
                               "advanced by BOCOH, so it is not formed here")
        buf = ctx.bc[key]
        f = fut[key]
        if buf.dtype != cp.float32 or f.dtype != cp.float32:
            raise ValueError(f"init-boco-tendencies: {key} is not float32")
        if not (buf.flags.c_contiguous and f.flags.c_contiguous):
            raise ValueError(f"init-boco-tendencies: {key} is not C-contiguous")
        if tuple(f.shape) != tuple(buf.shape[1:]):
            raise ValueError(f"init-boco-tendencies: {key} future shape {f.shape}, "
                             f"want {buf.shape[1:]}")
        buf_ptr.append(buf.data.ptr)
        fut_ptr.append(f.data.ptr)
        n_elem.append(int(buf[0].size))
        along_e.append(int(buf.shape[-1]))
        vmask.append(1 if key[0].startswith("v") else 0)

    bt = cp.asarray(np.array(buf_ptr, dtype=np.uint64))
    ft = cp.asarray(np.array(fut_ptr, dtype=np.uint64))
    nt = cp.asarray(np.array(n_elem, dtype=np.int64))
    at = cp.asarray(np.array(along_e, dtype=np.int32))
    vt = cp.asarray(np.array(vmask, dtype=np.int32))
    max_n = max(n_elem)
    grid = (min((max_n + BLOCK - 1) // BLOCK, MAX_GRID_X), len(keys), 1)

    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, (BLOCK, 1, 1), (
            bt, ft, nt, at, vt, np.int32(len(keys)), recip,
        ))
    return keys
