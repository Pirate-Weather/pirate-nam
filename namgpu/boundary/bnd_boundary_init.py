"""bnd-boundary-init: boundary working arrays from the full-domain edges (S, N, W, E).

Ported from nmm/module_CONTROL.F90:1145-1774 (boundary_init), NAM nam.v4.2.11, 83df4e13c1.
Kernel: namgpu/kernels/bnd_boundary_init.cu (bnd_boundary_init_strip).

Reads:  PD (2-D H), T, Q, CW (3-D H), U, V (3-D V) from ctx.st. These are the parent's hardwired
        boundary variables (module_SOLVER_GRID_COMP.F90:7658-7664: NVARS_BC_2D_H=1, 3D_H=3,
        4D_H=0, 2D_V=0, 3D_V=2). The Fortran loops over NV by registration order; here they are
        bound by name, which is the same set.
        lnsh, lnsv from ctx.cfg (parm/nam_configfile: 5 and 5).
Writes: ctx.bc, a dict keyed "<SIDE>_<NAME>" (SIDE first, the key bnd-bocoh-blend reads; SIDE in
        S, N, W, E). Each entry is float32 with a leading time-slot axis of length 2, as the
        bnd-bocoh-blend contract requires (CONTRACT section 6): index 0 is the value slot this unit
        writes, index 1 is the tendency slot that bnd-read-bc and bnd-bocoh-update own and that
        this unit leaves at zero. Value-slot shapes, with slot excluded:
          PD (2-D):      S, N: [lnsh, IM]          W, E: [JM, lnsh]
          T, Q, CW (3-D): S, N: [LM, lnsh, IM]      W, E: [LM, JM, lnsh]
        lnsh is the H width; the V entries (U, V) use lnsv and are not consumed by bocoh-blend.
        A buffer is reallocated when its shape changes (a changed lnsh, grid or level count), so
        a stale buffer is never written with a wider strip than it holds.
Scope: the strips cover the domain only (i in 1..IM, j in 1..JM, single process). The Fortran
        loops run over ims..ime and jms..jme, which include halo columns and rows; those halo
        entries are not written here, and no consumer reads them (bocoh-blend and bocoh-update
        loop over the domain).
Launch: one launch per (variable, side), 24 in all, each on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab

UNIT = "bnd_boundary_init"
KERNEL = "bnd_boundary_init_strip"

# The uppermost parent's hardwired boundary variables.
H_2D = ("PD",)
H_3D = ("T", "Q", "CW")
V_3D = ("U", "V")

SIDES = (("S", 0), ("N", 1), ("W", 2), ("E", 3))


def _src0(side: int, stagger: str, width: int, im: int, jm: int) -> int:
    """Zero-based full-domain index of strip position s = 0 (see the kernel header)."""
    if side == 0:                      # S: var(i, jds-1+n)
        return 0
    if side == 1:                      # N: var(i, jde-lnsh+n) for H, jde-1-lnsv+n for V
        return jm - width if stagger == "H" else jm - 1 - width
    if side == 2:                      # W: var(ids-1+n, j)
        return 0
    # E: var(ide-lnsh+n, j) for H, ide-1-lnsv+n for V
    return im - width if stagger == "H" else im - 1 - width


def _value_shape(name: str, side: int, nlev: int, width: int, im: int, jm: int) -> tuple:
    """Value-slot shape that bnd-bocoh-blend reads (no time-slot axis)."""
    two_d = name in H_2D
    if side < 2:                       # S, N: [lnsh, IM] or [LM, lnsh, IM]
        return (width, im) if two_d else (nlev, width, im)
    return (jm, width) if two_d else (nlev, jm, width)  # W, E: [JM, lnsh] or [LM, JM, lnsh]


def _buffer(ctx, side_name: str, name: str, side: int, nlev: int, width: int, im: int, jm: int):
    """Return the [2, ...] buffer for key "<SIDE>_<NAME>", allocating it when absent or when its
    shape no longer matches (a stale buffer of another width is never reused)."""
    import cupy as cp
    key = f"{side_name}_{name}"
    want = (2,) + _value_shape(name, side, nlev, width, im, jm)
    buf = ctx.bc.get(key)
    if buf is None or tuple(buf.shape) != want:
        buf = cp.zeros(want, dtype=np.float32)
        ctx.bc[key] = buf
    return buf


def run(ctx):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh, lnsv = int(cfg.lnsh), int(cfg.lnsv)
    if ctx.bc is None:
        ctx.bc = {}

    # (field name, nlev, stagger, width) in the Fortran order: 2-D H, 3-D H, 3-D V.
    plan = [(n, 1, "H", lnsh) for n in H_2D]
    plan += [(n, lm, "H", lnsh) for n in H_3D]
    plan += [(n, lm, "V", lnsv) for n in V_3D]

    kern = kernels.get(UNIT, KERNEL)
    s_bdy, n_bdy = slab.edges(ctx)   # row slab (namgpu/slab.py); W/E strips hold the local rows
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for name, nlev, stagger, width in plan:
            src = st[name].reshape(nlev, jm, im)
            for side_name, side in SIDES:
                extent = im if side < 2 else jm
                buf = _buffer(ctx, side_name, name, side, nlev, width, im, jm)
                if (side == 0 and not s_bdy) or (side == 1 and not n_bdy):
                    continue   # row slab: S/N strips are only filled (and used) on the edge ranks
                dst = buf[0]           # value slot; contiguous because the slot axis is outermost
                grid = ((width + kernels.BLOCK_2D[0] - 1) // kernels.BLOCK_2D[0],
                        (extent + kernels.BLOCK_2D[1] - 1) // kernels.BLOCK_2D[1],
                        nlev)
                kern(grid, kernels.BLOCK_2D, (
                    src, dst,
                    np.int32(side), np.int32(nlev), np.int32(im), np.int32(jm),
                    np.int32(width), np.int32(_src0(side, stagger, width, im, jm)),
                ))
