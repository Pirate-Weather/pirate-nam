"""dyn-cdwdt-boundary-smooth: CDWDT setup (early return), and smoothing of w on H points along
the lateral boundaries.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3278-3532 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/dyn_cdwdt_boundary_smooth.cu (cdwdt_setup, cdwdt_smooth_ww,
cdwdt_smooth_copy).

Reads : DSG2 [LM], PDSG1 [LM] (grid.dsg2, grid.pdsg1, uploaded once to ctx.tables),
        PD [JM,IM], W [LM,JM,IM], PINT [LM+1,JM,IM].
Writes: DWDT [LM,JM,IM] = 1 and PDWDT [LM,JM,IM] = 1 (setup path), PINT levels 2..LM+1 (setup
        path), W strips (smoothing path, only when NSMUD > 0 in the Fortran; NSMUD is 0 here).
Scratch: PDWDT (not in the NAM state tables) via st.add(); CDWDT_WW [LM,JM,IM] only when the
         smoothing path runs.

Branch selection (host scalars, no per-cell work):
  setup path  when hydro or (not restart and abs(ntimestep) < 2)   (Fortran 3427, ntsd = ABS(NTIMESTEP) from 4752)
  smoothing   otherwise, and only when the run is regional (not global).
  The DFI backward-step diffusion gate (diffw from dt > 0) is set in the Fortran after 3532 and is
  not used in this range, so it is not computed here.

Lateral-boundary flags: all four sides are domain boundaries on one card (single process).
Launch: one kernel on ctx.stream for the setup path; for smoothing, NSMUD(=0) x (copy, stencil)
pairs of kernels, each with grid z = LM. No host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab

UNIT = "dyn_cdwdt_boundary_smooth"
K_SETUP = "cdwdt_setup"
K_WW = "cdwdt_smooth_ww"
K_COPY = "cdwdt_smooth_copy"

NSMUD = 0                       # Fortran PARAMETER nsmud=00 (number of smoothing iterations)
FLAGS_ALL_SIDES = 1 | 2 | 4 | 8  # s_bdy | n_bdy | w_bdy | e_bdy, true on one card


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.dsg2 is None or g.dsg1 is None:
        raise RuntimeError(
            "dyn-cdwdt-boundary-smooth: vertical coordinate not set (grid.set_vcoord); dsg2 missing")

    hydro = bool(cfg.hydro)
    global_ = bool(cfg.global_)
    restart = bool(ctx.restart)                       # set from cfg.raw['restart'] in driver.Ctx
    # Fortran passes ABS(NTIMESTEP) to CDWDT (module_SOLVER_GRID_COMP.F90:4752): DFI backward
    # steps carry a negative counter and must not re-enter the setup branch.
    ntsd = abs(int(ctx.ntimestep))

    pdwdt = st.add("PDWDT", (lm, jm, im))
    dsg2 = _device_vector(ctx, "dsg2", g.dsg2)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern_setup = kernels.get(UNIT, K_SETUP)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    i32 = np.int32

    # Fortran 3427: the setup branch returns from CDWDT, so every later CDWDT unit (local-deriv,
    # diffusion, vert-adv, horiz-adv, extmode, filter) is skipped on that step. Set every step.
    ctx.cdwdt_skip = bool(hydro or (not restart and ntsd < 2))

    with scope:
        if ctx.cdwdt_skip:
            kern_setup(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
                i32(im), i32(jm), i32(lm),
                dsg2, pdsg1, st.PD, st.DWDT, pdwdt, st.PINT,
            ))
            return

        if global_ or NSMUD <= 0:
            # Global runs skip the lateral smoothing; nsmud = 0 means no smoothing iterations.
            return

        s_bdy, n_bdy = slab.edges(ctx)   # row slab: S/N strips only on the edge ranks
        flags = (FLAGS_ALL_SIDES & ~(0 if s_bdy else 1)) & ~(0 if n_bdy else 2)
        ww = st.add("CDWDT_WW", (lm, jm, im))
        kern_ww = kernels.get(UNIT, K_WW)
        kern_copy = kernels.get(UNIT, K_COPY)
        grid = kernels.grid_2d(im, jm, lm)
        for _ks in range(NSMUD):
            kern_ww(grid, kernels.BLOCK_2D, (
                i32(im), i32(jm), i32(lm), i32(flags), st.W, ww,
            ))
            kern_copy(grid, kernels.BLOCK_2D, (
                i32(im), i32(jm), i32(lm), i32(flags), st.W, ww,
            ))
