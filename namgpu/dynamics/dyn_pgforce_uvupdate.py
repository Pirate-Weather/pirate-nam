"""dyn-pgforce-uvupdate: PGFORCE first-pass switch and the pressure-gradient u/v tendency update.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:452-493 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_pgforce_uvupdate.cu (one thread per (i, j, l), grid_2d).

What the routine does: it adds the pressure-gradient force tendency to TCU and TCV (the u/v
tendencies). It does NOT advance U or V (UPDATEUV does that later), and it does not use the
Adams-Bashforth weights cfc/bfc (not present in lines 452-493). The UNITS.json purpose text
("U, V, UP, VP -> U, V") does not match these lines; the Fortran writes TCU/TCV.

Reads : PGX_3D, PGY_3D, PGNE_3D, PGNW_3D (pgf scratch from dyn-pgforce-pgf), PD, DSG2 from
        ctx.st; RDXV (1/dxv, [JM]) and RDYV (1/dyv) from ctx.grid, cached in ctx.tables;
        PDSG1 from ctx.grid (device copy cached in ctx.tables).
Writes: TCU, TCV in ctx.st (read-modify-write, as the Fortran does).
Flag  : UPDATE = (not ctx.first_step) or restart (cfg.raw['restart'], see _restart_flag), the
        Fortran `.not.first.or.restart`. When
        UPDATE is 0 (first step, no restart) TCU and TCV are zeroed over the same range.
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_pgforce_uvupdate"
KERNEL = "dyn_pgforce_uvupdate"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _restart_flag(ctx) -> bool:
    """Fortran `restart` (int_state%RESTART). The config keeps it in cfg.raw["restart"] as parsed
    text ('true', '.true.', ...); NamConfig has no restart attribute. ctx.restart wins when set."""
    v = getattr(ctx, "restart", None)
    if v is None:
        v = (getattr(ctx.cfg, "raw", None) or {}).get("restart", False)
    if isinstance(v, str):
        return v.strip().strip("'").lower() in ("true", ".true.", "t", "yes", "1")
    return bool(v)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pdsg1 is None:
        raise RuntimeError(
            "dyn-pgforce-uvupdate: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    # Scratch written by dyn-pgforce-pgf (same shapes; add is a no-op once they exist).
    st.add("PGX_3D", (lm, jm, im))
    st.add("PGY_3D", (lm, jm, im))
    st.add("PGNE_3D", (lm, jm, im))
    st.add("PGNW_3D", (lm, jm, im))

    rdxv = _device_vector(ctx, "rdxv", np.float32(1.0) / np.asarray(g.dxv, dtype=np.float32))
    rdyv = float(np.float32(1.0) / np.float32(g.dyv))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    restart = _restart_flag(ctx)
    update = (not ctx.first_step) or restart

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_2D, (
            st["PGX_3D"], st["PGY_3D"], st["PGNE_3D"], st["PGNW_3D"], st["PD"], rdxv,
            st["DSG2"], pdsg1, st["TCU"], st["TCV"],
            np.float32(rdyv), np.float32(ctx.dt), np.int32(1 if update else 0),
            np.int32(im), np.int32(jm), np.int32(lm),
        ))
