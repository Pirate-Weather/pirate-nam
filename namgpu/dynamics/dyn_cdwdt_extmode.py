"""dyn-cdwdt-extmode: cos^2 attenuation of dw/dt in the top layers, then external-mode removal.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3700-3756 (NAM nam.v4.2.11, 83df4e13c1), subroutine cdwdt.
Kernel: namgpu/kernels/dyn_cdwdt_extmode.cu (dwdt_extmode_col, one thread per column).

Reads : PSGML1 [LM], DSG2 [LM], PDSG1 [LM] (grid, cached in ctx.tables), PDO [JM,IM].
Writes: DWDT [LM,JM,IM] (inout: attenuated, then external mode removed).
Constants from the configfile (parm/nam_configfile): fwhy = 1.0, slpd = 1500.0 (Pa, 15 hPa).
Launch: one kernel on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_extmode"
KERNEL = "dwdt_extmode_col"
FWHY = np.float32(1.0)      # configfile fwhy (cdwdt dwdt control factor)
SLPD = np.float32(1500.0)   # configfile slpd (cdwdt control layer depth, Pa)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if getattr(g, "pdsg1", None) is None or getattr(g, "sgml1", None) is None:
        raise RuntimeError(
            "dyn-cdwdt-extmode: vertical coordinate not set (grid.set_vcoord); pdsg1/psgml1 missing")

    psgml1 = _device_vector(ctx, "psgml1", g.psgml1)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    dsg2 = st["DSG2"]

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        FWHY, SLPD,
        psgml1, dsg2, pdsg1, st["PDO"], st["DWDT"],
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
