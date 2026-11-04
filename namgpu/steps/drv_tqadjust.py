"""drv-tqadjust: TQADJUST after the physics (T, Q, CW and condensate fractions consistency).

Fortran: nmm/module_SOLVER_GRID_COMP.F90:11777-12113 (NAM nam.v4.2.11, 83df4e13c1), called from
SOLVER_RUN at module_SOLVER_GRID_COMP.F90:6022 (gfs: false branch, after CLTEND).
Kernel: namgpu/kernels/drv_tqadjust.cu (drv_tqadjust, one thread per (i, j) column, grid_cols).

Reads and writes (device arrays, CONTRACT names, Fortran (i,j,l) -> [l, j, i]):
  T, Q, QC, CW, F_ICE, F_RAIN   [LM, JM, IM] float32 (updated in place)
  PD                            [JM, IM]
  DSG2, SGML2                   [LM] (state)
  PDSG1, PSGML1                 [LM] (grid constants, uploaded once into ctx.tables)
Scalars: SPEC_ADV from cfg.raw['spec_adv'] (parm/nam_configfile:230, .false. in the 12 km parent),
RHgrd from cfg.rhgrd (parm/nam_configfile:267, 0.98).

Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
The column scratch lives in the kernel (per-thread, 1-based Fortran levels, LM <= 64).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_tqadjust"
KERNEL = "drv_tqadjust"
LMAX = 64          # must equal TQ_LMAX in kernels/drv_tqadjust.cu


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
    if lm > LMAX:
        raise ValueError(f"drv-tqadjust: LM={lm} exceeds the kernel column scratch (LMAX={LMAX})")
    if g.pdsg1 is None or g.dsg2 is None or g.psgml1 is None:
        raise RuntimeError("drv-tqadjust: vertical coordinate not set (grid.set_vcoord)")

    spec_adv = int(bool(cfg.raw.get("spec_adv", False)))
    rhgrd = np.float32(cfg.rhgrd)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    psgml1 = _device_vector(ctx, "psgml1", g.psgml1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        st.T, st.Q, st.QC, st.CW, st.F_ICE, st.F_RAIN,
        st.PD, st["DSG2"], pdsg1, psgml1, st["SGML2"],
        np.int32(spec_adv), rhgrd,
        np.int32(im), np.int32(jm), np.int32(lm),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
