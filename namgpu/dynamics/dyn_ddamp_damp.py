"""dyn-ddamp-damp: divergence damping of U and V (NMMB DDAMP, divergence-damping branch).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1000-1036 plus the ddamp preamble it depends on,
lines 890-898 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/dyn_ddamp_damp.cu (ddamp_apd, ddamp_damp; one thread per point).

Reads:   PD, PDO [JM,IM] from ctx.st (current step); DIV [LM,JM,IM], DSG2 [LM], SG1/SG2 [LM+1] from
         ctx.st; PDSG1 [LM] from ctx.grid; DDMPU [JM] and DDMPV derived from ctx.cfg (codamp, dt) and
         ctx.grid (dxv, dyv) with the module_CONTROL regional formula (device copies cached in
         ctx.tables); PDTOP from ctx.grid; FREERUN from ctx.cfg (parm/nam_configfile freerun).
Writes:  U, V [LM,JM,IM] in place (the damping increments are added to the existing U, V).
         APD [JM,IM] scratch, recomputed every call from the current PD/PDO.
Scalars: dfac=9., dpb=1500., fint=(1-dfac)/dpb, assimfc=5.0 (Fortran lines 885-887, 857-858).
NOTE:    extmod is the .true. parameter (Fortran line 855), so the else branch is not reached by
         NAM 4.2.11. Ported in full because the unit is defined as this line range.
Launch:  two kernels on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.dynamics.dyn_ddamp_extmode import _freerun

UNIT = "dyn_ddamp_damp"
K_APD = "ddamp_apd"
K_DAMP = "ddamp_damp"

DFAC = np.float32(9.0)
DPB = np.float32(1500.0)
FINT = np.float32((np.float32(1.0) - DFAC) / DPB)
ASSIMFC = np.float32(5.0)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _ddamp_coefficients(ctx):
    """DDMPU(j) and DDMPV from module_CONTROL grid_consts, regional branch (lines 905-946).

    cddamp = codamp*dt (line 535); ddmpv = cddamp*dyv/(2*dyv) (line 919);
    ddmpu(j) = cddamp*dyv/(2*dxv(j)) for j = jds..jde (line 944). The grid object does not carry
    these yet, so they are derived here from the same formula, in single precision.
    """
    g = ctx.grid
    cddamp = np.float32(ctx.cfg.codamp) * np.float32(ctx.cfg.dt)
    dyv = np.float32(g.dyv)
    ddmpv = np.float32(cddamp * dyv / (np.float32(2.0) * dyv))
    dxv = np.asarray(g.dxv, dtype=np.float32)
    ddmpu = (cddamp * dyv / (np.float32(2.0) * dxv)).astype(np.float32)
    return ddmpu, ddmpv


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pdsg1 is None:
        raise RuntimeError(
            "dyn-ddamp-damp: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    freerun = _freerun(ctx.cfg)
    apd = st.add("APD", (jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    ddmpu_host, ddmpv = _ddamp_coefficients(ctx)
    ddmpu = _device_vector(ctx, "ddmpu", ddmpu_host)

    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        # Fortran 890-898: APD from the current PD/PDO, every call (not only the first).
        kernels.get(UNIT, K_APD)(kernels.grid_2d(im, jm, 1), block, (
            np.int32(im), np.int32(jm), st.PD, st.PDO, apd,
        ))
        kernels.get(UNIT, K_DAMP)(grid, block, (
            np.int32(im), np.int32(jm), np.int32(lm),
            np.float32(ddmpv), np.float32(g.pdtop),
            DFAC, DPB, FINT, ASSIMFC, np.int32(1 if freerun else 0),
            st.DSG2, pdsg1, st.SG1, st.SG2, ddmpu, apd,
            st.DIV, st.U, st.V,
        ))
