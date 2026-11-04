"""dyn-mono-q2-back: Q2 interface interpolation at the end of MONO (q2_check block).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:5204-5235, subroutine mono (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_mono_q2_back.cu (dyn_mono_q2_back, one thread per column, one launch).

Fortran MONO receives s = int_state%TRACERS and tcs = int_state%TRACERS_TEND (the 4-D arrays) and
indexes them with INDX_Q2. The wrapper therefore binds the Q2 slot of those two arrays, not a
separate Q2 field:
Reads : TRACERS[indx_q2-1] (Q2 species, read and written in place),
        TRACERS_TEND[indx_q2-1] (Q2 species tendency; written by dyn-mono-fix, which the driver runs
        earlier in the same block), PD [JM,IM], DSG2 [LM] (state), PDSG1 [LM] (grid.pdsg1, uploaded
        once to ctx.tables), EPSQ2 [LM] (state, written by init-derive-state's init_epsq2).
Writes: TRACERS_TEND[indx_q2-1] (interior columns, rescaled and interpolated; lowest layer LM set to 0)
        and TRACERS[indx_q2-1] (interior columns: interface values, floored at EPSQ2; lowest level
        set to EPSQ2(LM)).
Range:  run(ctx, kss, kse) takes Fortran 1-based tracer slots. The driver passes KSS=INDX_CW and
        KSE=INDX_Q2 for the tracers block (module_SOLVER_GRID_COMP.F90:4459-4461, call at 4519). The
        Fortran block runs only when kss <= indx_q2 <= kse, and this wrapper returns in the same case.
        With no range given, the CW..Q2 default is used, as in dyn-adv2-q2-mid.
Slots:  ctx.tables['indx'] (drv-tracer-index) gives INDX_Q2; if it is not built yet, the host mirror
        is used (module_SOLVER_INTERNAL_STATE.F90:1382-1400, module_VARS_STATE.F90:797).
Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_tracer_index import host_index

UNIT = "dyn_mono_q2_back"
KERNEL = "dyn_mono_q2_back"
TEND = "TRACERS_TEND"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx, kss=None, kse=None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    indx = ctx.tables.get("indx")
    if indx is None:
        indx = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
    indx_q2 = int(indx["INDX_Q2"])
    if kss is None:
        kss = int(indx["INDX_CW"])
    if kse is None:
        kse = indx_q2
    nt = int(st.TRACERS.shape[0])
    if not (1 <= kss <= kse <= nt):
        raise ValueError(f"dyn-mono-q2-back: tracer group {kss}..{kse} outside 1..{nt}")
    if not (kss <= indx_q2 <= kse):
        return  # q2_check: Q2 is not in this species range, the Fortran block is skipped

    if g.dsg1 is None:
        raise RuntimeError(
            "dyn-mono-q2-back: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    epsq2 = st["EPSQ2"]
    tend = st.add(TEND, (nt, lm, jm, im))

    k = indx_q2 - 1   # Fortran 1-based slot -> zero-based slab of [NT, LM, JM, IM]
    q2 = st.TRACERS[k]
    tcs_q2 = tend[k]

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, (
            tcs_q2, q2, st["PD"], st["DSG2"], pdsg1, epsq2,
            np.int32(im), np.int32(jm), np.int32(lm),
        ))
