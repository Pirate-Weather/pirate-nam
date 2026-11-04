"""dyn-adv2-bndry: regional upstream advection of the tracer group along the four lateral boundary
strips (ADV2, regional branch).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:4609-4763 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_adv2_bndry.cu (dyn_adv2_bndry_upstream, one thread per (i, j), level in
blockIdx.z).

Reads:   UP, VP [LM,JM,IM] (ctx.st, previous-step winds); TRACERS_SQRT [NT,LM,JM,IM] (the
         intermediate species s1 from the horizontal ADV2 step, dyn-adv2-horiz); RDXH [JM] (1/dxh,
         uploaded once and cached in ctx.st); DT and RDYH (1/dyh) scalars; LNSH from ctx.cfg.
Writes:  TRACERS_TEND [NT,LM,JM,IM] (ctx.st), added to, as in Fortran. Scratch allocated once.
Calls:   kss, kse (1-based species, Fortran KSS, KSE) and idtadt (Fortran IDTADT or IDTADTQ) are
         required; there are no defaults. They are the arguments of the enclosing ADV2 call, passed
         by the driver plan (namgpu/driver.py):
           Q call      module_SOLVER_GRID_COMP.F90:4321-4325: KSS = KSE1 = INDX_Q, idtadt = IDTADTQ
           tracer call module_SOLVER_GRID_COMP.F90:4459-4461 (SPEC_ADV=F, the configured run):
                       KSS = INDX_CW, KSE1 = INDX_Q2, idtadt = IDTADT
         IDTADTQ is 1 on a regional run (module_SOLVER_GRID_COMP.F90:2858-2862). The SPEC_ADV=T branch
         (KSE1 = NUM_TRACERS_TOTAL, 4442-4443) is not what NAM runs and is not wired here.
Branch:  the regional strips only. global = true is refused (the global branch is not this unit).
         When adv_upstream is false (module_CONTROL.F90:264-269, subdomain too small for the strips)
         the Fortran branch is skipped and so is the launch.
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv2_bndry"
KERNEL = "dyn_adv2_bndry_upstream"


def run(ctx, kss: int, kse: int, idtadt: int):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if ctx.cfg.global_:
        raise RuntimeError(
            "dyn-adv2-bndry: global = true uses the global branch of ADV2, which is not this unit "
            "(CONTRACT section 2 assumes global = false)")

    lnsad = int(ctx.cfg.lnsh) + 2                  # module_CONTROL.F90:260
    # Single tile, full domain (ids = its = 1, ide = ite = IM, jds = jts = 1, jde = jte = JM).
    adv_upstream = (1 < 1 + 1 + lnsad or jm >= jm - 1 - lnsad
                    or 1 < 1 + 1 + lnsad or im >= im - 1 - lnsad)
    if not adv_upstream:                           # module_CONTROL.F90:264-269
        return

    tracers = st.TRACERS
    nt = int(tracers.shape[0])
    kss, kse = int(kss), int(kse)
    if not (1 <= kss <= kse <= nt):
        raise ValueError(f"dyn-adv2-bndry: species range {kss}..{kse} outside 1..{nt}")
    rec = ctx.tables.get("ADV2_HORIZ_RANGE")
    if rec != (kss, kse):
        raise RuntimeError(f"dyn-adv2-bndry: dyn-adv2-horiz must run on {kss}..{kse} first (last {rec}); "
                           "the regional outer rows are written after the standard block")

    up = st.UP
    vp = st.VP
    s1 = st.add("TRACERS_SQRT", tuple(tracers.shape))
    tcs = st.add("TRACERS_TEND", tuple(tracers.shape))
    rdxh = _rdxh(st, g)
    rdyh = np.float32(1.0) / np.float32(g.dyh)     # module_CONTROL.F90:566, rdyh = 1./dyh
    idtadt = int(idtadt)

    kernel = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernel(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
               (np.int32(im), np.int32(jm), np.int32(lm),
                np.int32(kss), np.int32(kse), np.int32(lnsad),
                np.float32(ctx.dt), rdyh, np.int32(idtadt),
                rdxh, up, vp, s1, tcs))


def _rdxh(st, g):
    """rdxh(j) = 1./dxh(j) (module_CONTROL.F90:636), single precision, cached on the device."""
    if "RDXH" not in st.fields:
        import cupy as cp
        dxh = np.asarray(g.dxh, dtype=np.float32)
        st.fields["RDXH"] = cp.asarray((np.float32(1.0) / dxh).astype(np.float32))
    return st.fields["RDXH"]
