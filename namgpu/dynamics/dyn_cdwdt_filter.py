"""dyn-cdwdt-filter: lateral spatial filter of DWDT, the cdwdt clamp and blend, DWDT = 1 on the
regional boundary H rows.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3773-3899 (NAM nam.v4.2.11, 83df4e13c1), regional case.
Kernels: namgpu/kernels/dyn_cdwdt_filter.cu
  dwdt_filter_ww     (only if wad > 0) 5-point stencil of DWDT into scratch CDWDT_WW
  dwdt_filter_apply  (only if wad > 0) DWDT += CDWDT_WW * wad
  cdwdt_clamp_bdy    clamp DWDT to [epsn, epsp], blend with PDWDT by wp, boundary H rows = 1
Reads : DWDT, PDWDT [LM,JM,IM]. PDWDT is the solver's previous correction factor (set by the
        solver, not a table row); it is added to the state here, zeroed, if not present yet.
Writes: DWDT (in place); CDWDT_WW [LM,JM,IM] scratch, allocated once and only when wad > 0.
Constants (Fortran module parameters of cdwdt): wad = 0 (line 3313), wp = 0 (3315),
epsn/epsp = -/+9.80 (3302-3303), G from NMMB module_CONSTANTS (kernels/nmmb_common.cuh).
Global-only swaphn/polehn (3819-3826) are skipped; the regional boundary flags are 1 for the
one-card, single-process layout.
Launch: kernels on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_filter"
NMMB_G = 9.8060226      # module_CONSTANTS (nmmb_common.cuh NMMB_G)
WAD = 0.0               # cdwdt lateral smoothing weight, line 3313 (active value)
WP = 0.0                # cdwdt time smoothing weight, line 3315 (active value)


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    dwdt = st["DWDT"]
    pdwdt = st.add("PDWDT", shape3)
    rg = np.float32(1.0) / np.float32(NMMB_G)           # rg = 1./g, single precision
    g2 = kernels.grid_2d(im, jm, lm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        if WAD > 0.0:
            ww = st.add("CDWDT_WW", shape3)
            kernels.get(UNIT, "dwdt_filter_ww")(g2, kernels.BLOCK_2D, (
                np.int32(im), np.int32(jm), np.int32(lm), dwdt, ww,
            ))
            kernels.get(UNIT, "dwdt_filter_apply")(g2, kernels.BLOCK_2D, (
                np.int32(im), np.int32(jm), np.int32(lm), np.float32(WAD), ww, dwdt,
            ))
        kernels.get(UNIT, "cdwdt_clamp_bdy")(g2, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm), rg, np.float32(WP),
            np.int32(1), np.int32(1), np.int32(1), np.int32(1),
            dwdt, pdwdt,
        ))
