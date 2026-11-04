"""phys-myj-prodq2: MYJ level 2.5 TKE production/dissipation (PRODQ2).

Ported from phys/module_BL_MYJPBL.F90:969-1173 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_myj_prodq2.cu (phys_myj_prodq2). Started from the ArWen PRODQ2 block in
third_party/arwen/core/kernels/myjpbl.cu (Apache-2.0). One launch, one thread per column
(grid_cols), no host sync, no per-step allocation.

Reads (CONTRACT names). Every input is a real producer's output; the wrapper never fills a
missing one (a missing or wrong-shape input raises, as phys-myj-difcof does):
    USTAR    float32 [JM, IM]      friction velocity (surface layer)
    EPSL     float32 [LM]          mixing-length floor, EPSL(K) (state table, rank 1)
    EPSQ2    float32 [LM]          TKE floor, EPSQ2(K) (state table, rank 1)
    Q2       float32 [LM, JM, IM]  2*TKE, read and written in place
    MYJ_GM   float32 [LM, JM, IM]  GM(K) from phys-myj-mixlen (levels 1..LM-1 used)
    MYJ_GH   float32 [LM, JM, IM]  GH(K) from phys-myj-mixlen (levels 1..LM-1 used)
    EL_PBL   float32 [LM, JM, IM]  EL(K), read and written in place (MYJPBL_INIT, then MIXLEN)

Writes: Q2 and EL_PBL (in place).

DTTURBL = DT * NPHS, with NPHS = cfg.nphs (the configfile value, module_BL_MYJPBL.F90 gets it
from int_state%NPHS). LMH = LM (MYJPBL sets LMH = LM at module_BL_MYJPBL.F90:364 and :605).

Driver contract (phys-myj-myjpbl must honour it; the wrapper cannot check it):
  * the call covers the whole IM x JM domain (the kernel has no ITS..ITE / JTS..JTE exclusion);
  * PRODQ2 runs in the Fortran order: after MIXLEN, before DIFCOF and VDIFQ, and the caller must
    not read Q2 between. The Fortran works on the column copy Q2K and floors it at line 516
    (Q2(I,J,K)=MAX(Q2K(K),EPSQ2(K))) after VDIFQ. This kernel writes the 3-D Q2 directly, so the
    driver has to keep that order or the result differs.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_prodq2"
KERNEL = "phys_myj_prodq2"
BLOCK = 128

PRODUCERS = {
    "USTAR": "the surface layer (phys-jsfc)",
    "EPSL": "the state table (init-derive-state)",
    "EPSQ2": "the state table (init-derive-state)",
    "Q2": "the state (initial condition or a prior step)",
    "MYJ_GM": "phys-myj-mixlen",
    "MYJ_GH": "phys-myj-mixlen",
    "EL_PBL": "phys-myj-init (MYJPBL_INIT), then phys-myj-mixlen",
}


def _require(st, name, shape):
    """Return the producer's array for `name`. Raise when it is missing or has the wrong shape."""
    if name not in st.fields:
        raise RuntimeError(
            f"phys-myj-prodq2: input {name} is not in the state; run {PRODUCERS[name]} first "
            "(no fallback values are generated)")
    a = st.fields[name]
    if tuple(a.shape) != tuple(shape):
        raise RuntimeError(
            f"phys-myj-prodq2: input {name} has shape {tuple(a.shape)}, contract needs {tuple(shape)}")
    return a


def run(ctx):
    """Launch PRODQ2 over every column. Returns {"Q2": ..., "EL_PBL": ...} (both in place)."""
    if ctx.cfg is None:
        raise RuntimeError("phys-myj-prodq2: ctx.cfg is required (NPHS is read from the configfile)")
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    ustar = _require(st, "USTAR", (jm, im))
    epsl = _require(st, "EPSL", (lm,))
    epsq2 = _require(st, "EPSQ2", (lm,))
    q2 = _require(st, "Q2", shape3)
    gm = _require(st, "MYJ_GM", shape3)
    gh = _require(st, "MYJ_GH", shape3)
    el = _require(st, "EL_PBL", shape3)

    nphs = int(ctx.cfg.nphs)
    dtturbl = np.float32(nphs * float(ctx.dt))
    k = kernels.get(UNIT, KERNEL)
    grid = ((im * jm + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (np.int32(im), np.int32(jm), np.int32(lm), np.int32(lm), dtturbl,
           ustar, gm, gh, epsl, epsq2, el, q2))

    return {"Q2": q2, "EL_PBL": el}
