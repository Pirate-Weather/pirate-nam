"""phys-myj-vdifq: VDIFQ, vertical diffusion of Q2 (2 x TKE) in the MYJ PBL, plus the EPSQ2 floor.

Ported from phys/module_BL_MYJPBL.F90:1342-1420 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE VDIFQ, and the
caller's floor Q2(I,J,K)=MAX(Q2K(K),EPSQ2(K)) for K=1..LM (module_BL_MYJPBL.F90:515-517).
Apache-2.0 origin of the copied arithmetic: ArWen third_party/arwen/core/kernels/myjpbl.cu, VDIFQ block.
Kernel: namgpu/kernels/phys_myj_vdifq.cu, entry phys_myj_vdifq, one thread per column (grid_cols),
one launch, on ctx.stream. No host sync, no per-step allocation.

Fortran call site: CALL VDIFQ(LMH,DTDIF,Q2K,EL,ZHK,I,J,LM) at module_BL_MYJPBL.F90:511. The floor is at :515.
LMH = LM is set at :364 and :605.

Reads (ctx.st, NMMB names, CuPy float32, [LM, JM, IM] unless noted; level 0 = model top):
    Q2        [LM, JM, IM]    TKE, updated in place (the 2*tke of the state table)
    EL_PBL    [LM, JM, IM]    EL(K) AFTER phys-myj-prodq2 (PRODQ2 sets EL = EPSL in places). Fortran EL(1:LM-1).
                              VDIFQ takes the post-PRODQ2 EL, the same array phys-myj-difcof reads.
    MYJ_ZINT  [LM+1, JM, IM]  ZINT = ZHK, interface heights, top-down (Fortran Z(K) at [K-1]).
                              ZINT(LM+1) = HT, the terrain height (module_BL_MYJPBL.F90:391, ZHK(LM+1)=HT(I,J)).
    EPSQ2     [LM] (rank 1)   2*TKE floor per level, the state table name (init-derive-state writes it).
Scalars: lmh (LMH = LM for the NAM calls), dtdif (the caller's DTDIF = DTTURBL; CONTRACT NPHS = 2 gives
DTPHS = 50 s).
PARAMETER ESQ = 5.0 and ELZ0 = 0. are fixed in the kernel (module_BL_MYJPBL.F90 line 58).

Writes: Q2, all LM levels (in place). Levels 1..LMH-1 are solved by VDIFQ, then every level K=1..LM is
floored by EPSQ2 (the caller's rule). EL_PBL and MYJ_ZINT are read only.

Ordering (not enforced by a kernel dependency, enforced by refusal): in the same step this unit runs after
phys-myj-mixlen (MYJ_ZINT and the EL producer), after phys-myj-prodq2 (writes Q2 and EL_PBL), and after
phys-myj-difcof. A missing producer raises; nothing is zero-filled.

run(ctx) is the done-rule entry. Without a ctx it builds a full-size random state (954 x 835 x 60) and
runs the kernel once. The __main__ block prints the count of non-finite values in Q2.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_vdifq"
KERNEL = "phys_myj_vdifq"
MYJ_LM = 60
MAXLEV = 72   # kernel VDIFQ_MAXLEV
EPSQ2_FLOOR = 0.02   # module_INIT_READ_BIN.F90:237-240 value, the same the init writes


def _require(st, name: str, shape: tuple, dtype: str | None = None):
    if name not in st.fields:
        raise RuntimeError(
            f"phys-myj-vdifq: {name} is not in ctx.st; EL_PBL comes from phys-myj-prodq2 and "
            "MYJ_ZINT from the MYJPBL driver, both must run earlier in the step")
    a = st.fields[name]
    if tuple(a.shape) != tuple(shape):
        raise RuntimeError(f"phys-myj-vdifq: {name} has shape {tuple(a.shape)}, contract needs {tuple(shape)}")
    return a


def launch(im, jm, lm, lmh, dtdif, q2, el, zint, epsq2, stream=None):
    """One launch over every column. q2, el: [LM, JM, IM]; zint: [LM+1, JM, IM]; epsq2: [LM]. float32 device arrays."""
    f32 = np.float32
    i32 = np.int32
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm), i32(lmh), f32(dtdif), q2, el, zint, epsq2,
        ))
    return q2


def run(ctx=None, *, im=None, jm=None, lm=None, dtdif: float = 50.0, seed: int = 1):
    """Done-rule entry: one launch on a full-size state. Returns {'Q2': q2} (updated in place).

    With a ctx: Q2 from ctx.st, EL_PBL and MYJ_ZINT required (no zero-fill), EPSQ2 from ctx.st, lmh = lm.
    Without one: a random full-size state (954 x 835 x 60). Q2 is drawn from [0, 4), so some values sit below
    the floor and the floor is exercised."""
    import cupy as cp

    if ctx is None:
        im = 954 if im is None else im
        jm = 835 if jm is None else jm
        lm = MYJ_LM if lm is None else lm
        rng = np.random.default_rng(seed)
        q2 = cp.asarray((4.0 * rng.random((lm, jm, im))).astype(np.float32))
        el = cp.asarray((1.0 + 299.0 * rng.random((lm, jm, im))).astype(np.float32))
        # interface heights, top-down: Z(1) highest, Z(LM+1) = 0 at the ground; strictly decreasing
        zlev = np.linspace(20000.0, 0.0, lm + 1)
        zj = zlev[:, None, None] + 10.0 * (rng.random((lm + 1, jm, im)) - 0.5)
        zj[-1] = 0.0
        zint = cp.asarray(zj.astype(np.float32))
        epsq2 = cp.full((lm,), EPSQ2_FLOOR, dtype=cp.float32)
        stream = None
    else:
        g, st = ctx.grid, ctx.st
        im, jm, lm = int(g.im), int(g.jm), int(g.lm)
        if lm != MYJ_LM:
            raise RuntimeError(f"phys-myj-vdifq: lm={lm}; the kernel is sized for {MYJ_LM} levels")
        shape3 = (lm, jm, im)
        q2 = _require(st, "Q2", shape3)
        el = _require(st, "EL_PBL", shape3)
        zint = _require(st, "MYJ_ZINT", (lm + 1, jm, im))
        epsq2 = st["EPSQ2"]
        if tuple(epsq2.shape) != (lm,):
            raise RuntimeError(f"phys-myj-vdifq: EPSQ2 has shape {tuple(epsq2.shape)}, contract needs ({lm},)")
        stream = getattr(ctx, "stream", None)

    launch(im, jm, lm, lm, dtdif, q2, el, zint, epsq2, stream=stream)
    return {"Q2": q2, "EPSQ2": epsq2}


if __name__ == "__main__":
    out = run()
    import cupy as cp
    n_bad = int(cp.count_nonzero(~cp.isfinite(out["Q2"])))
    print("non-finite Q2 values:", n_bad)
