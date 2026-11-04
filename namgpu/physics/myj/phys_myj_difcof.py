"""phys-myj-difcof: MYJ level 2.5 exchange coefficients (DIFCOF).

Fortran: phys/module_BL_MYJPBL.F90:1178-1337 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_myj_difcof.cu (phys_myj_difcof, one thread per column, grid_cols, one launch,
no host sync, no per-step allocation after the first call). Start point: ArWen DIFCOF block in
namgpu/third_party/arwen/core/kernels/myjpbl.cu (Apache-2.0). Notice: NOAA/NCEP NMMB source, kept in the
.cu header.

Reads (CONTRACT names, Fortran (i,j,l) -> [l, j, i]). Every input is a real producer's output; the
wrapper never fills a missing one:
    MYJ_GM    float32 [LM, JM, IM]    GM(K) from phys-myj-mixlen (levels 1..LM-1 used)
    MYJ_GH    float32 [LM, JM, IM]    GH(K) from phys-myj-mixlen (levels 1..LM-1 used)
    EL_PBL    float32 [LM, JM, IM]    EL(K) after phys-myj-prodq2 (levels 1..LM-1 used)
    Q2        float32 [LM, JM, IM]    Q2(K) (state)
    MYJ_ZINT  float32 [LM+1, JM, IM]  Z(K) interface heights, Z(LM+1) = ground (MYJPBL driver)
Writes:
    MYJ_AKM   float32 [LM, JM, IM]    AKM(K), K = 1..LMH-1
    MYJ_AKH   float32 [LM, JM, IM]    AKH(K), K = 1..LMH-1

LMH = LM: MYJPBL sets LMH = LM (module_BL_MYJPBL.F90:364 and :605), as phys-myj-prodq2 does.
Level LM of AKM/AKH is not written by the Fortran (undefined there). The kernel writes an explicit
zero there. Consumer check: the MYJPBL driver reads AKM and AKH only for K = 1..LM-1 (Fortran
lines 572 and 697 loop DO K=1,LM-1), and VDIFV/VDIFH take the CUDA-side RKM/RKH for levels 1..LM-1,
so the zero at level LM is never read.
T (a Fortran argument) is read only inside the commented-out inversion block and is not passed.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_difcof"
KERNEL = "phys_myj_difcof"

PRODUCERS = {
    "MYJ_GM": "phys-myj-mixlen",
    "MYJ_GH": "phys-myj-mixlen",
    "EL_PBL": "phys-myj-prodq2",
    "MYJ_ZINT": "the MYJPBL driver (interface heights)",
    "Q2": "the state (initial condition or a prior step)",
}


def _require(st, name, shape):
    """Return the producer's array for `name`. Raise when it is missing or has the wrong shape."""
    if name not in st.fields:
        raise RuntimeError(
            f"phys-myj-difcof: input {name} is not in the state; run {PRODUCERS[name]} first "
            "(no fallback values are generated)")
    a = st.fields[name]
    if tuple(a.shape) != tuple(shape):
        raise RuntimeError(
            f"phys-myj-difcof: input {name} has shape {tuple(a.shape)}, contract needs {tuple(shape)}")
    return a


def fixture_inputs(ctx):
    """Explicit test fixture, never called by run(). Fills GM, GH, EL_PBL and MYJ_ZINT with
    plausible values (GM/GH in [0, 0.05), EL in [10, 500), interface heights 30 km to ground) so
    the wrapper can run on a state with no MIXLEN, PRODQ2 or MYJPBL driver. Used only by the
    done-rule check tools/check-phys-myj-difcof.py."""
    import cupy as cp

    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    shape3 = (lm, jm, im)
    rng = cp.random.default_rng(21)
    st.fields["MYJ_GM"] = rng.uniform(0.0, 0.05, shape3).astype(cp.float32)
    st.fields["MYJ_GH"] = rng.uniform(0.0, 0.05, shape3).astype(cp.float32)
    st.fields["EL_PBL"] = rng.uniform(10.0, 500.0, shape3).astype(cp.float32)
    z = cp.linspace(30000.0, 0.0, lm + 1, dtype=cp.float32).reshape(lm + 1, 1, 1)
    st.fields["MYJ_ZINT"] = cp.broadcast_to(z, (lm + 1, jm, im)).copy()


def run(ctx):
    """Launch DIFCOF over every column. Returns {"MYJ_AKM": ..., "MYJ_AKH": ...}.

    Requires GM, GH and EL_PBL from the MIXLEN and PRODQ2 units, and MYJ_ZINT from the MYJPBL
    driver. A missing producer raises; nothing is filled in."""
    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    shape3 = (lm, jm, im)

    q2 = _require(st, "Q2", shape3)
    gm = _require(st, "MYJ_GM", shape3)
    gh = _require(st, "MYJ_GH", shape3)
    el = _require(st, "EL_PBL", shape3)
    zint = _require(st, "MYJ_ZINT", (lm + 1, jm, im))
    akm = st.add("MYJ_AKM", shape3)
    akh = st.add("MYJ_AKH", shape3)
    if tuple(akm.shape) != shape3 or tuple(akh.shape) != shape3:
        raise RuntimeError("phys-myj-difcof: MYJ_AKM/MYJ_AKH scratch has the wrong shape")

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (np.int32(im), np.int32(jm), np.int32(lm), np.int32(lm),
           gm, gh, el, q2, zint, akm, akh))

    return {"MYJ_AKM": akm, "MYJ_AKH": akh}
