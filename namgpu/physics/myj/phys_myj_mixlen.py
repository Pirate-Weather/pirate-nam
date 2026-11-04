"""phys-myj-mixlen: MIXLEN, master mixing length of the MYJ level 2.5 PBL scheme (NMMB
module_BL_MYJPBL.F90:758-965, nam.v4.2.11, 83df4e13c1).

Kernel: namgpu/kernels/phys_myj_mixlen.cu (phys_myj_mixlen). One launch, one thread per column
(grid_cols), the whole Fortran routine inside, no host sync, no per-step allocation after the
first call. MYJPBL (phys-myj-myjpbl) calls it once per column with the column arrays it builds;
this wrapper runs the same kernel on full-field arrays.

Required inputs (ctx.st; raise RuntimeError when absent or too small, never zero-filled):
    U, V       state table, wind on the V stagger                 [LM, JM, IM] float32
    THV, THE   MYJ workspace from MYJPBL                          [LM, JM, IM] float32
    Q2         state table, 2*tke                                 [LM, JM, IM] float32
    EPSQ2      floor of 2*tke, per level                          [>= LM] float32
    EPSL       floor mixing length, per level                     [>= LM] float32
    MYJ_ZINT   interface heights (MYJPBL ZHK = ZINT)              [LM+1, JM, IM] float32
    MYJ_P      mid-level pressure (MYJPBL PK = PMID)              [LM, JM, IM] float32
    MYJ_PSK    per-level PS(K) (MYJPBL PSK)                       [LM, JM, IM] float32
    MYJ_LMH    lowest mass level per column (MYJPBL sets LM)      [JM, IM] int32
Writes (st.add scratch, or the state table where it exists):
    MYJ_GM, MYJ_GH, MYJ_EL   float32 [LM, JM, IM]     Fortran GM, GH, EL (levels 1..LM-1 used)
    PBLH, MIXHT              state table [JM, IM]     PBL height, mixed-layer height
    MYJ_LPBL, MYJ_LMXL       int32 [JM, IM]
    MYJ_CT                   float32 [JM, IM]         CT (INOUT), zeroed by the kernel
Scalars: DT and NPHS come from ctx.cfg, the same source MYJPBL reads (phys_myj_myjpbl.py), and
RSQDT = SQRT(1 / (DT * NPHS)) exactly as module_BL_MYJPBL.F90:307-309.

RXNER is a Fortran argument of MIXLEN that its body never reads; it is not passed.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_mixlen"
KERNEL = "phys_myj_mixlen"
BLOCK = 128
MIXLEN_KMAX = 128          # kernel per-thread column scratch (phys_myj_mixlen.cu MIXLEN_KMAX)

OUT_GM = "MYJ_GM"
OUT_GH = "MYJ_GH"
OUT_EL = "MYJ_EL"
OUT_LPBL = "MYJ_LPBL"
OUT_LMXL = "MYJ_LMXL"
OUT_CT = "MYJ_CT"


def _required(st, name, min_size, dtype="float32"):
    """A required state array. Absent, wrong dtype or too small is a refusal, not a zero-fill."""
    if name not in st.fields:
        raise RuntimeError(f"phys-myj-mixlen: input {name!r} is not in the state")
    arr = st.fields[name]
    if str(arr.dtype) != dtype:
        raise RuntimeError(f"phys-myj-mixlen: input {name!r} is {arr.dtype}, expected {dtype}")
    if int(arr.size) < min_size:
        raise RuntimeError(f"phys-myj-mixlen: input {name!r} has {int(arr.size)} values, "
                           f"needs at least {min_size}")
    return arr


def _output(st, name, shape, dtype="float32"):
    """A unit output: the state table array when it exists, otherwise a unit-owned scratch array."""
    if name in st.fields:
        return st.fields[name]
    return st.add(name, shape, dtype)


def run(ctx):
    """Launch MIXLEN over every column. Returns the output arrays by name."""
    if ctx.cfg is None:
        raise RuntimeError("phys-myj-mixlen: ctx.cfg is required (DT and NPHS come from it, as MYJPBL)")
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if not 2 <= lm <= MIXLEN_KMAX:
        raise RuntimeError(f"phys-myj-mixlen: LM={lm} is outside 2..{MIXLEN_KMAX} "
                           "(the kernel column scratch size)")
    shape3 = (lm, jm, im)
    n3 = lm * jm * im

    # Fortran RSQDT = SQRT(RDTTURBL), RDTTURBL = 1/(DT*NPHS)   (module_BL_MYJPBL.F90:307-309)
    rsqdt = math.sqrt(1.0 / (int(ctx.cfg.nphs) * float(ctx.cfg.dt)))

    u = _required(st, "U", n3)
    v = _required(st, "V", n3)
    q2 = _required(st, "Q2", n3)
    thv = _required(st, "THV", n3)
    the = _required(st, "THE", n3)
    epsq2 = _required(st, "EPSQ2", lm)
    epsl = _required(st, "EPSL", lm)
    zint = _required(st, "MYJ_ZINT", (lm + 1) * jm * im)
    p = _required(st, "MYJ_P", n3)
    psk = _required(st, "MYJ_PSK", n3)
    lmh = _required(st, "MYJ_LMH", jm * im, dtype="int32")

    gm = st.add(OUT_GM, shape3)
    gh = st.add(OUT_GH, shape3)
    el = st.add(OUT_EL, shape3)
    lpbl = st.add(OUT_LPBL, (jm, im), "int32")
    lmxl = st.add(OUT_LMXL, (jm, im), "int32")
    ct = st.add(OUT_CT, (jm, im))
    pblh = _output(st, "PBLH", (jm, im))
    mixht = _output(st, "MIXHT", (jm, im))

    k = kernels.get(UNIT, KERNEL)
    ncol = im * jm
    grid = ((ncol + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (u, v, thv, the, q2, epsq2, epsl, zint, p, psk, lmh,
           np.float32(rsqdt),
           gm, gh, el, pblh, lpbl, lmxl, ct, mixht,
           np.int32(im), np.int32(jm), np.int32(lm)))

    return {OUT_GM: gm, OUT_GH: gh, OUT_EL: el, OUT_LPBL: lpbl, OUT_LMXL: lmxl,
            OUT_CT: ct, "PBLH": pblh, "MIXHT": mixht}
