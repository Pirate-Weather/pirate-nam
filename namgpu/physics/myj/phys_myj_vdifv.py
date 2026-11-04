"""phys-myj-vdifv: VDIFV, vertical diffusion of the horizontal wind by the MYJ PBL.

Fortran: phys/module_BL_MYJPBL.F90:1527-1607 (NAM nam.v4.2.11, 83df4e13c1), called from MYJPBL at
line 733 once per column. Kernel: namgpu/kernels/phys_myj_vdifv.cu (phys_myj_vdifv_columns, one
thread per column, grid_cols). Notice: NOAA/NCEP NMMB source, kept in the .cu header.

Inputs (device arrays, CONTRACT names; Fortran (i,j,l) -> [l, j, i]):
  U, V        [LM, JM, IM]  winds on which VDIFV runs. The kernel overwrites them in place, so the
              caller passes scratch copies (UK, VK), never the state U/V.
  Z           [LM+1, JM, IM] interface heights as used by VDIFV (decreasing with level index)
  UZ0, VZ0    [JM, IM]      top-of-viscous-sublayer winds (state, read only)
  AKMS_DENS   [JM, IM]      RKMS of the call (MYJPBL: AKMS*RHOK(LM))
  D, RKM      [LM, JM, IM]  DCOL and AKMK of the MYJPBL driver (levels 1..LM-1 used)
  RHO         [LM, JM, IM]  RHOK of the MYJPBL driver
  LMH         [JM, IM] int32 LMH of the column (MYJPBL sets LMH=LM, lines 364 and 605)
Scalar: dtdif = DTDIF = DTTURBL = DT*NPHS (module_BL_MYJPBL.F90:307-310), passed by the caller.

Caller contract (module_BL_MYJPBL.F90:740-742): the tendency is derived after VDIFV as
RUBLTEN = (UK - U) * RDTTURBL, with RDTTURBL = 1/DTTURBL. ``vdifv_tendency`` implements exactly that
for the unit: it copies the state winds to scratch, runs VDIFV on the copies, and writes the tendency.
The state U and V are only read. The production MYJPBL path is the fused kernel
namgpu/kernels/phys_myj_myjpbl.cu, which does the same copy and tendency at its lines 640-643.

Refusals (no silent fallback):
  - LM must be 2..LMAX (the kernel column scratch is LMAX levels; the NAM VDIFV needs CM(LMH-1)).
  - The kernel leaves a column with LMH < 2 unchanged. That guard is unreachable from the MYJPBL
    caller, which sets LMH = LM and refuses LM < 4 (phys_myj_myjpbl.py). NAM would read CM(0) and
    RKM(0), which do not exist, so the guard exists only to avoid that read.

run(ctx): done-rule entry. Test inputs live in their own scratch arrays (MYJ_TEST_*): positive RKM
and D, RHO near 1 kg m-3, decreasing interface heights, LMH = LM, DTDIF from cfg (DT*NPHS). The
kernel runs once on the full 954 x 835 x 60 state. The state U, V and Z are not written. Nothing is
compared; the check is that the state stays finite.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_vdifv"
KERNEL = "phys_myj_vdifv_columns"
LMAX = 64          # must equal VDIFV_LMAX in kernels/phys_myj_vdifv.cu


def vdifv_columns(IM, JM, LM, dtdif, lmh, uz0, vz0, akms_dens, D, RKM, Z, RHO, U, V, stream=None):
    """Run VDIFV over every column in place on U, V. All arguments are device arrays; no host sync."""
    if LM > LMAX:
        raise ValueError(f"phys-myj-vdifv: LM={LM} exceeds the kernel column scratch (LMAX={LMAX})")
    if LM < 2:
        raise ValueError(f"phys-myj-vdifv: LM={LM} < 2; NAM VDIFV reads CM(0) and RKM(0) and has no such level")
    grid, block = kernels.grid_cols(IM, JM), kernels.BLOCK_COL
    kern = kernels.get(UNIT, KERNEL)
    args = (
        np.int32(IM), np.int32(JM), np.int32(LM), np.float32(dtdif),
        lmh, uz0, vz0, akms_dens, D, RKM, Z, RHO, U, V,
    )
    kern(grid, block, args, stream=stream)
    return U, V


def vdifv_tendency(*, u, v, uk, vk, dudt, dvdt, uz0, vz0, akms_dens, D, RKM, Z, RHO, LMH,
                   dtdif, rdttturbl, stream=None):
    """VDIFV with the MYJPBL caller contract (module_BL_MYJPBL.F90:740-742).

    u, v are the state winds (read only). uk, vk are scratch copies that VDIFV updates in place.
    dudt, dvdt receive (UK - U) * RDTTURBL. dtdif and rdttturbl must be the caller's DTTURBL values
    (DT*NPHS and its reciprocal). No host sync.
    """
    import cupy as cp

    LM, JM, IM = (int(n) for n in u.shape)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        cp.copyto(uk, u)
        cp.copyto(vk, v)
        vdifv_columns(IM, JM, LM, dtdif, LMH, uz0, vz0, akms_dens, D, RKM, Z, RHO, uk, vk, stream=stream)
        dudt[...] = (uk - u) * np.float32(rdttturbl)
        dvdt[...] = (vk - v) * np.float32(rdttturbl)
    return dudt, dvdt


def _test_inputs(ctx):
    """Test scratch for the done-rule smoke. Never the state arrays."""
    import cupy as cp

    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    rng = cp.random.default_rng(11)
    D = st.add("MYJ_TEST_D", (lm, jm, im))
    RKM = st.add("MYJ_TEST_RKM", (lm, jm, im))
    RHO = st.add("MYJ_TEST_RHO", (lm, jm, im))
    AKMS = st.add("MYJ_TEST_AKMS_DENS", (jm, im))
    LMH = st.add("MYJ_TEST_LMH", (jm, im), dtype="int32")
    Z = st.add("MYJ_TEST_Z", (lm + 1, jm, im))
    UK = st.add("MYJ_TEST_UK", (lm, jm, im))
    VK = st.add("MYJ_TEST_VK", (lm, jm, im))
    DUDT = st.add("MYJ_TEST_DUDT", (lm, jm, im))
    DVDT = st.add("MYJ_TEST_DVDT", (lm, jm, im))
    D[...] = rng.uniform(0.0, 1.0, (lm, jm, im)).astype(np.float32)
    RKM[...] = rng.uniform(0.5, 50.0, (lm, jm, im)).astype(np.float32)
    RHO[...] = rng.uniform(0.9, 1.3, (lm, jm, im)).astype(np.float32)
    AKMS[...] = rng.uniform(0.01, 0.1, (jm, im)).astype(np.float32)
    LMH[...] = lm
    # interface heights decreasing with the level index (index 0 = model top), 30 km top
    Z[...] = cp.linspace(30000.0, 0.0, lm + 1, dtype=cp.float32).reshape(lm + 1, 1, 1)
    return dict(uk=UK, vk=VK, dudt=DUDT, dvdt=DVDT, D=D, RKM=RKM, RHO=RHO, akms_dens=AKMS,
                LMH=LMH, Z=Z)


def run(ctx):
    """Done-rule entry: one VDIFV launch on the full-size state (954 x 835 x 60). See module docstring."""
    st = ctx.st
    dttrbl = float(ctx.cfg.dt) * int(ctx.cfg.nphs)   # DTTURBL = DT*NPHS (module_BL_MYJPBL.F90:307)
    t = _test_inputs(ctx)
    return vdifv_tendency(u=st.U, v=st.V, uz0=st["UZ0"], vz0=st["VZ0"], dtdif=dttrbl,
                          rdttturbl=1.0 / dttrbl, stream=ctx.stream, **t)
