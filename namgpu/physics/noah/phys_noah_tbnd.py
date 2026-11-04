"""phys-noah-tbnd: TBND, soil layer boundary temperature by interpolation (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4945-4985. NOAA US
government work; the module header is kept as the notice in the kernel file. Start point:
ArWen core/kernels/noah.cu noah_tbnd (Apache-2.0, third_party/arwen/LICENSE).
Kernel: namgpu/kernels/phys_noah_tbnd.cu (phys_noah_tbnd, one thread per (i, j, layer), one launch,
grid.z = NSOIL). No host sync, no per-step allocation, current stream.

Inputs: STC [NSOIL, JM, IM] float32, TBOT [JM, IM] float32, ZSOIL [NSOIL] float32 (column
independent), ZBOT scalar. Output: TBND [NSOIL, JM, IM] float32; entry k (0-based) is the Fortran
TBND1 for K = k + 1, with TB = STC(k+1) and TB = TBOT for the last layer, as the three SFLX calls use it.

Caller bindings in run(ctx) (nothing is defaulted on the ctx path):
  STC    ctx.st.fields["STC"] (NMMB state table).
  TBOT   ctx.st.fields["TBOT"] (state table row, 2D). The SFLX caller's per-column bottom
         temperature: TMN where ICE == 0, 271.16 K where ICE is set (module_LS_NOAHLSM.F90:779-783).
         Missing -> RuntimeError.
  ZBOT   ctx.tables["noah_parm_host"].gen["ZBOT_DATA"] (GENPARM.TBL, read by phys-noah-parm-tables;
         the same value redprm passes to SFLX, module_LS_NOAHLSM.F90:3596). Missing -> RuntimeError.
  ZSOIL  ctx.st.fields["ZSOIL"] when present; otherwise registered with st.add and filled with the NAM
         layer bottoms, exactly as phys-noah-smflx does, so the two units agree.
  TBND   ctx.st.add("TBND", ...), unit-owned output.

Without ctx it builds a full-size random 954 x 835 batch with GENPARM ZBOT_DATA for the smoke.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_tbnd"
KERNEL = "phys_noah_tbnd"
BLOCK = (32, 8, 1)
NSOIL = 4
# NAM layer bottoms (m), cumulative SLDPTH 0.1/0.3/0.6/1.0 (module_LS_NOAHLSM.F90:1835-1837).
ZSOIL_NAM = (-0.1, -0.4, -1.0, -2.0)
# GENPARM.TBL ZBOT_DATA in the WRF-4.7.1 run directory copy on node-2. Used only by the no-ctx smoke.
ZBOT_SMOKE = -8.0

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, NSOIL)


def tbnd(stc, tbot, zsoil, zbot, out):
    """OUT[k] = TBND for layer k (0-based), from STC, TBOT, ZSOIL and ZBOT. Float32 device arrays."""
    _, jm, im = (int(s) for s in stc.shape)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (stc, tbot, zsoil, np.float32(zbot), out, np.int32(im), np.int32(jm)))
    return out


def _zbot_from_ctx(ctx) -> float:
    from namgpu.physics.noah import phys_noah_parm_tables as parm

    host = ctx.tables.get(parm.KEY_HOST)
    if host is None:
        raise RuntimeError("phys-noah-tbnd: ctx.tables has no 'noah_parm_host'; "
                           "phys-noah-parm-tables must run first (ZBOT_DATA)")
    if "ZBOT_DATA" not in host.gen:
        raise RuntimeError("phys-noah-tbnd: GENPARM host table has no ZBOT_DATA")
    return float(host.gen["ZBOT_DATA"])


def _require(st, name: str):
    if name not in st.fields:
        raise RuntimeError(f"phys-noah-tbnd: state has no {name}; the SFLX caller must bind it")
    return st.fields[name]


def run(ctx=None):
    """Done-rule entry. With ctx: bindings as in the module docstring, output TBND in ctx.st.
    Without ctx: one launch on a full-size 954 x 835 x 4 random soil column set."""
    import cupy as cp

    if ctx is not None:
        st = ctx.st
        stc = _require(st, "STC")
        tbot = _require(st, "TBOT")
        jm, im = int(stc.shape[1]), int(stc.shape[2])
        if tuple(tbot.shape) != (jm, im):
            raise RuntimeError(f"phys-noah-tbnd: TBOT shape {tuple(tbot.shape)}, need ({jm}, {im})")
        if "ZSOIL" in st.fields:
            zsoil = st.fields["ZSOIL"]
        else:
            zsoil = st.add("ZSOIL", (NSOIL,))
            zsoil[...] = cp.asarray(np.array(ZSOIL_NAM, dtype=np.float32))
        zbot = _zbot_from_ctx(ctx)
        out = st.add("TBND", (NSOIL, jm, im))
    else:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        stc = cp.asarray(rng.uniform(250.0, 271.0, size=(NSOIL, jm, im)).astype(np.float32))
        tbot = cp.asarray(rng.uniform(265.0, 285.0, size=(jm, im)).astype(np.float32))
        zsoil = cp.asarray(np.array(ZSOIL_NAM, dtype=np.float32))
        out = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
        zbot = ZBOT_SMOKE
    tbnd(stc, tbot, zsoil, zbot, out)
    return {"TBND": out}
