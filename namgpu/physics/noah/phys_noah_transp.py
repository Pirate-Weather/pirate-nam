"""phys-noah-transp: TRANSP, transpiration for the vegetation class (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:5200-5309 (TRANSP). NOAA US
government work; the NOAA header of that module is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_transp.cu (phys_noah_transp, one launch, one thread per column,
grid.z = 1). Started from the ArWen noah_transp (namgpu/third_party/arwen/core/kernels/noah.cu,
Apache-2.0), which matches the NAM routine term for term.

Caller gate (in the kernel, not left to the caller): the only call, EVAPO (module_LS_NOAHLSM.F90:
2600-2613), runs TRANSP only inside IF (ETP1 > 0.0) and IF (SHDFAC > 0.0), with ET(K) = 0 set before.
The kernel writes ET = 0 for any column failing either test, so ETP1 < 0 never gives negative ET.

NROOT contract: 1 <= NROOT <= NSOIL (module_LS_NOAHLSM.F90:2360). The VEGPARM loader enforces it
(phys_noah_parm_tables.py, NoahParmTableError on NROTBL outside 1..NSOIL). The kernel NaN-poisons an
active column whose NROOT is outside that range rather than indexing past its layer arrays.

ETP1 = ETP * 0.001 (NOPAC 3247, SNOPAC 4147/4152), formed per point from the ETP state field.

ctx path (run(ctx)): every input is bound from ctx.st by the field's producer (REQUIRED below). A
missing producer is a RuntimeError that names it; there is no stand-in value. ET is written to the
state field "ET" [NSOIL, JM, IM] and ETP1 to "ETP1" [JM, IM], both created once and reused.
SH2O is the TRANSP dummy SMC. No host sync, no per-step allocation after the first call.

Without ctx, run() builds a full-size 954 x 835 synthetic batch for the done-rule check only.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_transp"
KERNEL = "phys_noah_transp"
BLOCK = (32, 8, 1)
NSOIL = 4

# state field -> producer, for the ctx path.
REQUIRED = {
    "ETP": "phys-noah-penman (ETP, kg m-2 s-1)",
    "CMC": "NAM state (CMC)",
    "SH2O": "phys-noah-sflx / phys-noah-init",
    "RTDIS": "phys-noah-redprm",
    "NROOT": "phys-noah-redprm",
    "SHDFAC": "phys-noah-redprm",
    "SMCWLT": "phys-noah-redprm",
    "SMCREF": "phys-noah-redprm",
    "CMCMAX": "phys-noah-redprm",
    "CFACTR": "phys-noah-redprm",
    "PC": "phys-noah-canres",
}

_scratch: dict = {}


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def transp(smc, etp1, cmc, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, nroot, rtdis, et):
    """Writes ET [NSOIL, JM, IM] from SH2O (smc), ETP1 [JM, IM] and RTDIS [NSOIL, JM, IM] (device)."""
    nsoil, jm, im = (int(s) for s in smc.shape)
    if nsoil != NSOIL:
        raise ValueError(f"phys_noah_transp: smc must be [NSOIL={NSOIL}, JM, IM], got {tuple(smc.shape)}")
    for name, arr, dt in (("smc", smc, np.float32), ("etp1", etp1, np.float32), ("cmc", cmc, np.float32),
                          ("shdfac", shdfac, np.float32), ("smcwlt", smcwlt, np.float32),
                          ("cmcmax", cmcmax, np.float32), ("pc", pc, np.float32),
                          ("cfactr", cfactr, np.float32), ("smcref", smcref, np.float32),
                          ("nroot", nroot, np.int32), ("rtdis", rtdis, np.float32), ("et", et, np.float32)):
        if np.dtype(arr.dtype) != np.dtype(dt):
            raise TypeError(f"phys_noah_transp: {name} must be {np.dtype(dt).name}, got {arr.dtype}")
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (smc, etp1, cmc, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, nroot, rtdis, et,
       np.int32(im), np.int32(jm)))
    return et


def _require(st, names):
    missing = [n for n in names if n not in st.fields]
    if missing:
        detail = "; ".join(f"{n} from {REQUIRED[n]}" for n in missing)
        raise RuntimeError(f"phys-noah-transp: state fields not produced yet: {detail}")


def _ctx_run(ctx):
    import cupy as cp

    st = ctx.st
    _require(st, tuple(REQUIRED))
    jm, im = int(st.SH2O.shape[1]), int(st.SH2O.shape[2])
    etp1 = st.add("ETP1", (jm, im))
    cp.multiply(st.ETP, np.float32(0.001), out=etp1)   # ETP1 = ETP * 0.001 (NOPAC 3247)
    et = st.add("ET", (NSOIL, jm, im))
    transp(st.SH2O, etp1, st.CMC, st.SHDFAC, st.SMCWLT, st.CMCMAX, st.PC, st.CFACTR, st.SMCREF,
           st.NROOT, st.RTDIS, et)
    return {"ET": et}


def _synthetic_batch(jm: int, im: int, seed: int):
    import cupy as cp

    rng = np.random.default_rng(seed)
    rtdis = rng.uniform(0.0, 1.0, size=(NSOIL, jm, im))
    rtdis = rtdis / rtdis.sum(axis=0, keepdims=True)
    return dict(
        smc=cp.asarray(rng.uniform(0.02, 0.45, size=(NSOIL, jm, im)).astype(np.float32)),
        etp1=cp.asarray(rng.uniform(-2.0e-4, 5.0e-4, size=(jm, im)).astype(np.float32)),
        cmc=cp.asarray(rng.uniform(0.0, 0.5, size=(jm, im)).astype(np.float32)),
        shdfac=cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32)),
        smcwlt=cp.asarray(rng.uniform(0.02, 0.1, size=(jm, im)).astype(np.float32)),
        cmcmax=cp.asarray(rng.uniform(0.5, 1.0, size=(jm, im)).astype(np.float32)),
        pc=cp.asarray(rng.uniform(0.2, 1.0, size=(jm, im)).astype(np.float32)),
        cfactr=cp.asarray(rng.uniform(0.3, 0.7, size=(jm, im)).astype(np.float32)),
        smcref=cp.asarray(rng.uniform(0.15, 0.4, size=(jm, im)).astype(np.float32)),
        nroot=cp.asarray(rng.integers(1, NSOIL + 1, size=(jm, im)).astype(np.int32)),
        rtdis=cp.asarray(rtdis.astype(np.float32)),
    )


def run(ctx=None):
    """Done-rule entry. With ctx: bind ctx.st by producer name, write ET into ctx.st. Without ctx:
    one launch on a full-size 954 x 835 synthetic batch (the done-rule check). Returns the arrays written."""
    import cupy as cp

    if ctx is not None:
        return _ctx_run(ctx)
    jm, im = 835, 954
    key = (jm, im)
    if key not in _scratch:
        b = _synthetic_batch(jm, im, seed=3)
        b["et"] = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
        _scratch[key] = b
    b = _scratch[key]
    transp(b["smc"], b["etp1"], b["cmc"], b["shdfac"], b["smcwlt"], b["cmcmax"], b["pc"],
           b["cfactr"], b["smcref"], b["nroot"], b["rtdis"], b["et"])
    return {"ET": b["et"]}
