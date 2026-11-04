"""phys-noah-snopac: SNOPAC, snow energy and water balance (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4050-4374. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel files.
Kernel: namgpu/kernels/phys_noah_snopac.cu (phys_noah_snopac, one thread per column, grid_cols);
the device function nam_snopac lives in namgpu/kernels/phys_noah_snopac.cuh for other units.
Started from the ArWen noah.cu SNOPAC section (third_party/arwen/core/kernels/noah.cu, Apache-2.0).

SNOPAC calls EVAPO, SMFLX, SHFLX and SNOWPACK, which are CONTRACT units phys-noah-evapo,
phys-noah-smflx, phys-noah-shflx and phys-noah-snowpack. Until their device functions exist in
their headers the kernel does not link, so run() cannot pass the done rule yet.

Arrays are bound by their CONTRACT names (CONTRACT section 1): soil [NSOIL, JM, IM], columns
[JM, IM], integer flags int32. ctx binding is not written: the REDPRM soil-type values and
ZSOIL/RTDIS come from phys-noah-parm-tables and phys-noah-init, which are not landed.
One launch, no host sync, no per-step allocation, everything on the current stream.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snopac"
KERNEL = "phys_noah_snopac"
NSOIL = 4
IM_FULL, JM_FULL = 954, 835
BLOCK = (128, 1, 1)

# Launch order, identical to the kernel signature in phys_noah_snopac.cu.
ARG_ORDER = (
    "etp", "prcp", "prcpf", "snowng", "smcmax", "smcwlt", "smcref", "smcdry", "cmcmax",
    "sbeta", "df1", "q2", "sfctmp", "t24", "th2", "fdown", "f1", "bexp", "pc", "rch", "rr",
    "cfactr", "slope", "kdt", "frzfact", "psisat", "dwsat", "dksat", "tbot", "zbot", "shdfac",
    "nroot", "vegtyp", "ice", "quartz", "fxexp", "csoil", "emissi", "ribb", "flx2",
    "smc", "sh2o", "stc", "zsoil", "rtdis", "et",
    "cmc", "t1", "sncovr", "esd", "sndens", "snowh", "beta",
    "dew", "edir", "ec", "ett", "etns", "esnow", "flx1", "flx3", "ssoil", "snomlt", "drip",
    "runoff1", "runoff2", "runoff3",
)


INT_COLS = frozenset(("snowng", "nroot", "vegtyp", "ice"))
SOIL_ARRS = frozenset(("smc", "sh2o", "stc", "zsoil", "rtdis", "et"))


def _check(a: dict, im: int, jm: int):
    """Host-side metadata check (shape, dtype, layout). No device sync.

    The kernel reads raw device pointers, so a float64 or int64 array would be read as garbage.
    """
    for n in ARG_ORDER:
        if n not in a:
            raise KeyError(f"snopac: missing array {n}")
        arr = a[n]
        want_shape = (NSOIL, jm, im) if n in SOIL_ARRS else (jm, im)
        want_dtype = np.dtype(np.int32) if n in INT_COLS else np.dtype(np.float32)
        if tuple(arr.shape) != want_shape:
            raise ValueError(f"snopac: {n} shape {tuple(arr.shape)}, want {want_shape}")
        if np.dtype(arr.dtype) != want_dtype:
            raise TypeError(f"snopac: {n} dtype {arr.dtype}, want {want_dtype}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"snopac: {n} is not C-contiguous")


def snopac(a: dict, dt: float, ivegsrc: int, im: int, jm: int):
    """Run SNOPAC over every column. a maps CONTRACT names to CuPy arrays.

    Soil arrays are [NSOIL, JM, IM]; column arrays [JM, IM]. INOUT arrays are updated in place.
    Shape, dtype and layout are checked on the host before launch.
    """
    _check(a, im, jm)
    k = kernels.get(UNIT, KERNEL)
    args = [a[n] for n in ARG_ORDER]
    args += [np.float32(dt), np.int32(ivegsrc), np.int32(im), np.int32(jm)]
    k(kernels.grid_cols(im, jm), BLOCK, tuple(args))
    return a


def _random_batch(im: int, jm: int, seed: int = 1):
    import cupy as cp

    rng = np.random.default_rng(seed)

    def col(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(jm, im)).astype(np.float32))

    def soil(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(NSOIL, jm, im)).astype(np.float32))

    zs = np.array([-0.1, -0.4, -1.0, -2.0], np.float32)[:, None, None]
    a = {
        "etp": col(-5.0, 300.0), "prcp": col(0.0, 1.0e-3), "prcpf": col(0.0, 1.0e-3),
        "snowng": cp.asarray(rng.integers(0, 2, size=(jm, im)).astype(np.int32)),
        "smcmax": col(0.4, 0.5), "smcwlt": col(0.02, 0.1), "smcref": col(0.2, 0.3),
        "smcdry": col(0.02, 0.1), "cmcmax": col(0.0, 0.5e-3), "sbeta": col(-2.0, 2.0),
        "df1": col(0.1, 2.0), "q2": col(0.001, 0.02), "sfctmp": col(255.0, 285.0),
        "t24": col(255.0, 285.0), "th2": col(255.0, 285.0), "fdown": col(0.0, 400.0),
        "f1": col(0.1, 0.9), "bexp": col(2.0, 12.0), "pc": col(0.1, 0.9), "rch": col(0.001, 0.01),
        "rr": col(0.5, 2.0), "cfactr": col(0.1, 0.9), "slope": col(0.0, 1.0), "kdt": col(1.0, 3.0),
        "frzfact": col(0.1, 1.0), "psisat": col(0.05, 0.7), "dwsat": col(1.0e-6, 1.0e-4),
        "dksat": col(1.0e-6, 1.0e-4), "tbot": col(260.0, 285.0), "zbot": col(-3.0, -2.0),
        "shdfac": col(0.0, 1.0),
        "nroot": cp.asarray(rng.integers(1, 4, size=(jm, im)).astype(np.int32)),
        "vegtyp": cp.asarray(rng.integers(1, 14, size=(jm, im)).astype(np.int32)),
        "ice": cp.zeros((jm, im), dtype=cp.int32),
        "quartz": col(0.1, 0.9), "fxexp": col(1.0, 3.0), "csoil": col(1.0e6, 2.0e6),
        "emissi": col(0.9, 0.99), "ribb": col(-1.0, 1.0), "flx2": col(0.0, 10.0),
        "smc": soil(0.1, 0.4), "sh2o": soil(0.05, 0.3), "stc": soil(255.0, 285.0),
        "zsoil": cp.asarray(np.broadcast_to(zs, (NSOIL, jm, im)).copy()),
        "rtdis": soil(0.0, 0.5), "et": cp.zeros((NSOIL, jm, im), dtype=cp.float32),
        "cmc": col(0.0, 0.5e-3), "t1": col(255.0, 280.0), "sncovr": col(0.0, 1.0),
        "esd": col(0.0, 0.5), "sndens": col(0.1, 0.4), "snowh": col(0.0, 1.0),
        "beta": col(0.0, 1.0),
    }
    for n in ("dew", "edir", "ec", "ett", "etns", "esnow", "flx1", "flx3", "ssoil", "snomlt",
              "drip", "runoff1", "runoff2", "runoff3"):
        a[n] = cp.zeros((jm, im), dtype=cp.float32)
    return a


def run(ctx=None, *, dt: float = 25.0, ivegsrc: int = 1, im: int = IM_FULL, jm: int = JM_FULL):
    """Done-rule entry: one launch over a full-size random column set, then the finite check."""
    import cupy as cp

    if ctx is not None:
        raise NotImplementedError("snopac: ctx binding waits for phys-noah-parm-tables and phys-noah-init")
    a = _random_batch(im, jm)
    snopac(a, dt, ivegsrc, im, jm)
    cp.cuda.Stream.null.synchronize()
    for name, arr in a.items():
        if arr.dtype.kind == "f" and not bool(cp.isfinite(arr).all()):
            raise FloatingPointError(f"snopac: non-finite values in {name}")
    return a
