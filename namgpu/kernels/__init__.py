"""Kernel loader: one .cu per unit, compiled once per process with NVRTC through CuPy.

Convention (CONTRACT.md section 3):
- File ``kernels/<unit_id_with_underscores>.cu``; every ``extern "C" __global__`` function in it
  is a kernel of that unit.  Include ``nmmb_common.cuh`` for the index macros and constants.
- Fast numerics: NVRTC default FMA contraction ON (``--fmad=true``), IEEE division/sqrt default,
  no ``--use_fast_math`` (it changes expf/logf accuracy; units may opt in with a reason).
- Single precision throughout (``float``), as NMMB kfpt=single; double only where the Fortran
  declares kdbl.
- ``get(unit, kernel)`` returns a cached ``cupy.RawKernel``; ``launch_2d``/``launch_cols`` give
  the standard grid shapes.  No host synchronisation anywhere in a wrapper.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPTIONS = ("-std=c++17", "--fmad=true", f"-I{HERE}")


@lru_cache(maxsize=None)
def module(unit: str):
    import cupy as cp
    src = (HERE / f"{unit}.cu").read_text()
    return cp.RawModule(code=src, options=OPTIONS, name_expressions=None)


def get(unit: str, kernel: str):
    return module(unit).get_function(kernel)


def compile_all() -> dict[str, str]:
    """Compile every .cu here; returns {unit: 'ok' | error text}.  Used by the done rule."""
    out = {}
    for cu in sorted(HERE.glob("*.cu")):
        try:
            module(cu.stem).compile()
            out[cu.stem] = "ok"
        except Exception as e:  # noqa: BLE001
            out[cu.stem] = f"{type(e).__name__}: {e}"
    return out


# Standard launch shapes.  Block sizes are tuning knobs; keep them here, not in wrappers.
BLOCK_2D = (32, 8, 1)
BLOCK_COL = (128, 1, 1)


def grid_2d(im: int, jm: int, lm: int = 1):
    """One thread per (i, j[, l]) point: grid.z runs over levels."""
    bx, by, _ = BLOCK_2D
    return ((im + bx - 1) // bx, (jm + by - 1) // by, lm)


def grid_cols(im: int, jm: int):
    """One thread per column (vertical recurrences, physics columns)."""
    n = im * jm
    return ((n + BLOCK_COL[0] - 1) // BLOCK_COL[0], 1, 1)


def trim_local_memory(reason: str = "") -> float:
    """Give back the device local-memory (stack) reservation a high-lmem kernel left behind.

    The CUDA driver keeps ONE local-memory reservation for the whole device (bytes per thread x every
    resident thread), sized to the largest per-thread stack any launched kernel needed, until the stack
    limit is set again. phys_myj_tables:phys_myj_tablett needs 28,848 B/thread, so after init about 7 GiB
    of a 32 GB RTX 5090 stayed reserved for the whole forecast (memory census 2026-11-04: OOM with 22.5 GiB
    of pool on a 31.8 GiB card). Setting the limit back to 1 KiB frees it; later launches grow it again to
    what the step kernels need. Results are unchanged (local memory is per-thread scratch).
    Returns the GiB freed (device used delta).
    """
    import cupy as cp
    rt = cp.cuda.runtime
    cp.cuda.Device().synchronize()
    f0, _ = rt.memGetInfo()
    # Keep exactly what the step loop needs, reserved now while the CuPy pool is still small: if the driver
    # has to GROW the reservation later, at a launch, the pool already holds the free memory and the launch
    # fails with CUDA_ERROR_OUT_OF_MEMORY (poison run, phys_rrtm_lw_rtrnmr at 23.5 GiB of pool).
    # Largest step-loop kernel frame: phys_myj_myjpbl_column 10,920 B/thread (scan 2026-11-04); 11 KiB
    # reserves ~2.7 GiB on a 170-SM card. NAMGPU_LMEM_RESERVE overrides (bytes per thread).
    import os
    keep = int(os.environ.get("NAMGPU_LMEM_RESERVE", "11264"))
    rt.deviceSetLimit(getattr(rt, "cudaLimitStackSize", 0), keep)
    f1, _ = rt.memGetInfo()
    return round((f1 - f0) / 2**30, 2)
