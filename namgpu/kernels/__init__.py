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

from namgpu import cores as _cores

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


# The step-loop kernels with the largest per-thread frames (local memory). The device local-memory reservation is
# the largest frame launched x every resident thread (SMs x threads/SM), so the step loop needs the max of these.
# Scan 2026-11-05 (tools/kernel_report.py, RTX 5090): phys_gwd_driver 10,016 B, phys_bmj_bmjdrv_column 9,840 B,
# fer_hires 9,216 B, phys_rrtm_sw_mcica_subcol 5,088 B, phys_myj_myjpbl_column (10,920 B while its arrays were sized
# for 128 levels); every other step kernel is smaller. Read from the compiled kernels, so a frame that changes moves
# the reservation with it.
STEP_FRAME_KERNELS = (
    ("phys_gwd_driver", "phys_gwd_driver"),
    ("phys_bmj_bmjdrv", "phys_bmj_bmjdrv_column"),
    ("phys_fer_fer_hires", "fer_hires"),
    ("phys_myj_myjpbl", "phys_myj_myjpbl_column"),
    ("phys_rrtm_sw_mcica_subcol", "phys_rrtm_sw_mcica_subcol"),
)


def step_frame_bytes() -> int:
    """Largest per-thread local-memory frame among STEP_FRAME_KERNELS (bytes, rounded up to 16)."""
    b = max(int(get(u, k).attributes["local_size_bytes"]) for u, k in STEP_FRAME_KERNELS)
    return (b + 15) // 16 * 16


def trim_local_memory(reason: str = "") -> dict:
    """Set the device local-memory (stack) reservation to exactly what the step loop needs, now.

    The CUDA driver keeps ONE local-memory reservation for the whole device (bytes per thread x every
    resident thread), sized to the largest per-thread stack any launched kernel needed, until the stack
    limit is set again. Set here while the CuPy pool is still small: if the driver had to GROW the reservation
    later, at a launch, the pool may already hold the free memory and the launch fails with
    CUDA_ERROR_OUT_OF_MEMORY (poison run, phys_rrtm_lw_rtrnmr at 23.5 GiB of pool). Before the init-only table
    builds moved their work arrays to global memory this also gave back ~7 GiB (phys_myj_tablett 28.8 KB/thread);
    now it usually GROWS the reservation from the init kernels' frames to the step loop's.
    Bytes per thread: step_frame_bytes() (the compiled step kernels); NAMGPU_LMEM_RESERVE overrides.
    Results are unchanged (local memory is per-thread scratch).
    Returns {bytes_per_thread, reserved_gib (bytes x resident threads), delta_gib (device used change, + = grew)}.
    """
    import os

    import cupy as cp
    rt = cp.cuda.runtime
    keep = int(os.environ.get("NAMGPU_LMEM_RESERVE", "0")) or step_frame_bytes()
    cp.cuda.Device().synchronize()
    f0, _ = rt.memGetInfo()
    rt.deviceSetLimit(getattr(rt, "cudaLimitStackSize", 0), keep)
    f1, _ = rt.memGetInfo()
    props = rt.getDeviceProperties(cp.cuda.Device().id)
    threads = int(props["multiProcessorCount"]) * int(props["maxThreadsPerMultiProcessor"])
    return {"bytes_per_thread": keep, "reserved_gib": round(keep * threads / 2**30, 3),
            "delta_gib": round((f0 - f1) / 2**30, 3)}


# ---------------------------------------------------------------------------------------------------------------
# Parallel warm-up of the NVRTC cache (nest door, 2026-11-04). A cold first run compiled every .cu in the step and
# init chain one after another on one core, about 7 min while the card waited (STATUS 18:36Z). warm() compiles
# every module across processes into CuPy's on-disk kernel cache through the same RawModule path module() uses, so
# the cache keys are the ones the run looks up; the run then only loads cubins. The workers never open a CUDA
# context (no VRAM on a shared card): the arch comes from the parent and CuPy's Module.load is skipped there.

def _warm_worker(units, arch):
    import time

    import cupy.cuda.compiler as C

    class _NoLoadModule:                      # compile + cache save only; loading would need a CUDA context
        mapping = {}

        def load(self, *a, **k):
            pass

        def _set_mapping(self, m):
            self.mapping = m

    class _Function:
        Module = _NoLoadModule

    C.function = _Function
    C._get_arch = lambda: arch
    import cupy as cp
    out = {}
    for u in units:
        t = time.perf_counter()
        try:
            cp.RawModule(code=(HERE / f"{u}.cu").read_text(), options=OPTIONS, name_expressions=None).compile()
            out[u] = ("ok", round(time.perf_counter() - t, 2))
        except TypeError as e:
            # CuPy saves the cubin to the cache, then returns the module object; its Cython return type refuses the
            # stand-in, after the save. That is the end of a successful compile here.
            ok = "_NoLoadModule" in str(e)
            out[u] = ("ok" if ok else f"TypeError: {str(e)[:300]}", round(time.perf_counter() - t, 2))
        except Exception as e:  # noqa: BLE001
            out[u] = (f"{type(e).__name__}: {str(e)[:300]}", round(time.perf_counter() - t, 2))
    return out


def card_arch(index: str | None = None) -> str:
    """sm major+minor of the visible card from nvidia-smi (no CUDA context), clipped to NVRTC's maximum the way
    cupy.cuda.compiler._get_arch does."""
    import os
    import subprocess

    import cupy.cuda.compiler as C
    idx = index or (os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0")
    cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", idx],
                        capture_output=True, text=True, timeout=20).stdout.strip().replace(".", "")
    return min(cc, C._get_max_compute_capability(), key=int)


def warm(workers: int | None = None, units=None, log=print) -> dict:
    """Compile every kernel module into the on-disk cache on all cores. Modules already cached only cost a cache
    read. Returns {"wall_s", "workers", "compiled": n, "failed": {...}, "slowest": [...]}."""
    import multiprocessing as mp
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    units = sorted(units or [p.stem for p in HERE.glob("*.cu")])
    # 32 at most: 16 compile all 269 modules cold in ~21 s, and a 96-core rented container (2026-11-04)
    # ran out of threads when 96 workers each started a 64-thread OpenBLAS pool on import.
    workers = max(1, min(workers or min(_cores.usable(), 32), len(units)))
    arch = card_arch()
    # biggest sources first so the long compiles start at once
    units.sort(key=lambda u: -(HERE / f"{u}.cu").stat().st_size)
    chunks = [units[k::workers] for k in range(workers)]
    t0 = time.perf_counter()
    res = {}
    # the workers only compile: one BLAS thread each (spawn copies this environment); restored afterwards
    blas = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in blas}
    os.environ.update({k: "1" for k in blas})
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            for part in ex.map(_warm_worker, chunks, [arch] * len(chunks)):
                res.update(part)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    wall = round(time.perf_counter() - t0, 1)
    failed = {u: v[0] for u, v in res.items() if v[0] != "ok"}
    slow = sorted(((v[1], u) for u, v in res.items()), reverse=True)[:5]
    rec = {"wall_s": wall, "workers": workers, "arch": arch, "modules": len(units), "failed": failed,
           "compile_s_sum": round(sum(v[1] for v in res.values()), 1), "slowest": slow}
    log(f"kernels.warm: {len(units)} modules on {workers} processes in {wall} s "
        f"(sum of per-module time {rec['compile_s_sum']} s), {len(failed)} failed")
    return rec

