"""Device memory pool of a NAM run: one allocator per process, one way to hand unused blocks back, and the numbers.

setup() is called by State.__init__ before the first state array, so every entry (run_forecast, slab_init save,
the package door) gets the same pool. Settings (environment):

NAMGPU_POOL = cupy (default) | async
  cupy : CuPy's caching MemoryPool (best fit, large free blocks split for smaller requests).
  async: the CUDA stream-ordered pool (cudaMallocAsync through cupy.cuda.MemoryAsyncPool); freed ranges are reused
         in stream order with no host sync.
NAMGPU_POOL_CAP = all (default) | init | 0          (cupy pool)
  Grow only when needed. The pool's limit is the reservation it has needed so far, never above
  CUPY_GPU_MEMORY_LIMIT when that is set (the nest door's pre-flight pool cap, nest/cli.py): the grow-only cap used to
  replace CuPy's own limit, so a door run could grow past the budget its pre-flight checked on a shared card. A request that would pass it
  first makes CuPy give every whole unused block back to the driver and retry; only if it still does not fit is the
  limit raised, to the bytes needed plus NAMGPU_POOL_CAP_STEP_MIB (256): a growing working set (init uploads, decode)
  meets the cap once per step instead of at every allocation. Without the cap a request that no cached block fits takes a new block
  while the unused ones stay reserved, so the reservation grows to live + fragments.
  all : capped through init and the step loop. Default since 2026-11-05: with "init" the first forecast steps grew
        the reservation to 3.35 GiB against 2.36 GiB live (333x333x60), about 1 GiB of slack at the process peak.
        Measured on an RTX 5090, 333x333x60 1 h from one saved init (69ed12d): process peak 6,538 -> 5,672 MiB; the
        step loop met the cap twice in 432 steps (cap_hits 17 vs 15 through init alone, +0.2 s host in them).
        (The 2026-11-04 note "every step meets the cap about 1.4 times" predates the band-array cycle fix 7f9c582,
        when dead band arrays waited for the cyclic collector.)
  init: capped through init (decode, interpolation, table builds: one-off allocations), lifted at the first step
        (forecast_start), so the step loop allocates as without the cap. Init reservation 3.40 -> 2.92 GiB
        (100x100x60), 5.03 -> 4.51 GiB (333x333x60), measured 2026-11-04.

release(stream) syncs the stream and returns every unused block (cupy: free_all_blocks; async: trim to the live
bytes). Called after init, after each radiation call on the fit path, and after every hourly history write;
release_slack after each init stage that leaves more than 256 MiB unused (the GFS decode left 1.84 GiB cached at
100x100 while drv-tablep grew the local-memory reservation: a 3,824 MiB init spike against a 3,144 MiB forecast).
stats() gives the pool's live and reserved bytes and the reservation's high-water mark for the run receipt.
Which allocator hands out a block never changes a result: every array is written before it is read (the
scratch poison A/B, state.py), and the identity check runs both arms from one saved init.
"""
from __future__ import annotations

import os
import time

_STATE: dict = {}
_HITS = [0, 0.0]     # requests that met the cap (each returned the unused whole blocks first), host seconds in them


def _round(n: int) -> int:
    return (int(n) + 511) // 512 * 512          # CuPy's allocation unit


def setup() -> dict:
    """Install the run's allocator once per process; returns the settings (also in stats())."""
    if _STATE:
        return _STATE["cfg"]
    import cupy as cp

    mode = os.environ.get("NAMGPU_POOL", "cupy").strip().lower() or "cupy"
    if mode not in ("cupy", "async"):
        raise ValueError(f"NAMGPU_POOL={mode!r}: use cupy or async")
    cap = os.environ.get("NAMGPU_POOL_CAP", "all").strip().lower() or "all"
    if cap not in ("init", "all", "0", "1"):
        raise ValueError(f"NAMGPU_POOL_CAP={cap!r}: use init, all or 0")
    cap = {"1": "all"}.get(cap, cap)
    if mode != "cupy":
        cap = "0"
    plain = cp.cuda.get_allocator()
    if mode == "async":
        pool = cp.cuda.MemoryAsyncPool()
        cp.cuda.set_allocator(pool.malloc)
    else:
        pool = cp.get_default_memory_pool()
        if cap != "0":
            import threading
            OOM = cp.cuda.memory.OutOfMemoryError
            ceiling = int(pool.get_limit())  # CUPY_GPU_MEMORY_LIMIT as CuPy read it (0 = none)
            _STATE["ceiling"] = ceiling
            pool.set_limit(size=max(pool.total_bytes(), 1 << 20) if not ceiling
                           else min(max(pool.total_bytes(), 1 << 20), ceiling))
            pmalloc = pool.malloc
            hits = _HITS
            step = int(float(os.environ.get("NAMGPU_POOL_CAP_STEP_MIB", "256")) * 2**20)
            grow = threading.Lock()          # init uploads allocate from several threads (slab_init, decode)

            clock = time.perf_counter

            def malloc(n):
                t0 = clock()
                try:
                    p = pmalloc(n)
                except OOM:
                    hits[0] += 1
                    # CuPy has already handed every whole unused block back. Raise the cap only while the request
                    # still passes it; a real out-of-memory (it fits the cap, the driver has no memory) raises.
                    with grow:
                        while True:
                            need = pool.total_bytes() + _round(n)
                            if need <= pool.get_limit():
                                try:
                                    p = pmalloc(n)
                                    break
                                except OOM:
                                    if pool.total_bytes() + _round(n) <= pool.get_limit():
                                        raise
                                    continue
                            if ceiling and need > ceiling:
                                raise                # past CUPY_GPU_MEMORY_LIMIT: CuPy's own out-of-memory
                            pool.set_limit(size=need + step if not ceiling else min(need + step, ceiling))
                    hits[1] += clock() - t0
                return p

            cp.cuda.set_allocator(malloc)
            _STATE["capped"] = malloc
    _STATE.update(pool=pool, mode=mode, cap=cap, plain=plain, lifted=cap not in ("init", "all"),
                  cfg={"pool": mode, "cap": cap, "how": "NAMGPU_POOL / NAMGPU_POOL_CAP"})
    return _STATE["cfg"]


def forecast_start(stream=None) -> None:
    """Called at the top of every step (Driver.step): the first call ends init. It hands the unused init blocks back.
    NAMGPU_POOL_CAP=all: the cap starts again from the forecast's own reservation (the init's decode and table builds
    had raised it, to 3.74 GiB at 333x333 from a GFS start, which the step loop then filled with 1.48 GiB of slack at
    the process peak). NAMGPU_POOL_CAP=init: the cap is lifted, so the step loop runs on the plain CuPy allocator."""
    if _STATE.get("lifted", True):
        return
    import cupy as cp

    _STATE["lifted"] = True
    release(stream)
    _STATE["init_reserved_high_gib"] = round(_STATE["pool"].get_limit() / 2**30, 3)
    if _STATE["cap"] == "all":
        p = _STATE["pool"]
        p.set_limit(size=max(p.total_bytes(), 1 << 20))
        return
    _STATE["pool"].set_limit(size=_STATE.get("ceiling", 0))   # CuPy's own limit again (0 = none)
    if cp.cuda.get_allocator() is _STATE["capped"]:  # not wrapped by someone else (tools/vram_audit.py hooks it)
        cp.cuda.set_allocator(_STATE["plain"])


def pool():
    setup()
    return _STATE["pool"]


def release(stream=None) -> int:
    """Sync `stream` (or the device) and give every unused pool block back to the driver. Returns the bytes freed."""
    import cupy as cp

    setup()
    p = _STATE["pool"]
    if stream is not None:
        stream.synchronize()
    else:
        cp.cuda.Device().synchronize()
    before = p.total_bytes()
    p.free_all_blocks()
    return int(before - p.total_bytes())


def release_slack(stream=None, min_mib: int = 256) -> int:
    """release(stream) only when the pool holds more than `min_mib` MiB of unused blocks (no sync otherwise)."""
    setup()
    p = _STATE["pool"]
    if p.total_bytes() - p.used_bytes() <= (int(min_mib) << 20):
        return 0
    return release(stream)


def stats() -> dict:
    """Live and reserved pool bytes now and their high-water marks (GiB), plus the settings."""
    import cupy as cp

    setup()
    p = _STATE["pool"]
    g = lambda b: round(b / 2**30, 3)  # noqa: E731
    out = dict(_STATE["cfg"], used_gib=g(p.used_bytes()), reserved_gib=g(p.total_bytes()))
    if _STATE["mode"] == "async":
        rt = cp.cuda.runtime
        h = rt.deviceGetDefaultMemPool(cp.cuda.Device().id)
        out.update(used_high_gib=g(rt.memPoolGetAttribute(h, rt.cudaMemPoolAttrUsedMemHigh)),
                   reserved_high_gib=g(rt.memPoolGetAttribute(h, rt.cudaMemPoolAttrReservedMemHigh)))
    elif _STATE["cap"] != "0":
        # the cap only rises (to what a request needed plus one step) and the reservation never passes it: while the
        # cap is on it bounds the reservation's high-water mark from above by at most one step
        if _STATE["cap"] == "all":
            out.update(reserved_high_gib=g(p.get_limit()))       # since the first step (the forecast's cap)
        if "init_reserved_high_gib" in _STATE:
            out.update(init_reserved_high_gib=_STATE["init_reserved_high_gib"])
        out.update(cap_hits=_HITS[0], cap_hit_s=round(_HITS[1], 3))
    return out
