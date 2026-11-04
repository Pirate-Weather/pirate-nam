"""Per-unit device-memory census of a real NAM forecast (32 GB fit work).

    source env.sh
    python tools/mem_census.py --case 2025-03-14_12z --hours 0.5 --out scratch/c32/census1 [--managed]

Hooks the CuPy allocator: for every plan unit (driver stamp label) it records the peak live pool bytes
reached inside it, the live bytes it leaves behind (persistent growth) and its allocations >= 32 MB.
--managed backs the pool with CUDA managed memory so the census itself cannot OOM on a 32 GB card.
Writes <out>/mem-census.json. Census only: no physics change.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _devbytes(v, seen, depth=0):
    """Device bytes reachable from v (each allocation counted once; containers visited once)."""
    if depth > 6:
        return 0
    if hasattr(v, "nbytes") and hasattr(v, "device") and hasattr(v, "data") and hasattr(v.data, "mem"):
        key = ("m", v.data.mem.ptr)
        if key in seen:
            return 0
        seen.add(key)
        return int(v.data.mem.size)
    if isinstance(v, (str, bytes, int, float, type(None))) or hasattr(v, "__array_interface__"):
        return 0
    if id(v) in seen:
        return 0
    seen.add(id(v))
    if isinstance(v, dict):
        return sum(_devbytes(x, seen, depth + 1) for x in list(dict.values(v)))
    if isinstance(v, (list, tuple, set)):
        return sum(_devbytes(x, seen, depth + 1) for x in list(v))
    d = getattr(v, "__dict__", None)
    if isinstance(d, dict) and not isinstance(v, type) and type(v).__module__.startswith("namgpu"):
        return sum(_devbytes(x, seen, depth + 1) for x in list(d.values()))
    if type(v).__name__ == "SimpleNamespace":
        return sum(_devbytes(x, seen, depth + 1) for x in list(vars(v).values()))
    return 0


def _dump_holders(ctx, st, path):
    from namgpu import state as sm
    table = set(sm.all_specs())
    seen = set()
    extra = sorted(((k, _devbytes(v, seen)) for k, v in st.fields.items() if k not in table), key=lambda r: -r[1])
    tab = sorted(((k, _devbytes(v, seen)) for k, v in ctx.tables.items()), key=lambda r: -r[1])
    bc = _devbytes(ctx.bc, seen)
    rec = {"extra_fields_gib": round(sum(r[1] for r in extra) / 2**30, 3),
           "tables_gib": round(sum(r[1] for r in tab) / 2**30, 3), "bc_gib": round(bc / 2**30, 3),
           "extra_fields": [(k, round(b / 2**20)) for k, b in extra[:80] if b],
           "tables": [(k, round(b / 2**20)) for k, b in tab[:40] if b]}
    path.write_text(json.dumps(rec, indent=1))
    print("HOLDERS", json.dumps({k: rec[k] for k in ("extra_fields_gib", "tables_gib", "bc_gib")}),
          "top extra", rec["extra_fields"][:25], "top tables", rec["tables"][:12], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--hours", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--managed", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="run this many steps instead of --hours")
    a = ap.parse_args()
    import cupy as cp
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed) if a.managed else cp.get_default_memory_pool()
    big = 32 << 20
    cur = ["setup"]
    peak = collections.defaultdict(int)
    allocs = collections.defaultdict(lambda: collections.Counter())
    gpeak = [0, "setup"]

    def alloc(n):
        p = pool.malloc(n)
        u = pool.used_bytes()
        lab = cur[0]
        if u > peak[lab]:
            peak[lab] = u
        if u > gpeak[0]:
            gpeak[0], gpeak[1] = u, lab
        if n >= big:
            allocs[lab][n] += 1
        return p

    cp.cuda.set_allocator(alloc)

    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver
    from namgpu.steps import drv_timing_receipt as timing

    before = {}
    after = collections.defaultdict(list)
    orig_stamp = timing.stamp

    nonpool = collections.defaultdict(int)

    def stamp(ctx, label):
        prev = cur[0]
        u = pool.used_bytes()
        fr, tt = cp.cuda.runtime.memGetInfo()
        npb = (tt - fr) - pool.total_bytes()
        if npb > nonpool[prev]:
            nonpool[prev] = npb
        if prev in before:
            after[prev].append(u - before[prev])
        base = str(label).split("/")[0]
        cur[0] = base
        before[base] = u
        return orig_stamp(ctx, label)

    timing.stamp = stamp

    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    cur[0] = "state-alloc"
    st = statemod.State(g)
    state_alloc = pool.used_bytes()
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.out_dir = out
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    chain = [u for u in init_case_driver.CHAIN if u != "init-fetch-case"]
    cur[0] = "init"
    t0 = time.perf_counter()
    with ctx.stream:
        init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    init_s = time.perf_counter() - t0
    import gc
    gc.collect()
    pool.free_all_blocks()
    after_init = pool.used_bytes()
    fields = sorted(((k, int(v.nbytes), str(v.dtype), list(v.shape)) for k, v in st.fields.items()
                     if hasattr(v, "nbytes")), key=lambda r: -r[1])
    tabs = []
    for k, v in ctx.tables.items():
        nb = 0
        if hasattr(v, "nbytes") and hasattr(v, "device"):
            nb = int(v.nbytes)
        elif isinstance(v, dict):
            nb = sum(int(x.nbytes) for x in v.values() if hasattr(x, "nbytes") and hasattr(x, "device"))
        if nb:
            tabs.append((k, nb))
    tabs.sort(key=lambda r: -r[1])
    print(f"init {init_s:.1f}s state {st.nbytes()/2**30:.2f} GiB live after init {after_init/2**30:.2f} GiB "
          f"tables {sum(t[1] for t in tabs)/2**30:.2f} GiB", flush=True)
    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()
    cur[0] = "loop"
    t1 = time.perf_counter()
    err = None
    try:
        hrs = a.steps * cfg.dt / 3600.0 if a.steps else a.hours
        _orig_step = drv.step

        def _step():
            _orig_step()
            ctx.stream.synchronize()
            fr, tt = cp.cuda.runtime.memGetInfo()
            print(f"step {ctx.ntimestep} card {(tt-fr)/2**30:.2f} GiB pool {pool.total_bytes()/2**30:.2f} live {pool.used_bytes()/2**30:.2f} GiB peak {gpeak[0]/2**30:.2f} GiB in {gpeak[1]} t {time.perf_counter()-t1:.1f}s", flush=True)
            if ctx.ntimestep == 1:
                _dump_holders(ctx, st, out / "holders-step1.json")
        drv.step = _step
        drv.run(hrs, log=lambda m: print(m, flush=True))
    except Exception as e:  # noqa: BLE001
        import traceback
        err = traceback.format_exc()[-3000:]
        print("RUN ERROR", err, flush=True)
        try:
            pool.free_all_blocks()
        except Exception:  # noqa: BLE001
            pass
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    end_live = pool.used_bytes()
    try:
        nonfinite = init_case_driver.census(ctx) if err is None else None
    except Exception as e:  # noqa: BLE001
        nonfinite = f"census failed: {e}"
    SANE = ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "T2", "TSKIN", "ACPREC", "RLWTT", "RSWTT", "CLDFRA")
    stats = {}
    for nm in SANE:
        v = st.fields.get(nm)
        if v is not None:
            stats[nm] = [float(v.min()), float(v.max()), float(v.astype("float64").sum())]
    print("nonfinite_end", nonfinite, flush=True)
    print("stats_end", json.dumps(stats), flush=True)
    rec = {
        "nonfinite_end": nonfinite, "stats_end": stats, "managed": a.managed, "init_s": round(init_s, 1), "steps": ctx.ntimestep,
        "ms_per_step": round(wall / max(1, ctx.ntimestep) * 1e3, 1), "error": err,
        "state_alloc_gib": round(state_alloc / 2**30, 3), "state_gib": round(st.nbytes() / 2**30, 3),
        "live_after_init_gib": round(after_init / 2**30, 3), "live_end_gib": round(end_live / 2**30, 3),
        "pool_total_end_gib": round(pool.total_bytes() / 2**30, 3),
        "global_peak_gib": round(gpeak[0] / 2**30, 3), "global_peak_unit": gpeak[1],
        "unit_peak_gib": {k: round(v / 2**30, 3) for k, v in sorted(peak.items(), key=lambda r: -r[1])},
        "unit_nonpool_gib": {k: round(v / 2**30, 2) for k, v in sorted(nonpool.items(), key=lambda r: -r[1])[:30]},
        "unit_retained_mib": {k: round(max(v) / 2**20, 1) for k, v in after.items() if v and max(v) > 2**20},
        "unit_big_allocs": {k: {str(n): c for n, c in v.most_common(12)} for k, v in allocs.items()},
        "fields_top": fields[:60], "n_fields": len(fields),
        "tables_top": tabs[:30],
    }
    (out / "mem-census.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("state_gib", "live_after_init_gib", "live_end_gib", "global_peak_gib",
                                          "global_peak_unit", "ms_per_step")}), flush=True)
    top = list(rec["unit_peak_gib"].items())[:25]
    for k, v in top:
        print(f"  peak {v:7.2f} GiB  {k}", flush=True)
    rd.__exit__(None, None, None)
    return 0 if err is None else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(rc)        # no CUDA teardown after an OOM/illegal-address (it hung holding the card lock)
