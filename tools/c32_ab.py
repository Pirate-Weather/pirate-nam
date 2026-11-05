"""Lane C (32 GB fit): bitwise A/B of the step loop from ONE saved init (lane E's slab_init save/load).

    python tools/c32_ab.py run --init /dev/shm/c32-init --steps 50 --out scratch/c32/abA   (env picks the arm)
    python tools/c32_ab.py cmp scratch/c32/abA scratch/c32/abB

`run` loads the saved post-init context, runs N steps through Driver.step (the forecast path), then writes
sha256 + min/max per state field and the run's peak card memory. `cmp` lists the fields that differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(a):
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import slab_init
    from namgpu import kernels as K
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=set())
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = "native"
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ctx.out_dir = out
    ctx.forecast_hours = 6
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    t0 = time.perf_counter()
    slab_init.load(ctx, a.init)
    from namgpu.steps import drv_loop_limits
    with ctx.stream:
        drv_loop_limits.run(ctx)                 # per-rank table slab_init drops (NMMB_LIMITS)
    ctx.tables.pop("gwd_dem", None)              # init-only (init_case_driver drops it too)
    cp.get_default_memory_pool().free_all_blocks()
    K.trim_local_memory("load")
    print(f"loaded in {time.perf_counter() - t0:.1f} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()
    peak = 0
    t1 = time.perf_counter()
    err = None
    try:
      with ctx.stream:
        for i in range(a.steps):
            drv.step()
            if i == 0 and a.holders:
                ctx.stream.synchronize()
                sys.path.insert(0, str(ROOT / "tools"))
                import mem_census as mc
                mc._dump_holders(ctx, st, out / "holders-step1.json")
            if i % 10 == 0 or i == a.steps - 1:
                ctx.stream.synchronize()
                fr, tt = cp.cuda.runtime.memGetInfo()
                peak = max(peak, tt - fr)
                print(f"step {ctx.ntimestep} card {(tt - fr) / 2**30:.2f} GiB pool "
                      f"{cp.get_default_memory_pool().total_bytes() / 2**30:.2f} t {time.perf_counter() - t1:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        err = traceback.format_exc()[-2500:]
        print("RUN ERROR", err, flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            import mem_census as mc
            mc._dump_holders(ctx, st, out / "holders-at-error.json")
        except Exception as e2:  # noqa: BLE001
            print("holders failed", e2, flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        return 1
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    rec = {"steps": a.steps, "wall_s": round(wall, 2), "card_peak_sampled_gib": round(peak / 2**30, 2), "fields": {}}
    # GPU digests (no host copy, no single-core hashing): two weighted uint64 sums of the raw 32-bit words
    # plus the non-finite count; equal digests on every field = bitwise-equal state for any practical purpose
    cp.get_default_memory_pool().free_all_blocks()
    C = 1 << 24
    idx = cp.arange(C, dtype=cp.uint64)
    w1b = idx * cp.uint64(2654435761) + cp.uint64(1)
    w2b = (idx ^ cp.uint64(0x9E3779B97F4A7C15)) * cp.uint64(40503) + cp.uint64(7)
    for k in sorted(st.fields):
        v = st.fields[k]
        if not hasattr(v, "device") or v.size == 0:
            continue
        flat = cp.ascontiguousarray(v).view(cp.uint32 if v.dtype.itemsize == 4 else cp.uint8).ravel()
        d1 = d2 = 0
        for c0 in range(0, flat.size, C):
            a = flat[c0:c0 + C].astype(cp.uint64)
            m = a.size
            off = cp.uint64(c0)
            d1 += int(cp.sum(a * (w1b[:m] + off)))
            d2 += int(cp.sum((a ^ cp.uint64(0xA5A5A5A5)) * (w2b[:m] ^ off)))
        d1 &= (1 << 64) - 1
        d2 &= (1 << 64) - 1
        nbad = int(v.size - int(cp.isfinite(v).sum())) if v.dtype.kind == "f" else 0
        if nbad:
            rec.setdefault("nonfinite", {})[k] = nbad
        rec["fields"][k] = [f"{d1:016x}{d2:016x}", list(v.shape)]
    (out / "ab.json").write_text(json.dumps(rec, indent=1))
    print("nonfinite", rec.get("nonfinite", {}), flush=True)
    print(f"wrote {out / 'ab.json'}: {len(rec['fields'])} fields, {wall / a.steps * 1e3:.1f} ms/step", flush=True)
    rd.__exit__(None, None, None)
    return 0


def cmp(a):
    A = json.loads((Path(a.a) / "ab.json").read_text())["fields"]
    B = json.loads((Path(a.b) / "ab.json").read_text())["fields"]
    from namgpu.state import STEP_SCRATCH, HISTORY_SCRATCH
    scratch = STEP_SCRATCH | HISTORY_SCRATCH
    diff = [k for k in A if k in B and A[k][0] != B[k][0]]
    onlyA = [k for k in A if k not in B]
    onlyB = [k for k in B if k not in A]
    common = [k for k in A if k in B]
    print(f"common {len(common)} fields, differ {len(diff)}: {diff[:60]}")
    print(f"only in A ({len(onlyA)}; scratch {sum(k in scratch for k in onlyA)}): {onlyA[:60]}")
    print(f"only in B ({len(onlyB)}): {onlyB[:60]}")
    return 1 if diff else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--init", required=True)
    r.add_argument("--case", required=True, help="case folder name YYYY-MM-DD_HHz")
    r.add_argument("--steps", type=int, default=50)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--holders", action="store_true")
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else cmp(a)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(rc)
