"""Init + io subsystem on the real case: build the NAM state from an init source, write the 0 h
history file, draw maps with the Rust renderer (rw_wrfbatch via gpuwm.rustwx).

    source env-box.sh
    python tools/run_init.py --case 2025-03-14_12z --source native --out runs/init_native
    python tools/run_init.py --case 2025-03-14_12z --source gfs    --out runs/init_gfs

Runs the init-case-driver CHAIN (state, boundaries from GFS, physics tables), io-wind-rotate,
io-history-fields, io-history (0 h file), then maps.py. Raises on any NaN/Inf.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma list of chain stages (debug)")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--hours", type=int, default=48, help="forecast length the boundaries cover")
    a = ap.parse_args(argv)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.init import init_case_driver
    from namgpu.io import history, io_history_fields, io_quasipost_driver, io_wind_rotate

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.forecast_hours = a.hours
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ctx.out_dir = out
    t0 = time.perf_counter()
    only = a.only.split(",") if a.only else None
    with ctx.stream:
        rec = init_case_driver.run(ctx, only=only)
        io_wind_rotate.run(ctx)
        io_quasipost_driver.run(ctx)
        fields = io_history_fields.run(ctx)
    ctx.stream.synchronize()
    path = out / f"wrfout_d01_{ctx.start:%Y-%m-%d_%H:%M:%S}"
    history.write_history(path, ctx, fields, ctx.start)
    used = cp.cuda.Device().mem_info
    print(f"run_init: {a.source} state built and 0 h history written {path} in "
          f"{time.perf_counter() - t0:.1f} s; card used {(used[1]-used[0])/2**30:.1f} GiB", flush=True)
    rec["source"] = a.source
    rec["history"] = str(path)
    (out / "init-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    if not a.no_maps:
        from namgpu.io import io_maps            # the driver's map unit: CF route, MRMS overlay, done rule
        print(f"run_init: maps in {io_maps.run(ctx)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
