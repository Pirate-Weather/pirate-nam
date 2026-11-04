"""Done-rule check for unit io-maps: full-size state -> hourly history -> io-maps renders each hour.

Runs the real history chain per output hour (io-wind-rotate, io-history-fields, io-history) and
then io-maps, so the field set is the one the driver writes, not a hand-made subset.  Two hours,
so the second call proves that only the new frame is drawn.  Every history field must be finite
(no NaN) before it is written.

    /opt/gpu-mutex/run.sh NAME --cards 1 --vram-mib 12000 --est 900 -- \
        bash -c "source env.sh && python tools/check_io_maps.py"
"""
from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOURS = 2


def main():
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.io import io_history, io_history_fields, io_maps, io_wind_rotate

    t0 = time.perf_counter()
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    out_dir = Path(__file__).resolve().parents[1] / "runs-check" / "io-maps"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.out_dir = out_dir
    ctx.start_time = datetime(2023, 3, 31, 12)
    steps_per_hour = int(round(3600.0 / cfg.dt))
    total_png = 0
    for hour in range(HOURS):
        ctx.ntimestep = hour * steps_per_hour
        st.randomize(seed=1 + hour)
        io_wind_rotate.run(ctx)
        ctx.history_fields = io_history_fields.run(ctx)
        required = {"PSFC", "HGT", "T2", "Q2", "U10", "V10", "RAINNC", "RAINC", "COMPOSITE",
                    "REFL_10CM", "T", "PRES", "PH", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "U", "V", "W"}
        absent = sorted(required - set(ctx.history_fields))
        if absent:
            raise SystemExit(f"history fields missing at hour {hour}: {absent}")
        nonfinite = [name for name, (arr, _dims) in ctx.history_fields.items()
                     if not bool(cp.isfinite(arr).all())]
        if nonfinite:
            raise SystemExit(f"NaN/inf in history fields at hour {hour}: {nonfinite}")
        t1 = time.perf_counter()
        path = io_history.run(ctx)
        pending = io_maps.history_files(ctx)
        if [p.name for p in pending] != [path.name]:
            raise SystemExit(f"hour {hour}: io-maps would render {[p.name for p in pending]}, expected only {path.name}")
        maps_dir = io_maps.run(ctx)
        if io_maps.history_files(ctx):
            raise SystemExit(f"hour {hour}: frames still unrendered after io-maps")
        t2 = time.perf_counter()
        pngs = sorted(maps_dir.rglob("*.png"))
        total_png = len(pngs)
        print(f"hour {hour}: history {path.name} fields={len(ctx.history_fields)} "
              f"io-maps {t2 - t1:.1f} s, {len(pngs)} png in {maps_dir}", flush=True)
    print(f"total {time.perf_counter() - t0:.1f} s, {total_png} png", flush=True)
    return 0 if total_png else 1


if __name__ == "__main__":
    sys.exit(main())
