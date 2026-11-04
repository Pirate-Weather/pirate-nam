"""Done-rule check for unit io-history (with io-wind-rotate and io-history-fields in front of it).

    source ~/nam-gpu/env.sh && python tools/check_io_history.py

Full-size random State (954 x 835 x 60), the three unit wrappers run in driver order, the history
file is read back: every written variable is finite, the attributes and dimensions are the ones
history.py promises, and the device->host copy time and the background write time are printed.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.io import io_history, io_history_fields, io_wind_rotate

    t0 = time.perf_counter()
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    out_dir = ROOT / "runs-check" / "io-history"
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    ctx.out_dir = out_dir
    ctx.start = datetime(2023, 3, 31, 12)
    ctx.start_time = ctx.start
    ctx.ntimestep = 0
    t1 = time.perf_counter()
    with stream:
        io_wind_rotate.run(ctx)
        ctx.history_fields = io_history_fields.run(ctx)
    stream.synchronize()
    t2 = time.perf_counter()
    bad_dev = [n for n, a in st.fields.items()
               if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    t3 = time.perf_counter()
    path = io_history.run(ctx)            # device->host copy on this thread
    t4 = time.perf_counter()
    io_history.drain(ctx)                 # background encode + write
    t5 = time.perf_counter()

    import netCDF4
    with netCDF4.Dataset(path) as nc:
        names = sorted(nc.variables)
        bad_file = []
        for n in names:
            v = nc.variables[n]
            if v.dtype.kind == "f":
                a = np.asarray(v[:])
                if not np.isfinite(a).all():
                    bad_file.append(n)
        attrs = {k: nc.getncattr(k) for k in ("MAP_PROJ", "CEN_LAT", "CEN_LON", "POLE_LAT",
                                             "POLE_LON", "DX", "DY", "WEST_EAST_GRID_DIMENSION",
                                             "SOUTH_NORTH_GRID_DIMENSION", "START_DATE")}
        dims = {k: len(d) for k, d in nc.dimensions.items()}
        shapes = {n: tuple(nc.variables[n].shape) for n in ("PRES", "U", "V", "W", "T2", "Q2",
                                                             "PH", "XTIME")}
        xtime = float(nc.variables["XTIME"][0])
    print(f"file {path}")
    print(f"variables ({len(names)}): {' '.join(names)}")
    print(f"attrs {attrs}")
    print(f"dims {dims}")
    print(f"shapes {shapes}")
    print(f"non-finite device fields before write: {bad_dev or 'none'}")
    print(f"non-finite file variables: {bad_file or 'none'}")
    print(f"timing: state {t1 - t0:.1f} s, wind-rotate+fields {t2 - t1:.2f} s, "
          f"device check {t3 - t2:.2f} s, host copy {t4 - t3:.2f} s, write drained {t5 - t4:.2f} s")
    ok = not bad_dev and not bad_file and xtime == 0.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
