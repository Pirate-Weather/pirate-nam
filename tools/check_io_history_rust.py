"""Done-rule check for unit io-history-rust (CONTRACT.md section 6, parts 1 and 2).

Run:  source env.sh; python tools/check_io_history_rust.py
1. the kernel compiles (kernels.module(...).compile());
2. the wrapper runs once on a FULL-SIZE random history set (954 x 835 x 60, the CONTRACT sizes),
   writes one file through the Rust writer, and the file reads back with no NaN/Inf.
3. the level flip is a value check: the top and bottom levels of T and PH read back exactly equal
   to the host reference h[::-1] (a reversed or transposed order fails here).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "check-io-history-rust"


def main() -> int:
    from namgpu.steps.drv_gpu_lease import gpu_lease

    with gpu_lease(lane="nam-gpu-io-history-rust", vram_mib=2500, est_s=300,
                   what="io-history-rust done-rule check") as cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        import cupy as cp
        import netCDF4
        import numpy as np
        from namgpu import config, grid as gridmod, kernels
        from namgpu.io import io_history_rust as unit

        t0 = time.perf_counter()
        kernels.module(unit.UNIT).compile()
        print(f"kernel {unit.UNIT} compiled in {time.perf_counter() - t0:.2f} s", flush=True)

        cfg = config.load()
        g = gridmod.build(cfg)
        im, jm, lm = int(g.im), int(g.jm), int(g.lm)
        print(f"grid {im} x {jm} x {lm}", flush=True)

        rng = np.random.default_rng(1)
        def rand3(nl):
            return cp.asarray(rng.standard_normal((nl, jm, im), dtype=np.float32) * 10.0 + 280.0)
        def rand2():
            return cp.asarray(rng.standard_normal((jm, im), dtype=np.float32) * 5.0 + 1000.0)

        fields = {
            "T": (rand3(lm), ("bottom_top", "south_north", "west_east")),
            "QVAPOR": (rand3(lm), ("bottom_top", "south_north", "west_east")),
            "PH": (rand3(lm + 1), ("bottom_top_stag", "south_north", "west_east")),
            "PSFC": (rand2(), ("south_north", "west_east")),
        }
        host = {k: v[0].get() for k, v in fields.items()}
        stream = cp.cuda.Stream(non_blocking=True)
        ctx = SimpleNamespace(cfg=cfg, grid=g, stream=stream, ntimestep=0,
                              history_fields=fields, start_time=datetime(2023, 3, 31, 12),
                              out_dir=OUT)
        t0 = time.perf_counter()
        path = unit.run(ctx)
        dt = time.perf_counter() - t0
        print(f"wrote {path} in {dt:.2f} s", flush=True)

        with netCDF4.Dataset(path) as ds:
            assert ds.dimensions["south_north"].size == jm and ds.dimensions["west_east"].size == im
            assert ds.dimensions["bottom_top"].size == lm
            assert ds.MAP_PROJ == 203
            bad = []
            for name in ("XLAT", "XLONG", "T", "QVAPOR", "PH", "PSFC"):
                v = ds.variables[name]
                v.set_auto_maskandscale(False)
                if not np.isfinite(np.asarray(v[:])).all():
                    bad.append(name)
            print("read back: dims ok, variables", list(ds.variables), flush=True)
            for name in ("T", "PH"):
                v = ds.variables[name]
                v.set_auto_maskandscale(False)
                nl = host[name].shape[0]
                ref = host[name][::-1]
                for lev in (0, nl - 1):
                    if not np.array_equal(np.asarray(v[0, lev]), ref[lev]):
                        bad.append(f"{name} level {lev} flip")
            print("flip value check done", flush=True)
        if bad:
            print(f"NaN/Inf in {bad}", flush=True)
            return 1
        print("ok: no NaN/Inf in any written variable", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
