"""Done-rule check for one unit (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_unit.py dyn-pgforce-geopot [more ids ...]

1. compiles the unit's kernel file;
2. runs its wrapper ONCE on a full-size random State (954 x 835 x 60) and checks every float
   array for NaN/Inf afterwards.
Part 3 (Fortran-to-CUDA diff review) is done by reading, and recorded in the unit's kernel header.
Take the GPU pool lease first (README.md).  No column test, no reduced size.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def call_kwargs(ctx, uid):
    """The kwargs the driver plan passes to this unit's wrapper, one dict per plan entry (or one bare call)."""
    from namgpu.driver import Driver
    drv = Driver(ctx)
    out = [{k: drv._resolve(v) for k, v in e[2].items()} if len(e) > 2 else {}
           for e in drv.plan if e[0] == uid]
    return out or [{}]


def main(ids):
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx

    units = {u["id"]: u for u in json.loads((ROOT / "UNITS.json").read_text())["units"]}
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ok = True
    for uid in ids:
        u = units[uid]
        if u["mode"] == "skip":
            print(f"{uid}: skip ({u.get('reason', '')})")
            continue
        t0 = time.perf_counter()
        kfile = ROOT / u["kernel"]
        if kfile.exists():
            kernels.module(kfile.stem).compile()
        with ctx.stream:
            mod = importlib.import_module(u["wrapper"])
            for kw in call_kwargs(ctx, uid):
                mod.run(ctx, **kw)
        ctx.stream.synchronize()
        bad = [n for n, a in st.fields.items()
               if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
        dt = time.perf_counter() - t0
        if bad:
            ok = False
            print(f"{uid}: NaN/Inf in {bad[:10]} ({dt:.2f} s)")
        else:
            print(f"{uid}: ok, compiled={kfile.exists()}, {dt:.2f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
