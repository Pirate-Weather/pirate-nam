"""Done-rule check for unit phys-bmj-init (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-bmj-init.py

1. compiles namgpu/kernels/phys_bmj_init.cu through NVRTC;
2. runs the wrapper once on a full-size random State (954 x 835 x 60) and asserts that no
   float array (the BMJ tables and CLDEFI) contains NaN or Inf.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.bmj import phys_bmj_init

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    t0 = time.perf_counter()
    print("compile:", kernels.module("phys_bmj_init").compile() or "ok", flush=True)
    t1 = time.perf_counter()
    phys_bmj_init.run(ctx)
    ctx.stream.synchronize()
    t2 = time.perf_counter()

    bad = []
    for name, a in ctx.tables.items():
        if name.startswith("bmj_") and not bool(cp.isfinite(a).all()):
            bad.append(name)
    if not bool(cp.isfinite(st.fields["CLDEFI"]).all()):
        bad.append("CLDEFI")
    for name in sorted(k for k in ctx.tables if k.startswith("bmj_")):
        a = ctx.tables[name]
        print(f"{name:12s} shape={tuple(a.shape)} min={float(a.min()):.6g} max={float(a.max()):.6g}")
    print(f"CLDEFI interior={float(st.fields['CLDEFI'][0, 0]):.4f} "
          f"edge(last col)={float(st.fields['CLDEFI'][0, -1]):.4f}")
    print(f"compile {t1 - t0:.2f} s, run {t2 - t1:.3f} s")
    if bad:
        print("NaN/Inf in:", bad)
        return 1
    print("ok: no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
