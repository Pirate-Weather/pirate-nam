"""Done-rule check for unit phys-gwd-init (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-gwd-init.py

1. compiles namgpu/kernels/phys_gwd_init.cu through NVRTC;
2. runs the wrapper once on a full-size State (954 x 835 x 60) with the grid's real GLAT/GLON, and
   asserts that CROT, SROT, HANGL and CLEFF contain no NaN or Inf.
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
    from namgpu.physics.gwd import phys_gwd_init

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    st["GLAT"][...] = cp.asarray(g.glat_h, dtype=cp.float32)
    st["GLON"][...] = cp.asarray(g.glon_h, dtype=cp.float32)
    st.add("HANGL", (g.jm, g.im))[...] = cp.random.default_rng(2).random((g.jm, g.im), dtype=cp.float32) * 3.0
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    t0 = time.perf_counter()
    print("compile:", kernels.module("phys_gwd_init").compile() or "ok", flush=True)
    t1 = time.perf_counter()
    tbl = phys_gwd_init.run(ctx)
    ctx.stream.synchronize()
    t2 = time.perf_counter()

    cleff = tbl["cleff"]
    arrays = {"CROT": st["CROT"], "SROT": st["SROT"], "HANGL": st["HANGL"], "CLEFF": cleff}
    bad = [k for k, a in arrays.items() if not bool(cp.isfinite(a).all())]
    for k, a in arrays.items():
        print(f"{k:6s} shape={tuple(a.shape)} min={float(a.min()):.6g} max={float(a.max()):.6g}")
    print(f"CLEFF={float(cleff[0]):.6g} deltim={tbl['deltim']} rdeltim={tbl['rdeltim']} nmtvr={tbl['nmtvr']}")
    print(f"compile {t1 - t0:.2f} s, run {t2 - t1:.3f} s")
    if bad:
        print("NaN/Inf in:", bad)
        return 1
    print("ok: no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
