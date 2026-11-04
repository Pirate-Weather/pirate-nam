"""Done-rule check for io-quasipost-driver (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-io-quasipost-driver.py

1. compiles namgpu/kernels/io_quasipost_driver.cu;
2. runs the wrapper ONCE on a full-size State (954 x 835 x 60) and checks every float array
   for NaN/Inf afterwards.

The three dependency units (io-quasipost-3d, io-quasipost-interp, io-quasipost-mslp) are not
on disk yet. For any that is missing, a stand-in run() fills the scratch arrays named in the
wrapper docstring with plausible values (mask nonzero on the bottom levels only, so the level
loop both relaxes and exits). The report says which were stand-ins.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_standins(missing):
    import cupy as cp
    import numpy as np

    def s3d(ctx):
        st, g = ctx.st, ctx.grid
        im, jm = int(g.im), int(g.jm)
        npres = 33
        k = np.arange(npres)[:, None, None]
        rng = np.random.default_rng(3)
        mask = (rng.random((npres, jm, im)) < 0.5) & (k >= 26)
        st.add("QP_PRESTV", (npres, jm, im))[...] = cp.asarray(
            (255.0 + 40.0 * rng.random((npres, jm, im))).astype(np.float32))
        st.add("QP_PRESZ", (npres, jm, im))[...] = cp.asarray(
            np.broadcast_to(np.linspace(1000.0, 30000.0, npres)[:, None, None],
                            (npres, jm, im)).astype(np.float32))
        st.add("QP_GROUND_MASK", (npres, jm, im), dtype="int32")[...] = cp.asarray(
            mask.astype(np.int32))
        st.add("QP_GROUND_LEVEL", (jm, im), dtype="int32")[...] = 0

    def sint(ctx):
        st, g = ctx.st, ctx.grid
        im, jm = int(g.im), int(g.jm)
        st.add("QP_PMSL", (jm, im))[...] = 101325.0

    def smslp(ctx):
        return None

    stand = {"io_quasipost_3d": s3d, "io_quasipost_interp": sint, "io_quasipost_mslp": smslp}
    for name in missing:
        mod = types.ModuleType(f"namgpu.io.{name}")
        mod.run = stand[name]
        sys.modules[f"namgpu.io.{name}"] = mod


def main() -> int:
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps.drv_gpu_lease import gpu_lease

    deps = ["io_quasipost_3d", "io_quasipost_interp", "io_quasipost_mslp"]
    missing = [d for d in deps if importlib.util.find_spec(f"namgpu.io.{d}") is None]
    _install_standins(missing)

    kernels.module("io_quasipost_driver").compile()
    print("kernel compiled: io_quasipost_driver.cu")

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    drv = importlib.import_module("namgpu.io.io_quasipost_driver")

    with gpu_lease(lane="nam-gpu-io-quasipost-driver", vram_mib=12000, est_s=600):
        t0 = time.perf_counter()
        with ctx.stream:
            drv.run(ctx)
        ctx.stream.synchronize()
        dt = time.perf_counter() - t0

    running = int(st.add("QP_RUNNING", (1,), dtype="int32").get()[0])
    bad = [n for n, a in st.fields.items()
           if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print(f"standins: {missing or 'none'}")
    print(f"level loop finished with running={running} (0 = exited early)")
    print(f"wall {dt:.2f} s (includes first-call setup)")
    if bad:
        print(f"NaN/Inf in {bad[:10]}")
        return 1
    print("io-quasipost-driver: ok, no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
