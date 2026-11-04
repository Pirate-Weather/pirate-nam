"""Done-rule check for io-quasipost-3d (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_io_quasipost_3d.py

Compiles the unit's kernels (NVRTC) and runs its wrapper ONCE on a full-size random State
(954 x 835 x 60), then checks every float array the unit writes for NaN/Inf and prints the
int32 abort counters (nbad[0] weight out of range, nbad[1] NaN m10rv). Take the GPU pool lease
first (README.md). No column test, no reduced size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WRITTEN = ["QP_PRESTV", "QP_PRESZ", "P500U", "P500V", "P700WIND", "P700RV", "P700U", "P700V",
           "P850WIND", "P850RV", "P850U", "P850V", "M10WIND", "M10RV"]


def main():
    import cupy as cp
    import importlib
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    t0 = time.perf_counter()
    for kname in ["io_quasipost_3d_column", "io_quasipost_3d_interior", "io_quasipost_3d_edge_i",
                  "io_quasipost_3d_edge_j", "io_quasipost_3d_m10"]:
        kernels.get("io_quasipost_3d", kname)          # NVRTC compile of every kernel
    print("compiled 5 kernels", flush=True)
    with ctx.stream:
        importlib.import_module("namgpu.io.io_quasipost_3d").run(ctx)
    ctx.stream.synchronize()
    print(f"wrapper ran once in {time.perf_counter() - t0:.2f} s (includes compile)", flush=True)

    bad = []
    for name in WRITTEN:
        a = st.fields[name]
        if not bool(cp.isfinite(a).all()):
            bad.append(name)
    nbad = st.fields["QP_NBAD"].get()
    print("NaN/Inf in:", bad if bad else "none", flush=True)
    print("abort counters nbad[0] weight, nbad[1] NaN m10rv:", nbad.tolist(), flush=True)
    for name in ["QP_PRESTV", "QP_PRESZ", "P850U", "M10WIND", "M10RV"]:
        a = st.fields[name]
        print(f"  {name}: min {float(a.min()):.4g} max {float(a.max()):.4g}", flush=True)
    ok = not bad and int(nbad[1]) == 0
    print("RESULT", "ok" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
