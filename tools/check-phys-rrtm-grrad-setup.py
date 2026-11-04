"""Done-rule check for unit phys-rrtm-grrad-setup (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-phys-rrtm-grrad-setup.py

1. compiles namgpu/kernels/phys_rrtm_grrad_setup.cu through NVRTC;
2. runs the wrapper ONCE on the full-size random State (954 x 835 x 60) and asserts no NaN/Inf in
   any float array it touches afterwards.
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

    t0 = time.perf_counter()
    kernels.module("phys_rrtm_grrad_setup").compile()
    print(f"compile phys_rrtm_grrad_setup: ok ({time.perf_counter() - t0:.2f} s)")

    cfg = config.load()
    g = gridmod.build(cfg)
    names = {"PINT", "PD", "SGML2", "T", "Q", "TRACERS", "CW", "O3", "CLDFRA", "F_ICE", "F_RAIN",
             "F_RIMEF", "TSKIN"}
    st = statemod.State(g, names=names).randomize(seed=1)
    st["PD"][...] = cp.clip(st["PD"], 50000.0, 100000.0)
    # plausible interface pressures in Pa (randomize leaves PINT unbuilt without the grid sigma)
    st["PINT"][...] = 1000.0 + 99000.0 * st["PINT"]

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    from namgpu.physics.rrtm import phys_rrtm_grrad_setup as wrapper

    t0 = time.perf_counter()
    with ctx.stream:
        out = wrapper.run(ctx, dtsw=300.0, dtlw=300.0)   # np3d, ntoz, ivflip and TSKIN from cfg/state
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    bad = []
    for n, a in list(out.items()) + [(k, v) for k, v in st.fields.items()]:
        if isinstance(a, float):
            continue
        if a.dtype.kind == "f" and not bool(cp.isfinite(a).all()):
            bad.append(n)
    if bad:
        print(f"phys-rrtm-grrad-setup: NaN/Inf in {sorted(set(bad))[:10]} ({dt:.2f} s)")
        return 1
    print(f"phys-rrtm-grrad-setup: ok, no NaN/Inf in any float array, {dt:.2f} s, raddt={out['raddt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
