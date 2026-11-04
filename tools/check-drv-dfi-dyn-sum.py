"""Done-rule check for drv-dfi-dyn-sum (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-drv-dfi-dyn-sum.py

Runs the real chain on a full-size random State (954 x 835 x 60): drv-dfi-dyn-init builds ctx.filt
(zeroed sums, kstep = -nstep-1, Dolph weights, filtered names from parm/nam_filt_vars_norad.txt),
then drv-dfi-dyn-sum runs once. No hand-built state. Checks: no NaN/Inf in any float array, the
sums and totalsum moved, and kstep advanced by exactly one (it is the single counter in ctx.filt).
Both dolph and the device non-dolph digfil are compiled; the non-dolph kernel is run once.
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
    from namgpu.steps import drv_dfi_dyn_init as init_unit
    from namgpu.steps import drv_dfi_dyn_sum as unit
    from namgpu.steps import drv_tracer_index as tidx

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    kernels.module(init_unit.UNIT).compile()
    kernels.module(unit.UNIT).compile()
    t0 = time.perf_counter()
    with ctx.stream:
        tidx.run(ctx)                              # ctx.tables["indx"] for dyn-init
        filt = init_unit.run(ctx, ndfistep=75)
        k0 = int(filt["kstep"])
        unit.run(ctx, mean_on=0, num_water=4, num_tracers=int(st["TRACERS"].shape[0])
                 if "TRACERS" in st.fields else 4, its=1, ite=int(g.im), jts=1, jte=int(g.jm))
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    filt = ctx.filt
    nstep = int(filt["nstep"])

    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    sums = {"sum2d": filt["sum2d"], "sum3d": filt["sum3d"]}
    bad += [k for k, s in sums.items() if not bool(cp.isfinite(s).all())]
    if not bool(cp.isfinite(filt["totalsum"]).all()):
        bad.append("totalsum")
    total = float(filt["totalsum"][0])
    moved = {k: float(cp.abs(s).max()) for k, s in sums.items() if s.size}
    print(f"2-D {filt['name_save_2d']}")
    print(f"3-D {filt['name_save_3d']}")
    print(f"kstep {k0} -> {filt['kstep']} (nstep {nstep}), totalsum {total:.6g}, "
          f"sum max |.| {moved}, {dt:.2f} s")
    if int(filt["kstep"]) != k0 + 1 or bad or total == 0.0 or not all(v > 0 for v in moved.values()):
        print("FAIL: kstep/NaN/Inf in", bad[:10], "or sums/totalsum did not move")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
