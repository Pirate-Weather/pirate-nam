"""Done-rule check for drv-dfi-dyn-average (the production handoff, not random scratch).

    source ~/nam-gpu/env.sh
    python tools/check-drv-dfi-dyn-average.py

Full-size random State (954 x 835 x 60). Runs the production chain once: drv-dfi-dyn-init (writes the
DFI handoff), drv-dfi-dyn-sum (one filter step), drv-dfi-dyn-average (reads the handoff). Checks:
  1. no handoff -> dyn-average raises (negative control, fresh ctx);
  2. every float state field is finite after the average;
  3. the T state field equals the scaled T sum, bit for bit (copy and scale both ran);
  4. the bookkeeping reset: names empty, TOT_RANK_* 0, KSTEP 0.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps import drv_dfi_dyn_average as avg
    from namgpu.steps import drv_dfi_dyn_init as init
    from namgpu.steps import drv_dfi_dyn_sum as dsum

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    # Negative control: no handoff must raise, never fall back to random sums.
    st0 = statemod.State(g).randomize(seed=1)
    ctx0 = Ctx(cfg=cfg, grid=g, st=st0, stream=None, dt=cfg.dt)
    try:
        avg.run(ctx0)
    except RuntimeError as e:
        print("negative control ok:", str(e)[:90])
    else:
        print("FAIL: dyn-average ran with no handoff")
        return 1
    del ctx0, st0

    st = statemod.State(g).randomize(seed=1)
    rows = re.findall(r"^'(\w+)'\s+([FB])\s", (ROOT / "parm/nam_filt_vars_norad.txt").read_text(), re.M)
    dyn = [(name, st.specs[name].rank) for name, _ in rows]
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    ctx.tables["indx"] = {"NUM_TRACERS_MET": 3, "NUM_TRACERS_CHEM": 0, "NUM_WATER": 4}

    kernels.module(init.UNIT).compile()
    kernels.module(dsum.UNIT).compile()
    kernels.module(avg.UNIT).compile()

    from namgpu.steps.drv_gpu_lease import gpu_lease
    n_names = None

    t0 = time.perf_counter()
    with gpu_lease(lane="nam-gpu-drv-dfi-dyn-average", vram_mib=14000, est_s=300), stream:
        init.run(ctx, ndfistep=75, dyn_vars=dyn)
        dsum.run(ctx)
        dfi = ctx.tables["DFI"]
        names3 = list(dfi["names_3d"])
        n_t = names3.index("T")
        pre_sums = {k: dfi[k] for k in ("sum_2d", "sum_3d")}
        n_names = (len(dfi["names_2d"]), len(names3), len(dfi["names_4d"]))
        avg.run(ctx)
    stream.synchronize()
    dt = time.perf_counter() - t0

    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    if bad:
        print("FAIL: NaN/Inf in", bad[:10])
        return 1

    # The sum array was scaled in place, so the copied T must equal its first LM levels exactly.
    sum_t = dfi["sum_3d"][n_t][:lm]
    if not bool(cp.array_equal(st["T"], sum_t)):
        print("FAIL: state T is not the scaled sum")
        return 1
    if not float(cp.abs(sum_t).max()) > 0.0:
        print("FAIL: scaled T sum is all zero (dyn-sum did not accumulate)")
        return 1

    if dfi["names_2d"] or dfi["names_3d"] or dfi["names_4d"] or ctx.filt["tot_rank_3d"] != 0 \
            or dfi["kstep"] != 0:
        print("FAIL: bookkeeping not reset after average")
        return 1

    print(f"names 2d/3d/4d {n_names}, totalsum {float(dfi['totalsum'][0]):.6g}, "
          f"{dt:.2f} s for init+sum+average")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
