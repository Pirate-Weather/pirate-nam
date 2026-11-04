"""Review check for drv-dfi-dyn-init: filter clock dt/taus, F+B cold-start bundle, full-size no-NaN run.

    source env.sh && python -I tools/check-drv-dfi-dyn-init-review.py
"""
from __future__ import annotations

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
    from namgpu.steps import drv_dfi_dyn_init as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    from namgpu.steps import drv_tracer_index
    drv_tracer_index.run(ctx)                 # ctx.tables["indx"] for NUM_TRACERS_MET/CHEM/WATER
    kernels.module(unit.UNIT).compile()

    fields = unit.dyn_filter_vars()
    names = [n for n, _ in fields]
    t0 = time.perf_counter()
    filt = unit.run(ctx)
    ctx.stream.synchronize()
    dt_wall = time.perf_counter() - t0

    fdt = float(cfg.filt_dt)
    nstep = int(filt["nstep"])
    bad = []
    for key in ("sum2d", "sum3d", "totalsum", "dolph"):
        arr = cp.asnumpy(filt[key]) if hasattr(filt[key], "get") else np.asarray(filt[key])
        if not np.all(np.isfinite(arr)):
            bad.append(key)
    dolph = cp.asnumpy(filt["dolph"])

    print(f"model dt {cfg.dt} s, filter dt {fdt} s")
    print(f"dt {filt['dt']} (want {fdt}), taus {filt['taus']} (want {2*nstep*fdt}), nstep {nstep}")
    print(f"dolph sum {float(dolph.sum()):.6f} (want ~1), finite-bad {bad}")
    print(f"bundle: {len(fields)} fields (2-D {filt['tot_rank_2d']}, 3-D {filt['tot_rank_3d']}, 4-D {filt['tot_rank_4d']})")
    print(f"B rows included: {[n for n in ('TSHLTR','QSHLTR','U10','V10','TH10') if n in names]}")
    print(f"R rows excluded: {[n for n in ('STC','SMC','TSKIN') if n not in names]}")
    print(f"wall {dt_wall:.2f} s")

    ok = (abs(filt["dt"] - fdt) < 1e-6 and abs(filt["taus"] - 2 * nstep * fdt) < 1e-3
          and abs(float(dolph.sum()) - 1.0) < 1e-3 and not bad
          and all(n in names for n in ("TSHLTR", "QSHLTR", "U10", "V10", "TH10"))
          and not any(n in names for n in ("STC", "SMC", "TSKIN")))
    print("ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
