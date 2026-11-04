"""Done-rule check for io-quasipost-interp (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_io_quasipost_interp.py

Compiles the kernel and runs its wrapper ONCE on a full-size random State (954 x 835 x 60), with the
QP_ inputs filled the way io-quasipost-3d leaves them and the ordering token 3d sets (this check
stands in for io-quasipost-3d.run). QP_GROUND_LEVEL is the 1-based index of the lowest standard level
below the surface pressure, npres+1 if none. Checks: check_err passes (no NAM abort path hit), no
NaN/Inf in QP_PRESZ or QP_PMSL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POST = np.arange(20000.0, 100000.0 + 1, 2500.0, dtype=np.float32)  # 33 levels, 20000..100000 Pa


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.io import io_quasipost_interp as uq
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    rng = np.random.default_rng(2)

    pres_tv = st.add("QP_PRESTV", (uq.NPRES, jm, im))
    pres_z = st.add("QP_PRESZ", (uq.NPRES, jm, im))
    ground = st.add("QP_GROUND_LEVEL", (jm, im), dtype="int32")
    pres_tv[...] = cp.asarray(250.0 + 40.0 * rng.random((uq.NPRES, jm, im), dtype=np.float32))
    pres_z[...] = cp.asarray(np.broadcast_to(
        np.linspace(1000.0, 30000.0, uq.NPRES).astype(np.float32)[:, None, None],
        (uq.NPRES, jm, im)).copy())
    ps = st.PINT[lm]                                              # surface interface pressure
    post = cp.asarray(POST)
    ground[...] = 1 + (post[:, None, None] <= ps[None]).sum(axis=0).astype(cp.int32)
    st.FIS[...] = cp.asarray(rng.random((jm, im), dtype=np.float32) * 2000.0 * 9.81)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    kernels.module(uq.UNIT).compile()
    ctx.tables[uq.TOKEN] = True                                   # stands in for io-quasipost-3d.run
    with gpu_lease(lane="nam-gpu-io-quasipost-interp", vram_mib=6000, est_s=300):
        with ctx.stream:
            uq.run(ctx)
        ctx.stream.synchronize()
        uq.check_err(ctx)                                         # raises on any abort path

    err = int(st.fields["QPI_ERR"][0])
    bad = [n for n in ("QP_PRESZ", "QP_PMSL") if not bool(cp.isfinite(st.fields[n]).all())]
    pm = st.fields["QP_PMSL"]
    print(f"io-quasipost-interp: err={err} ground_level range={int(ground.min())}..{int(ground.max())} "
          f"PMSL range={float(pm.min()):.1f}..{float(pm.max()):.1f} Pa "
          f"nonfinite={bad or 'none'}")
    return 0 if (err == 0 and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
