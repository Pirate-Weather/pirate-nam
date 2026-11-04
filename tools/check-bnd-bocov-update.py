"""Done-rule check for bnd-bocov-update (CONTRACT.md section 6, parts 1 and 2).

    source env.sh && python tools/check-bnd-bocov-update.py

Compiles namgpu/kernels/bnd_bocov_update.cu and runs the wrapper once on full-size boundary arrays
(954 x 835 x 60, LNSV = 5, two 2-D and two 3-D V-point variables on all four strips) filled with
random values. Asserts no NaN/Inf and that the value slots changed (the launches ran).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels
    from namgpu.boundary import bnd_bocov_update as w
    from namgpu.driver import Ctx
    from namgpu import state as statemod

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm, lnsv = int(g.im), int(g.jm), int(g.lm), int(cfg.lnsv)
    kernels.module("bnd_bocov_update").compile()

    rng = cp.random.default_rng(1)

    def rnd(shape):
        return rng.random(shape, dtype=cp.float32)

    def var2(): return types.SimpleNamespace(
        south=rnd((2, lnsv, im)), north=rnd((2, lnsv, im)),
        west=rnd((2, jm, lnsv)), east=rnd((2, jm, lnsv)))

    def var3(): return types.SimpleNamespace(
        south=rnd((2, lm, lnsv, im)), north=rnd((2, lm, lnsv, im)),
        west=rnd((2, lm, jm, lnsv)), east=rnd((2, lm, jm, lnsv)))

    v2 = [var2(), var2()]
    v3 = [var3(), var3()]
    before = [float(v.south[0, 0, 1].get()) for v in v2]
    bc = types.SimpleNamespace(v=types.SimpleNamespace(var_2d=v2, var_3d=v3))
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True),
              dt=float(cfg.dt), bc=bc)
    with ctx.stream:
        w.run(ctx)
    ctx.stream.synchronize()

    arrays = [a for v in v2 + v3 for a in (v.south, v.north, v.west, v.east)]
    bad = [i for i, a in enumerate(arrays) if not bool(cp.isfinite(a).all())]
    changed = float(v2[0].south[0, 0, 1].get()) != before[0]
    print(f"bnd-bocov-update: IM={im} JM={jm} LM={lm} LNSV={lnsv} dt={cfg.dt}; "
          f"nonfinite arrays={len(bad)}; launches ran={changed}")
    return 0 if (not bad and changed) else 1


if __name__ == "__main__":
    sys.exit(main())
