"""Done-rule check for unit phys-bmj-ttblex (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-phys-bmj-ttblex.py

Compiles namgpu/kernels/phys_bmj_ttblex.cu and runs the wrapper once on a FULL-SIZE random State
(954 x 835 x 60 grid; the columns are full size). run() builds the BMJ tables itself through phys-bmj-init (no synthetic tables,
no drv-tablet dependency), then evaluates both sets. The check is NaN/Inf on the output only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.bmj import phys_bmj_ttblex as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=())  # no NAM fields needed: BMJ tables + [JM, IM] scratch only
    IM, JM = g.im, g.jm
    rng = cp.random.default_rng(1)

    for name in unit.SCRATCH_2D:
        st.add(name, (JM, IM))
    st.fields["BMJ_TTLEX_PRSMID"][...] = rng.uniform(10000.0, 100000.0, (JM, IM), dtype=cp.float32)
    st.fields["BMJ_TTLEX_THESP"][...] = rng.uniform(200.0, 350.0, (JM, IM), dtype=cp.float32)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    kernels.module(unit.UNIT).compile()
    bad = False
    with ctx.stream:
        for tableset in ("lo", "q"):
            out = unit.run(ctx, tableset)
            ctx.stream.synchronize()
            ok = bool(cp.isfinite(out).all())
            bad = bad or not ok
            print(f"phys-bmj-ttblex {tableset}: {'ok' if ok else 'NaN/Inf in TREF'}, shape {tuple(out.shape)}, "
                  f"min {float(out.min()):.2f} max {float(out.max()):.2f}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
