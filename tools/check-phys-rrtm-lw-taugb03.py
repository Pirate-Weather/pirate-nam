"""Done-rule check for phys-rrtm-lw-taugb03 (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-phys-rrtm-lw-taugb03.py

1. compiles the kernel (NVRTC through CuPy);
2. fills the setcoef-side inputs with random but plausible values at FULL size (954 x 835 x 60),
   runs the wrapper once, and checks every output for NaN/Inf.
The inputs are filled here, in the check, because setcoef is not ported yet; the wrapper does not fill them.
Take the GPU pool lease first (see README.md). No column test, no reduced size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, kernels, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_lw_taugb03 as W  # noqa: E402


def main():
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=[])   # no NAM tables: this unit owns only its own fields (memory)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    io = W.alloc(ctx)
    rs = cp.random.RandomState(1)
    im, jm, lm = g.im, g.jm, g.lm
    lmk = lm + 1                       # LMK: nlay of the kernel (LTP = 1)
    lay = (lmk, jm, im)

    def uni(shape, lo, hi, dtype=cp.float32):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(dtype)

    def ints(shape, lo, hi):
        return rs.randint(lo, hi, size=shape).astype(cp.int32)

    io["LW_LAYTROP"][...] = ints((jm, im), 10, 41)
    k = cp.arange(1, lmk + 1, dtype=cp.int32)[:, None, None]
    lower = k <= io["LW_LAYTROP"][None]
    io["LW_JP"][...] = cp.where(lower, ints(lay, 1, 11), ints(lay, 13, 59))   # lower 1..10, upper 13..58
    io["LW_JT"][...] = ints(lay, 1, 6)
    io["LW_JT1"][...] = ints(lay, 1, 6)
    io["LW_INDSELF"][...] = ints(lay, 1, 10)      # selfref has 10 rows, indsp = inds+1
    io["LW_INDFOR"][...] = ints(lay, 1, 4)        # forref has 4 rows, indfp = indf+1
    io["LW_INDMINOR"][...] = ints(lay, 1, 19)     # ka_mn2o/kb_mn2o have 19 rows, indmp = indm+1
    io["LW_COLAMT"][...] = uni((W.MAXGAS, lmk, jm, im), 1.0e18, 1.0e20)
    io["LW_RFRATE"][...] = uni((2, lmk, jm, im), 0.1, 10.0)
    io["LW_COLDRY"][...] = uni(lay, 1.0e19, 1.0e20)
    for n in ("LW_FAC00", "LW_FAC01", "LW_FAC10", "LW_FAC11", "LW_SELFFRAC", "LW_FORFRAC",
              "LW_MINORFRAC", "LW_FORFAC", "LW_SELFFAC"):
        io[n][...] = uni(lay, 0.0, 1.0)
    cp.cuda.runtime.deviceSynchronize()

    kernels.module(W.UNIT).compile()
    t0 = time.perf_counter()
    with ctx.stream:
        out = W.run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    bad = {n: int((~cp.isfinite(out[n])).sum()) for n in W.OUTPUTS}
    print(f"{W.UNIT}: full size {im}x{jm}x{lmk} (LM {lm} + top layer), wall {dt:.2f} s (includes first compile/table upload)")
    for n in W.OUTPUTS:
        a = out[n]
        print(f"  {n}: non-finite {bad[n]}, min {float(cp.nanmin(a)):.4g}, max {float(cp.nanmax(a)):.4g}")
    return 0 if all(v == 0 for v in bad.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
