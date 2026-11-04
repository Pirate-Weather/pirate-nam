"""Done-rule check for phys-rrtm-lw-params (CONTRACT section 6).

Compiles the kernel, runs the wrapper once on the full-size 954 x 835 x 60 state, then checks the
written tables against the Fortran values (radlw_param_nmmb.f:57-175) and that nothing is NaN/Inf.

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-lw-params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu import kernels
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_lw_params as glue

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)

    kernels.module(glue.UNIT).compile()

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    glue.run(ctx)
    ctx.stream.synchronize()

    tbl = ctx.tables[glue.KEY]
    for name, a in tbl.items():
        if a.dtype.kind == "f":
            assert bool(cp.isfinite(a).all()), f"NaN/Inf in {name}"

    got = glue.check_status(ctx)
    assert got["ctrl"] == glue.CTRL, f"ctrl {got['ctrl']} != {glue.CTRL}"
    assert got["dims"] == glue.DIMS, f"dims {got['dims']} != {glue.DIMS}"
    assert got["ng"] == glue.NG, f"ng {got['ng']} != {glue.NG}"
    assert got["ns"] == glue.NS, f"ns {got['ns']} != {glue.NS}"
    assert got["wvn1"] == glue.WVN1, f"wvn1 {got['wvn1']} != {glue.WVN1}"
    assert got["wvn2"] == glue.WVN2, f"wvn2 {got['wvn2']} != {glue.WVN2}"
    import numpy as np
    ngb = cp.asnumpy(ctx.tables["rrtm_lw_ngb"])
    assert ngb.shape == (140,) and ngb[0] == 1 and ngb[9] == 1 and ngb[10] == 2 and ngb[-1] == 16, "ngb wrong"
    print(f"phys-rrtm-lw-params: ok, NBANDS={glue.NBANDS} NGPTLW={glue.NGPTLW} NPLNK={glue.DIMS[7]}, "
          f"ctrl={got['ctrl']}, ng sum={sum(got['ng'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
