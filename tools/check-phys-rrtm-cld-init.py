"""Done-rule check for phys-rrtm-cld-init (after the review fix).

    source env.sh
    python tools/check-phys-rrtm-cld-init.py

1. no SGM header: the wrapper refuses, naming the breakage (no stand-in).
2. full-size 954 x 835 x 60 random State, rad_init_args recorded as drv_physics_initialize records them
   (iflip 0, num_p3d 3, ntcw 3, iovr 1/1), header SGM supplied. Compile, one launch, read llyr and iovr.
   The header SGM here is a TEST input (the grid SG2 values, top first); the real run takes the NAM SGM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_cld_init as w
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.tables["rad_init_args"] = RadInitArgs(num_p3d=int(cfg.np3d), iaer=int(cfg.iaer), iems=int(cfg.iems),
                                              ntcw=int(cfg.ntcw), ntoz=int(cfg.ntoz))
    kernels.module("phys_rrtm_cld_init").compile()

    try:
        w.run(ctx)
        print("FAIL: run without SGM did not refuse")
        return 1
    except RuntimeError as e:
        print("refusal ok:", str(e)[:120])

    ctx.tables["vcoord_header"] = {"SGM": np.asarray(g.sg2, np.float64)}   # TEST input, see docstring
    with ctx.stream:
        w.run(ctx)
    ctx.stream.synchronize()
    res = w.check_status(ctx)
    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print("llyr, iovr =", res, "| NaN/Inf fields:", bad[:10])
    return 0 if (not bad and 1 <= res["llyr"] <= g.lm) else 1


if __name__ == "__main__":
    sys.exit(main())
