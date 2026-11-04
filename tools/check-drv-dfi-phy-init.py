"""Done-rule check for drv-dfi-phy-init (review fix): restart run, full size, per-array presence.

    source ~/nam-gpu/env.sh
    python tools/check-drv-dfi-phy-init.py

One full-size random State (954 x 835 x 60), cfg.restart = True. The 2-D save array is pre-created with a
sentinel value, so the 3-D array is the only one absent. Runs drv-dfi-phy-init once, then checks:
  - the 2-D sentinel array is kept (per-array gating: its presence did not block the 3-D allocation);
  - DFI_PHY_SAVE_3D is created with shape (n3, LM, JM, IM), float32, zeroed, no NaN/Inf;
  - the counts equal the restore bundle split by rank, and ctx.tables["dfi_phy_init"] records the same names;
  - drv-dfi-phy-restore accepts the same restart flag (its record check passes).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps import drv_dfi_phy_init as init, drv_dfi_phy_restore as restore

    cfg = config.load()
    cfg.restart = True
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    names2, names3 = init.bundle_fields(ctx)
    n2, n3 = len(names2), len(names3)
    print("restart bundle: 2-D", names2, "3-D", names3)

    sentinel = st.add(init.NAME_2D, (n2, jm, im), "float32")
    sentinel.fill(7.0)

    kernels.module(init.UNIT).compile()
    with ctx.stream:
        init.run(ctx)
    ctx.stream.synchronize()

    s2, s3 = st.fields[init.NAME_2D], st.fields[init.NAME_3D]
    kept = bool(cp.all(s2 == 7.0))
    shape3_ok = tuple(s3.shape) == (n3, lm, jm, im) and str(s3.dtype) == "float32"
    zero3 = bool(cp.all(s3 == 0.0))
    finite = bool(cp.isfinite(s3).all()) and bool(cp.isfinite(s2).all())
    rec = ctx.tables["dfi_phy_init"]
    rec_ok = rec["name_save_2d"] == names2 and rec["name_save_3d"] == names3 \
        and rec["tot_rank_2d"] == n2 and rec["tot_rank_3d"] == n3
    print("2-D sentinel kept:", kept, "| 3-D shape/dtype ok:", shape3_ok, "| 3-D zeroed:", zero3,
          "| finite:", finite, "| record ok:", rec_ok)

    with ctx.stream:
        restore.run(ctx)
    ctx.stream.synchronize()
    print("restore accepted the record")

    if not (kept and shape3_ok and zero3 and finite and rec_ok):
        print("FAIL")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
