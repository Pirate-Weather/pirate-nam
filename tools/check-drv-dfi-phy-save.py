"""Done-rule check for drv-dfi-phy-save (review fix): restart run, full size.

    source env.sh
    python tools/check-drv-dfi-phy-save.py

One full-size random State (954 x 835 x 60). cfg.restart = True, so the save list is the restore
bundle (R rows plus B rows: TSHLTR, QSHLTR, U10, V10, TH10). Runs drv-dfi-phy-save once, then checks:
every listed field has its slot in DFI_PHY_SAVE_2D / _3D equal to the source (bit for bit, levels
0..NL-1), no NaN/Inf in the save arrays, and drv-dfi-phy-restore accepts the packed arrays.
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
    from namgpu.steps import drv_dfi_phy_save as save, drv_dfi_phy_restore as restore

    cfg = config.load()
    cfg.restart = True
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    names = save.save_names(True)
    print("restart save list:", names)
    kernels.module(save.UNIT).compile()
    with ctx.stream:
        save.run(ctx)
    ctx.stream.synchronize()

    s2, s3 = st.fields["DFI_PHY_SAVE_2D"], st.fields["DFI_PHY_SAVE_3D"]
    k2 = k3 = 0
    bad_eq = []
    for name in names:
        src = st.fields[name]
        if src.ndim == 2:
            ok = bool(cp.array_equal(s2[k2], src)); k2 += 1
        else:
            nl = src.shape[0]
            ok = bool(cp.array_equal(s3[k3, :nl], src)); k3 += 1
        if not ok:
            bad_eq.append(name)
    nonfinite = [n for n, a in (("DFI_PHY_SAVE_2D", s2), ("DFI_PHY_SAVE_3D", s3))
                 if not bool(cp.isfinite(a).all())]
    print("2-D slots", s2.shape, "3-D slots", s3.shape)
    print("mismatch:", bad_eq, "nonfinite:", nonfinite)

    with ctx.stream:
        restore.run(ctx)
    ctx.stream.synchronize()
    if bad_eq or nonfinite:
        print("FAIL")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
