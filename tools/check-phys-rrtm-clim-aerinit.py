"""Done-rule check for phys-rrtm-clim-aerinit (CONTRACT.md section 6, parts 1 and 2, plus the review gaps).

    source env.sh
    python tools/check-phys-rrtm-clim-aerinit.py

1. compiles the kernel file (NVRTC via CuPy);
2. builds the full-size State (config, grid, State(grid).randomize(), 954 x 835 x 60) as CONTRACT section 6.2 asks;
3. chains the real producers: phys-rrtm-rad-initialize, then phys-rrtm-aer-init, so solfwv/eirfwv are the
   aer_init fluxes (not random numbers);
4. runs clim_aerinit once with the defaults (laswflg, lalwflg True): no NaN/Inf in any float table, ssa in [0, 1],
   asy in [-1, 1], 30 band rows with sw rows 0..13 and lw rows 14..29, and every lw row filled;
5. runs clim_aerinit once with laswflg False (the LW index-offset defect): 16 band rows, lw band nb in row nb,
   no sw rows, no NaN/Inf.
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
    from namgpu.physics.rrtm import phys_rrtm_aer_init as aer_init
    from namgpu.physics.rrtm import phys_rrtm_clim_aerinit as clim
    from namgpu.physics.rrtm import phys_rrtm_rad_initialize as radinit

    kernels.module("phys_rrtm_clim_aerinit").compile()
    print("compile ok", flush=True)

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    print(f"state {st.nbytes() / 2**30:.2f} GiB, grid {int(g.im)} x {int(g.jm)} x {int(g.lm)}", flush=True)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    with ctx.stream:
        radinit.run(ctx)
        aer_init.run(ctx)
    ctx.stream.synchronize()
    flags = aer_init.check_status(ctx)
    assert flags["nswbnd"] == 14 and flags["nlwbnd"] == 16, flags
    print("aer_init flags", flags, flush=True)

    # ---- run A: defaults (the NAM 12 km configuration: laswflg, lalwflg True)
    with ctx.stream:
        clim.run(ctx)
    ctx.stream.synchronize()
    t = ctx.tables
    bad = [n for n, a in t.items() if hasattr(a, "dtype") and a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    assert not bad, f"NaN/Inf in {bad}"
    for name in ("ssarhi", "ssarhd"):
        a = t[name]
        assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0, f"{name} outside [0,1]: {float(a.min())} {float(a.max())}"
    for name in ("asyrhi", "asyrhd"):
        a = t[name]
        assert float(a.min()) >= -1.0 and float(a.max()) <= 1.0, f"{name} outside [-1,1]"
    assert t["extrhi"].shape == (30, 6), t["extrhi"].shape
    lw_rows_zero = [r for r in range(14, 30) if float(cp.abs(t["extrhi"][r]).sum()) == 0.0]
    assert not lw_rows_zero, f"lw rows empty: {lw_rows_zero}"
    print("run A defaults: rows 30; NaN/Inf none; ssa/asy in range; all lw rows filled", flush=True)
    print("  extrhi[0,0] sw1", float(t["extrhi"][0, 0]), "extrhi[14,0] lw1", float(t["extrhi"][14, 0]), flush=True)
    ref_lw = t["extrhi"][14:30].get().copy()

    # ---- run B: laswflg False (lw band nb must land in row nb, not 14 + nb)
    with ctx.stream:
        clim.run(ctx, laswflg=False)
    ctx.stream.synchronize()
    t = ctx.tables
    assert t["extrhi"].shape == (16, 6), t["extrhi"].shape
    bad = [n for n, a in t.items() if hasattr(a, "dtype") and a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    assert not bad, f"NaN/Inf in {bad}"
    rows = [float(cp.abs(t["extrhi"][r]).sum()) for r in range(16)]
    assert all(v > 0.0 for v in rows), f"lw rows empty with laswflg False: {rows}"
    got = t["extrhi"].get()
    assert np.array_equal(got, ref_lw), "lw band values moved when laswflg is False"
    print("run B laswflg False: rows 16; lw band nb in row nb; NaN/Inf none", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
