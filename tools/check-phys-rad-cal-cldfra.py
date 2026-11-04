"""Review-fix check for phys-rad-cal-cldfra: full size (954 x 835 x 60), one card.

    source env.sh
    python tools/check-phys-rad-cal-cldfra.py

Checks: the gate closes on cadence and on LW=SW=0; gfs runs with both flags (QC and QI thresholded);
an unknown microphysics name is refused; the kernel writes only the interior and no NaN/Inf.
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
    from namgpu.physics.driver import phys_rad_cal_cldfra as cf

    cfg = config.load()
    g = gridmod.build(cfg)
    # Only the three fields this unit touches: the full NAM state does not fit beside other lanes
    # on a shared card. QC/QI are uniform on [0, 2e-6) so both sides of the 1e-6 threshold occur.
    st = statemod.State(g, names={"CLDFRA", "QC", "QI"})
    cp.random.seed(1)
    st["QC"][...] = cp.random.uniform(0.0, 2.0e-6, st["QC"].shape, dtype=cp.float32)
    st["QI"][...] = cp.random.uniform(0.0, 2.0e-6, st["QI"].shape, dtype=cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nrad = int(cfg.nrads)
    print("grid", im, jm, lm, "nrads", nrad, "microphysics", cfg.microphysics)

    assert cf.cldfra_gate(1, nrad, 3, 3) and cf.cldfra_gate(nrad, nrad, 3, 3)
    assert not cf.cldfra_gate(nrad + 1, nrad, 3, 3)
    assert not cf.cldfra_gate(nrad, nrad, 0, 0)
    assert cf.cldfra_flags("gfs") == (True, True)
    try:
        cf.cldfra_flags("nosuch")
        raise SystemExit("unknown microphysics was not refused")
    except ValueError as exc:
        print("refused:", exc)

    kernels.module(cf.UNIT).compile()
    st["CLDFRA"].fill(-1.0)
    with ctx.stream:
        out = cf.run(ctx, itimestep=nrad + 1, sw_physics=3, lw_physics=3)
    assert out is None, "gate should be closed at cadence miss"
    with ctx.stream:
        out = cf.run(ctx, itimestep=nrad, sw_physics=3, lw_physics=3, microphysics="gfs")
    ctx.stream.synchronize()
    cld = st["CLDFRA"]
    interior = cld[:, 1:jm - 1, 1:im - 1]
    edge_kept = bool(cp.all(cld[:, 0, :] == -1.0)) and bool(cp.all(cld[:, :, 0] == -1.0))
    vals = bool(cp.all((interior == 0.0) | (interior == 1.0)))
    finite = bool(cp.isfinite(cld).all())
    print("gfs run ran:", out is not None, "| interior 0/1:", vals, "| boundary kept:", edge_kept,
          "| finite:", finite, "| cloudy fraction:", float(interior.mean()))
    assert out is not None and vals and edge_kept and finite


if __name__ == "__main__":
    main()
