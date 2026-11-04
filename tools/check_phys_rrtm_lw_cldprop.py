"""Done-rule check for phys-rrtm-lw-cldprop (CONTRACT section 6, parts 1 and 2).

    source env.sh
    python tools/check_phys_rrtm_lw_cldprop.py

Compiles the kernel (NVRTC), fills RAD_CLOUDS (the state array phys-rrtm-grrad-clouds writes) with a
full-size plausible random state (954 x 835 x LMK), and runs the wrapper once for isubclw=0 and once for
isubclw=1. Checks taucld and cldf for NaN/Inf and checks the gate and the refusals. No column test,
no bitwise comparison.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fill_clouds(cp, clouds):
    """Plausible LW cloud fields in state order. Channel 0 cfrac in [0,1] with a clear majority of zeros."""
    rs = cp.random.RandomState(7)
    shape = clouds.shape[1:]
    u = rs.uniform(0.0, 1.0, shape).astype(cp.float32)
    cf = rs.uniform(0.0, 1.0, shape).astype(cp.float32)
    cf[cf < 0.4] = 0.0
    clouds[0] = cf
    clouds[1] = (rs.uniform(0.0, 1.0, shape) * 200.0 * (u > 0.3)).astype(cp.float32)   # clwp
    clouds[2] = (5.0 + 15.0 * u).astype(cp.float32)                                   # relw
    clouds[3] = (rs.uniform(0.0, 1.0, shape) * 100.0 * (u > 0.5)).astype(cp.float32)  # ciwp
    clouds[4] = (20.0 + 60.0 * u).astype(cp.float32)                                  # reiw
    clouds[5] = (0.5 * u).astype(cp.float32)                                          # cda1
    clouds[6] = (0.5 * u).astype(cp.float32)                                          # cda2
    clouds[7] = (5.0 * u).astype(cp.float32)                                          # cda3
    clouds[8] = (10.0 + 100.0 * u).astype(cp.float32)                                 # cda4


def main():
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import config, grid as gridmod, kernels

    # Only the state this unit reads is allocated (RAD_CLOUDS, about 1.9 GB at full size). The full
    # 10.6 GB State is not needed for this unit and does not fit beside the other pool leases on the card.
    cfg = config.load()
    g = gridmod.build(cfg)
    lmk = int(g.lm) + 1                      # grrad LTP = 1 extra top layer
    clouds = cp.zeros((9, lmk, int(g.jm), int(g.im)), dtype=cp.float32)
    _fill_clouds(cp, clouds)
    st = SimpleNamespace(fields={"RAD_CLOUDS": clouds})
    ctx = SimpleNamespace(st=st, grid=g, cfg=cfg, tables={}, stream=cp.cuda.Stream(non_blocking=True))
    wrap = importlib.import_module("namgpu.physics.rrtm.phys_rrtm_lw_cldprop")
    kernels.module("phys_rrtm_lw_cldprop").compile()
    ok = True
    for isub in (0, 1):
        t0 = time.perf_counter()
        ctx.tables["rrtm_lw_ctl_host"] = {"ilwcliq": 1, "isubclw": isub}   # rlwinit copy (NAM values)
        with ctx.stream:
            taucld, cldf = wrap.run(ctx)
        ctx.stream.synchronize()
        nbad = int((~cp.isfinite(taucld)).sum())
        if cldf is not None:
            nbad += int((~cp.isfinite(cldf)).sum())
        tmin = float(taucld.min())
        tmax = float(taucld.max())
        tmean = float(taucld.mean())
        dt = time.perf_counter() - t0
        print(f"isubclw={isub}: nonfinite={nbad} taucld min={tmin:.4g} max={tmax:.4g} mean={tmean:.4g} "
              f"cldf={'none' if cldf is None else 'ok'} ({dt:.2f} s)")
        if nbad:
            ok = False
    # refusals: undefined Fortran branches must raise, not compute
    for kw in ({"ilwcliq": 2}, {"ilwcliq": 1, "ilwcice": 0}):
        try:
            wrap.run(ctx, **kw)
            print(f"refusal missing for {kw}")
            ok = False
        except ValueError as e:
            print(f"refused {kw}: ok")
    # lcf1 gate: column (j=0, i=0) with every layer at 0 < cfrac <= 1e-6 must give taucld = 0 (radlw_main_nmmb.f:924-945)
    clouds[0][:, 0, 0] = 1.0e-7
    ctx.tables["rrtm_lw_ctl_host"] = {"ilwcliq": 1, "isubclw": 0}
    taucld, _ = wrap.run(ctx)
    gmax = float(cp.abs(taucld[:, :, 0, 0]).max())
    print(f"gate column (cfrac 1e-7 everywhere): taucld max={gmax:.3g} ok={gmax == 0.0}")
    if gmax != 0.0:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
