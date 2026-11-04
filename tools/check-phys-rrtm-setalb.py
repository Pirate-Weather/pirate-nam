"""Done-rule check for phys-rrtm-setalb (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-rrtm-setalb.py

Compiles the kernel and runs the wrapper once per ialbflg (0, 1, 2) on the full 954 x 835 grid with
a plausible random state. Checks SFCALB for NaN/Inf and the albedo range [0, 2.5] (direct-beam rfcs reaches about 2.14 near the terminator, by design).
The state follows the NAM caller's inputs: SM is a 0/1 mask (1 sea), SI is snow depth (mm), STDH is
terrain standard deviation (m), and coszen_loc comes from the dayparts table, as in grrad_sw.
Counted as the unit's one run, not a column test.
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
    from namgpu.physics.rrtm import phys_rrtm_setalb as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    rng = cp.random.default_rng(7)
    shp = (g.jm, g.im)
    # plausible surface state (randomize() fills everything uniform [0, 1))
    st.TSKIN[...] = 250.0 + 60.0 * rng.random(shp, dtype=cp.float32)
    st.Z0[...] = 0.01 + 0.5 * rng.random(shp, dtype=cp.float32)
    st.SI[...] = 300.0 * rng.random(shp, dtype=cp.float32)            # snow depth, mm
    st.STDH[...] = 2000.0 * rng.random(shp, dtype=cp.float32)         # terrain std dev, m
    st.SM[...] = cp.floor(2.0 * rng.random(shp, dtype=cp.float32))    # 0 land, 1 sea
    st.SICE[...] = rng.random(shp, dtype=cp.float32)
    # inputs that the NAM state tables do not carry yet (named gaps in the wrapper)
    for name, lo, hi in (("ALVSF", 0.0, 0.5), ("ALNSF", 0.0, 0.5), ("ALVWF", 0.0, 0.5),
                         ("ALNWF", 0.0, 0.5), ("FACSF", 0.0, 1.0), ("FACWF", 0.0, 1.0)):
        st.add(name, shp)
        st.fields[name][...] = lo + (hi - lo) * rng.random(shp, dtype=cp.float32)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    # coszen_loc as phys-rrtm-dayparts publishes it (flat, row-major p = j*IM + i)
    ctx.tables["rrtm_dayparts"] = {"coszen": (1.2 * rng.random(g.jm * g.im, dtype=cp.float32) - 0.2)}
    kernels.module(unit.KERNEL).compile()
    ok = True
    for ialb in (2, 0, 1):
        with ctx.stream:
            out = unit.run(ctx, ialbflg=ialb)
        ctx.stream.synchronize()
        finite = bool(cp.isfinite(out).all())
        lo, hi = float(out.min()), float(out.max())
        print(f"ialbflg={ialb}: finite={finite} min={lo:.4f} max={hi:.4f} shape={tuple(out.shape)}")
        if not finite or lo < 0.0 or hi > 2.5:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
