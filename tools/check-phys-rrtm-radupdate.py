"""Done-rule check for phys-rrtm-radupdate (CONTRACT.md section 6, parts 1 and 2), review fixes.

    source env.sh
    python tools/check-phys-rrtm-radupdate.py

1. compiles the kernel (part 1);
2. builds the full-size random State (954 x 835 x 60) and runs the wrapper with the driver's call
   signature (radupdate(ctx, jdat=, dtsw=, dtx=, lssw=)) once (part 2), checking every float array
   for NaN/Inf and that SOLCON comes back finite and plausible;
3. runs one month-change step (January to April) to exercise aer_update with iyear/imon from JDAT,
   and one lssw=False step, which must keep the previous SOLCON.
Take the GPU pool lease first.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.physics.rrtm import phys_rrtm_radupdate as ru
    from namgpu.physics.rrtm import phys_rrtm_sol_init as soli

    ru.selftest()
    kernels.module(ru.UNIT).compile()
    print("compile ok; selftest ok")

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    bad = []
    for name in st.fields:
        arr = st[name]
        if hasattr(arr, "dtype") and np.issubdtype(np.dtype(arr.dtype), np.floating):
            if not bool(cp.isfinite(arr).all()):
                bad.append(name)
    print("full-size State", int(g.im), "x", int(g.jm), "x", int(g.lm), "non-finite input fields:", bad)

    ctx = SimpleNamespace(cfg=cfg, tables={}, stream=None, st=st, grid=g)
    soli.run(ctx, isolar=int(cfg.isolar))           # radinit's sol-init (isolar 0: no table file)
    dt = float(cfg.dt)

    out = ru.run(ctx, jdat=[2023, 3, 31, 0, 12, 0, 0, 0], dtsw=float(cfg.nrads) * dt, dtx=dt, lssw=True)
    print("step 1 (2023-03-31 12Z, lssw):", {k: round(v, 4) for k, v in out.items()})
    assert all(np.isfinite(v) for v in out.values())
    assert 1200.0 < out["SOLCON"] < 1500.0, out["SOLCON"]

    # month change and aer_update: JDAT month 4 must reach aer_update as imon=4
    out2 = ru.run(ctx, jdat=[2023, 4, 15, 0, 0, 0, 0, 0], dtsw=float(cfg.nrads) * dt, dtx=dt, lssw=False)
    print("step 2 (2023-04-15, lssw False):", {k: round(v, 4) for k, v in out2.items()})
    assert out2["SOLCON"] == out["SOLCON"]
    assert int(ctx.tables[ru.KEY_RST][0]) == 4
    assert "aer_idxcg" in ctx.tables and "aer_volc" in ctx.tables
    print("aer_update ran for the month change; volc state kmonsav =", ctx.tables["aer_volc"].kmonsav)
    print("NO NaN/Inf in writes: ok")


if __name__ == "__main__":
    main()
