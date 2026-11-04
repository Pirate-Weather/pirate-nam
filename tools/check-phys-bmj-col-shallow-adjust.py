"""Done-rule check for phys-bmj-col-shallow-adjust and its zeroing producer phys-bmj-col-prep.

    source ~/nam-gpu/env.sh
    python tools/check-phys-bmj-col-shallow-adjust.py

Compiles phys_bmj_col_prep and phys_bmj_col_shallow_adjust (NVRTC) and runs them on a FULL-SIZE state
(954 x 835 x 60) under one pool-v3 lease on the shared card. Three checks:
  1. refusal: run(ctx) with no producers in ctx.st raises, naming the missing inputs (no stand-ins).
  2. prep zeroing: prep zeroes BMJ_DTDT and BMJ_DQDT over the whole field (a sentinel of 7.0 is gone).
  3. shallow adjust, done-rule path (standins=True): finite outputs, and no level outside ltop..lbot
     of a shallow column changes from its sentinel (the wrapper no longer fills whole fields).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod
    from namgpu import kernels
    from namgpu.driver import Ctx
    from namgpu.physics.bmj import phys_bmj_col_prep as prep
    from namgpu.physics.bmj import phys_bmj_col_shallow_adjust as unit
    from namgpu.steps.drv_gpu_lease import gpu_lease

    kernels.module(prep.UNIT).compile()
    kernels.module(unit.UNIT).compile()

    cfg = config.load()
    g = gridmod.build(cfg)
    IM, JM, LM = g.im, g.jm, g.lm
    rng = cp.random.default_rng(3)

    class _Fields:
        def __init__(self):
            self.fields = {}

        def add(self, name, shape, dtype="float32"):
            if name not in self.fields:
                self.fields[name] = cp.zeros(shape, dtype=dtype)
            return self.fields[name]

        def __getitem__(self, name):
            return self.fields[name]

        def __contains__(self, name):
            return name in self.fields

    st = _Fields()
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    bad = []
    with gpu_lease(vram_mib=4096, est_s=900, cards=1, what="phys-bmj-col-shallow-adjust check"):
        # 1. refusal with an empty state.
        try:
            unit.run(ctx)
            bad.append("refusal: run(ctx) did not raise with no producers")
        except RuntimeError as e:
            print("refusal ok: " + str(e).split(": ", 1)[-1][:160])

        # 2. prep zeroing. Driver inputs (PHMID, EXNER) are provided so prep does not need PD.
        f = st.fields
        lvl3 = (LM + 1, JM, IM)
        f["T"] = rng.uniform(220.0, 305.0, (LM, JM, IM), dtype=cp.float32)
        f["Q"] = rng.uniform(1e-6, 0.02, (LM, JM, IM), dtype=cp.float32)
        f["PINT"] = cp.sort(rng.uniform(2.0e4, 1.02e5, lvl3, dtype=cp.float32), axis=0)
        f["PHMID"] = cp.sort(rng.uniform(3.0e3, 1.0e5, (LM, JM, IM), dtype=cp.float32), axis=0)
        f["EXNER"] = cp.power(f["PHMID"] * cp.float32(1e-5), cp.float32(287.04 / 1004.6))
        for name in ("BMJ_DTDT", "BMJ_DQDT", "BMJ_DUDT", "BMJ_DVDT"):
            f[name] = cp.full((LM, JM, IM), 7.0, dtype=cp.float32)
        prep.run(ctx)
        cp.cuda.Device().synchronize()
        for name in ("BMJ_DTDT", "BMJ_DQDT", "BMJ_DUDT", "BMJ_DVDT"):
            n7 = int((f[name] == 7.0).sum())
            if n7:
                bad.append(f"prep left {n7} sentinel values in {name}")
        print("prep zeroing: DTDT/DQDT/DUDT/DVDT sentinel values left: "
              f"{[int((f[n] == 7.0).sum()) for n in ('BMJ_DTDT', 'BMJ_DQDT', 'BMJ_DUDT', 'BMJ_DVDT')]}")

        # 3. shallow adjust on the done-rule path, with the tendency sentinel set again.
        for name in ("BMJ_DTDT", "BMJ_DQDT", "BMJ_DUDT", "BMJ_DVDT"):
            f[name][...] = 7.0
        out = unit.run(ctx, standins=True)
        cp.cuda.Device().synchronize()
        for name, arr in out.items():
            if arr.dtype.kind == "f" and not bool(cp.isfinite(arr).all()):
                bad.append(f"non-finite {name}")
        lbot, ltop = out["BMJ_LBOT"], out["BMJ_LTOP"]
        lev = cp.arange(1, LM + 1, dtype=cp.int32)[:, None, None]
        written = (lev >= ltop[None]) & (lev <= lbot[None]) & (lbot[None] >= 1)
        for name in ("BMJ_DTDT", "BMJ_DQDT", "BMJ_DUDT", "BMJ_DVDT"):
            changed_outside = int(((f[name] != 7.0) & ~written).sum())
            if changed_outside:
                bad.append(f"{name}: {changed_outside} values changed outside ltop..lbot")
            print(f"shallow {name}: written levels {int(written.sum())}, changed outside {changed_outside}")
        print(f"shallow columns with lbot>=1: {int((lbot >= 1).sum())} of {JM * IM}")
    print("NaN/Inf/extra: " + (", ".join(bad) if bad else "none"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
