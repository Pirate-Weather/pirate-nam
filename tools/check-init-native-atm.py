"""Done-rule check for init-native-atm (CONTRACT.md section 6, parts 1 and 2), full size.

    source env.sh
    python tools/check-init-native-atm.py

1. compiles the kernel file (NVRTC through CuPy);
2. runs the wrapper once on a full-size (954 x 835 x 60) state with plausible random decoded
   bgrd3d00 fields (some condensate values fall below the 1e-11 floor);
3. checks every array the wrapper writes for NaN/Inf.
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
    from namgpu.init import init_native_atm as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    # Only the fields this unit touches (full shapes); the whole NMMB state would not fit the shared card.
    st = statemod.State(g, names={"T", "Q", "U", "V", "CW", "W", "Q2", "F_ICE", "F_RAIN", "F_RIMEF", "PD"})
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    rng = cp.random.default_rng(3)

    def f3(lo, hi):
        return (lo + (hi - lo) * rng.random((lm, jm, im), dtype=cp.float32)).astype(cp.float32)

    st.add("TMP", (lm, jm, im))[...] = f3(200.0, 300.0)
    st.add("SPFH", (lm, jm, im))[...] = f3(0.0, 0.02)
    st.add("UGRD", (lm, jm, im))[...] = f3(-30.0, 30.0)
    st.add("VGRD", (lm, jm, im))[...] = f3(-30.0, 30.0)
    for n in ("CLMR", "CICE", "RWMR", "SNMR"):
        st.add(n, (lm, jm, im))[...] = f3(0.0, 2e-3) * (rng.random((lm, jm, im), dtype=cp.float32) > 0.5)
    st.add("DZDT", (lm, jm, im))[...] = f3(-1.0, 1.0)
    st.add("PSFC", (jm, im))[...] = (80000.0 + 21000.0 * rng.random((jm, im), dtype=cp.float32)).astype(cp.float32)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    kernels.module(unit.UNIT).compile()
    unit.run(ctx)
    ctx.stream.synchronize()

    written = ["T", "Q", "U", "V", "CW", "W", "Q2", "F_ICE", "F_RAIN", "F_RIMEF", "PD"]
    bad = [n for n in written if not bool(cp.isfinite(st.fields[n]).all())]
    print("NaN/Inf arrays:", bad if bad else "none")
    print("written shapes:", {n: tuple(st.fields[n].shape) for n in written})
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
