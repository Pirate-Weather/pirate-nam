"""Done-rule check for unit phys-bmj-driver (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-phys-bmj-driver.py

1. compiles namgpu/kernels/phys_bmj_driver.cu through NVRTC;
2. runs the wrapper ONCE on the full-size random State (954 x 835 x 60) at an output-window step
   (ntimestep 2, so the MNTO reset runs) and asserts no NaN/Inf in any float array afterwards.

The BMJ column scheme (unit phys-bmj-bmjdrv) is not built yet. This check installs a stand-in
for namgpu.physics.bmj.phys_bmj_bmjdrv that writes plausible values into the arrays bmjdrv
owns (RAINCV, CUTOP, CUBOT, RTHCUTEN, RQCUTEN, DUDT_PHY, DVDT_PHY). It exercises only this glue.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CALLS = {"bmjdrv": 0}


def _stub_bmjdrv(ctx):
    """Plausible bmjdrv outputs (stand-in only; not the BMJ scheme)."""
    import cupy as cp
    CALLS["bmjdrv"] += 1
    st, g = ctx.st, ctx.grid
    lm = int(g.lm)
    rng = cp.random.default_rng(3)
    st["RAINCV"][...] = 5.0 * rng.random(st["RAINCV"].shape, dtype=cp.float32)      # mm
    st["CUTOP"][...] = 1.0 + (lm + 1) * rng.random(st["CUTOP"].shape, dtype=cp.float32)
    st["CUBOT"][...] = 1.0 + lm * rng.random(st["CUBOT"].shape, dtype=cp.float32)
    st["RTHCUTEN"][...] = 1.0e-4 * (rng.random(st["RTHCUTEN"].shape, dtype=cp.float32) - 0.5)
    st["RQCUTEN"][...] = 1.0e-7 * rng.random(st["RQCUTEN"].shape, dtype=cp.float32)
    st["DUDT_PHY"][...] = 0.0
    st["DVDT_PHY"][...] = 0.0


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx

    t0 = time.perf_counter()
    kernels.module("phys_bmj_driver").compile()
    print(f"compile phys_bmj_driver: ok ({time.perf_counter() - t0:.2f} s)")

    cfg = config.load()
    g = gridmod.build(cfg)
    # Only the fields this unit touches, so the full 954 x 835 x 60 size fits the shared card
    # (the full 80-field State is about 10.6 GB; the card had about 11.5 GB free).
    names = {"PD", "T", "Q", "U", "V", "FIS", "SM", "TCUCN", "AVCNVC", "ACUTIM", "HTOP", "HBOT",
             "CUPPT", "CNVBOT", "CNVTOP", "HBOTD", "HTOPD", "HBOTS", "HTOPS", "PREC",
             "ACPREC", "ACPREC_TOT", "CUPREC", "CPRATE", "RAINCV", "CUTOP", "CUBOT",
             "RTHCUTEN", "RQCUTEN", "DUDT_PHY", "DVDT_PHY"}
    st = statemod.State(g, names=names).randomize(seed=1)
    # LPBL is written by the PBL unit; here a plausible PBL-top layer index (int32, 1..LM).
    st.add("LPBL", (int(g.jm), int(g.im)), dtype="int32")[...] = 12

    # Keep the physical pressure/temperature bounds sane for the glue's divisions.
    st["PD"][...] = cp.clip(st["PD"], 50000.0, 100000.0)

    sys.modules["namgpu.physics.bmj.phys_bmj_bmjdrv"] = types.SimpleNamespace(run=_stub_bmjdrv)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.ntimestep = 2   # call step (NCNVC = 2) and output-window step (MNTO = 2 <= NCNVC)
    ctx.tables["cu_physics"] = 2   # set by BMJ_INIT (Fortran 10941-10946)

    from namgpu.physics.driver import phys_bmj_driver as wrapper

    t0 = time.perf_counter()
    with ctx.stream:
        wrapper.run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    if CALLS["bmjdrv"] != 1:
        print(f"phys-bmj-driver: bmjdrv called {CALLS['bmjdrv']} times with CU_PHYSICS=2, want 1")
        return 1

    # CU_PHYSICS gate (Fortran 445): with CU_PHYSICS 0 the glue runs but bmjdrv must not be called.
    ctx.tables["cu_physics"] = 0
    with ctx.stream:
        wrapper.run(ctx)
    ctx.stream.synchronize()
    if CALLS["bmjdrv"] != 1:
        print(f"phys-bmj-driver: bmjdrv called with CU_PHYSICS=0 (count {CALLS['bmjdrv']})")
        return 1

    bad = [n for n, a in st.fields.items()
           if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    if bad:
        print(f"phys-bmj-driver: NaN/Inf in {bad[:10]} ({dt:.2f} s)")
        return 1
    print(f"phys-bmj-driver: ok, no NaN/Inf in any float array, {dt:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
