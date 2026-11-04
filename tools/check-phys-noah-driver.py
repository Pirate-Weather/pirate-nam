"""Done-rule check for phys-noah-driver (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-noah-driver.py

1. compiles namgpu/kernels/phys_noah_driver.cu through the NVRTC loader;
2. runs the wrapper ONCE on a full-size 954 x 835 x 60 state (only the fields this unit touches)
   and asserts no NaN or Inf in any array it writes. The inputs are physically shaped (monotone
   pressure, land-sea mask, soil moisture 0.1-0.4, soil temperature 270-290 K, snow 0-20 mm).

phys-noah-sflx is a separate unit. Until its header ``namgpu/kernels/phys_noah_sflx.cuh`` lands,
this check compiles against a stub noah_sflx() that writes plausible fixed values from
NOAH_SFLX_STUB (default scratch/noah-unit-test/stub). The check then proves the driver's
own code only, and says so in its output. Part 3 (Fortran-to-CUDA diff review) is recorded in
the kernel header.

Take the GPU pool lease first (README.md). No column test, no reduced size.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES_3D = ["DZ", "T", "Q", "SMC", "STC", "SH2O"]          # SOIL ones are [NSOIL, JM, IM]
NAMES_PINT = ["PHINT"]
NAMES_2D_F32 = [
    "TSFC", "RLW_DN_SFC", "RSW_DN_SFC", "RAINBL", "QGH", "EPSR", "ALBEDO", "ALBASE", "Z0BASE",
    "TG", "MXSNAL", "SR", "RIMEF", "CHS", "CQS2", "CHKLOWQ", "SNOW", "SNOWH", "SNOWC", "QSH",
    "CMC", "RIB", "ACSNOM", "ACSNOW", "SNOPCX", "POTEVP", "GRNFLX", "ELFLX", "VGFRCK", "SICE",
    "XLAND", "TWBS", "QWBS", "SSROFF", "BGROFF", "SMSTAV", "SMSTOT", "CHS2",
]
NAMES_2D_INT = ["IVGTYP", "ISLTYP"]
OUT_NAMES = ["TSFC", "TWBS", "QWBS", "ELFLX", "GRNFLX", "SMSTAV", "SMSTOT", "SSROFF", "BGROFF",
             "ALBEDO", "SNOWC", "QSH", "CHS2", "CQS2", "CHKLOWQ", "SMC", "STC", "SH2O", "SNOW",
             "SNOWH", "CMC", "ACSNOM", "ACSNOW", "SNOPCX", "POTEVP"]


def _stub_dir() -> Path:
    return Path(os.environ.get("NOAH_SFLX_STUB",
                               "scratch/noah-unit-test/stub"))


def main() -> int:
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.noah import phys_noah_driver as wrap

    real_header = Path(kernels.HERE) / "phys_noah_sflx.cuh"
    if real_header.exists():
        print("SFLX: real phys_noah_sflx.cuh in the kernels folder")
    else:
        stub = _stub_dir()
        print(f"SFLX: STUB noah_sflx from {stub} (phys-noah-sflx not landed; driver-only proof)")
        kernels.OPTIONS = kernels.OPTIONS + (f"-I{stub}",)

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    print(f"grid {im} x {jm} x {lm}, dt={cfg.dt}, nphs={cfg.nphs}")

    wanted = set(NAMES_3D + NAMES_PINT + NAMES_2D_F32 + NAMES_2D_INT + ["SLDPTH"])
    st = statemod.State(g, names=wanted)
    rng = cp.random.default_rng(11)
    f32 = cp.float32

    print("compiling phys_noah_driver ...")
    t0 = time.time()
    kernels.module(wrap.UNIT).compile()
    print(f"  compiled in {time.time() - t0:.1f} s")

    def add(name, shape, dtype="float32"):
        if name in st.fields:
            return st.fields[name]
        return st.add(name, shape, dtype)

    pint = cp.linspace(2.0e4, 1.0e5, lm + 1, dtype=f32)[:, None, None] * cp.ones((1, jm, im), f32)
    add("PHINT", (lm + 1, jm, im))[...] = pint
    add("DZ", (lm, jm, im))[...] = f32(300.0)
    add("T", (lm, jm, im))[...] = f32(270.0) + f32(25.0) * rng.random((lm, jm, im), dtype=f32)
    add("Q", (lm, jm, im))[...] = f32(0.005) + f32(0.01) * rng.random((lm, jm, im), dtype=f32)
    add("SMC", (4, jm, im))[...] = f32(0.1) + f32(0.3) * rng.random((4, jm, im), dtype=f32)
    add("SH2O", (4, jm, im))[...] = f32(0.9) * add("SMC", (4, jm, im))
    add("STC", (4, jm, im))[...] = f32(270.0) + f32(20.0) * rng.random((4, jm, im), dtype=f32)
    xland = (1.0 + (rng.random((jm, im), dtype=f32) < 0.3)).astype(f32)
    add("XLAND", (jm, im))[...] = xland
    add("SICE", (jm, im))[...] = (rng.random((jm, im), dtype=f32) < 0.05).astype(f32)
    add("IVGTYP", (jm, im), "int32")[...] = (1 + rng.integers(0, 20, (jm, im))).astype(np.int32)
    add("ISLTYP", (jm, im), "int32")[...] = (1 + rng.integers(0, 12, (jm, im))).astype(np.int32)
    add("VGFRCK", (jm, im))[...] = f32(100.0) * rng.random((jm, im), dtype=f32)
    add("TSFC", (jm, im))[...] = f32(280.0) + f32(20.0) * rng.random((jm, im), dtype=f32)
    add("RLW_DN_SFC", (jm, im))[...] = f32(300.0)
    add("RSW_DN_SFC", (jm, im))[...] = f32(800.0) * rng.random((jm, im), dtype=f32)
    add("RAINBL", (jm, im))[...] = f32(2.0) * rng.random((jm, im), dtype=f32)
    add("QGH", (jm, im))[...] = f32(0.01)
    add("EPSR", (jm, im))[...] = f32(0.95)
    add("ALBEDO", (jm, im))[...] = f32(0.2)
    add("ALBASE", (jm, im))[...] = f32(0.2)
    add("Z0BASE", (jm, im))[...] = f32(0.1)
    add("TG", (jm, im))[...] = f32(280.0)
    add("MXSNAL", (jm, im))[...] = f32(0.7)
    add("SR", (jm, im))[...] = (rng.random((jm, im), dtype=f32) < 0.2).astype(f32)
    add("RIMEF", (jm, im))[...] = f32(1.0)
    add("CHS", (jm, im))[...] = f32(0.01)
    add("CQS2", (jm, im))[...] = f32(0.01)
    add("CHKLOWQ", (jm, im))[...] = f32(1.0)
    add("SNOW", (jm, im))[...] = f32(20.0) * rng.random((jm, im), dtype=f32)
    add("SNOWH", (jm, im))[...] = f32(0.1) * rng.random((jm, im), dtype=f32)
    add("SNOWC", (jm, im))[...] = rng.random((jm, im), dtype=f32)
    add("QSH", (jm, im))[...] = f32(0.01)
    add("CMC", (jm, im))[...] = f32(0.5) * rng.random((jm, im), dtype=f32)
    add("RIB", (jm, im))[...] = f32(0.1)
    for n in ("ACSNOM", "ACSNOW", "SNOPCX", "POTEVP", "SSROFF", "BGROFF", "GRNFLX", "ELFLX",
              "TWBS", "QWBS", "SMSTAV", "SMSTOT", "CHS2"):
        add(n, (jm, im))[...] = f32(0.0)
    add("SLDPTH", (4,))[...] = cp.asarray(np.array([0.1, 0.3, 0.6, 1.0], np.float32))

    ctx = Ctx(cfg=cfg, grid=g, st=st, dt=float(cfg.dt))
    print("running phys_noah_driver once on the full-size state ...")
    cp.cuda.Device().synchronize()
    t0 = time.time()
    wrap.run(ctx)
    cp.cuda.Device().synchronize()
    print(f"  wall {time.time() - t0:.2f} s (includes first-call NVRTC load)")

    # the kernel must have run: every land point of the B1 region carries the SFLX stub value
    xl = st["XLAND"][1:-1, 1:-1]
    land = cp.asnumpy(xl < 1.5)
    flx = cp.asnumpy(st["ELFLX"][1:-1, 1:-1])
    n_land, n_set = int(land.sum()), int((flx == np.float32(100.0))[land].sum())
    print(f"land points in B1 region {n_land}, written by the driver {n_set}")
    if n_set != n_land:
        print("driver did not write every land point")
        return 1
    bad = []
    for name in OUT_NAMES:
        a = st[name]
        n_bad = int((~cp.isfinite(a)).sum())
        if n_bad:
            bad.append((name, n_bad))
    if bad:
        print("NaN/Inf in outputs:", bad)
        return 1
    print(f"no NaN or Inf in {len(OUT_NAMES)} written arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
