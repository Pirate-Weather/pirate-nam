"""Done-rule check for drv-tqadjust only (CONTRACT.md section 6, parts 1 and 2).

Full 954 x 835 x 60 domain, but only the fields this unit reads and writes are allocated (the
full 48-field State does not fit beside the other tenants on the shared 5090 right now).
Compiles the kernel, runs the wrapper once, checks every written array for NaN/Inf.
"""
import sys
import time
from pathlib import Path

ROOT = Path.home() / "nam-gpu"
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

from namgpu import config, grid as gridmod, kernels, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.steps import drv_tqadjust as w  # noqa: E402

cfg = config.load()
g = gridmod.build(cfg)
names = ["T", "Q", "QC", "CW", "F_ICE", "F_RAIN", "PD", "DSG2", "SGML2"]
st = statemod.State(g, names=names)
rng = cp.random.default_rng(1)
for n in ["T", "Q", "QC", "CW", "F_ICE", "F_RAIN", "PD"]:
    st[n][...] = rng.random(st[n].shape, dtype=cp.float32)
st["T"][...] = 200.0 + 110.0 * st["T"]
st["Q"][...] = 0.02 * st["Q"]
st["QC"][...] = 1e-3 * st["QC"]
st["CW"][...] = 2e-3 * st["CW"]
st["PD"][...] = 50000.0 + 50000.0 * st["PD"]
st["DSG2"][...] = cp.asarray(np.asarray(g.dsg2, np.float32))
st["SGML2"][...] = cp.asarray(np.asarray(g.sgml2, np.float32))
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

t0 = time.perf_counter()
kernels.module("drv_tqadjust").compile()
print("compile ok", f"{time.perf_counter() - t0:.2f} s", flush=True)
t0 = time.perf_counter()
with ctx.stream:
    w.run(ctx)
ctx.stream.synchronize()
print(f"run ok {time.perf_counter() - t0:.2f} s  grid={g.im}x{g.jm}x{g.lm}", flush=True)
bad = [n for n in ["T", "Q", "QC", "CW", "F_ICE", "F_RAIN"]
       if not bool(cp.isfinite(st[n]).all())]
print("NaN/Inf in:", bad if bad else "none")
print("T range", float(st["T"].min()), float(st["T"].max()))
print("CW range", float(st["CW"].min()), float(st["CW"].max()))
print("F_ICE range", float(st["F_ICE"].min()), float(st["F_ICE"].max()))
print("F_RAIN range", float(st["F_RAIN"].min()), float(st["F_RAIN"].max()))
sys.exit(1 if bad else 0)
