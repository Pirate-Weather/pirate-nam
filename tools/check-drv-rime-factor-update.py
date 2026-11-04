"""Done-rule check for drv-rime-factor-update (CONTRACT.md section 6): full 954 x 835 x 60 domain.

Allocates only QS, QG, F_RIMEF (3 x 191 MB). Part 1: with the configured parm (spec_adv .false.)
the gate must be off and the wrapper must leave QG and F_RIMEF untouched. Part 2: spec_adv forced
to .true. (the gate's other input, microphysics fer_hires, is the configured one), so both kernels
run at full size; every written array is checked for NaN/Inf and the rime-factor bounds
(1 <= F_RIMEF <= 50 where QG and QS exceed EPSQ, F_RIMEF = 1 elsewhere).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from namgpu import config, grid as gridmod, kernels, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.steps import drv_rime_factor_update as w  # noqa: E402

EPSQ = 1.0e-12
cfg = config.load()
g = gridmod.build(cfg)
st = statemod.State(g, names=["QS", "QG", "F_RIMEF"])
rng = cp.random.default_rng(1)
st["QS"][...] = 1e-3 * rng.random(st["QS"].shape, dtype=cp.float32)
st["QG"][...] = 1e-3 * rng.random(st["QG"].shape, dtype=cp.float32)
st["QG"][:, :10, :10] = 0.0          # some cells below EPSQ, exercises the else branch
st["F_RIMEF"][...] = 1.0 + 9.0 * rng.random(st["F_RIMEF"].shape, dtype=cp.float32)
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
print(f"grid {g.im}x{g.jm}x{g.lm}  microphysics={cfg.microphysics}  spec_adv={cfg.raw.get('spec_adv')}", flush=True)

t0 = time.perf_counter()
kernels.module("drv_rime_factor_update").compile()
print("compile ok", f"{time.perf_counter() - t0:.2f} s", flush=True)

# Part 1: configured parm, gate must be off and both branches must be no-ops.
qg0 = st["QG"].copy()
rf0 = st["F_RIMEF"].copy()
assert not w.advect_enabled(cfg), "gate should be off with spec_adv .false."
with ctx.stream:
    w.run(ctx, branch="before")
    w.run(ctx, branch="after")
ctx.stream.synchronize()
assert bool(cp.array_equal(st["QG"], qg0)) and bool(cp.array_equal(st["F_RIMEF"], rf0))
print("part 1 ok: gate off, both branches no-op", flush=True)

# Part 2: gate on (spec_adv forced .true.), full-size kernels.
cfg.raw["spec_adv"] = True
assert w.advect_enabled(cfg)
qs = st["QS"].copy()
t0 = time.perf_counter()
with ctx.stream:
    w.run(ctx, branch="before")
ctx.stream.synchronize()
print(f"before run {time.perf_counter() - t0:.3f} s", flush=True)
qg_expect = qs * rf0
err_before = float(cp.abs(st["QG"] - qg_expect).max())
print("before max |QG - QS*F_RIMEF| =", err_before, flush=True)

qg_before = st["QG"].copy()
t0 = time.perf_counter()
with ctx.stream:
    w.run(ctx, branch="after")
ctx.stream.synchronize()
print(f"after run {time.perf_counter() - t0:.3f} s", flush=True)

rf = st["F_RIMEF"]
ok_mask = (qg_before > EPSQ) & (qs > EPSQ)
ratio = cp.where(ok_mask, qg_before / cp.maximum(qs, EPSQ), 1.0)
expect = cp.minimum(50.0, cp.maximum(1.0, ratio)).astype(cp.float32)
err_after = float(cp.abs(rf - expect).max())
print("after max |F_RIMEF - expected| =", err_after, flush=True)

bad = [n for n in ["QG", "F_RIMEF"] if not bool(cp.isfinite(st[n]).all())]
print("NaN/Inf in:", bad if bad else "none", flush=True)
print("F_RIMEF range", float(rf.min()), float(rf.max()), flush=True)
sys.exit(1 if bad or err_before > 1e-6 or err_after > 1e-5 else 0)
