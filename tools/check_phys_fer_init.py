"""One-shot check for unit phys-fer-init: compile, run the wrapper on a full-size state (cold start and
restart), assert no NaN/Inf, the cold-start fill, and that a restart neither fills F_* nor writes the
restart-state words."""
import sys
import time
from types import SimpleNamespace

import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp  # noqa: E402

from namgpu import kernels  # noqa: E402
from namgpu.physics.fer_hires import phys_fer_init as U  # noqa: E402

IM, JM, LM = 954, 835, 60

mod = kernels.module(U.UNIT)
for name in ("phys_fer_init_gpvs", "phys_fer_init_sdens",
             "phys_fer_init_scalars", "phys_fer_init_fill"):
    mod.get_function(name)
from namgpu.physics.fer_hires import phys_fer_growth_rates as G  # noqa: E402
kernels.module(G.UNIT).get_function(G.KERNEL)
print("compile ok: 4 init kernels + phys_fer_growth_rates")

rng = np.random.default_rng(0)
shape = (LM, JM, IM)
grid = SimpleNamespace(im=IM, jm=JM, lm=LM)


def make_ctx(restart):
    st = SimpleNamespace(
        F_ICE=cp.asarray(rng.random(shape, dtype=np.float32)),
        F_RAIN=cp.asarray(rng.random(shape, dtype=np.float32)),
        F_RIMEF=cp.asarray(rng.random(shape, dtype=np.float32)),
    )
    return SimpleNamespace(cfg=SimpleNamespace(nprecip=2, dt=25.0), st=st, grid=grid, stream=None,
                           tables={}, restart=restart)


NONFINITE = ("tbpvs", "tbpvs0", "tbpvs_state", "tbpvs0_state", "growth", "sdens", "sc",
             "mp_restart", "massr", "massi", "ventr1", "ventr2", "accrr", "vrain", "rrate",
             "venti1", "venti2", "accri", "vsnowi", "vel_rf")

# Cold start: full run, fill and restart-state words written.
ctx = make_ctx(False)
t0 = time.time()
fer = U.run(ctx, restart=False)
cp.cuda.Stream.null.synchronize()
print(f"cold run wall {time.time() - t0:.3f} s  dtph={fer['dtph']}")
bad = [k for k in NONFINITE if not np.all(np.isfinite(cp.asnumpy(fer[k])))]
for key in ("F_ICE", "F_RAIN", "F_RIMEF"):
    if not np.all(np.isfinite(cp.asnumpy(getattr(ctx.st, key)))):
        bad.append(key)
print("NaN/Inf arrays:", bad if bad else "none")
assert not bad
assert float(cp.asnumpy(ctx.st.F_ICE).max()) == 0.0 and float(cp.asnumpy(ctx.st.F_RAIN).max()) == 0.0
assert float(cp.asnumpy(ctx.st.F_RIMEF).min()) == 1.0 and float(cp.asnumpy(ctx.st.F_RIMEF).max()) == 1.0
assert cp.array_equal(fer["tbpvs_state"], fer["tbpvs"]) and cp.array_equal(fer["tbpvs0_state"], fer["tbpvs0"])
assert float(cp.asnumpy(fer["mp_restart"])[0:35].sum()) == float(cp.asnumpy(fer["growth"]).sum())
print("cold: fill ok, TBPVS_STATE/TBPVS0_STATE = tables, MP_RESTART_STATE written")
print("RFMAX =", fer["scalars"]["RFMAX"], "(1.1**40 = 45.2592...)", " ESW0 =", fer["scalars"]["ESW0"])

# Restart: a new table dict; the state the restart file supplies must survive untouched.
ctx2 = make_ctx(True)
f_ice0 = cp.asnumpy(ctx2.st.F_ICE).copy()
fer2 = U.run(ctx2, restart=True)
cp.cuda.Stream.null.synchronize()
assert np.array_equal(cp.asnumpy(ctx2.st.F_ICE), f_ice0), "restart fill touched F_ICE"
assert not np.any(cp.asnumpy(fer2["mp_restart"])), "restart wrote MP_RESTART_STATE"
assert not np.any(cp.asnumpy(fer2["tbpvs_state"])) and not np.any(cp.asnumpy(fer2["tbpvs0_state"]))
assert fer2["restart"] is True and fer2["dtph"] == 50.0
bad_r = [k for k in ("tbpvs", "tbpvs0", "sc", "sdens") if not np.all(np.isfinite(cp.asnumpy(fer2[k])))]
assert not bad_r, bad_r
print("restart: F_* untouched, MP_RESTART_STATE and TBPVS(0)_STATE left for the restart reader")
print("OK")
