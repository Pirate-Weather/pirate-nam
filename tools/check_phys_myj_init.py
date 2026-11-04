"""One-shot check for unit phys-myj-init (MYJPBL_INIT): compile, then the wrapper on a full-size state.

Covers the review gaps: the three table builders run inside the init unit (cold and restart, as in
Fortran), the restart flag binds from ctx.restart when not passed, a context with no restart flag
raises, EXCH_H is [LM+1, JM, IM] with Fortran levels 1..LM zeroed (interior only), and nothing is NaN/Inf.
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import cupy as cp  # noqa: E402

from namgpu import kernels  # noqa: E402
from namgpu.physics.myj import phys_myj_init as U  # noqa: E402
from namgpu.physics.myj import phys_myj_tables as T  # noqa: E402

IM, JM, LM = 954, 835, 60

kernels.module(U.UNIT).get_function("phys_myj_init_exch_h")
kernels.module(T.UNIT).get_function("phys_myj_herftbl")
kernels.module(T.UNIT).get_function("phys_myj_tablept")
kernels.module(T.UNIT).get_function("phys_myj_tablett")
print("compile ok: phys_myj_init_exch_h + phys_myj_tables kernels")

rng = np.random.default_rng(0)
grid = SimpleNamespace(im=IM, jm=JM, lm=LM)


def make_ctx(restart=None, has_restart=True):
    exch = cp.asarray(rng.random((LM + 1, JM, IM), dtype=np.float32) + 1.0)
    attrs = dict(st={"EXCH_H": exch}, grid=grid, stream=None, tables=None)
    if has_restart:
        attrs["restart"] = restart
    return SimpleNamespace(**attrs)


def finite(a):
    return bool(np.all(np.isfinite(cp.asnumpy(a))))


# Cold start, restart bound from ctx.restart (no argument passed).
ctx = make_ctx(restart=False)
orig = cp.asnumpy(ctx.st["EXCH_H"]).copy()
t0 = time.time()
exch = U.run(ctx)
cp.cuda.Stream.null.synchronize()
print(f"cold run wall {time.time() - t0:.3f} s")
out = cp.asnumpy(exch)
assert out.shape == (LM + 1, JM, IM), out.shape
assert not np.any(out[0:LM, 0:JM - 1, 0:IM - 1]), "interior levels 1..LM not zeroed"
assert np.array_equal(out[LM], orig[LM]), "level LM (outside Fortran 1:LM) was written"
assert np.array_equal(out[:, JM - 1, :], orig[:, JM - 1, :]), "row JM-1 (JTF = JM-1) was written"
assert np.array_equal(out[:, :, IM - 1], orig[:, :, IM - 1]), "column IM-1 (ITF = IM-1) was written"
tb = ctx.tables
for k in ("myj_herff", "myj_ptbl", "myj_ttbl", "myj_qs0", "myj_sqs", "myj_the0", "myj_sthe",
          "myj_rdxe", "myj_rdth", "myj_rdq", "myj_rdp", "myj_rdthe"):
    assert k in tb, k
    assert finite(tb[k]), k
assert abs(float(cp.asnumpy(tb["myj_herff"])[0]) - 0.5) < 1e-6  # HERFF(1)=0.5 after the final 0.5-HERFF loop
assert float(cp.asnumpy(tb["myj_ptbl"]).max()) > 0.0 and float(cp.asnumpy(tb["myj_ttbl"]).max()) > 0.0
print("cold: EXCH_H zeroed on 1..LM interior, boundary and level LM untouched; MYJ tables built and finite")

# Restart, restart bound from ctx.restart: EXCH_H untouched, tables still built (Fortran unconditional).
ctx2 = make_ctx(restart=True)
keep = cp.asnumpy(ctx2.st["EXCH_H"]).copy()
U.run(ctx2)
cp.cuda.Stream.null.synchronize()
assert np.array_equal(cp.asnumpy(ctx2.st["EXCH_H"]), keep), "restart zeroed EXCH_H"
assert finite(ctx2.tables["myj_ptbl"]) and finite(ctx2.tables["myj_ttbl"])
print("restart (ctx.restart=True, no argument): EXCH_H untouched, tables built")

# No restart flag and no argument: must refuse, not default to a cold start.
try:
    U.run(make_ctx(has_restart=False))
except ValueError as e:
    print("no restart flag: refused as intended:", str(e)[:80])
else:
    raise AssertionError("missing restart flag did not raise")

print("NaN/Inf check: none" if finite(exch) else "NaN/Inf check: FAILED")
print("OK")
