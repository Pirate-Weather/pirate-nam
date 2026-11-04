"""init-grid-consts review check: full-size 954x835x60 regional run (no NaN/Inf in any float output)
and the global ddv path (boundary rows 1,2,JM-2,JM-1 equal dyv)."""
import copy
import sys
from types import SimpleNamespace

import cupy as cp
import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu import config  # noqa: E402
from namgpu.init import init_grid_consts as M  # noqa: E402

cfg0 = config.load()
assert cfg0.im == 954 and cfg0.jm == 835, (cfg0.im, cfg0.jm)


def run_once(cfg):
    ctx = SimpleNamespace(cfg=cfg, stream=None, tables={})
    out = M.run(ctx)
    return out


# 1. regional, full size
out = run_once(cfg0)
bad = []
for k, v in out.items():
    if isinstance(v, cp.ndarray) and v.dtype == cp.float32:
        n = int(cp.count_nonzero(~cp.isfinite(v)))
        if n:
            bad.append((k, n))
print("regional", cfg0.im, cfg0.jm, "global_=", cfg0.global_, "non-finite arrays:", bad or "none")
assert not bad
print("regional ddv[0:3]", cp.asnumpy(out["ddv"][:3]), "rddv[-2:]", cp.asnumpy(out["rddv"][-2:]))

# 2. global ddv path, full size (the race defect)
gcfg = copy.copy(cfg0)
gcfg.global_ = True
gout = run_once(gcfg)
dyv = float(cp.asnumpy(gout["scalars"])[M.SC_DYV])
ddv = cp.asnumpy(gout["ddv"]).astype(np.float64)
rddv = cp.asnumpy(gout["rddv"]).astype(np.float64)
JM = cfg0.jm
bnd = {1: ddv[0], 2: ddv[1], JM - 2: ddv[JM - 3], JM - 1: ddv[JM - 2]}
print("global dyv", dyv)
for j, val in bnd.items():
    print("ddv row", j, val, "rddv", rddv[j - 1])
assert all(abs(v - dyv) < 1e-3 for v in bnd.values()), bnd
print("global ddv(JM) row (unwritten, documented zero)", ddv[JM - 1], rddv[JM - 1])
print("CHECK OK")
