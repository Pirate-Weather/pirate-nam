"""Done-rule check for bnd-bocoh-update (CONTRACT.md section 6): compile, run once on the full-size
12 km shape with random boundary strips, assert no NaN/Inf, and confirm the strip update equals
working + tendency * dt in NumPy for every variable (indexing check on the full strips)."""
import sys
from types import SimpleNamespace as NS

import cupy as cp
import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu import kernels  # noqa: E402
from namgpu.boundary import bnd_bocoh_update as unit  # noqa: E402

IM, JM, LM, LNSH = 954, 835, 60, 5
DT = np.float32(25.0)

print("compile:", kernels.module("bnd_bocoh_update").compile() is None or "ok")


def strip_shape(rank_extra, ni, nj, nl=None):
    # Layout: [NL?, 2, LM?, nj, ni]
    shp = []
    if nl is not None:
        shp.append(nl)
    shp.append(2)
    if rank_extra:
        shp.append(LM)
    shp += [nj, ni]
    return tuple(shp)


def var(rank_extra, nl=None, lbnd=None, ubnd=None):
    d = {}
    for edge in unit.EDGES:
        ns = edge in ("south", "north")
        ni, nj = (IM, LNSH) if ns else (LNSH, JM)
        d[edge] = cp.random.random(strip_shape(rank_extra, ni, nj, nl), dtype=cp.float32)
    v = NS(**d)
    if nl is not None:
        v.lbnd, v.ubnd = lbnd, ubnd
    return v


bc = NS(h2d=[var(False) for _ in range(3)],
        h3d=[var(True) for _ in range(2)],
        h4d=[var(True, nl=3, lbnd=1, ubnd=3), var(True, nl=2, lbnd=5, ubnd=6)])
# keep a host copy of the pre-update working values for the reference
ref = {}
for grp in ("h2d", "h3d", "h4d"):
    for k, v in enumerate(getattr(bc, grp)):
        for edge in unit.EDGES:
            ref[(grp, k, edge)] = cp.asnumpy(getattr(v, edge)).copy()

cfg = NS(lnsh=LNSH)
grid = NS(im=IM, jm=JM, lm=LM)
ctx = NS(cfg=cfg, grid=grid, st=None, stream=None, dt=float(DT), bc=bc)

unit.run(ctx)
cp.cuda.Stream.null.synchronize()

bad = 0
for grp in ("h2d", "h3d", "h4d"):
    for k, v in enumerate(getattr(bc, grp)):
        for edge in unit.EDGES:
            out = cp.asnumpy(getattr(v, edge))
            if not np.isfinite(out).all():
                bad += 1
                print("NaN/Inf", grp, k, edge)
            r = ref[(grp, k, edge)]
            # working index k=0 is at [..., 0, :, :]; tendency at [..., 1, :, :]
            ax = 1 if grp == "h4d" else 0
            exp = np.take(r, 0, axis=ax) + np.take(r, 1, axis=ax) * DT
            got = np.take(out, 0, axis=ax)
            if not np.allclose(got, exp, rtol=1e-6, atol=1e-6):
                bad += 1
                print("MISMATCH", grp, k, edge, float(np.abs(got - exp).max()))
            tend_same = np.array_equal(np.take(out, 1, axis=ax), np.take(r, 1, axis=ax))
            if not tend_same:
                bad += 1
                print("TENDENCY CHANGED", grp, k, edge)
print("RESULT", "PASS" if bad == 0 else f"FAIL {bad}")
