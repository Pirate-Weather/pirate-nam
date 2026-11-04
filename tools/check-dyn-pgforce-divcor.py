"""Full-size check of dyn-pgforce-divcor on the 12 km grid (954 x 835 x 60), both branches.

Runs the real wrapper and NVRTC kernel (tile table from drv-loop-limits, wpdar from
ctx.tables["grid"]) on random PGF scratch, and compares every cell with a vectorised numpy
transcription of the Fortran lines 290-451 (module_DYNAMICS_ROUTINES.F90), for:
  - global = .true.  (rows jw, jcl, jch, interior)
  - global = .false. (regional s/n/w/e rows, four corners, interior)
Also checks:
  - no NaN/Inf in DIV;
  - a half-width tile (its..ite = 1..IM/2 in the loop-limit table) writes only its own cells
    and matches the full-domain result inside the tile.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu.driver import Ctx  # noqa: E402
from namgpu.dynamics import dyn_pgforce_divcor as w  # noqa: E402
from namgpu.state import State  # noqa: E402
from namgpu.steps import drv_loop_limits as dll  # noqa: E402

IM, JM, LM = 954, 835, 60
rng = np.random.default_rng(1)
grid = types.SimpleNamespace(im=IM, jm=JM, lm=LM)
wpdar_host = rng.uniform(-2.0e-3, -1.0e-3, JM).astype(np.float32)

st = State(grid, names=["DIV"])
shape3 = (LM, JM, IM)
for name in ("PGX_3D", "PGY_3D", "PGNE_3D", "PGNW_3D"):
    st.add(name, shape3)
    st[name][...] = cp.asarray(rng.standard_normal(shape3, dtype=np.float32) * 1e-3)
div0 = cp.asarray(rng.standard_normal(shape3, dtype=np.float32))
W = cp.asarray(wpdar_host)


def ref_divcor(global_flag, div_in):
    """Vectorised Fortran transcription. Arrays are [LM, JM, IM]; slices are 0-based."""
    px, py = st["PGX_3D"], st["PGY_3D"]
    pne, pnw = st["PGNE_3D"], st["PGNW_3D"]
    out = div_in.copy()
    if global_flag:
        out[:, 0:1, :] = 0.0                                   # row jds (jw=1)
        out[:, JM - 1:JM, :] = 0.0                             # row jde
        # row jcl = 2 (Fortran), i = 2..IM-1
        out[:, 1, 1:IM - 1] = (px[:, 1, 2:IM] - px[:, 1, 1:IM - 1]
                               + py[:, 2, 1:IM - 1]
                               - (pne[:, 2, 2:IM] + pnw[:, 2, 1:IM - 1]) * 0.5) * W[1]
        # row jch = JM-1 (Fortran), i = 2..IM-1
        out[:, JM - 2, 1:IM - 1] = (px[:, JM - 2, 2:IM] - px[:, JM - 2, 1:IM - 1]
                                    - py[:, JM - 2, 1:IM - 1]
                                    - (-pne[:, JM - 2, 1:IM - 1] - pnw[:, JM - 2, 2:IM]) * 0.5) * W[JM - 2]
        # interior Fortran j = 3..JM-2, i = 2..IM-1
        out[:, 2:JM - 2, 1:IM - 1] = (px[:, 2:JM - 2, 2:IM] - px[:, 2:JM - 2, 1:IM - 1]
                                      + py[:, 3:JM - 1, 1:IM - 1] - py[:, 2:JM - 2, 1:IM - 1]
                                      - (pne[:, 3:JM - 1, 2:IM] - pne[:, 2:JM - 2, 1:IM - 1]
                                         + pnw[:, 3:JM - 1, 1:IM - 1] - pnw[:, 2:JM - 2, 2:IM]) * 0.5
                                      ) * W[2:JM - 2][None, :, None]
    else:
        # s_bdy: j = 2, i = 3..IM-2
        out[:, 1, 2:IM - 2] = (px[:, 1, 3:IM - 1] - px[:, 1, 2:IM - 2]
                               + py[:, 2, 2:IM - 2]
                               - (pne[:, 2, 3:IM - 1] + pnw[:, 2, 2:IM - 2]) * 0.5) * W[1]
        # n_bdy: j = JM-1, i = 3..IM-2
        out[:, JM - 2, 2:IM - 2] = (px[:, JM - 2, 3:IM - 1] - px[:, JM - 2, 2:IM - 2]
                                    - py[:, JM - 2, 2:IM - 2]
                                    - (-pne[:, JM - 2, 2:IM - 2] - pnw[:, JM - 2, 3:IM - 1]) * 0.5) * W[JM - 2]
        # w_bdy: i = 2, j = 3..JM-2
        out[:, 2:JM - 2, 1] = (px[:, 2:JM - 2, 2]
                               + py[:, 3:JM - 1, 1] - py[:, 2:JM - 2, 1]
                               - (pne[:, 3:JM - 1, 2] - pnw[:, 2:JM - 2, 2]) * 0.5) * W[2:JM - 2]
        # e_bdy: i = IM-1, j = 3..JM-2
        out[:, 2:JM - 2, IM - 2] = (-px[:, 2:JM - 2, IM - 2]
                                    + py[:, 3:JM - 1, IM - 2] - py[:, 2:JM - 2, IM - 2]
                                    - (-pne[:, 2:JM - 2, IM - 2] + pnw[:, 3:JM - 1, IM - 2]) * 0.5) * W[2:JM - 2]
        # corners
        out[:, 1, 1] = (px[:, 1, 2] + py[:, 2, 1] - pne[:, 2, 2] * 0.5) * W[1]
        out[:, 1, IM - 2] = (-px[:, 1, IM - 2] + py[:, 2, IM - 2] - pnw[:, 2, IM - 2] * 0.5) * W[1]
        out[:, JM - 2, 1] = (px[:, JM - 2, 2] - py[:, JM - 2, 1] + pnw[:, JM - 2, 2] * 0.5) * W[JM - 2]
        out[:, JM - 2, IM - 2] = (-px[:, JM - 2, IM - 2] - py[:, JM - 2, IM - 2]
                                  + pne[:, JM - 2, IM - 2] * 0.5) * W[JM - 2]
        # interior i = 3..IM-2, j = 3..JM-2
        out[:, 2:JM - 2, 2:IM - 2] = (px[:, 2:JM - 2, 3:IM - 1] - px[:, 2:JM - 2, 2:IM - 2]
                                      + py[:, 3:JM - 1, 2:IM - 2] - py[:, 2:JM - 2, 2:IM - 2]
                                      - (pne[:, 3:JM - 1, 3:IM - 1] - pne[:, 2:JM - 2, 2:IM - 2]
                                         + pnw[:, 3:JM - 1, 2:IM - 2] - pnw[:, 2:JM - 2, 3:IM - 1]) * 0.5
                                      ) * W[2:JM - 2][None, :, None]
    return out


def run_case(global_flag, limits_override=None):
    cfg = types.SimpleNamespace(raw={"global": global_flag}, global_=global_flag, dt=60.0)
    ctx = Ctx(cfg=cfg, grid=grid, st=st, stream=None, first_step=False, dt=60.0)
    ctx.tables["grid"] = {"wpdar": W}          # what init-grid-consts writes
    lim = dll.run(ctx)                          # what drv-loop-limits writes
    if limits_override is not None:
        host = lim.get()
        for name, val in limits_override.items():
            host[dll.NAMES.index(name)] = val
        lim[...] = cp.asarray(host)
    st["DIV"][...] = div0
    w.run(ctx)
    cp.cuda.Stream.null.synchronize()
    return st["DIV"].copy(), lim.get()


def compare(tag, got, ref):
    finite = bool(cp.isfinite(got).all())
    scale = float(cp.abs(ref).max())
    diff = cp.abs(got - ref)
    max_diff = float(diff.max())
    n_bad = int((diff > 1e-5 * scale).sum())
    print(f"{tag}: finite={finite} max|ref|={scale:.3e} max|got-ref|={max_diff:.3e} "
          f"cells_over_1e-5_scale={n_bad}")
    assert finite and n_bad == 0, tag


# regional (the NAM 12 km parent): s/n/w/e rows, corners, interior
got_r, lim_r = run_case(False)
compare("regional full-size", got_r, ref_divcor(False, div0))
assert lim_r[dll.NAMES.index("ide")] == IM and lim_r[dll.NAMES.index("jde")] == JM

# global = .true.: the wrapper refuses (drv-loop-limits gives the ghost-line bounds IM+2, JM+2,
# and the state arrays have no ghost lines)
try:
    run_case(True)
except RuntimeError as exc:
    print("global refusal:", str(exc)[:160])
    assert "global = .true. is refused" in str(exc) and f"ide={IM + 2}" in str(exc)
else:
    raise AssertionError("global run was not refused")

# global formulas: the kernel directly, with the bounds the array shape implies (ide=IM, jde=JM)
host_lim = np.array([dll.host_limits(IM, JM, False)[n] for n in dll.NAMES], dtype=np.int32)
lim_dev = cp.asarray(host_lim)
st["DIV"][...] = div0
kern = w.kernels.get(w.UNIT, w.KERNEL)
kern(w.kernels.grid_2d(IM, JM, LM), w.kernels.BLOCK_2D,
     (lim_dev, st["PGX_3D"], st["PGY_3D"], st["PGNE_3D"], st["PGNW_3D"], W, st["DIV"],
      np.int32(IM), np.int32(JM), np.int32(LM), np.int32(1)))
cp.cuda.Stream.null.synchronize()
compare("global formulas full-size", st["DIV"].copy(), ref_divcor(True, div0))

# half-width tile on the regional branch: i = 1..IM/2 only
half = IM // 2
got_t, _ = run_case(False, {"ite": half, "ite_b1": min(half, IM - 1), "ite_b2": min(half, IM - 2)})
ref_r = ref_divcor(False, div0)
outside = bool((got_t[:, :, half:] == div0[:, :, half:]).all())
inside = float(cp.abs(got_t[:, :, :half] - ref_r[:, :, :half]).max())
print(f"half tile: outside_unchanged={outside} inside_max|got-ref|={inside:.3e}")
assert outside and inside <= 1e-5 * float(cp.abs(ref_r).max())
print("dyn-pgforce-divcor: full-size checks ok")
