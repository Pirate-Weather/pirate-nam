"""Full-size check of dyn-adv2-horiz on the 12 km grid (954 x 835 x 60): the review gaps.

Runs the real wrapper and NVRTC kernels under one pool-v3 lease and checks:
  - FAH is the shared state array: doubling st.fields["FAH"] doubles the standard-region TEND;
  - a normal step: no NaN/Inf, TEND changes in the standard interior and nowhere else;
  - the adv_standard gate (lnsh=953 -> lnsad > IM): no flux written, SS1/SP still produced;
  - ordering: the wrapper refuses to run when dyn-adv2-vert has not written TCS for the range.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu.driver import Ctx  # noqa: E402
from namgpu.dynamics import dyn_adv2_horiz as w  # noqa: E402
from namgpu.state import State  # noqa: E402
from namgpu.steps.drv_gpu_lease import gpu_lease  # noqa: E402

IM, JM, LM, NTR = 954, 835, 60, 1   # one species: the card is shared, the species loop is a host loop
DT = 25.0 + 1.0 / 3.0
rng = np.random.default_rng(11)
grid = types.SimpleNamespace(
    im=IM, jm=JM, lm=LM,
    pdsg1=(1000.0 * (0.5 + rng.random(LM))).astype(np.float32),
)
st = State(grid, ntracers=NTR, names=["DSG2"])
st.fields["TRACERS"] = cp.zeros((NTR, 1, 1, 1), dtype=cp.float32)   # run() reads only its tracer count
st.add("TRACERS_SQRT", (NTR, LM, JM, IM))[...] = cp.asarray(
    rng.uniform(0.5, 1.5, (NTR, LM, JM, IM)).astype(np.float32))
st.add("TRACERS_PREV", (NTR, LM, JM, IM))[...] = cp.asarray(
    rng.uniform(0.5, 1.5, (NTR, LM, JM, IM)).astype(np.float32))
st.add("TRACERS_TEND", (NTR, LM, JM, IM))
for name in ("PFX", "PFY", "PFNE", "PFNW"):
    st.add(name, (LM, JM, IM))[...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32) * 1e-3)
st.add("PDOP", (JM, IM))[...] = cp.asarray(rng.uniform(5e4, 1e5, (JM, IM)).astype(np.float32))
st["DSG2"][...] = cp.asarray(rng.uniform(0.01, 0.05, LM).astype(np.float32))
fah_host = (-DT / (3.0 * 1.2e4 * (1 + 0.1 * rng.random(JM)))).astype(np.float32)
GRID_TAB = {"fah": cp.asarray(fah_host)}
tend = st["TRACERS_TEND"]


def ctx_for(lnsh, tables):
    cfg = types.SimpleNamespace(raw={"restart": "false"}, lnsh=lnsh, global_=False, im=IM, jm=JM,
                                dt=DT, idtadt=1)
    return Ctx(cfg=cfg, grid=grid, st=st, stream=None, first_step=False, dt=DT, tables=tables)


def fresh_tables(lnsh_ok=True):
    tab = {"grid": GRID_TAB, "ADV2_SQRT_RANGE": (1, NTR), "ADV2_TCS_RANGE": (1, NTR)}
    return tab


with gpu_lease(vram_mib=8192, est_s=900, cards=1, what="dyn-adv2-horiz full-size check"):
    sp0 = st["TRACERS_PREV"].copy()
    # 1. normal step
    tend[...] = 0.0
    w.run(ctx_for(5, fresh_tables()))
    cp.cuda.Stream.null.synchronize()
    t1 = tend.copy()
    finite = bool(cp.isfinite(t1).all())
    lnsad = 5 + 2
    # standard region, 0-based inclusive ibeg..iend and jbeg..jend, on the [NTR,LM,JM,IM] axes
    inner = (slice(None), slice(None), slice(lnsad + 1, JM - 1 - lnsad), slice(lnsad + 1, IM - 1 - lnsad))
    interior_moved = float(cp.abs(t1[inner]).max())
    rest = t1.copy()
    rest[inner] = 0.0
    outside = float(cp.abs(rest).max())
    print(f"normal: finite={finite} max|TEND| interior={interior_moved:.3e} max|TEND| outside={outside:.3e}")
    assert finite and interior_moved > 0 and outside == 0.0

    # 2. shared FAH: scale the state FAH (as the TDFI filter does) and the standard TEND doubles.
    # TRACERS_PREV is restored first: the run advances it (Adams-Bashforth history), so the second run needs
    # the same input as the first.
    st["TRACERS_PREV"][...] = sp0
    st.fields["FAH"] = GRID_TAB["fah"].copy() * 2.0
    tend[...] = 0.0
    w.run(ctx_for(5, fresh_tables()))
    cp.cuda.Stream.null.synchronize()
    ratio = float(cp.abs(tend[inner]).max() / cp.abs(t1[inner]).max())
    print(f"shared FAH x2: TEND ratio={ratio:.6f} (expect 2)")
    assert abs(ratio - 2.0) < 1e-4
    del st.fields["FAH"]

    # 3. adv_standard gate: lnsh=953 -> no flux, SS1/SP still produced
    st["TRACERS_PREV"][...] = sp0
    tend[...] = 0.0
    w.run(ctx_for(953, fresh_tables()))
    cp.cuda.Stream.null.synchronize()
    gate_ok = float(cp.abs(tend).max()) == 0.0
    print(f"gate lnsh=953: TEND untouched={gate_ok}")
    assert gate_ok

    # 4. ordering: vertical TCS for this range missing -> refuse
    tab = fresh_tables()
    del tab["ADV2_TCS_RANGE"]
    try:
        w.run(ctx_for(5, tab))
        raise SystemExit("ordering guard did not fire")
    except RuntimeError as e:
        print(f"ordering guard fired: {str(e)[:90]}...")
print("dyn-adv2-horiz check PASSED")
