"""Full-size check of dyn-adv1-horiz-t on the 12 km grid (954 x 835 x 60): the review defects.

Runs the real wrapper and NVRTC kernels and checks:
  - PDOP is produced by the unit: a -999 sentinel is overwritten with (PD+PDO)*0.5 exactly, every step;
  - the adv_standard gate: with lnsh=953 (lnsad=955, IM < 2+lnsad) the standard-region flux block is
    skipped (TCT unchanged) while PDOP is still written;
  - the TDFI FAH state machine: filter start gives -DTRATIO*FAH (backward), direction change flips it,
    filter end restores FAH_SV exactly; the scaled FAH changes the standard-region TCT;
  - normal full-size step: no NaN/Inf, TCT updated in the standard interior.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu.driver import Ctx  # noqa: E402
from namgpu.dynamics import dyn_adv1_horiz_t as w  # noqa: E402
from namgpu.state import State  # noqa: E402

IM, JM, LM = 954, 835, 60
LNSH = 5
DT, FDT = 25.0 + 1.0 / 3.0, 8.0 + 1.0 / 3.0
rng = np.random.default_rng(7)
grid = types.SimpleNamespace(
    im=IM, jm=JM, lm=LM,
    fah=(-DT / (3.0 * 1.2e4 * (1 + 0.1 * rng.random(JM)))).astype(np.float32),
    pdsg1=(1000.0 * (0.5 + rng.random(LM))).astype(np.float32),
)
st = State(grid, names=["T", "TP", "PD", "PDO", "DSG2", "OMGALF", "TCT"])
for name, shape in [("PFX", (LM, JM, IM)), ("PFY", (LM, JM, IM)),
                    ("PFNE", (LM, JM, IM)), ("PFNW", (LM, JM, IM))]:
    st.add(name, shape)
    st[name][...] = cp.asarray(rng.standard_normal(shape, dtype=np.float32) * 1e-2)
st["T"][...] = cp.asarray(rng.uniform(200.0, 310.0, (LM, JM, IM)).astype(np.float32))
st["TP"][...] = cp.asarray(rng.uniform(200.0, 310.0, (LM, JM, IM)).astype(np.float32))
st["PD"][...] = cp.asarray(rng.uniform(5.0e4, 1.0e5, (JM, IM)).astype(np.float32))
st["PDO"][...] = cp.asarray(rng.uniform(5.0e4, 1.0e5, (JM, IM)).astype(np.float32))
st["DSG2"][...] = cp.asarray(rng.uniform(0.01, 0.05, LM).astype(np.float32))
st["OMGALF"][...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32) * 1e-4)
st["TCT"][...] = cp.asarray(rng.standard_normal((LM, JM, IM), dtype=np.float32))
st.add("PDOP", (JM, IM))
tct0 = st["TCT"].copy()
pdop_ref = (st["PD"] + st["PDO"]) * np.float32(0.5)


def ctx_for(lnsh):
    cfg = types.SimpleNamespace(raw={"restart": "false"}, lnsh=lnsh, global_=False, im=IM, jm=JM,
                                dt=DT, filt_dt=FDT)
    return Ctx(cfg=cfg, grid=grid, st=st, stream=None, first_step=False, dt=DT)


def run_case(lnsh):
    ctx = ctx_for(lnsh)
    st["TCT"][...] = tct0
    st["PDOP"][...] = -999.0            # sentinel: the unit itself must write PDOP
    w.run(ctx)
    cp.cuda.Stream.null.synchronize()
    return st["TCT"].copy(), st["PDOP"].copy()   # copies: the next call overwrites TCT in place


# 1. normal full-size step: PDOP produced, no NaN/Inf, TCT updated in the standard interior
tct_n, pdop_n = run_case(LNSH)
pdop_ok = bool(cp.array_equal(pdop_n, pdop_ref))
finite = bool(cp.isfinite(tct_n).all())
inner = (slice(None), slice(1 + LNSH + 2, JM - 2 - LNSH - 2), slice(1 + LNSH + 2, IM - 2 - LNSH - 2))
moved = float(cp.abs(tct_n[inner] - tct0[inner]).max())
print(f"normal step: pdop_exact={pdop_ok} finite={finite} max|dTCT| interior={moved:.3e}")
assert pdop_ok and finite and moved > 0

# 2. adv_standard gate: lnsh=953 -> lnsad=955 > IM, standard block skipped, PDOP still written
tct_g, pdop_g = run_case(953)
gate_ok = bool(cp.array_equal(tct_g, tct0))
pdop_g_ok = bool(cp.array_equal(pdop_g, pdop_ref))
print(f"adv_standard false: TCT unchanged={gate_ok} pdop_exact={pdop_g_ok}")
assert gate_ok and pdop_g_ok

# 3. TDFI FAH state machine on the shared FAH array
ctx = ctx_for(LNSH)
w.run(ctx)                                          # creates the cached FAH
fah_base = st["FAH"].copy()
ratio = np.float32(abs(FDT / DT))
w.fah_filter_start(ctx, -FDT)                       # backward leg, first filter pass
fah_bck_ok = bool(cp.allclose(st["FAH"], -ratio * fah_base, rtol=1e-6))
w.fah_direction_change(ctx)
fah_fwd_ok = bool(cp.allclose(st["FAH"], ratio * fah_base, rtol=1e-6))
print(f"FAH backward=-DTRATIO*base: {fah_bck_ok}; forward=+DTRATIO*base: {fah_fwd_ok} (DTRATIO={float(ratio):.6f})")
assert fah_bck_ok and fah_fwd_ok
tct_f, _ = run_case(LNSH)
dfi_diff = float(cp.abs(tct_f[inner] - tct_n[inner]).max())
finite_f = bool(cp.isfinite(tct_f).all())
print(f"forward DFI step: finite={finite_f} max|TCT_dfi - TCT_normal|={dfi_diff:.3e}")
assert dfi_diff > 0 and finite_f
w.fah_filter_end(ctx)
restored = bool(cp.array_equal(st["FAH"], fah_base)) and "FAH_SV" not in st.fields
print(f"filter end: FAH restored exactly={restored}")
assert restored
print("dyn-adv1-horiz-t: full-size checks ok")
