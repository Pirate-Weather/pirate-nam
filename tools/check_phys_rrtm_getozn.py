"""Done-rule check for phys-rrtm-getozn (review fix, binding and refusals).

    source env.sh
    python tools/check_phys_rrtm_getozn.py

Full size 954 x 835 x 60 (LMK 61). Runs under a pool lease. Checks, each by the input that showed the defect:
  1. self-test run(None): the one-launch full-size path, no NaN;
  2. bound climatology: tables from the gas_init and gas_update names, no NaN;
  3. refusal: no climatology bound (the old stand-in path);
  4. refusal: XLAT all zero (the old silent latitude 0);
  5. refusal: RAD_PRSLK1 not produced (the old synthetic exner profile);
  6. refusal: gas_update has not set the time indices.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_getozn as unit  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_gas_update as gupd  # noqa: E402
from namgpu.steps.drv_gpu_lease import gpu_lease  # noqa: E402

IM, JM, LM = 954, 835, 60
LMK = LM + 1


class FakeState:
    def __init__(self):
        self.fields = {}

    def add(self, name, shape):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=cp.float32)
        return self.fields[name]


def make_ctx(*, xlat_zero=False, tables=None):
    st = FakeState()
    p = np.linspace(200.0, 1.0e5, LMK).astype(np.float32)
    ex = (p / np.float32(1.0e5)) ** np.float32(0.2857142)
    st.fields["RAD_PRSLK1"] = cp.asarray(np.broadcast_to(ex[:, None, None], (LMK, JM, IM)).copy())
    if xlat_zero:
        xl = np.zeros((JM, IM), np.float32)
    else:
        xl = np.broadcast_to(np.linspace(-1.2, 1.2, JM, dtype=np.float32)[:, None], (JM, IM)).copy()
    st.fields["XLAT"] = cp.asarray(xl)
    grid = SimpleNamespace(im=IM, jm=JM, lm=LM)
    return SimpleNamespace(grid=grid, st=st, tables=tables if tables is not None else {}, stream=None)


def bound_tables():
    """gas_init and gas_update names. Climatology values are the explicit self-test set (the ozone file
    global_o3clim.txt is not on this box), bound under the names gas_init writes."""
    oz = unit._selftest_climatology(cp, unit.JMR_NAM, unit.LOZ_NAM)
    gs = gupd.GasState()
    gupd.ozone_factor(gs, 5, 10, 0, True)
    return {"gas_o3r": oz["o3r"], "gas_pkstr": oz["pkstr"], "rrtm_gas_update": gs}


def expect_refusal(name, fn, needle):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"{name}: refused with the wrong message: {e}"
        print(f"refused ok [{name}]: {str(e)[:110]}")
        return
    raise AssertionError(f"{name}: did not refuse")


def nan_free(name, arr):
    a = arr.get()
    assert np.isfinite(a).all(), f"{name}: non-finite values"
    print(f"{name}: shape {a.shape}, min {a.min():.4e}, max {a.max():.4e}, finite")


def main():
    out = unit.run(None)
    nan_free("selftest RAD_OLYR", out["RAD_OLYR"])
    print("selftest source:", out["climatology_source"], out["kind_phys_build_flag"])

    ctx = make_ctx(tables=bound_tables())
    out = unit.run(ctx)
    nan_free("bound RAD_OLYR", out["RAD_OLYR"])
    print("bound source:", out["climatology_source"], out["kind_phys_build_flag"])

    expect_refusal("no climatology bound", lambda: unit.run(make_ctx(tables={})),
                   "ozone climatology not bound")
    expect_refusal("XLAT all zero", lambda: unit.run(make_ctx(xlat_zero=True, tables=bound_tables())),
                   "XLAT is all zero")
    ctx = make_ctx(tables=bound_tables())
    ctx.st.fields["RAD_PRSLK1"][...] = 0.0
    expect_refusal("RAD_PRSLK1 not produced", lambda: unit.run(ctx), "RAD_PRSLK1 is not positive")
    t = bound_tables()
    t["rrtm_gas_update"] = gupd.GasState()
    expect_refusal("gas_update not run", lambda: unit.run(make_ctx(tables=t)), "time indices")


if __name__ == "__main__":
    with gpu_lease(vram_mib=4000, est_s=900, what="phys-rrtm-getozn review check") as card:
        print("card:", card)
        main()
    print("PASS phys-rrtm-getozn review check")
