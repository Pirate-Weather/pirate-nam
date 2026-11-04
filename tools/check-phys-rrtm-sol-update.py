"""Done-rule check for phys-rrtm-sol-update: compile, run the wrapper once per hour branch, NaN/Inf check.

The unit is scalar (no grid arrays), so the full-size State is not allocated; the wrapper reads only
ctx.tables["sol_solc0"] and writes its own scratch.
"""
import math
import sys
from types import SimpleNamespace

import cupy as cp
import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu import kernels  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_sol_update as w  # noqa: E402


class Scratch(dict):
    def add(self, name, shape, dtype="float32"):
        if name not in self:
            self[name] = cp.zeros(shape, dtype=dtype)
        return self[name]


def main():
    print("compile:", type(kernels.module(w.UNIT).compile()).__name__, "ok")
    ctx = SimpleNamespace(tables={"sol_solc0": cp.asarray([1360.8], dtype=cp.float64)},
                          st=Scratch(), stream=None)
    ok = True
    cases = {
        "2023-03-31 12Z": [2023, 3, 31, 0, 12, 0, 0, 0],   # hr >= 12 branch
        "2022-12-22 00Z": [2022, 12, 22, 0, 0, 0, 0, 0],   # hr < 12 branch
    }
    for label, jd in cases.items():
        out, state, nstp = w.run(ctx, jd, jd[0], 1200.0, 25.0, False, 0, 0)
        cp.cuda.Stream.null.synchronize()
        o = out.get()
        s = state.get()
        n = int(nstp.get()[0])
        vals = list(o) + list(s) + [float(n)]
        bad = [v for v in vals if not math.isfinite(v)]
        ok &= not bad
        print(f"{label}: slag={o[0]:.5f} sdec={o[1]:.5f} cdec={o[2]:.5f} solcon={o[3]:.2f} "
              f"anginc={s[3]:.6e} nstp={n} nonfinite={len(bad)}")
    # Consumer binding at full size: dayparts and coszmn read the sol_update state (SOL_STATE) on the device.
    from namgpu.physics.rrtm import phys_rrtm_dayparts as dp
    from namgpu.physics.rrtm import phys_rrtm_coszmn as cz
    im, jm = 954, 835
    rng = np.random.default_rng(0)
    lat = rng.uniform(np.deg2rad(20.0), np.deg2rad(55.0), im * jm).astype(np.float32)
    lon = rng.uniform(np.deg2rad(-125.0), np.deg2rad(-65.0), im * jm).astype(np.float32)
    dpo = dp.dayparts(cp.asarray(lon), cp.asarray(np.sin(lat)), cp.asarray(np.cos(lat)),
                      solhr=13.0, dtswav=48 * 25.0 / 3600.0, nrads=48, sol=state, im=im)
    nd = dpo["ndayparts"].get()
    dp_fin = bool(np.isfinite(dpo["coszen"].get()).all() and np.isfinite(dpo["coszdg"].get()).all())
    dp_ok = dp_fin and int(nd.min()) >= 1
    ok &= dp_ok
    print(f"dayparts full-size {im}x{jm}: runs/row min={int(nd.min())} max={int(nd.max())} finite={dp_fin}")
    glon = cp.asarray(lon.reshape(jm, im))
    glat = cp.asarray(lat.reshape(jm, im))
    coszen = cp.zeros((jm, im), dtype=cp.float32)
    coszdg = cp.zeros((jm, im), dtype=cp.float32)
    cz.coszmn(glon, glat, coszen, coszdg, solhr=13.0, dtswav=48 * 25.0 / 3600.0, nrads=48, sol=state)
    cp.cuda.Stream.null.synchronize()
    cz_fin = bool(np.isfinite(coszen.get()).all() and np.isfinite(coszdg.get()).all())
    ok &= cz_fin
    print(f"coszmn full-size {im}x{jm}: finite={cz_fin} coszen mean={float(coszen.get().mean()):.4f}")
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
