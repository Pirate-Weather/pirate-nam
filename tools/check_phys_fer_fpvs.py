"""Done-rule check for phys-fer-fpvs (CONTRACT section 6, parts 1 and 2). Full-size T, one wrapper run.

    source ~/nam-gpu/env.sh && python tools/check_phys_fer_fpvs.py

Compiles namgpu/kernels/phys_fer_fpvs.cu through NVRTC, runs the wrapper once on a full-size
float32 [60, 835, 954] temperature array and asserts all four outputs are finite. The GPVS tables
(TBPVS, TBPVS0 and the C1/C2 coefficients) are built here with the GPVS_hr formula in numpy as a fixture,
because phys-fer-gpvs is a separate unit; the test only needs finite inputs. The state is a real
namgpu-style st object (add() allocates scratch once), so the wrapper's st.add path is what runs.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

NX, XMIN, XMAX = 7501, 180.0, 330.0
TTP, HVAP, PSAT, CLIQ, CVAP, CICE, HSUB, RV = 2.7316e2, 2.5e6, 6.1078e2, 4.1855e3, 1.846e3, 2.106e3, 2.834e6, 461.6


class _St(dict):
    """Minimal stand-in for namgpu.state.State: fields by name plus add(name, shape) (allocate once)."""

    def add(self, name, shape, dtype="float32"):
        import cupy as cp

        if name not in self:
            self[name] = cp.zeros(shape, dtype=dtype)
        return self[name]


def fpvsx(T):
    psatk = PSAT * 1e-3
    xa = -(CVAP - CLIQ) / RV
    xb = xa + HVAP / (RV * TTP)
    xai = -(CVAP - CICE) / RV
    xbi = xai + HSUB / (RV * TTP)
    tr = TTP / T
    liq = psatk * tr ** xa * np.exp(xb * (1 - tr))
    ice = psatk * tr ** xai * np.exp(xbi * (1 - tr))
    return np.where(T >= TTP, liq, ice)


def fpvsx0(T):
    psatk = PSAT * 1e-3
    xa = -(CVAP - CLIQ) / RV
    xb = xa + HVAP / (RV * TTP)
    tr = TTP / T
    return psatk * tr ** xa * np.exp(xb * (1 - tr))


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.fer_hires import phys_fer_fpvs

    xinc = (XMAX - XMIN) / (NX - 1)
    c1, c2 = 1.0 - XMIN / xinc, 1.0 / xinc
    xs = XMIN + np.arange(NX) * xinc
    tbpvs = cp.asarray(fpvsx(xs).astype(np.float32))
    tbpvs0 = cp.asarray(fpvsx0(xs).astype(np.float32))
    coef = cp.asarray(np.array([c1, c2, c1, c2], dtype=np.float32))

    rng = np.random.default_rng(1)
    T = cp.asarray(rng.uniform(190.0, 320.0, size=(60, 835, 954)).astype(np.float32))

    tables = {"fer_gpvs_tbpvs": tbpvs, "fer_gpvs_tbpvs0": tbpvs0, "fer_gpvs_coef": coef}
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = SimpleNamespace(st=_St({"T": T}), stream=stream, tables=tables)
    kernels.module(phys_fer_fpvs.UNIT).compile()
    t0 = time.perf_counter()
    with stream:
        outs = phys_fer_fpvs.run(ctx)
    stream.synchronize()
    dt = time.perf_counter() - t0
    names = ["fpvs", "fpvs0", "fpvsx", "fpvsx0"]
    ok = True
    for n, a in zip(names, outs):
        finite = bool(cp.isfinite(a).all())
        ok &= finite
        print(f"{n}: shape={tuple(a.shape)} finite={finite} min={float(a.min()):.4e} max={float(a.max()):.4e}")
    # 273.16 K sanity probe on a fresh state (scratch is allocated once per state, so the probe needs its own)
    probe = cp.asarray(np.array([273.16, 300.0], dtype=np.float32))
    ctx2 = SimpleNamespace(st=_St({"T": probe}), stream=stream, tables=tables)
    with stream:
        p = phys_fer_fpvs.run(ctx2)
    stream.synchronize()
    print("at 273.16 K fpvs, fpvsx:", float(p[0][0]), float(p[2][0]), "(expect ~0.611)")
    print(f"wrapper wall {dt:.3f} s, full size {T.size} points")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
