"""Done-rule check for phys-noah-hstep (review fix).

1. NVRTC compile and one full-size 954 x 835 launch: outputs finite.
2. Solve check, independent of the kernel: the GPU CI (solution) must satisfy the tridiagonal system
   A x = RHS built from the scaled inputs in float64 (sub-diagonal AI*DT, diagonal 1+BI*DT,
   super-diagonal CI*DT with the last layer zeroed as ROSR12 does), and STCF = STCIN + CI.
3. Guards: dt omitted raises; ctx call without bound coefficients raises; NSOIL=3 input raises.

The batch is diagonally dominant (|AI*DT| + |CI*DT| < 1 + BI*DT), so the solve is well conditioned and
the residual test is meaningful. Not covered: the coefficients are random, not HRT/HRTICE output
(named gap in the receipt).
"""
import time
from types import SimpleNamespace

import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_hstep as w

NSOIL = 4
JM, IM = 835, 954
DT = 50.0  # test value only; the real caller passes the DT it receives from NOAHLSM

kernels.get(w.UNIT, w.KERNEL)
rng = np.random.default_rng(1)


def rnd(lo, hi):
    return rng.uniform(lo, hi, size=(NSOIL, JM, IM)).astype(np.float32)


stcin_h = rnd(250.0, 290.0)
rhs_h = rnd(-0.5, 0.5)
ai_h = rnd(-0.2, 0.2) / np.float32(DT)     # AI*DT in [-0.2, 0.2]
ci_h = rnd(-0.2, 0.2) / np.float32(DT)     # CI*DT in [-0.2, 0.2]
bi_h = rnd(0.0, 0.05)                      # 1 + BI*DT >= 1 > |AI*DT| + |CI*DT|

t0 = time.time()
out = w.run(stcin=cp.asarray(stcin_h), rhsts=cp.asarray(rhs_h), ai=cp.asarray(ai_h),
            bi=cp.asarray(bi_h), ci=cp.asarray(ci_h), dt=DT)
cp.cuda.Stream.null.synchronize()
print(f"run ok {time.time() - t0:.2f}s", flush=True)

ok = True
for name in ("STCF", "RHSTS", "AI", "BI", "CI"):
    fin = bool(np.isfinite(cp.asnumpy(out[name])).all())
    ok &= fin
    print(f"{name:6s} finite={fin}", flush=True)

# Solve residual in float64 from the scaled inputs.
a = ai_h.astype(np.float64) * DT
b = 1.0 + bi_h.astype(np.float64) * DT
c = ci_h.astype(np.float64) * DT
c[NSOIL - 1] = 0.0
d = rhs_h.astype(np.float64) * DT
x = cp.asnumpy(out["CI"]).astype(np.float64)
res = b * x - d
res[1:] += a[1:] * x[:-1]
res[:-1] += c[:-1] * x[1:]
rel = float(np.max(np.abs(res)) / np.max(np.abs(d)))
solve_ok = rel < 1e-5
ok &= solve_ok
print(f"solve residual max/|rhs|={rel:.3g} match={solve_ok}", flush=True)

stcf_err = float(np.max(np.abs(cp.asnumpy(out["STCF"]).astype(np.float64) - (stcin_h + x))))
stcf_ok = stcf_err < 1e-3
ok &= stcf_ok
print(f"STCF = STCIN + CI max abs err={stcf_err:.3g} match={stcf_ok}", flush=True)


def expect_raise(label, fn):
    global ok
    try:
        fn()
    except (ValueError, TypeError) as e:
        print(f"guard {label}: raised ({str(e)[:80]})", flush=True)
        return
    ok = False
    print(f"guard {label}: NOT raised", flush=True)


expect_raise("dt omitted", lambda: w.run(stcin=cp.asarray(stcin_h), rhsts=cp.asarray(rhs_h),
                                         ai=cp.asarray(ai_h), bi=cp.asarray(bi_h), ci=cp.asarray(ci_h)))
ctx = SimpleNamespace(st=SimpleNamespace(STC=cp.asarray(stcin_h)))
expect_raise("ctx without coefficients", lambda: w.run(ctx, dt=DT))
expect_raise("NSOIL=3", lambda: w.hstep(cp.asarray(stcin_h[:3]), cp.zeros((3, JM, IM), cp.float32),
                                        cp.asarray(rhs_h[:3]), cp.asarray(ai_h[:3]),
                                        cp.asarray(bi_h[:3]), cp.asarray(ci_h[:3]), DT))
print("PASS" if ok else "FAIL", flush=True)
