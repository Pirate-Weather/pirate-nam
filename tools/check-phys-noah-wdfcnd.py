"""Done-rule check for phys-noah-wdfcnd (CONTRACT.md section 6, parts 1 and 2), plus the review gaps.

1. compiles the kernel file (NVRTC via CuPy);
2. value check: the ctx path (SMC, SH2O from the state, SMCMAX/BEXP/DKSAT/DWSAT from REDPRM fields)
   against a float32 numpy reference of the Fortran WDFCND, on a small grid that has both the
   SICEMAX = 0 branch (no frozen water) and the SICEMAX > 0 branch (frozen water in a column);
3. refusals by name: a missing REDPRM field, and a layer count other than NSOIL, raise (no fill);
4. the full-size 954 x 835 x NSOIL wrapper run once: no NaN/Inf in WDF, WCND.
Run under the pool lease: GPU_MUTEX_POOL=0 ~/gpuwm-work/pool-v3/run.sh --cards 1 --vram-mib 2000 --est 300 ...
"""
import sys, time, types
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.noah import phys_noah_wdfcnd as w

kernels.module("phys_noah_wdfcnd").compile()
print("compile ok")


def ref_wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    f32 = np.float32
    factr1 = f32(0.2) / smcmax
    factr2 = smc / smcmax
    expon = bexp + f32(2.0)
    wdf = dwsat * np.power(factr2, expon)
    vk = f32(1.0) / (f32(1.0) + np.power(f32(500.0) * sicemax, f32(3.0)))
    wdf_frz = vk * wdf + (f32(1.0) - vk) * dwsat * np.power(factr1, expon)
    wdf = np.where(sicemax > f32(0.0), wdf_frz, wdf).astype(f32)
    expon = f32(2.0) * bexp + f32(3.0)
    wcnd = (dksat * np.power(factr2, expon)).astype(f32)
    return wdf, wcnd


class _St:
    def __init__(self, fields):
        self.fields = fields

    def __getattr__(self, name):
        return self.fields[name]


NSOIL = w.NSOIL
JM, IM = 8, 16
rng = np.random.default_rng(3)
smc = rng.uniform(0.10, 0.40, (NSOIL, JM, IM)).astype(np.float32)
sh2o = (smc * rng.uniform(0.5, 1.0, (NSOIL, JM, IM))).astype(np.float32)
sh2o[:, 0, 0] = smc[:, 0, 0]          # column with no frozen water: SICEMAX = 0
smcmax = rng.uniform(0.40, 0.50, (JM, IM)).astype(np.float32)
bexp = rng.uniform(2.0, 12.0, (JM, IM)).astype(np.float32)
dksat = rng.uniform(1e-6, 1e-5, (JM, IM)).astype(np.float32)
dwsat = rng.uniform(1e-6, 1e-5, (JM, IM)).astype(np.float32)
fields = {"SMC": cp.asarray(smc), "SH2O": cp.asarray(sh2o), "SMCMAX": cp.asarray(smcmax),
          "BEXP": cp.asarray(bexp), "DKSAT": cp.asarray(dksat), "DWSAT": cp.asarray(dwsat)}
ctx = types.SimpleNamespace(st=_St(dict(fields)))

failures = []
out = w.run(ctx)
sice = np.maximum(smc - sh2o, 0.0).max(axis=0).astype(np.float32)
print("columns with SICEMAX > 0:", int((sice > 0).sum()), "of", JM * IM)
assert (sice > 0).any() and (sice == 0).any(), "test grid must cover both branches"
ref_d, ref_k = ref_wdfcnd(smc, smcmax[None], bexp[None], dksat[None], dwsat[None], sice[None])
for name, ref in (("WDF", ref_d), ("WCND", ref_k)):
    got = out[name].get()
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    print(name, "max rel err vs Fortran-formula reference:", float(rel.max()))
    if not (rel.max() < 1e-5):
        failures.append(f"{name} differs from the Fortran formula (max rel {float(rel.max()):.3e})")

for missing in ("SMCMAX", "DWSAT"):
    f = dict(fields)
    del f[missing]
    try:
        w.run(types.SimpleNamespace(st=_St(f)))
        failures.append(f"missing {missing} did not raise")
    except KeyError as e:
        print("refused missing", missing, ":", str(e)[:60])

try:
    w.wdfcnd(cp.asarray(smc[:3]), cp.asarray(smcmax), cp.asarray(bexp), cp.asarray(dksat),
             cp.asarray(dwsat), cp.asarray(sice), cp.zeros((3, JM, IM), cp.float32),
             cp.zeros((3, JM, IM), cp.float32))
    failures.append("nsoil=3 did not raise")
except ValueError as e:
    print("refused nsoil=3:", str(e)[:80])

t0 = time.perf_counter()
full = w.run()
cp.cuda.Stream.null.synchronize()
print("full-size wrapper time_s", round(time.perf_counter() - t0, 4))
for name, arr in full.items():
    a = arr.get()
    print(name, tuple(a.shape), "finite", bool(np.isfinite(a).all()),
          "min", float(a.min()), "max", float(a.max()))
    if not np.isfinite(a).all():
        failures.append(f"NaN/Inf in {name}")

print("FAIL" if failures else "PASS", failures)
sys.exit(1 if failures else 0)
