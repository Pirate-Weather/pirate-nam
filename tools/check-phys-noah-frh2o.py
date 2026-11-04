"""Check for phys-noah-frh2o (review gaps on the ctx path).

Full-size 954 x 835 x NSOIL batch on the GPU, no full model state:
1. ctx path binds TAVG (not STC) and the REDPRM arrays SMCMAX, BEXP, PSISAT: its output equals an
   explicit frh2o() call on those arrays, bitwise, and differs from a call on STC where T < T0;
2. no NaN or Inf in FREE with positive PSISAT (REDPRM convention), in the ctx path and the random path;
3. the ctx path refuses by name when TAVG is missing (no zero fill).
Soil-type values are synthetic in the REDPRM range (the NAM fix tables are not on node-2).
"""
import sys
import types

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu.physics.noah import phys_noah_frh2o as w

NSOIL, JM, IM = 4, 835, 954
rng = np.random.default_rng(7)
shape3 = (NSOIL, JM, IM)
shape2 = (JM, IM)


class _St:
    def __init__(self, fields):
        self.fields = fields

    def __getattr__(self, name):
        return self.fields[name]


smc = rng.uniform(0.05, 0.45, shape3).astype(np.float32)
fields = {
    "SMC": cp.asarray(smc),
    "SH2O": cp.asarray((smc * rng.uniform(0.5, 1.0, shape3)).astype(np.float32)),
    "TAVG": cp.asarray(rng.uniform(240.0, 300.0, shape3).astype(np.float32)),
    "STC": cp.asarray(rng.uniform(250.0, 300.0, shape3).astype(np.float32)),
    "SMCMAX": cp.asarray(rng.uniform(0.35, 0.5, shape2).astype(np.float32)),
    "BEXP": cp.asarray(rng.uniform(2.0, 12.0, shape2).astype(np.float32)),
    "PSISAT": cp.asarray(rng.uniform(0.03, 0.7, shape2).astype(np.float32)),
}
ctx = types.SimpleNamespace(st=_St(dict(fields)))

failures = []

out = w.run(ctx)["FREE"]
ref = cp.zeros(shape3, dtype=cp.float32)
w.frh2o(fields["TAVG"], fields["SMC"], fields["SH2O"], fields["SMCMAX"], fields["BEXP"],
        fields["PSISAT"], ref)
ndiff = int(cp.count_nonzero(out != ref).get())
print("ctx vs explicit(TAVG, REDPRM) differing values:", ndiff)
if ndiff:
    failures.append("ctx path does not bind TAVG/REDPRM arrays")

cold = fields["TAVG"] < 273.15
stc_ref = cp.zeros(shape3, dtype=cp.float32)
w.frh2o(fields["STC"], fields["SMC"], fields["SH2O"], fields["SMCMAX"], fields["BEXP"],
        fields["PSISAT"], stc_ref)
n_cold = int(cold.sum().get())
n_stc = int(cp.count_nonzero((out != stc_ref) & cold).get())
print("cold values:", n_cold, " ctx output differs from STC-driven call at:", n_stc)
if n_cold and n_stc == 0:
    failures.append("ctx output is indistinguishable from STC-driven output")

nonfinite = int((~cp.isfinite(out)).sum().get())
print("ctx path non-finite values:", nonfinite,
      " min", float(out.min().get()), " max", float(out.max().get()))
if nonfinite:
    failures.append(f"ctx path produced {nonfinite} non-finite FREE values")

rnd = w.run()["FREE"]
rnd_bad = int((~cp.isfinite(rnd)).sum().get())
print("random full-size path non-finite values:", rnd_bad)
if rnd_bad:
    failures.append(f"random path produced {rnd_bad} non-finite values")

no_tavg = dict(fields)
del no_tavg["TAVG"]
try:
    w.run(types.SimpleNamespace(st=_St(no_tavg)))
    failures.append("ctx path ran without TAVG (should refuse)")
except KeyError as e:
    print("refusal without TAVG:", str(e)[:90])

print("RESULT", "FAIL" if failures else "PASS", failures)
sys.exit(1 if failures else 0)
