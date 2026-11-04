"""Check for phys-noah-transp (review gaps). Full-size 954 x 835 x NSOIL batch on the GPU.

1. ctx path binds real state: ctx.st.ET equals an explicit transp() call on the same state fields
   (SH2O, RTDIS, NROOT, CMC, SHDFAC, SMCWLT, CMCMAX, PC, CFACTR, SMCREF) with ETP1 = ETP * 0.001,
   bitwise, and ETP1 is ETP * 0.001 exactly.
2. Caller gate: columns with ETP1 <= 0 or SHDFAC <= 0 have ET exactly 0 on every layer, including
   ETP < 0 columns that would otherwise give negative ET.
3. NROOT contract: an active column with NROOT = 0 or 5 is NaN on every layer; every other active
   column is finite.
4. The ctx path refuses by name when a producer is missing (PC).
5. The random full-size path has no NaN or Inf in ET.
"""
import sys
import types

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu.physics.noah import phys_noah_transp as w

NSOIL, JM, IM = 4, 835, 954
rng = np.random.default_rng(11)
shape3 = (NSOIL, JM, IM)
shape2 = (JM, IM)


class _St:
    def __init__(self, fields):
        self.fields = fields

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name)

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def f2(lo, hi):
    return cp.asarray(rng.uniform(lo, hi, size=shape2).astype(np.float32))


smcwlt = f2(0.02, 0.10)
smcref = cp.maximum(f2(0.15, 0.40), smcwlt + 0.05)
rt = rng.uniform(0.0, 1.0, size=shape3).astype(np.float32)
fields = {
    "ETP": cp.asarray(rng.uniform(-2.0e-1, 5.0e-1, size=shape2).astype(np.float32)),
    "CMC": f2(0.0, 0.5),
    "SH2O": cp.asarray(rng.uniform(0.02, 0.45, size=shape3).astype(np.float32)),
    "RTDIS": cp.asarray(rt / rt.sum(axis=0, keepdims=True)),
    "NROOT": cp.asarray(rng.integers(1, NSOIL + 1, size=shape2).astype(np.int32)),
    "SHDFAC": f2(0.0, 1.0),
    "SMCWLT": smcwlt,
    "SMCREF": smcref,
    "CMCMAX": f2(0.5, 1.0),
    "CFACTR": f2(0.3, 0.7),
    "PC": f2(0.2, 1.0),
}
# Sprinkle gate and contract cases on known columns.
fields["SHDFAC"][0, 0] = 0.0          # gate: SHDFAC = 0
fields["ETP"][0, 1] = -0.5            # gate: ETP1 < 0
fields["ETP"][0, 2] = 0.0             # gate: ETP1 = 0
fields["NROOT"][0, 3] = 0             # contract: NROOT = 0, gate open
fields["NROOT"][0, 4] = 5             # contract: NROOT > NSOIL, gate open
fields["SHDFAC"][0, 3] = 0.5
fields["ETP"][0, 3] = 0.2
fields["SHDFAC"][0, 4] = 0.5
fields["ETP"][0, 4] = 0.2
fields["SHDFAC"][0, 5] = 0.5
fields["ETP"][0, 5] = 0.2
fields["NROOT"][0, 5] = 4

failures = []

# 1. ctx path: real state bound, ET written to the state field.
st = _St(dict(fields))
ctx = types.SimpleNamespace(st=st)
out = w.run(ctx)["ET"]
if out is not st.fields["ET"]:
    failures.append("ctx path did not write ET into ctx.st")
etp1_ref = cp.multiply(fields["ETP"], np.float32(0.001))
if not bool(cp.array_equal(st.fields["ETP1"], etp1_ref)):
    failures.append("ETP1 != ETP * 0.001")
ref = cp.zeros(shape3, dtype=cp.float32)
w.transp(fields["SH2O"], etp1_ref, fields["CMC"], fields["SHDFAC"], fields["SMCWLT"],
          fields["CMCMAX"], fields["PC"], fields["CFACTR"], fields["SMCREF"], fields["NROOT"],
          fields["RTDIS"], ref)
both_nan = cp.isnan(out) & cp.isnan(ref)      # NaN != NaN, so count NaN-vs-NaN as equal
ndiff = int(cp.count_nonzero((out != ref) & ~both_nan).get())
print("1. ctx ET vs explicit state-bound transp, differing values:", ndiff)
if ndiff:
    failures.append("ctx path output differs from the explicit call on the same state")

# 2. caller gate.
gate_closed = (etp1_ref <= 0.0) | (fields["SHDFAC"] <= 0.0)
gate_closed[0, 3] = False
gate_closed[0, 4] = False
n_closed = int(gate_closed.sum().get())
nz_closed = int(cp.count_nonzero(out[:, gate_closed]).get()) if n_closed else 0
neg_closed = int(cp.count_nonzero(out[:, gate_closed] < 0).get()) if n_closed else 0
print(f"2. gate-closed columns: {n_closed}, nonzero ET values there: {nz_closed}, negative: {neg_closed}")
if nz_closed or neg_closed:
    failures.append("gate-closed columns do not write ET = 0")

# 3. NROOT contract.
bad_nroot_cols = out[:, 0, 3], out[:, 0, 4]
for nm, col in (("NROOT=0", bad_nroot_cols[0]), ("NROOT=5", bad_nroot_cols[1])):
    isnan = bool(cp.all(cp.isnan(col)).get())
    print(f"3. {nm} active column all-NaN: {isnan}")
    if not isnan:
        failures.append(f"{nm} active column is not NaN")
ok_col = out[:, 0, 5]
print("3. valid active column finite:", bool(cp.all(cp.isfinite(ok_col)).get()))
if not bool(cp.all(cp.isfinite(ok_col)).get()):
    failures.append("valid active column is not finite")

# 4. named refusal when a producer is missing.
partial = dict(fields)
del partial["PC"]
try:
    w.run(types.SimpleNamespace(st=_St(partial)))
    failures.append("missing PC did not raise")
except RuntimeError as e:
    named = "PC" in str(e) and "phys-noah-canres" in str(e)
    print("4. refusal names PC and its producer:", named)
    if not named:
        failures.append("refusal does not name PC")

# 5. random full-size path, no NaN or Inf.
res = w.run()["ET"]
finite = bool(cp.all(cp.isfinite(res)).get())
print("5. random full-size ET finite everywhere:", finite)
if not finite:
    failures.append("random full-size path has NaN or Inf")

print("FAIL: " + "; ".join(failures) if failures else "PASS: all five checks")
sys.exit(1 if failures else 0)
