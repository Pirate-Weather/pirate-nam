"""Extract the LW band-3 k-tables and chi_mls from NAM radlw_datatb.f into an npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f
  module_radlw_ref   (lines 733-905)   -> chi_mls(7,59)
  module_radlw_kgb03 (lines 3228-10025) -> absa, absb, selfref, forref, fracrefa, fracrefb,
                                           ka_mn2o, kb_mn2o
The taugb03 routine in radlw_main_nmmb.f (USE module_radlw_kgb03) indexes absa(ig,id), so the
radlw_datatb.f layout is the one that matches the Fortran. The module_radlw_kgb03_nmmb table in
radlw_datatb_nmmb.f has a different layout (absa(650,NG03)) and is not what taugb03 reads.

Fortran (i,j,k) arrays are stored C-order reversed: [k, j, i], per CONTRACT section 1.
DATA statements fill in Fortran column-major order (first index fastest); this script does the same.
Every element of every declared array must be filled exactly once (asserted).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LINES = SRC.read_text(errors="replace").splitlines()

# Declared Fortran shapes (radlw_datatb.f:3228-3245, :733-800).
DIMS = {
    "absa": (16, 585), "absb": (16, 1175), "selfref": (16, 10), "forref": (16, 4),
    "fracrefa": (16, 9), "fracrefb": (16, 5), "ka_mn2o": (16, 9, 19), "kb_mn2o": (16, 5, 19),
    "chi_mls": (7, 59),
}


def module_body(name):
    lo = next(i for i, l in enumerate(LINES) if l.lower().startswith(f"      module {name}"))
    hi = next(i for i in range(lo, len(LINES)) if LINES[i].lower().startswith(f"      end module {name}"))
    return LINES[lo:hi]


def joined(body):
    parts = []
    for raw in body:
        s = raw.split("!")[0].strip()
        if not s:
            continue
        s = s.rstrip("&").strip()
        if s.startswith("&"):
            s = s[1:].strip()
        parts.append(s)
    return " ".join(parts)


def ranges(spec, shape):
    dims = [d.strip() for d in spec.split(",")]
    assert len(dims) == len(shape), (spec, shape)
    out = []
    for d, n in zip(dims, shape):
        if d == ":":
            out.append(list(range(n)))
        elif ":" in d:
            a, b = d.split(":")
            out.append(list(range(int(a) - 1, int(b))))
        else:
            out.append([int(d) - 1])
    return out


def fill(text, store, mask):
    pat = re.compile(r"data\s+(\w+)\s*(?:\(([^)]*)\))?\s*/([^/]*)/", re.I)
    n_stmt = 0
    for m in pat.finditer(text):
        name = m.group(1).lower()
        if name not in DIMS:
            continue
        shape = DIMS[name]
        spec = m.group(2) or ",".join([":"] * len(shape))
        vals = [float(v.strip().replace("d", "e").replace("D", "e"))
                for v in m.group(3).split(",") if v.strip()]
        idx_ranges = ranges(spec, shape)
        sel = list(itertools.product(*reversed(idx_ranges)))
        assert len(sel) == len(vals), (name, spec, len(sel), len(vals))
        for tup, v in zip(sel, vals):
            idx = tuple(reversed(tup))            # Fortran index order (i, j, k)
            assert not mask[name][idx], ("double fill", name, idx)
            mask[name][idx] = True
            store[name][idx] = np.float32(v)
        n_stmt += 1
    return n_stmt


kgb = module_body("module_radlw_kgb03")
ref = module_body("module_radlw_ref")
store = {n: np.zeros(s, dtype=np.float32) for n, s in DIMS.items()}
mask = {n: np.zeros(s, dtype=bool) for n, s in DIMS.items()}
n1 = fill(joined(kgb), store, mask)
n2 = fill(joined(ref), store, mask)
for n in ("absa", "absb", "selfref", "forref", "fracrefa", "fracrefb", "ka_mn2o", "kb_mn2o"):
    assert mask[n].all(), f"{n}: {int((~mask[n]).sum())} elements never filled"
assert mask["chi_mls"].all(), "chi_mls not fully filled"
print(f"data statements: kgb03 {n1}, ref {n2}")

out = {}
for n in DIMS:
    out[n] = np.ascontiguousarray(store[n].T)   # C-order reversed: [k, j, i]
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for n, a in out.items():
    print(f"{n:10s} C-shape {a.shape} min {a.min():.6g} max {a.max():.6g}")
print("wrote", OUT)
