"""Extract the LW band-11 k-tables from NAM radlw_datatb.f into an npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_kgb11 (lines 28426-29089).
The taugb11 routine in radlw_main_nmmb.f (USE module_radlw_kgb11) indexes absa(ig,id) with
ig = 1..NG11 (8) and id = 1..65 (lower) or 1..235 (upper), so the radlw_datatb.f layout is the one
that matches the Fortran. The module_radlw_kgb11_nmmb table in radlw_datatb_nmmb.f has a different
layout (absa(65,NG11)) and is not what taugb11 reads.

Fortran (i,j) arrays are stored C-order reversed: [j, i], per CONTRACT section 1.
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

NG11 = 8
DIMS = {
    "absa": (NG11, 65), "absb": (NG11, 235), "selfref": (NG11, 10), "forref": (NG11, 4),
    "fracrefa": (NG11,), "fracrefb": (NG11,), "ka_mo2": (NG11, 19), "kb_mo2": (NG11, 19),
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
            a = int(a) if a.strip() else 1
            b = int(b) if b.strip() else n
            out.append(list(range(a - 1, b)))
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
            idx = tuple(reversed(tup))            # Fortran index order (i, j)
            assert not mask[name][idx], ("double fill", name, idx)
            mask[name][idx] = True
            store[name][idx] = np.float32(v)
        n_stmt += 1
    return n_stmt


kgb = module_body("module_radlw_kgb11")
store = {n: np.zeros(s, dtype=np.float32) for n, s in DIMS.items()}
mask = {n: np.zeros(s, dtype=bool) for n, s in DIMS.items()}
n1 = fill(joined(kgb), store, mask)
for n in DIMS:
    assert mask[n].all(), f"{n}: {int((~mask[n]).sum())} elements never filled"
print(f"data statements: kgb11 {n1}")

out = {}
for n in DIMS:
    out[n] = np.ascontiguousarray(store[n].T)   # C-order reversed: [j, i]
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for n, a in out.items():
    print(f"{n:10s} C-shape {a.shape} min {a.min():.6g} max {a.max():.6g}")
print("wrote", OUT)
