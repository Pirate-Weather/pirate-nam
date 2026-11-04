"""Extract the LW band-16 k-tables from NAM radlw_datatb.f into an npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_kgb16 (lines 31616-32031).
The taugb16 routine in radlw_main_nmmb.f (USE module_radlw_kgb16, line 6195) indexes absa(ig,id),
absb(ig,id), forref(ig,id), selfref(ig,id), fracrefa(ig,id) and fracrefb(ig), which is the
radlw_datatb.f layout with NG16 as the first index. The module_radlw_kgb16_nmmb block in
radlw_datatb_nmmb.f (lines 29207-29499) has absa(MSA16,NG16) and no absb or forref, and is not the
table taugb16 reads. The module lines are matched exactly, so the _nmmb module is never picked.

Fortran (i,j) arrays are stored C-order reversed: [j, i], per CONTRACT section 1.
DATA statements fill in Fortran column-major order (first index fastest); this script does the same.
Every element of every declared array must be filled exactly once (asserted).

Usage: python3 tools/extract_rrtm_lw_kgb16.py <radlw_datatb.f> <out.npz>
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LINES = SRC.read_text(errors="replace").splitlines()

NG16 = 2
# Declared Fortran shapes (radlw_datatb.f:31626-31630).
DIMS = {
    "absa": (NG16, 585), "absb": (NG16, 235), "selfref": (NG16, 10), "forref": (NG16, 4),
    "fracrefa": (NG16, 9), "fracrefb": (NG16,),
}


def module_body(name):
    head = re.compile(rf"^\s{{6}}module\s+{name}\s*(!.*)?$", re.I)
    tail = re.compile(rf"^\s{{6}}end\s+module\s+{name}\s*(!.*)?$", re.I)
    lo = next(i for i, l in enumerate(LINES) if head.match(l))
    hi = next(i for i in range(lo, len(LINES)) if tail.match(LINES[i]))
    return LINES[lo:hi + 1]


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
            idx = tuple(reversed(tup))            # Fortran index order (ig, m)
            assert not mask[name][idx], ("double fill", name, idx)
            mask[name][idx] = True
            store[name][idx] = np.float32(v)
        n_stmt += 1
    return n_stmt


kgb = module_body("module_radlw_kgb16")
store = {n: np.zeros(s, dtype=np.float32) for n, s in DIMS.items()}
mask = {n: np.zeros(s, dtype=bool) for n, s in DIMS.items()}
n1 = fill(joined(kgb), store, mask)
for n in DIMS:
    assert mask[n].all(), f"{n}: {int((~mask[n]).sum())} elements never filled"
print(f"module lines {kgb[0].strip()} .. {kgb[-1].strip()}; data statements: {n1}")

out = {}
for n in DIMS:
    out[n] = np.ascontiguousarray(store[n].T)   # C-order reversed: [m, ig] or [ig] for 1-D
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for n, a in out.items():
    print(f"{n:10s} C-shape {a.shape} min {a.min():.6g} max {a.max():.6g}")
print("wrote", OUT)
