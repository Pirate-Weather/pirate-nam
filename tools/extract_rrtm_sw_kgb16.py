"""Extract the SW band-16 k-tables from NAM radsw_datatb_nmmb.f into an npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f, module_radsw_kgb16_nmmb (lines 1384-2528).
taumol16 (radsw_main_nmmb.f:3384-3490) indexes absa(ind,j), absb(ind,j), selfref(ind,j) and forref(ind,j)
with j = 1..NG16 (NG16 = 6 for band 16, radsw_param_nmmb.f:161) as the second index.

Also taken from the same file: rayl (module_radsw_kgb16_nmmb parameter, 2.91e-10) and strrat(16)
(module_radsw_sflux_nmmb, the data statement at radsw_datatb_nmmb.f:1108-1110; the first entry
is strrat(NBLOW=16)).

Fortran (i,j) arrays are stored C-order reversed: [j, i], per CONTRACT section 1.
DATA statements fill in Fortran column-major order (first index fastest); this script does the same.
Every element of every declared array must be filled exactly once (asserted).

Usage: python3 tools/extract_rrtm_sw_kgb16.py <radsw_datatb_nmmb.f> <out.npz>
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LINES = SRC.read_text(errors="replace").splitlines()

NG16 = 6
DIMS = {
    "absa": (585, NG16), "absb": (235, NG16), "selfref": (10, NG16), "forref": (3, NG16),
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
            idx = tuple(reversed(tup))            # Fortran index order (m, ig)
            assert not mask[name][idx], ("double fill", name, idx)
            mask[name][idx] = True
            store[name][idx] = np.float32(v)
        n_stmt += 1
    return n_stmt


kgb = module_body("module_radsw_kgb16_nmmb")
ktxt = joined(kgb)
store = {n: np.zeros(s, dtype=np.float32) for n, s in DIMS.items()}
mask = {n: np.zeros(s, dtype=bool) for n, s in DIMS.items()}
n1 = fill(ktxt, store, mask)
for n in DIMS:
    assert mask[n].all(), f"{n}: {int((~mask[n]).sum())} elements never filled"
print(f"module lines {kgb[0].strip()} .. {kgb[-1].strip()}; data statements: {n1}")

m = re.search(r"rayl\s*=\s*([0-9.]+[eEdD][+-]?[0-9]+|[0-9.]+)", ktxt)
assert m, "rayl parameter not found"
rayl = np.float32(float(m.group(1).replace("d", "e").replace("D", "e")))

sflux = joined(module_body("module_radsw_sflux_nmmb"))
m = re.search(r"data\s+strrat\s*/([^/]*)/", sflux, re.I)
assert m, "strrat data statement not found"
strrat = [float(v.strip().replace("d", "e").replace("D", "e")) for v in m.group(1).split(",") if v.strip()]
assert len(strrat) == 14, len(strrat)          # strrat(16:29)
strrat16 = np.float32(strrat[0])               # strrat(NBLOW=16)

out = {}
for n in DIMS:
    out[n] = np.ascontiguousarray(store[n].T)   # C-order reversed: [j, m] = [ig, index]
out["rayl"] = np.array(rayl, dtype=np.float32)
out["strrat16"] = np.array(strrat16, dtype=np.float32)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for n, a in out.items():
    print(f"{n:10s} C-shape {np.shape(a)} min {float(np.min(a)):.6g} max {float(np.max(a)):.6g}")
print("wrote", OUT)
