"""Extract the NMMB RRTM shortwave band-25 k-tables (taumol25) into namgpu/data/rrtm_sw/kgb25.npz.

Source: module_radsw_kgb25_nmmb in phys/radsw_datatb_nmmb.f (nam.v4.2.11, 83df4e13c1), lines 16972-17119,
the module taumol25 USEs (radsw_main_nmmb.f:4393). Band 25 = 16000-22650 cm-1, NG25 = 6, MSA25 = 65.

Stored in C order with the g-point first, as taumol16 stores its tables:
  absa    (NG25, MSA25)  absa[g, ms-1]  = Fortran absa(ms, g)   (65 x 6 in Fortran order)
  rayl    (NG25,)        Fortran rayl(g)      (data statement, lines 17002-17004)
  abso3a  (NG25,)        Fortran abso3a(g)    (data statement, lines 17006-17007)
  abso3b  (NG25,)        Fortran abso3b(g)    (data statement, lines 17009-17010)

Usage:  python3 tools/extract_rrtm_sw_kgb25.py <path-to-radsw_datatb_nmmb.f> <out.npz>
"""
import itertools
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG25, MSA25 = 6, 65
SPEC = {"absa": (MSA25, NG25), "rayl": (NG25,), "abso3a": (NG25,), "abso3b": (NG25,)}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines)
              if re.match(r"^\s*module\s+module_radsw_kgb25_nmmb\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines))
           if re.search(r"end\s+module\s+module_radsw_kgb25_nmmb\b", lines[i], re.I))
body = "\n".join(s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1])

stmt = re.compile(r"data\s+(\w+)\s*\(([^)]*)\)\s*/([^/]*)/", re.I)
vals = {k: [] for k in SPEC}
spans = {k: [] for k in SPEC}
for m in stmt.finditer(body):
    name = m.group(1).lower()
    if name not in SPEC:
        continue
    idx = []
    for part in m.group(2).split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":")
            idx.append((int(a), int(b)))
        else:
            idx.append((int(part), int(part)))
    spans[name].append(idx)
    for t in m.group(3).split(","):
        t = t.strip()
        if t:
            vals[name].append(float(t.replace("d", "e").replace("D", "e")))

out = {}
for k, shape in SPEC.items():
    arr = np.zeros(shape, dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)
    v = vals[k]
    p = 0
    for idx in spans[k]:
        dims = [range(a, b + 1) for a, b in idx]
        # Fortran order: first subscript varies fastest.
        for combo in itertools.product(*reversed(dims)):
            pos = tuple(c - 1 for c in reversed(combo))
            arr[pos] = v[p]
            filled[pos] = True
            p += 1
    if p != len(v) or not filled.all():
        sys.exit(f"kgb25 extract: {k} filled {p} of {len(v)} values, full={bool(filled.all())}")
    out[k] = np.ascontiguousarray(arr.T if k == "absa" else arr, dtype=np.float32)
    print(k, out[k].shape, "first", float(out[k].ravel()[0]), "last", float(out[k].ravel()[-1]))
np.savez(OUT, **out, source=np.array(f"{SRC}:{start + 1}-{end + 1} module_radsw_kgb25_nmmb"))
print("wrote", OUT)
