"""Extract the NMMB RRTM shortwave band-29 k-tables (taumol29) into namgpu/data/rrtm_sw/kgb29.npz.

Source: module_radsw_kgb29_nmmb in phys/radsw_datatb_nmmb.f (nam.v4.2.11, 83df4e13c1), lines 20065-20973,
the block radsw_main_nmmb.f taumol29 USEs (`use module_radsw_kgb29`, line 4685).

Stored with the Fortran shapes (column-major order preserved, as the kernel indexes them):
  absa     (65, 12)  absa(ms, g)    -> absa[ms-1, g-1]     (MSA29 = 65, NG29 = 12)
  absb     (235, 12) absb(ms, g)    -> absb[ms-1, g-1]     (MSB29 = 235)
  selfref  (10, 12)  selfref(m, g)  -> selfref[m-1, g-1]   (MSF29 = 10)
  forref   (4, 12)   forref(m, g)   -> forref[m-1, g-1]    (MFR29 = 4)
  absh2o   (12,)     absh2o(g)      -> absh2o[g-1]          (water-vapour continuum-band coefficient, lower
                                                              and upper atmosphere; Fortran taug uses absh2o(j))
  absco2   (12,)     absco2(g)      -> absco2[g-1]          (CO2 coefficient, lower atmosphere)
  rayl     ()        9.30e-11, the module parameter (rayleigh extinction coefficient at 2200 cm-1)

Usage:  python3 tools/extract-rrtm-sw-kgb29.py <path-to-radsw_datatb_nmmb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG29, MSA29, MSB29, MSF29, MFR29 = 12, 65, 235, 10, 4
SPEC = {
    "absa": (MSA29, NG29),
    "absb": (MSB29, NG29),
    "selfref": (MSF29, NG29),
    "forref": (MFR29, NG29),
    "absh2o": (NG29,),
    "absco2": (NG29,),
}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines)
              if re.match(r"^\s*module\s+module_radsw_kgb29_nmmb\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines))
           if re.search(r"end\s+module\s+module_radsw_kgb29_nmmb\b", lines[i], re.I))
body_lines = [s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1]]
body = "\n".join(body_lines)

rayl_m = None
for s in body_lines:
    m = re.search(r"rayl\s*=\s*([-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?)", s, re.I)
    if m:
        rayl_m = float(m.group(1).replace("d", "e").replace("D", "e"))
        break
assert rayl_m is not None, "rayl parameter not found in module_radsw_kgb29_nmmb"

# data NAME(r1:r2[, c1:c2]) / v, v, ... /   (Fortran implied-DO order: first index fastest)
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
    toks = []
    for t in m.group(3).split(","):
        t = t.strip()
        if not t:
            continue
        if "*" in t:
            rep, v = t.split("*")
            toks += [v.strip()] * int(rep)
        else:
            toks.append(t)
    vals[name].extend(float(t.replace("d", "e").replace("D", "e")) for t in toks)

out = {}
for k, shape in SPEC.items():
    arr = np.zeros(shape, dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)
    v = vals[k]
    p = 0
    for idx in spans[k]:
        if len(idx) == 1:
            (r1, r2), = idx
            for r in range(r1, r2 + 1):
                arr[r - 1] = v[p]
                filled[r - 1] = True
                p += 1
        else:
            (r1, r2), (c1, c2) = idx
            for c in range(c1, c2 + 1):
                for r in range(r1, r2 + 1):
                    arr[r - 1, c - 1] = v[p]
                    filled[r - 1, c - 1] = True
                    p += 1
    if p != len(v) or not filled.all():
        sys.exit(f"kgb29 extract: {k} filled {p} of {len(v)} values, full={bool(filled.all())}")
    out[k] = arr.astype(np.float32)
    print(k, out[k].shape, "first", out[k].ravel(order="F")[0], "last", out[k].ravel(order="F")[-1])
out["rayl"] = np.float32(rayl_m)
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radsw_kgb29_nmmb"))
print("rayl", rayl_m)
print("wrote", OUT)
