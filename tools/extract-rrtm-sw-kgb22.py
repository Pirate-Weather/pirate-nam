"""Extract the NMMB RRTM shortwave band-22 k-tables (taumol22) into namgpu/data/rrtm_sw/kgb22.npz.

Source: module_radsw_kgb22_nmmb in phys/radsw_datatb_nmmb.f (nam.v4.2.11, 83df4e13c1), lines 14760-15190,
the block radsw_main_nmmb.f taumol22 USEs (`use module_radsw_kgb22`, line 4092).

Stored with the Fortran shapes (column-major order preserved, as the kernel indexes them):
  absa    (585, 2)  absa(ms, g)    -> absa[ms-1, g-1]     (MSA22 = 585, NG22 = 2)
  absb    (235, 2)  absb(ms, g)    -> absb[ms-1, g-1]     (MSB22 = 235)
  selfref (10, 2)   selfref(m, g)  -> selfref[m-1, g-1]   (MSF22 = 10)
  forref  (3, 2)    forref(m, g)   -> forref[m-1, g-1]    (MFR22 = 3)
  rayl    ()        1.54e-08, line 14782 (module parameter)
  strrat22 ()       3.63328e-02, radsw_datatb_nmmb.f line 1108 (strrat(22), o2adj-multiplied; strrat is
                    declared in module_radsw_sflux_nmmb, lines 1070-1378, and host-associated in taumol22)

Usage:  python3 tools/extract-rrtm-sw-kgb22.py <path-to-radsw_datatb_nmmb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG22, MSA22, MSB22, MSF22, MFR22 = 2, 585, 235, 10, 3
SPEC = {
    "absa": (MSA22, NG22),
    "absb": (MSB22, NG22),
    "selfref": (MSF22, NG22),
    "forref": (MFR22, NG22),
}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines)
              if re.match(r"^\s*module\s+module_radsw_kgb22_nmmb\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines))
           if re.search(r"end\s+module\s+module_radsw_kgb22_nmmb\b", lines[i], re.I))
body = "\n".join(s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1])

# data NAME(r1:r2, c1:c2) / v, v, ... /   (Fortran implied-DO order: first index fastest)
stmt = re.compile(r"data\s+(\w+)\s*\(([^)]*)\)\s*/([^/]*)/", re.I)
num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
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
for k, (n1, n2) in SPEC.items():
    arr = np.zeros((n1, n2), dtype=np.float64)
    filled = np.zeros((n1, n2), dtype=bool)
    v = vals[k]
    p = 0
    for idx in spans[k]:
        (r1, r2), (c1, c2) = idx
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                arr[r - 1, c - 1] = v[p]
                filled[r - 1, c - 1] = True
                p += 1
    if p != len(v) or not filled.all():
        sys.exit(f"kgb22 extract: {k} filled {p} of {len(v)} values, full={bool(filled.all())}")
    out[k] = arr.astype(np.float32)
    print(k, out[k].shape, "first", out[k].ravel(order="F")[0], "last", out[k].ravel(order="F")[-1])
out["rayl"] = np.float32(1.54e-08)
out["strrat22"] = np.float32(3.63328e-02)
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radsw_kgb22_nmmb"))
print("wrote", OUT)
