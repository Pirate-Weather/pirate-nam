"""Extract the RRTM LW band-14 k-tables used by taugb14 into namgpu/data/rrtm_lw/kgb14.npz.

Source: module_radlw_kgb14 in phys/radlw_datatb.f (nam.v4.2.11, 83df4e13c1, lines 31025-31220).
That is the module phys/radlw_main_nmmb.f taugb14 actually USEs (radlw_main_nmmb.f:5919). The
module_radlw_kgb14_nmmb block of radlw_datatb_nmmb.f (lines 28721-28903) is NOT what the routine
reads; its values differ (fracrefa 0.9476 vs 0.9520) and it has no forref array. It is not extracted.

Fortran layouts (column-major, NG14 fastest) are flattened into C order and stored as:
  absa     (65, 2)   absa(ig, m)    -> absa[m-1, ig-1]
  absb     (235, 2)  absb(ig, m)    -> absb[m-1, ig-1]
  selfref  (10, 2)   selfref(ig, m) -> selfref[m-1, ig-1]
  forref   (4, 2)    forref(ig, m)  -> forref[m-1, ig-1]
  fracrefa (2,), fracrefb (2,)

Usage:  python3 tools/extract-rrtm-lw-kgb14.py <path-to-radlw_datatb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG14 = 2
SHAPE = {"absa": (65, NG14), "absb": (235, NG14), "selfref": (10, NG14), "forref": (4, NG14),
         "fracrefa": (NG14,), "fracrefb": (NG14,)}
EXPECT = {k: int(np.prod(v)) for k, v in SHAPE.items()}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines) if re.match(r"^\s*module\s+module_radlw_kgb14\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines)) if re.search(r"end\s+module\s+module_radlw_kgb14\b", lines[i], re.I))
body = "\n".join(s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1])

vals = {k: [] for k in EXPECT}
pat = re.compile(r"data\s+(\w+)\s*(\([^)]*\))?\s*/([^/]*)/", re.I)
num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
for m in pat.finditer(body):
    name = m.group(1).lower()
    if name not in vals:
        continue
    vals[name].extend(float(t.replace("d", "e").replace("D", "e")) for t in num.findall(m.group(3)))

out = {}
for k, n in EXPECT.items():
    if len(vals[k]) != n:
        sys.exit(f"kgb14 extract: {k} has {len(vals[k])} values, expected {n}")
    out[k] = np.asarray(vals[k], dtype=np.float32).reshape(SHAPE[k])
print("absa[0,0] =", out["absa"][0, 0], " absa[64,1] =", out["absa"][64, 1],
      " absb[234,1] =", out["absb"][234, 1])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radlw_kgb14"))
print("wrote", OUT, {k: v.shape for k, v in out.items()})
