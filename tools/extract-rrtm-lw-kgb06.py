"""Extract the RRTM LW band-6 k-tables used by taugb06 into namgpu/data/rrtm_lw/kgb06.npz.

Source: module_radlw_kgb06 in phys/radlw_datatb.f (nam.v4.2.11, 83df4e13c1), the module that
radlw_main_nmmb.f taugb06 actually USEs. The module_radlw_kgb06_nmmb block of radlw_datatb_nmmb.f is
not used by any routine in the tree and is not extracted here.

Arrays are stored in C order with the Fortran column-major flattening preserved (NG06 fastest):
  absa     (65, 8)  absa(ig, m)    -> absa[m-1, ig-1]
  selfref  (10, 8)  selfref(ig, m) -> selfref[m-1, ig-1]
  forref   (4, 8)   forref(ig, m)  -> forref[m-1, ig-1]
  ka_mco2  (19, 8)  ka_mco2(ig, m) -> ka_mco2[m-1, ig-1]
  fracrefa, cfc11adj, cfc12 (8,)

Usage:  python3 tools/extract-rrtm-lw-kgb06.py <path-to-radlw_datatb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG06 = 8
EXPECT = {"absa": 65 * NG06, "selfref": 10 * NG06, "forref": 4 * NG06,
          "ka_mco2": 19 * NG06, "fracrefa": NG06, "cfc11adj": NG06, "cfc12": NG06}
SHAPE = {"absa": (65, NG06), "selfref": (10, NG06), "forref": (4, NG06),
         "ka_mco2": (19, NG06), "fracrefa": (NG06,), "cfc11adj": (NG06,), "cfc12": (NG06,)}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines) if re.match(r"^\s*module\s+module_radlw_kgb06\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines)) if re.search(r"end\s+module\s+module_radlw_kgb06\b", lines[i], re.I))
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
        sys.exit(f"kgb06 extract: {k} has {len(vals[k])} values, expected {n}")
    out[k] = np.asarray(vals[k], dtype=np.float32).reshape(SHAPE[k])
print("absa[0,0] =", out["absa"][0, 0], " absa[64,7] =", out["absa"][64, 7])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radlw_kgb06"))
print("wrote", OUT, {k: v.shape for k, v in out.items()})
