"""Extract the RRTM LW band-15 k-tables used by taugb15 into namgpu/data/rrtm_lw/kgb15.npz.

Source: module_radlw_kgb15 in phys/radlw_datatb.f (nam.v4.2.11, 83df4e13c1), lines 31225-31611. This is
the module radlw_main_nmmb.f taugb15 actually USEs (line 5988). The module_radlw_kgb15_nmmb block of
radlw_datatb_nmmb.f (28909-29201) has no ka_mn2 and is not the routine's table source, so it is not used.

Arrays are stored with the Fortran column-major flattening preserved (NG15 = 2 fastest):
  absa     (585, 2)     absa(ig, m)        -> absa[m-1, ig-1]
  selfref  (10, 2)      selfref(ig, m)     -> selfref[m-1, ig-1]
  forref   (4, 2)       forref(ig, m)      -> forref[m-1, ig-1]
  fracrefa (9, 2)       fracrefa(ig, j)    -> fracrefa[j-1, ig-1]
  ka_mn2   (19, 9, 2)   ka_mn2(ig, j, m)   -> ka_mn2[m-1, j-1, ig-1]

Usage:  python3 tools/extract-rrtm-lw-kgb15.py <path-to-radlw_datatb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG15 = 2
MSA15, MSF15, MFR15, MAF15, MMN15 = 585, 10, 4, 9, 19
EXPECT = {"absa": MSA15 * NG15, "selfref": MSF15 * NG15, "forref": MFR15 * NG15,
          "fracrefa": MAF15 * NG15, "ka_mn2": MMN15 * MAF15 * NG15}
SHAPE = {"absa": (MSA15, NG15), "selfref": (MSF15, NG15), "forref": (MFR15, NG15),
         "fracrefa": (MAF15, NG15), "ka_mn2": (MMN15, MAF15, NG15)}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines) if re.match(r"^\s*module\s+module_radlw_kgb15\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines)) if re.search(r"end\s+module\s+module_radlw_kgb15\b", lines[i], re.I))
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
        sys.exit(f"kgb15 extract: {k} has {len(vals[k])} values, expected {n}")
    out[k] = np.asarray(vals[k], dtype=np.float32).reshape(SHAPE[k])
print("absa[0,0] =", out["absa"][0, 0], " absa[584,1] =", out["absa"][584, 1])
print("ka_mn2[0,0,0] =", out["ka_mn2"][0, 0, 0], " ka_mn2[18,8,1] =", out["ka_mn2"][18, 8, 1])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radlw_kgb15"))
print("wrote", OUT, {k: v.shape for k, v in out.items()})
