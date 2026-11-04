"""Extract the RRTM LW band-13 k-tables used by taugb13 into namgpu/data/rrtm_lw/kgb13.npz.

Source: module_radlw_kgb13 in phys/radlw_datatb.f (nam.v4.2.11, 83df4e13c1), the module that
radlw_main_nmmb.f taugb13 actually USEs (`use module_radlw_kgb13`, not the _nmmb block).
The module_radlw_kgb13_nmmb block of radlw_datatb_nmmb.f declares only absa, selfref and fracrefa,
and taugb13 needs forref, fracrefb, ka_mco2, ka_mco and kb_mo3, which only the RRTM3 block has.
The same choice was made for band 6 (tools/extract-rrtm-lw-kgb06.py).

Arrays are stored in C order with Fortran column-major order preserved (NG13 = 4 fastest):
  absa     (585, 4)     absa(ig, id)        -> absa[id-1, ig-1]
  selfref  (10, 4)      selfref(ig, m)      -> selfref[m-1, ig-1]
  forref   (4, 4)       forref(ig, m)       -> forref[m-1, ig-1]
  fracrefa (9, 4)       fracrefa(ig, m)     -> fracrefa[m-1, ig-1]
  fracrefb (4,)         fracrefb(ig)
  ka_mco2  (19, 9, 4)   ka_mco2(ig, j, m)   -> ka_mco2[m-1, j-1, ig-1]
  ka_mco   (19, 9, 4)   ka_mco(ig, j, m)    -> ka_mco[m-1, j-1, ig-1]
  kb_mo3   (19, 4)      kb_mo3(ig, m)       -> kb_mo3[m-1, ig-1]

Usage:  python3 tools/extract-rrtm-lw-kgb13.py <path-to-radlw_datatb.f> <out.npz>
"""
import re
import sys

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
NG13, MSA13, MSF13, MFR13, MAF13, MMO13 = 4, 585, 10, 4, 9, 19
# name -> (expected flat count, C shape)
SPEC = {
    "absa": (MSA13 * NG13, (MSA13, NG13)),
    "selfref": (MSF13 * NG13, (MSF13, NG13)),
    "forref": (MFR13 * NG13, (MFR13, NG13)),
    "fracrefa": (MAF13 * NG13, (MAF13, NG13)),
    "fracrefb": (NG13, (NG13,)),
    "ka_mco2": (NG13 * MAF13 * MMO13, (MMO13, MAF13, NG13)),
    "ka_mco": (NG13 * MAF13 * MMO13, (MMO13, MAF13, NG13)),
    "kb_mo3": (NG13 * MMO13, (MMO13, NG13)),
}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines)
              if re.match(r"^\s*module\s+module_radlw_kgb13\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines))
           if re.search(r"end\s+module\s+module_radlw_kgb13\b", lines[i], re.I))
body = "\n".join(s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1])

vals = {k: [] for k in SPEC}
pat = re.compile(r"data\s+(\w+)\s*(\([^)]*\))?\s*/([^/]*)/", re.I)
num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
for m in pat.finditer(body):
    name = m.group(1).lower()
    if name not in vals:
        continue
    vals[name].extend(float(t.replace("d", "e").replace("D", "e")) for t in num.findall(m.group(3)))

out = {}
for k, (n, shape) in SPEC.items():
    if len(vals[k]) != n:
        sys.exit(f"kgb13 extract: {k} has {len(vals[k])} values, expected {n}")
    out[k] = np.asarray(vals[k], dtype=np.float32).reshape(shape)
print("absa[0,0] =", out["absa"][0, 0], " absa[584,3] =", out["absa"][584, 3])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start+1}-{end+1} module_radlw_kgb13"))
print("wrote", OUT, {k: v.shape for k, v in out.items()})
