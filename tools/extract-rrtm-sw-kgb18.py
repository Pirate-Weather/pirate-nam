"""Extract the NMMB RRTM SW band-18 k-tables used by taumol18 into namgpu/data/rrtm_sw/kgb18.npz.

Source: a module_radsw_kgb18 or module_radsw_kgb18_nmmb block of the NAM 4.2.11 radsw_datatb*.f file
(nam.v4.2.11, 83df4e13c1). Stored as C-order arrays with the Fortran index first in the name:
  absa    (585, 8)  absa(ind, g)    -> absa[ind-1, g-1]
  absb    (235, 8)  absb(ind, g)    -> absb[ind-1, g-1]
  selfref (10, 8)   selfref(ind, g) -> selfref[ind-1, g-1]
  forref  (3, 8)    forref(ind, g)  -> forref[ind-1, g-1]
  rayl    scalar    parameter rayl (Rayleigh extinction at 4325 cm-1)

Usage: python3 -I extract-rrtm-sw-kgb18.py <radsw_datatb*.f> <module name> <out.npz>
"""
import re
import sys

import numpy as np

SRC, MODNAME, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NG18 = 8
MSA18, MSB18, MSF18, MFR18 = 585, 235, 10, 3
SIZES = {"absa": (MSA18, NG18), "absb": (MSB18, NG18), "selfref": (MSF18, NG18), "forref": (MFR18, NG18)}

lines = open(SRC, encoding="latin-1").read().splitlines()
start = next(i for i, s in enumerate(lines)
             if re.match(r"^\s*module\s+" + re.escape(MODNAME) + r"\s*!?\s*$", s, re.I))
end = next(i for i in range(start, len(lines)) if re.search(r"end\s+module\s+" + re.escape(MODNAME) + r"\b", lines[i], re.I))
body = "\n".join(s.split("!", 1)[0].replace("&", " ") for s in lines[start:end + 1])

num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
pat = re.compile(r"data\s+(\w+)\s*(\([^)]*\))?\s*/([^/]*)/", re.I)
vals = {k: [] for k in SIZES}
for m in pat.finditer(body):
    name = m.group(1).lower()
    if name in vals:
        assert "*" not in m.group(3), f"repeat count in data statement for {name}"
        vals[name].extend(float(t.replace("d", "e").replace("D", "e")) for t in num.findall(m.group(3)))

out = {}
for k, (n_ind, n_g) in SIZES.items():
    want = n_ind * n_g
    if len(vals[k]) != want:
        sys.exit(f"kgb18 extract ({MODNAME}): {k} has {len(vals[k])} values, expected {want}")
    # Fortran data fill is column-major: flat[ind + n_ind * g]. Reshape to [g, ind], then transpose to [ind, g].
    flat = np.asarray(vals[k], dtype=np.float32)
    out[k] = np.ascontiguousarray(flat.reshape(n_g, n_ind).T)

rm = re.search(r"rayl\s*=\s*([-+.\deEdD]+)", body, re.I)
assert rm, "rayl parameter not found"
out["rayl"] = np.float32(float(rm.group(1).replace("d", "e").replace("D", "e")))

print("absa[0,0] =", out["absa"][0, 0], " absa[584,7] =", out["absa"][584, 7])
print("absb[0,0] =", out["absb"][0, 0], " absb[234,7] =", out["absb"][234, 7])
print("rayl =", out["rayl"])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start + 1}-{end + 1} {MODNAME}"))
print("wrote", OUT, {k: v.shape for k, v in out.items() if hasattr(v, "shape")}, "lines", start + 1, end + 1)
