"""Extract the NMMB RRTM SW band-17 k-tables used by taumol17 into namgpu/data/rrtm_sw/kgb17.npz.

Source: a module_radsw_kgb17 or module_radsw_kgb17_nmmb block of the NAM 4.2.11 radsw_datatb*.f file
(nam.v4.2.11, 83df4e13c1). radsw_main_nmmb.f taumol17 (line 3502) USEs module_radsw_kgb17, which the NMMB
build provides from phys/radsw_datatb.f (MODULES_GFSF); the _nmmb block is extracted too so both can be
compared. Stored as C-order arrays with the Fortran index first in the name:
  absa    (585, 12)   absa(ind, g)    -> absa[ind-1, g-1]
  absb    (1175, 12)  absb(ind, g)    -> absb[ind-1, g-1]
  selfref (10, 12)    selfref(ind, g) -> selfref[ind-1, g-1]
  forref  (4, 12)     forref(ind, g)  -> forref[ind-1, g-1]

Usage: python3 -I extract-rrtm-sw-kgb17.py <radsw_datatb*.f> <module name> <out.npz>
"""
import re
import sys

import numpy as np

SRC, MODNAME, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NG17 = 12
MSA17, MSB17, MSF17, MFR17 = 585, 1175, 10, 4
SIZES = {"absa": (MSA17, NG17), "absb": (MSB17, NG17), "selfref": (MSF17, NG17), "forref": (MFR17, NG17)}

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
        vals[name].extend(float(t.replace("d", "e").replace("D", "e")) for t in num.findall(m.group(3)))

out = {}
for k, (n_ind, n_g) in SIZES.items():
    want = n_ind * n_g
    if len(vals[k]) != want:
        sys.exit(f"kgb17 extract ({MODNAME}): {k} has {len(vals[k])} values, expected {want}")
    # Fortran data fill is column-major: flat[ind + n_ind * g]. Reshape to [g, ind], then transpose to [ind, g].
    flat = np.asarray(vals[k], dtype=np.float32)
    out[k] = np.ascontiguousarray(flat.reshape(n_g, n_ind).T)

print("absa[0,0] =", out["absa"][0, 0], " absa[584,11] =", out["absa"][584, 11])
print("absb[0,0] =", out["absb"][0, 0], " absb[1174,11] =", out["absb"][1174, 11])
np.savez(OUT, **out, source=np.array(f"{SRC}:{start + 1}-{end + 1} {MODNAME}"))
print("wrote", OUT, {k: v.shape for k, v in out.items()}, "lines", start + 1, end + 1)
