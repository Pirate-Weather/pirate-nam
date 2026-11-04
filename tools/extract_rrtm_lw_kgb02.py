"""Extract the RRTM LW band-2 k-tables (taugb02 inputs) from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_kgb02 (lines 2385-3223).
This is the module that phys/radlw_main_nmmb.f taugb02 actually binds (`use module_radlw_kgb02`,
radlw_main_nmmb.f:3723), together with module_radlw_parameters (radlw_param.f: NG02=12, NS02=10).
The _nmmb variant (radlw_datatb_nmmb.f:1753-2777, module_radlw_kgb02_nmmb) is not referenced by
taugb02 and is not extracted here.

Output namgpu/data/rrtm_lw/kgb02.npz, C-order, Fortran (ig, ind) stored as [ind, ig]:
  absa     [65, 12]   absa(ig, ind)     ind = 1..65 (lower atmosphere)
  absb     [235, 12]  absb(ig, ind)     ind = 1..235 (upper atmosphere)
  selfref  [10, 12]   selfref(ig, ind)
  forref   [4, 12]    forref(ig, ind)
  fracrefa [12]       fracrefa(ig)
  fracrefb [12]       fracrefb(ig)
Usage: python tools/extract_rrtm_lw_kgb02.py <radlw_datatb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NG02, MSA02, MSB02, MSF02, MFR02 = 12, 65, 235, 10, 4

lines = SRC.read_text(errors="replace").splitlines()
lo = next(i for i, l in enumerate(lines) if l.lower().startswith("      module module_radlw_kgb02"))
hi = next(i for i in range(lo, len(lines)) if lines[i].lower().startswith("      end module module_radlw_kgb02"))
body = []
for l in lines[lo:hi]:
    s = l.lstrip()
    if s.startswith("!"):
        continue
    if s.startswith("&"):
        s = s[1:]
    body.append(s.split("!")[0] if "!" in s else s)
text = " ".join(body)

stmts = re.findall(r"data\s+(\w+)\s*(\([^)]*\))?\s*/([^/]*)/", text, flags=re.I)
vals = {}
for name, _idx, payload in stmts:
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+]?\d+(?:[eEdD][-+]?\d+)", payload)
    nums = [float(n.replace("d", "e").replace("D", "e")) for n in nums]
    key = name.lower()
    vals.setdefault(key, []).extend(nums)

expect = {"absa": NG02 * MSA02, "absb": NG02 * MSB02, "selfref": NG02 * MSF02,
          "forref": NG02 * MFR02, "fracrefa": NG02, "fracrefb": NG02}
for k, n in expect.items():
    if len(vals.get(k, [])) != n:
        raise SystemExit(f"{k}: got {len(vals.get(k, []))} values, expected {n}")

# Fortran DATA fills column-major (ig fastest), so the flat list reshapes to [ind, ig] in C order.
out = {
    "absa": np.array(vals["absa"], np.float32).reshape(MSA02, NG02),
    "absb": np.array(vals["absb"], np.float32).reshape(MSB02, NG02),
    "selfref": np.array(vals["selfref"], np.float32).reshape(MSF02, NG02),
    "forref": np.array(vals["forref"], np.float32).reshape(MFR02, NG02),
    "fracrefa": np.array(vals["fracrefa"], np.float32),
    "fracrefb": np.array(vals["fracrefb"], np.float32),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for k, v in out.items():
    print(k, v.shape, float(v.min()), float(v.max()), bool(np.isfinite(v).all()))
