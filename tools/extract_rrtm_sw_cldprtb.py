"""Extract the NMMB SW cloud-property tables (module_radsw_cldprtb) from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radsw_datatb.f, module_radsw_cldprtb (lines 125-1817). This is the
module that phys/radsw_main_nmmb.f (cldprop) uses; the NMMB build links radsw_datatb.f.
Writes namgpu/data/rrtm_sw_cldprtb.npz, band-major (Fortran (index, ib) -> [ib, index]), bands 16..29 (14):
  extliq1, ssaliq1, asyliq1  [14, 58]   hu & stamnes liquid
  extice2, ssaice2, asyice2  [14, 43]   streamer ice
  extice3, ssaice3, asyice3  [14, 46]   fu ice (also used for the iswcice=4 snow)
The small Ebert-Curry / rain / snow constants are written in the kernel (with their line numbers) and are
printed here so they can be checked against the source.

Usage: python tools/extract_rrtm_sw_cldprtb.py <radsw_datatb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
lines = SRC.read_text(errors="replace").splitlines()
lo = next(i for i, l in enumerate(lines) if re.match(r"\s*module module_radsw_cldprtb\s*(!.*)?$", l, re.I))
hi = next(i for i in range(lo, len(lines)) if re.match(r"\s*end module module_radsw_cldprtb", lines[i], re.I))

# Join the module body into one string: drop comment lines and trailing comments, join '&' continuations.
joined = []
for raw in lines[lo:hi]:
    s = raw.strip()
    if not s or s.startswith("!"):
        continue
    if "!" in s:
        s = s.split("!", 1)[0].strip()
    if s.endswith("&"):
        s = s[:-1]
    if s.startswith("&"):
        s = s[1:]
    joined.append(s)
body = " ".join(joined)

pat = re.compile(r"\bdata\s+(\w+)\s*(?:\(\s*:\s*,?\s*(\d*)\s*\)|\(\s*:\s*\))?\s*/([^/]*)/", re.I)
stmts = {}
for m in pat.finditer(body):
    name = m.group(1).lower()
    col = int(m.group(2)) if m.group(2) else 0
    vals = []
    for tok in m.group(3).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            rep, v = tok.split("*", 1)
            vals.extend([float(v.strip().replace("d", "e").replace("D", "e"))] * int(rep))
        else:
            vals.append(float(tok.replace("d", "e").replace("D", "e")))
    stmts.setdefault(name, {})[col] = vals


def cols(name, n, ncol=14, base=16):
    out = np.zeros((ncol, n), dtype=np.float64)
    for ib in range(ncol):
        v = stmts[name][base + ib]
        assert len(v) == n, (name, base + ib, len(v))
        out[ib, :] = v
    return out


tables = {
    "extliq1": cols("extliq1", 58), "ssaliq1": cols("ssaliq1", 58), "asyliq1": cols("asyliq1", 58),
    "extice2": cols("extice2", 43), "ssaice2": cols("ssaice2", 43), "asyice2": cols("asyice2", 43),
    "extice3": cols("extice3", 46), "ssaice3": cols("ssaice3", 46), "asyice3": cols("asyice3", 46),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **{k: v.astype(np.float32) for k, v in tables.items()})
print("wrote", OUT, {k: v.shape for k, v in tables.items()})
for name in ("abari", "bbari", "cbari", "dbari", "ebari", "fbari", "b0r", "c0r", "b0s", "b1s", "c0s"):
    print(name, stmts.get(name, {}).get(0))
