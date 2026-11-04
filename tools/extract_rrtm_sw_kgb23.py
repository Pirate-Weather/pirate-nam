"""Extract module_radsw_kgb23_nmmb DATA tables to namgpu/data/rrtm_sw/kgb23.npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f, module_radsw_kgb23_nmmb (band 23,
8050-12850 cm-1). The module range is located by its own begin and end lines, so a moved file fails
the range check. Fortran DATA is column-major; every array is stored with the Fortran shape, so a
numpy index [a, b] equals the Fortran element (a+1, b+1):
  absa    (MSA23=65, NG23=10)   absa[ind, g]
  forref  (MFR23=3,  NG23=10)   forref[n, g]
  selfref (MSF23=10, NG23=10)   selfref[t, g]
  rayl    (NG23=10,)            rayl[g]
givfac = 1.029 is a PARAMETER of the module (kind_phys = real*4, rounded to float32) and is stored
as a scalar. Usage: python3 extract_rrtm_sw_kgb23.py <radsw_datatb_nmmb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NG23, MSA23, MSF23, MFR23 = 10, 65, 10, 3

lines = SRC.read_text(errors="replace").splitlines()
beg = next(i for i, s in enumerate(lines) if re.match(r"\s*module module_radsw_kgb23_nmmb\b", s, re.I))
end = next(i for i, s in enumerate(lines) if re.match(r"\s*end module module_radsw_kgb23_nmmb\b", s, re.I))
assert 15190 <= beg + 1 <= 15200 and 15430 <= end + 1 <= 15440, (beg + 1, end + 1)

body = []
for raw in lines[beg:end + 1]:
    if raw.lstrip().startswith("!"):
        continue
    body.append(raw.split("!", 1)[0].replace("&", " "))
text = re.sub(r"\s+", " ", " ".join(body))


def floats(s):
    return [float(v.strip().replace("d", "e").replace("D", "e")) for v in s.split(",") if v.strip()]


assert re.search(r"givfac\s*=\s*1\.029\b", text), "givfac parameter not found"
assert re.search(r"MSA23\s*=\s*65\b.*MSF23\s*=\s*10\b.*MFR23\s*=\s*3\b", text, re.I), "dims not found"

absa = np.zeros((MSA23, NG23), dtype=np.float32)
seen = set()
for m in re.finditer(r"\bdata\s+absa\s*\(\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]*)/", text, re.I):
    a, b, g = int(m.group(1)), int(m.group(2)), int(m.group(3))
    vals = floats(m.group(4))
    assert (a, b) == (1, MSA23) and len(vals) == MSA23, (a, b, len(vals))
    assert g not in seen, g
    seen.add(g)
    absa[:, g - 1] = np.asarray(vals, dtype=np.float32)
assert seen == set(range(1, NG23 + 1)), sorted(seen)


def whole(name, shape):
    m = re.search(r"\bdata\s+%s\s*\(([^)]*)\)\s*/([^/]*)/" % name, text, re.I)
    assert m, name
    vals = floats(m.group(2))
    n = int(np.prod(shape))
    assert len(vals) == n, (name, len(vals), n)
    # Fortran column-major: the first index runs fastest.
    return np.asarray(vals, dtype=np.float32).reshape(shape, order="F")


forref = whole("forref", (MFR23, NG23))
selfref = whole("selfref", (MSF23, NG23))
rayl = whole("rayl", (NG23,))
givfac = np.float32(1.029)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, absa=absa, forref=forref, selfref=selfref, rayl=rayl, givfac=givfac,
         source=np.array("NAM nam.v4.2.11 83df4e13c1 phys/radsw_datatb_nmmb.f module_radsw_kgb23_nmmb "
                         f"lines {beg + 1}-{end + 1}"))
print("wrote", OUT, "absa", absa.shape, "forref", forref.shape, "selfref", selfref.shape,
      "rayl", rayl.shape, "absa range", float(absa.min()), float(absa.max()))
