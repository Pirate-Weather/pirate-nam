"""Extract module_radlw_cldprlw_nmmb DATA tables to namgpu/data/rrtm_lw/cldprlw.npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radlw_datatb_nmmb.f lines 733-1144 (module_radlw_cldprlw_nmmb).
Fortran (i,j) arrays are stored C-order as [j, i] per CONTRACT section 1.
Single precision (kind_phys = real*4 in NAM): every literal is rounded to float32.

Fortran -> NPZ:
  absice0 (2)        -> absice0 [2]
  absice1 (2,5)      -> absice1 [5, 2]   absice1[ia-1, m-1] = absice1_F(m, ia)  (DATA is column-major)
  absice2 (40,16)    -> absice2 [16, 40] absice2[ib-1, k-1] = absice2_F(k, ib)
  absliq3 (58,16)    -> absliq3 [16, 58]
  absliq1, absliq2, absrain, abssnow0, abssnow1 -> 0-d float32
  ipat (NBANDS=16)   -> ipat int32 [16]
"""
import re
import sys
from pathlib import Path

import numpy as np

from f32_literal import f32_literal

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LO, HI = 733, 1144  # 1-based inclusive, module_radlw_cldprlw_nmmb

lines = SRC.read_text(errors="replace").splitlines()[LO - 1:HI]
assert lines[0].lower().lstrip().startswith("module module_radlw_cldprlw_nmmb"), lines[0]
assert lines[-1].lower().lstrip().startswith("end module module_radlw_cldprlw_nmmb"), lines[-1]

# Join continuation lines; strip comments. A DATA statement ends at its closing slash.
stmts = []
buf = None
for raw in lines:
    s = raw.split("!", 1)[0] if not raw.lstrip().startswith("!") else ""
    if not s.strip():
        continue
    s = s.strip()
    if buf is None:
        if not re.match(r"data\b", s, re.I):
            continue
        buf = s
    else:
        buf += " " + s
    if buf.count("/") >= 2:
        stmts.append(buf)
        buf = None
    s2 = buf
assert buf is None, "unterminated DATA statement"

def values(body):
    body = body.replace("&", " ")
    return [f32_literal(v) for v in body.split(",") if v.strip()]

tables = {}
for st in stmts:
    m = re.match(r"data\s+(\w+)\s*(\(\s*:\s*,\s*(\d+)\s*\))?\s*/(.*)/\s*$", st, re.I)
    assert m, st
    name = m.group(1).lower()
    col = int(m.group(3)) if m.group(3) else None
    vals = values(m.group(4))
    if col is None:
        tables[name] = vals
    else:
        tables.setdefault(name + "_cols", {})[col] = vals

# ipat: integer DATA, not a float; parse separately to keep it exact.
ipat = [int(v) for v in re.findall(r"\d+", next(s for s in stmts if re.match(r"data\s+ipat", s, re.I))
                                   .split("/", 1)[1].rsplit("/", 1)[0])]
assert len(ipat) == 16, ipat
assert ipat == [1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], ipat

absice2_cols = tables.pop("absice2_cols")
absliq3_cols = tables.pop("absliq3_cols")
assert sorted(absice2_cols) == list(range(1, 17)) and sorted(absliq3_cols) == list(range(1, 17))
absice2 = np.array([absice2_cols[k] for k in range(1, 17)], dtype=np.float32)
absliq3 = np.array([absliq3_cols[k] for k in range(1, 17)], dtype=np.float32)
assert absice2.shape == (16, 40), absice2.shape
assert absliq3.shape == (16, 58), absliq3.shape

absice0 = np.array(tables["absice0"], dtype=np.float32)
absice1_flat = np.array(tables["absice1"], dtype=np.float32)
assert absice0.shape == (2,), absice0.shape
assert absice1_flat.shape == (10,), absice1_flat.shape
absice1 = absice1_flat.reshape((2, 5), order="F").T.copy()  # [ia, m] from column-major (2,5)
assert absice1.shape == (5, 2)

def scalar(name):
    v = tables[name]
    assert len(v) == 1, (name, v)
    return np.float32(v[0])

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    OUT,
    absice0=absice0,
    absice1=absice1,
    absice2=absice2,
    absliq1=scalar("absliq1"),
    absliq2=scalar("absliq2"),
    absliq3=absliq3,
    absrain=scalar("absrain"),
    abssnow0=scalar("abssnow0"),
    abssnow1=scalar("abssnow1"),
    ipat=np.array(ipat, dtype=np.int32),
)
print("wrote", OUT)
print("absice0", absice0.tolist())
print("absice1[ia=1]", absice1[0].tolist(), "absice1[ia=5]", absice1[4].tolist())
print("absice2 shape", absice2.shape, "first", absice2[0, :3].tolist(), "last", absice2[15, -1])
print("absliq3 shape", absliq3.shape, "first", absliq3[0, :3].tolist(), "last", absliq3[15, -1])
print("absliq1", scalar("absliq1"), "absliq2", scalar("absliq2"), "absrain", scalar("absrain"),
      "abssnow0", scalar("abssnow0"), "abssnow1", scalar("abssnow1"))
print("ipat", ipat)
