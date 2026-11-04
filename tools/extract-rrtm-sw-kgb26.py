"""Extract the NMMB RRTM shortwave band-26 table (taumol26) into namgpu/data/rrtm_sw/kgb26.npz.

Source: module_radsw_kgb26_nmmb in phys/radsw_datatb_nmmb.f (nam.v4.2.11, 83df4e13c1), lines 17124-17154,
the module radsw_main_nmmb.f taumol26 USEs (`use module_radsw_kgb26`, line 4455 area).

Band 26 has no absorption table: the module holds only the Rayleigh coefficient rayl(NG26), six values,
one per g-point (NG26 = 6, radsw_param_nmmb.f:163). Stored as float32 rayl[6] in the Fortran order (index j-1).
DATA values are read in order and the count is asserted.

Usage:  python3 tools/extract-rrtm-sw-kgb26.py <path-to-radsw_datatb_nmmb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LINES = SRC.read_text(errors="replace").splitlines()

NG26 = 6
head = re.compile(r"^\s{6}module\s+module_radsw_kgb26_nmmb\s*(!.*)?$", re.I)
tail = re.compile(r"^\s{6}end\s+module\s+module_radsw_kgb26_nmmb\s*(!.*)?$", re.I)
lo = next(i for i, l in enumerate(LINES) if head.match(l))
hi = next(i for i in range(lo, len(LINES)) if tail.match(LINES[i]))

parts = []
for raw in LINES[lo:hi + 1]:
    s = raw.split("!")[0].strip()
    if not s:
        continue
    s = s.rstrip("&").strip()
    if s.startswith("&"):
        s = s[1:].strip()
    parts.append(s)
text = " ".join(parts)

m = re.search(r"data\s+rayl\s*\(\s*1\s*:\s*6\s*\)\s*/([^/]*)/", text, re.I)
assert m, "data statement for rayl(1:6) not found"
vals = [float(v.strip().replace("d", "e").replace("D", "e")) for v in m.group(1).split(",") if v.strip()]
assert len(vals) == NG26, f"rayl has {len(vals)} values, expected {NG26}"

rayl = np.asarray(vals, dtype=np.float32)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, rayl=rayl)
print(f"module lines {LINES[lo].strip()} .. {LINES[hi].strip()}")
print(f"rayl {rayl.shape} {rayl.dtype}: {rayl.tolist()}")
print(f"wrote {OUT}")
