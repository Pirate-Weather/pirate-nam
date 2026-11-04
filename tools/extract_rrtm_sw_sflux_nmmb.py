"""Extract module_radsw_sflux_nmmb DATA tables to namgpu/data/rrtm_sw/sflux.npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f lines 1070-1378 (module_radsw_sflux_nmmb,
NOAA/NCEP, US government work). The module range is located by its own begin/end lines, so the
range check below also catches a moved file. Band constants from phys/radsw_param.f (NBLOW=16,
NBHGH=29, NGMAX=16). Single precision (kind_phys = real*4): every literal is rounded to float32.

Fortran -> NPZ (Fortran index order is column-major, so Fortran A(g,s,b) -> npz [b, s, g]):
  sfluxref01 (NGMAX, MFS01=1, MFB01=7)  -> sfluxref01 [7, 1, 16]
  sfluxref02 (NGMAX, MFS02=5, MFB02=2)  -> sfluxref02 [2, 5, 16]
  sfluxref03 (NGMAX, MFS03=9, MFB03=5)  -> sfluxref03 [5, 9, 16]
  band_sfluxref0x                        -> int32 band numbers for the third index b
  strrat, specwt (NBLOW:NBHGH)          -> float32 [14]   entry k = band 16+k
  layreffr, ix1, ix2, ibx               -> int32 [14]
  scalekur = 50.15/48.37 (parameter)    -> 0-d float32   (the Fortran comment says the band-27
                                           sfluxref01 values are scaled by it; the DATA values are stored
                                           unscaled, exactly as in the Fortran)
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])            # .../phys/radsw_datatb_nmmb.f
PARAM = Path(sys.argv[2])          # .../phys/radsw_param.f
OUT = Path(sys.argv[3])            # namgpu/data/rrtm_sw/sflux.npz

NGMAX, NBLOW, NBHGH = 16, 16, 29
MFS = {"sfluxref01": 1, "sfluxref02": 5, "sfluxref03": 9}
MFB = {"sfluxref01": 7, "sfluxref02": 2, "sfluxref03": 5}
BANDS = {
    "sfluxref01": [16, 20, 23, 25, 26, 27, 29],
    "sfluxref02": [17, 28],
    "sfluxref03": [18, 19, 21, 22, 24],
}
NB = NBHGH - NBLOW + 1

ptxt = PARAM.read_text(errors="replace")
for name, val in (("NBLOW", NBLOW), ("NBHGH", NBHGH), ("NGMAX", NGMAX)):
    m = re.search(rf"integer,\s*parameter\s*::\s*{name}\s*=\s*(\d+)", ptxt, re.I)
    assert m and int(m.group(1)) == val, (name, m and m.group(1))

lines = SRC.read_text(errors="replace").splitlines()
beg = next(i for i, s in enumerate(lines) if re.match(r"\s*module module_radsw_sflux_nmmb\b", s, re.I))
end = next(i for i, s in enumerate(lines) if re.match(r"\s*end module module_radsw_sflux_nmmb\b", s, re.I))
assert beg + 1 == 1070 and end + 1 == 1378, (beg + 1, end + 1)

body = []
for raw in lines[beg:end + 1]:
    if raw.lstrip().startswith("!"):
        continue
    body.append(raw.split("!", 1)[0].replace("&", " "))
text = re.sub(r"\s+", " ", " ".join(body))

stmts = re.findall(r"\bdata\s+(\w+)\s*(\(\s*:\s*,\s*:\s*,\s*(\d+)\s*\))?\s*/([^/]*)/", text, re.I)
assert stmts, "no DATA statements found"

def floats(s):
    return [float(v.strip().replace("d", "e").replace("D", "e")) for v in s.split(",") if v.strip()]

def ints(s):
    return [int(v.strip()) for v in s.split(",") if v.strip()]

sflux = {k: {} for k in MFS}
d = {}
for name, _, bidx, vals in stmts:
    name = name.lower()
    if name in MFS:
        v = floats(vals)
        assert len(v) == NGMAX * MFS[name], (name, bidx, len(v))
        sflux[name][int(bidx)] = v
    else:
        d[name] = vals

out = {}
for name in MFS:
    assert sorted(sflux[name]) == list(range(1, MFB[name] + 1)), (name, sorted(sflux[name]))
    arr = np.zeros((MFB[name], MFS[name], NGMAX), dtype=np.float32)
    for b in range(1, MFB[name] + 1):
        # Fortran flat index g + NGMAX*s (g fastest) -> C-order [s, g].
        arr[b - 1] = np.array(sflux[name][b], dtype=np.float32).reshape((MFS[name], NGMAX))
    out[name] = arr
    out["band_" + name] = np.array(BANDS[name], dtype=np.int32)

def one_d(name, conv, dtype):
    v = np.array(conv(d[name]), dtype=dtype)
    assert v.shape == (NB,), (name, v.shape)
    return v

out["strrat"] = one_d("strrat", floats, np.float32)
out["specwt"] = one_d("specwt", floats, np.float32)
out["layreffr"] = one_d("layreffr", ints, np.int32)
out["ix1"] = one_d("ix1", ints, np.int32)
out["ix2"] = one_d("ix2", ints, np.int32)
out["ibx"] = one_d("ibx", ints, np.int32)
out["scalekur"] = np.float32(np.float32(50.15) / np.float32(48.37))

assert abs(float(out["strrat"][22 - NBLOW]) - 3.63328e-2) < 1e-7   # o2adj=1.6 already applied in the DATA

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
print("wrote", OUT)
for k in sorted(out):
    print(k, out[k].shape, out[k].dtype)
