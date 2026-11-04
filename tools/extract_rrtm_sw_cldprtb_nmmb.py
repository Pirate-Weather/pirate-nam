"""Extract module_radsw_cldprtb_nmmb DATA tables to namgpu/data/rrtm_sw/cldprtb.npz.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f lines 52-1064 (module_radsw_cldprtb_nmmb,
NOAA/NCEP, US government work). The module range is located by its own begin/end lines. Band range
NBLOW=16, NBHGH=29 are checked against module_radsw_parameters_nmmb in phys/radsw_param_nmmb.f:147-148.
Single precision: NAM's production build defines -DSINGLE_PREC, so kind_phys = real*4 (phys/machine.f:20); every
literal is rounded to float32.

Fortran -> NPZ (Fortran (k, b) column-major DATA, band b = NBLOW..NBHGH -> npz row b-NBLOW):
  extdatliq1, ssadatliq1, asydatliq1 (58, NBLOW:NBHGH) -> float32 [14, 58]
  extdatice3, ssadatice3, asydatice3, fdldatice3 (27, NBLOW:NBHGH) -> float32 [14, 27]
  b0r, c0r, b0s, b1s, c0s (NBLOW:NBHGH)                 -> float32 [14]
  a0r, a1r, a0s, a1s, absdatliq1 (scalars)              -> 0-d float32
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])            # .../phys/radsw_datatb_nmmb.f
PARAM = Path(sys.argv[2])          # .../phys/radsw_param_nmmb.f
OUT = Path(sys.argv[3])            # namgpu/data/rrtm_sw/cldprtb.npz

NBLOW, NBHGH = 16, 29
NB = NBHGH - NBLOW + 1

ptxt = PARAM.read_text(errors="replace")
for name, val in (("NBLOW", NBLOW), ("NBHGH", NBHGH)):
    m = re.search(rf"integer,\s*parameter\s*::\s*{name}\s*=\s*(\d+)", ptxt, re.I)
    assert m and int(m.group(1)) == val, (name, m and m.group(1))

lines = SRC.read_text(errors="replace").splitlines()
beg = next(i for i, s in enumerate(lines) if re.match(r"\s*module module_radsw_cldprtb_nmmb\b", s, re.I))
end = next(i for i, s in enumerate(lines) if re.match(r"\s*end module module_radsw_cldprtb_nmmb\b", s, re.I))
assert beg + 1 == 52 and end + 1 == 1064, (beg + 1, end + 1)

body = []
for raw in lines[beg:end + 1]:
    if raw.lstrip().startswith("!"):
        continue
    body.append(raw.split("!", 1)[0].replace("&", " "))
text = re.sub(r"\s+", " ", " ".join(body))

m = re.search(r"absdatliq1\s*=\s*([0-9.eEdD+-]+)", text)
assert m, "absdatliq1 parameter not found"
absdatliq1 = np.float32(float(m.group(1).replace("d", "e").replace("D", "e")))


def vals(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            r, v = tok.split("*")
            out += [np.float32(float(v.strip().replace("d", "e").replace("D", "e")))] * int(r)
        else:
            out.append(np.float32(float(tok.replace("d", "e").replace("D", "e"))))
    return out


cols = {}   # name -> {band: values}
vec = {}    # 1-D band tables
sca = {}    # scalars
for chunk in re.split(r"\bdata\b", text, flags=re.I)[1:]:
    for lhs, rhs in re.findall(r"([^/]+?)\s*/([^/]*)/", chunk):
        lhs = lhs.strip().lstrip(",").strip().lower()
        v = vals(rhs)
        cm = re.fullmatch(r"(\w+)\s*\(\s*:\s*,\s*(\d+)\s*\)", lhs)
        if cm:
            cols.setdefault(cm.group(1), {})[int(cm.group(2))] = v
            continue
        names = [n.strip() for n in lhs.split(",")]
        if len(names) == 1:
            vec[names[0]] = v
        else:
            assert len(names) == len(v), (lhs, len(v))
            for n, x in zip(names, v):
                sca[n] = x


def stack(name, n):
    c = cols[name]
    assert sorted(c) == list(range(NBLOW, NBHGH + 1)), (name, sorted(c))
    arr = np.zeros((NB, n), dtype=np.float32)
    for b in range(NBLOW, NBHGH + 1):
        assert len(c[b]) == n, (name, b, len(c[b]))
        arr[b - NBLOW] = np.array(c[b], dtype=np.float32)
    return arr


def band_vec(name):
    v = np.array(vec[name], dtype=np.float32)
    assert v.shape == (NB,), (name, v.shape)
    return v


out = {
    "extdatliq1": stack("extdatliq1", 58),
    "ssadatliq1": stack("ssadatliq1", 58),
    "asydatliq1": stack("asydatliq1", 58),
    "extdatice3": stack("extdatice3", 27),
    "ssadatice3": stack("ssadatice3", 27),
    "asydatice3": stack("asydatice3", 27),
    "fdldatice3": stack("fdldatice3", 27),
    "b0r": band_vec("b0r"), "c0r": band_vec("c0r"),
    "b0s": band_vec("b0s"), "b1s": band_vec("b1s"), "c0s": band_vec("c0s"),
    "a0r": np.float32(sca["a0r"]), "a1r": np.float32(sca["a1r"]),
    "a0s": np.float32(sca["a0s"]), "a1s": np.float32(sca["a1s"]),
    "absdatliq1": absdatliq1,
}

# Spot checks against the Fortran source text.
assert out["extdatliq1"][0, 0] == np.float32(8.981463e-01)
assert out["asydatliq1"][27 - NBLOW, 0] == np.float32(8.430819e-01)
assert out["a1s"] == np.float32(1.5) and out["a0r"] == np.float32(3.07e-3)
assert out["b1s"][7] == np.float32(1.62e-5) and out["b1s"][6] == 0.0

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
print("wrote", OUT)
for k in sorted(out):
    print(k, np.shape(out[k]), np.asarray(out[k]).dtype)
