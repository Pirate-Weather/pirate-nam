"""Extract the LW band-8 k-tables for phys-rrtm-lw-taugb08 to namgpu/data/rrtm_lw/kgb08.npz.

Usage:  python tools/extract_rrtm_lw_kgb08.py <nam phys dir> <out npz>

Sources (nam.v4.2.11, 83df4e13c1, sorc/nam_nems_nmmb.fd/src/atmos/phys):
  module_radlw_kgb08       radlw_datatb.f:24545-25324   USED by taugb08 (radlw_main_nmmb.f:4954)
  module_radlw_kgb08_nmmb  radlw_datatb_nmmb.f:22997-23638  (unit spec; not used by taugb08)
  chi_mls(2,:)             radlw_datatb.f:791-     (module_radlw_ref; CO2 column of the reference profile)

Storage: Fortran table T(NG08, N) -> npz array [N, NG08] (row = Fortran second index, C order),
so the kernel reads T[row*8 + ig]. The DATA values run ig fastest (first index of the Fortran
array varies fastest), so the flat list reshapes to [N, NG08] in C order: T[r, ig] = flat[r*NG08 + ig].
Single precision: every literal is rounded to float32 (kind_phys = real*4 in NAM).
"""
import re
import sys
from pathlib import Path

import numpy as np

PHYS = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def module_lines(path, name):
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines)
                 if re.match(rf"\s*module\s+{name}\b", l, re.I))
    end = next(i for i in range(start + 1, len(lines))
               if re.match(rf"\s*end\s+module\s+{name}\b", lines[i], re.I))
    return lines[start:end + 1]


def data_statements(lines):
    """Join continued DATA statements; return list of (name, subscript or None, values)."""
    out = []
    buf = None
    for raw in lines:
        if raw.lstrip().startswith("!"):
            continue
        s = raw.split("!", 1)[0].replace("&", " ").strip()
        if not s:
            continue
        if buf is None:
            if not re.match(r"data\b", s, re.I):
                continue
            buf = s
        else:
            buf += " " + s
        if buf.count("/") >= 2:
            m = re.match(r"data\s+(\w+)\s*(?:\(([^)]*)\))?\s*/(.*)/\s*$", buf, re.I)
            assert m, buf[:120]
            vals = [np.float32(float(v.strip().replace("d", "e").replace("D", "e")))
                    for v in m.group(3).split(",") if v.strip()]
            out.append((m.group(1).lower(), (m.group(2) or "").replace(" ", ""), vals))
            buf = None
    assert buf is None, "unterminated DATA statement"
    return out


def gather(stmts, name, count, subscript_ok=lambda s: True):
    vals = []
    for n, sub, v in stmts:
        if n == name and subscript_ok(sub):
            vals.extend(v)
    assert len(vals) == count, (name, len(vals), count)
    return np.array(vals, dtype=np.float32)


# ---- module_radlw_kgb08 (regular; used by taugb08): declared T(NG08, N) ----
NG = 8
reg = data_statements(module_lines(PHYS / "radlw_datatb.f", "module_radlw_kgb08"))
full = data_statements(module_lines(PHYS / "radlw_datatb.f", "module_radlw_ref"))   # chi_mls
REG_SHAPES = {  # name: (rows N, cols NG) as stored; Fortran declaration in brackets
    "absa": 65,      # absa(NG08,MSA08=65)
    "absb": 235,     # absb(NG08,MSB08=235)
    "selfref": 10,   # selfref(NG08,MSF08=10)
    "forref": 4,     # forref(NG08,MFR08=4)
    "ka_mco2": 19,   # ka_mco2(NG08,MMC08=19)
    "kb_mco2": 19,
    "ka_mo3": 19,
    "ka_mn2o": 19,
    "kb_mn2o": 19,
}
tables = {}
for name, rows in REG_SHAPES.items():
    flat = gather(reg, name, NG * rows)
    tables[name] = flat.reshape((rows, NG))   # C order: [row, ig] = flat[row*NG + ig]
for name in ("fracrefa", "fracrefb", "cfc12", "cfc22adj"):
    tables[name] = gather(reg, name, NG)

# ---- chi_mls: the CO2 column (second row), 59 reference levels ----
chi_co2 = gather(full, "chi_mls", 59, subscript_ok=lambda s: s.startswith("2,"))
tables["chi_mls_co2"] = chi_co2

# ---- module_radlw_kgb08_nmmb (unit spec; kept for the record, not used by taugb08) ----
nm = data_statements(module_lines(PHYS / "radlw_datatb_nmmb.f", "module_radlw_kgb08_nmmb"))
NM_SHAPES = {"absa": (35, NG), "absb": (265, NG), "selfref": (10, NG)}
for name, (r, c) in NM_SHAPES.items():
    flat = gather(nm, name, r * c)
    tables["nmmb_" + name] = flat.reshape((r, c))   # same ig-fastest DATA order
for name in ("absco2a", "absco2b", "absn2oa", "absn2ob", "cfc12", "cfc22adj", "fracrefa", "fracrefb"):
    tables["nmmb_" + name] = gather(nm, name, NG)

# ---- hand-evaluated reference values (radlw_datatb.f:24575, first DATA line of absa) ----
# Fortran absa(ig, row) with ig fastest in DATA: absa(1,1)=1.958396E-05, absa(2,1)=3.183247E-05,
# absa(1,2)=1.958781E-05, absa(2,2)=4.539081E-05. Stored [row-1, ig-1].
assert tables["absa"][0, 0] == np.float32(1.958396e-05)
assert tables["absa"][0, 1] == np.float32(3.183247e-05)
assert tables["absa"][1, 0] == np.float32(1.958781e-05)
assert tables["absa"][1, 1] == np.float32(4.539081e-05)
assert tables["chi_mls_co2"].shape == (59,)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **tables)
for k in sorted(tables):
    print(f"{k:16s} {tables[k].shape} min={tables[k].min():.4g} max={tables[k].max():.4g}")
print("wrote", OUT)
