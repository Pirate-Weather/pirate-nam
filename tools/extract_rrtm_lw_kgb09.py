"""Extract the LW band-9 k-tables for phys-rrtm-lw-taugb09 to namgpu/data/rrtm_lw/kgb09.npz.

Usage:  python tools/extract_rrtm_lw_kgb09.py <nam phys dir> <out npz>

Sources (nam.v4.2.11, 83df4e13c1, sorc/nam_nems_nmmb.fd/src/atmos/phys):
  module_radlw_kgb09   radlw_datatb.f:25328-27963   USED by taugb09 (radlw_main_nmmb.f:5077, `use module_radlw_kgb09`)
  module_radlw_ref     radlw_datatb.f:733-905       chi_mls(7,59) (reference profile; chi_mls(g, lev))
  NOTE: radlw_datatb_nmmb.f:23644-26117 (module_radlw_kgb09_nmmb) is NOT what taugb09 calls. It
  declares absa(MSA09,NG09) (g index last) and has no forref/ka_mn2o/kb_mn2o, so it cannot be read
  with taugb09's absa(ig,id) indexing.

Storage: every Fortran array A(d1,...,dn) is stored in the npz with the axes reversed, so the C
index is A[dn-1,...,d1-1] and the g index (NG09=12) is the last (fastest) axis:
  absa     (12,585) -> [585,12]     absb    (12,235) -> [235,12]
  selfref  (12,10)  -> [10,12]      forref  (12,4)   -> [4,12]
  fracrefa (12,9)   -> [9,12]       fracrefb (12)    -> [12]
  ka_mn2o  (12,9,19)-> [19,9,12]    kb_mn2o (12,19)  -> [19,12]
  chi_mls  (7,59)   -> [59,7]       (chi_mls[lev-1, gas-1])

DATA is applied with Fortran semantics: each DATA element goes to the subscript it names, with the
first subscript varying fastest (column-major). Every element of every table must be set exactly
once; the extractor asserts that. Values are rounded to float32 (kind_phys = real*4 in NAM).
"""
import itertools
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
    """Join continued DATA statements; return list of (name, subscript string or '', value tokens)."""
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
            assert m, buf[:160]
            toks = [t.strip() for t in m.group(3).split(",") if t.strip()]
            out.append((m.group(1).lower(), (m.group(2) or "").replace(" ", ""), toks))
            buf = None
    assert buf is None, "unterminated DATA statement"
    return out


def to_f32(tok):
    tok = tok.replace("d", "e").replace("D", "e")
    return np.float32(float(tok))


def expand(toks):
    vals = []
    for t in toks:
        if "*" in t:                       # Fortran repeat count r*value
            r, v = t.split("*", 1)
            vals.extend([to_f32(v)] * int(r))
        else:
            vals.append(to_f32(t))
    return vals


def section_indices(sub, dims):
    """Column-major (1-based) subscript tuples selected by a DATA subscript string."""
    toks = sub.split(",") if sub else [":"] * len(dims)
    assert len(toks) == len(dims), (sub, dims)
    ranges = []
    for tok, d in zip(toks, dims):
        if tok == ":":
            ranges.append(list(range(1, d + 1)))
        elif ":" in tok:
            a, b = tok.split(":")
            ranges.append(list(range(int(a), int(b) + 1)))
        else:
            ranges.append([int(tok)])
    # first subscript must vary fastest: iterate the last subscript slowest.
    combos = itertools.product(*reversed(ranges))
    return [tuple(reversed(c)) for c in combos]


def gather(stmts, name, dims):
    """Fill a Fortran array of shape dims from every DATA statement that names it."""
    size = int(np.prod(dims))
    flat = np.full(size, np.nan, dtype=np.float32)
    seen = np.zeros(size, dtype=bool)
    for n, sub, toks in stmts:
        if n != name:
            continue
        vals = expand(toks)
        idx = section_indices(sub, dims)
        assert len(vals) == len(idx), (name, sub, len(vals), len(idx))
        for v, ix in zip(vals, idx):
            lin = 0
            stride = 1
            for i, d, dim in zip(ix, range(len(dims)), dims):
                lin += (i - 1) * stride
                stride *= dim
            assert not seen[lin], (name, ix)
            seen[lin] = True
            flat[lin] = v
    assert seen.all(), (name, int((~seen).sum()), "elements unset")
    return flat.reshape(dims, order="F")       # Fortran shape, values in Fortran order


kgb = data_statements(module_lines(PHYS / "radlw_datatb.f", "module_radlw_kgb09"))
ref = data_statements(module_lines(PHYS / "radlw_datatb.f", "module_radlw_ref"))

NG = 12
absa = gather(kgb, "absa", (NG, 585))
absb = gather(kgb, "absb", (NG, 235))
selfref = gather(kgb, "selfref", (NG, 10))
forref = gather(kgb, "forref", (NG, 4))
fracrefa = gather(kgb, "fracrefa", (NG, 9))
fracrefb = gather(kgb, "fracrefb", (NG,))
ka_mn2o = gather(kgb, "ka_mn2o", (NG, 9, 19))
kb_mn2o = gather(kgb, "kb_mn2o", (NG, 19))
chi_mls = gather(ref, "chi_mls", (7, 59))

tables = {
    "absa": np.ascontiguousarray(absa.T),                      # [585, 12]
    "absb": np.ascontiguousarray(absb.T),                      # [235, 12]
    "selfref": np.ascontiguousarray(selfref.T),                # [10, 12]
    "forref": np.ascontiguousarray(forref.T),                  # [4, 12]
    "fracrefa": np.ascontiguousarray(fracrefa.T),              # [9, 12]
    "fracrefb": np.ascontiguousarray(fracrefb),                # [12]
    "ka_mn2o": np.ascontiguousarray(ka_mn2o.transpose(2, 1, 0)),  # [19, 9, 12]
    "kb_mn2o": np.ascontiguousarray(kb_mn2o.T),                # [19, 12]
    "chi_mls": np.ascontiguousarray(chi_mls.T),                # [59, 7]
}

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **tables)
for k in sorted(tables):
    a = tables[k]
    print(f"{k:10s} {a.shape} min={a.min():.4g} max={a.max():.4g}")
print("wrote", OUT)
