"""Extract the SW band-19 k-tables for phys-rrtm-sw-taumol19 to namgpu/data/rrtm_sw/kgb19.npz.

Usage:  python tools/extract_rrtm_sw_kgb19.py <nam phys dir> <out npz>

Source (nam.v4.2.11, 83df4e13c1, sorc/nam_nems_nmmb.fd/src/atmos/phys):
  module_radsw_kgb19_nmmb   radsw_datatb_nmmb.f:8639-10141   USED by taumol19 (radsw_main_nmmb.f:3750)
  The NMMB copy is the one taumol19 calls (the non-nmmb radsw_datatb.f copy is not referenced).

Storage: each Fortran array keeps its declared shape and index order, so Fortran A(i,j) is
numpy A[i-1, j-1] (the g index is the last axis, as in kgb09.npz):
  absa     (585, 8)   absb (235, 8)   selfref (10, 8)   forref (3, 8)
Every DATA element goes to the subscript it names (first subscript fastest); the extractor asserts
that every element is set exactly once. Values are rounded to float32 (kind_phys = real*4).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

PHYS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODULE = "module_radsw_kgb19_nmmb"
SRC = "radsw_datatb_nmmb.f"


def module_lines(path, name):
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines)
                 if re.match(rf"\s*module\s+{name}\b", l, re.I))
    end = next(i for i in range(start + 1, len(lines))
               if re.match(rf"\s*end\s+module\s+{name}\b", lines[i], re.I))
    return lines[start:end + 1]


def data_statements(lines):
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
        if "*" in t:
            r, v = t.split("*", 1)
            vals.extend([to_f32(v)] * int(r))
        else:
            vals.append(to_f32(t))
    return vals


def section_indices(sub, dims):
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
    combos = itertools.product(*reversed(ranges))
    return [tuple(reversed(c)) for c in combos]


def gather(stmts, name, dims):
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
            for i, dim in zip(ix, dims):
                lin += (i - 1) * stride
                stride *= dim
            assert not seen[lin], (name, ix)
            seen[lin] = True
            flat[lin] = v
    assert seen.all(), (name, int((~seen).sum()), "elements unset")
    return flat.reshape(dims, order="F")


stmts = data_statements(module_lines(PHYS / SRC, MODULE))
NG = 8
tables = {
    "absa": gather(stmts, "absa", (585, NG)),
    "absb": gather(stmts, "absb", (235, NG)),
    "selfref": gather(stmts, "selfref", (10, NG)),
    "forref": gather(stmts, "forref", (3, NG)),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **tables)
for k in sorted(tables):
    a = tables[k]
    print(f"{k:10s} {a.shape} min={a.min():.4g} max={a.max():.4g}")
print("wrote", OUT)
