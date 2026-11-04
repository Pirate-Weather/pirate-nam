"""Extract the SW band-21 k-tables for phys-rrtm-sw-taumol21 to namgpu/data/rrtm_sw/kgb21.npz.

Usage:  python tools/extract_rrtm_sw_kgb21.py <nam phys dir> <out npz>

Sources (nam.v4.2.11, 83df4e13c1, sorc/nam_nems_nmmb.fd/src/atmos/phys):
  module_radsw_kgb21_nmmb   radsw_datatb_nmmb.f:10915-14755   (the table named by the unit)
  module_radsw_kgb21        radsw_datatb.f:11670-15510        (the module radsw_main_nmmb.f:3955 `use`s)
  Both are extracted; the script asserts they hold identical values and writes the _nmmb one.

Declarations: absa(MSA21=585, NG21=10), absb(MSB21=1175, NG21), selfref(MSF21=10, NG21),
forref(MFR21=4, NG21). The DATA statements fill them in Fortran (column-major) order; every element
must be set exactly once. Stored as C arrays [row, g] (g fastest), i.e. absa[585, 10] etc.
Values are rounded to float32 (kind_phys = 4 in NAM machine.f).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

PHYS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NG21 = 10


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


def tables_from(module):
    stmts = data_statements(module_lines(PHYS / ("radsw_datatb_nmmb.f" if module.endswith("_nmmb") else "radsw_datatb.f"), module))
    return {
        "absa": gather(stmts, "absa", (585, NG21)),
        "absb": gather(stmts, "absb", (1175, NG21)),
        "selfref": gather(stmts, "selfref", (10, NG21)),
        "forref": gather(stmts, "forref", (4, NG21)),
    }


nmmb = tables_from("module_radsw_kgb21_nmmb")
plain = tables_from("module_radsw_kgb21")
for k in nmmb:
    assert np.array_equal(nmmb[k], plain[k]), f"{k}: _nmmb and plain module differ"
print("module_radsw_kgb21_nmmb and module_radsw_kgb21 hold identical tables")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **{k: np.ascontiguousarray(v) for k, v in nmmb.items()})
for k in sorted(nmmb):
    a = nmmb[k]
    print(f"{k:10s} {a.shape} min={a.min():.4g} max={a.max():.4g}")
print("wrote", OUT)
