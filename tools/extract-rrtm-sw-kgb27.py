"""Extract the NMMB RRTM shortwave band-27 table (taumol27) into namgpu/data/rrtm_sw/kgb27.npz.

Source: module_radsw_kgb27_nmmb in phys/radsw_datatb_nmmb.f (nam.v4.2.11, 83df4e13c1), lines 17159-17749.
Declarations: absa(MSA27=65, NG27=8), absb(MSB27=235, NG27), rayl(NG27=8). DATA statements fill them
in Fortran (column-major) order; every element must be set exactly once.
Stored as C arrays [row, g] (g fastest): absa[65, 8], absb[235, 8], rayl[8].
Values are rounded to float32 (kind_phys = 4 in NAM machine.f).

Usage:  python3 tools/extract-rrtm-sw-kgb27.py <nam phys dir> <out npz>
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

PHYS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NG27 = 8
MSA27 = 65
MSB27 = 235


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


stmts = data_statements(module_lines(PHYS / "radsw_datatb_nmmb.f", "module_radsw_kgb27_nmmb"))
tabs = {
    "absa": gather(stmts, "absa", (MSA27, NG27)),
    "absb": gather(stmts, "absb", (MSB27, NG27)),
    "rayl": gather(stmts, "rayl", (NG27,)),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **{k: np.ascontiguousarray(v) for k, v in tabs.items()})
for k in sorted(tabs):
    a = tabs[k]
    print(f"{k:6s} {a.shape} min={a.min():.4g} max={a.max():.4g}")
print("wrote", OUT)
