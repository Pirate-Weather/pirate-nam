"""Extract the RRTM LW band-4 k-tables and the module_radlw_ref reference tables from the NAM Fortran.

Source: nam.v4.2.11 (83df4e13c1), phys/radlw_datatb.f. The NMMB longwave driver (radlw_main_nmmb.f,
taugb04 at lines 4098-4337) USES module_radlw_kgb04 from radlw_datatb.f (lines 10030-15204), not the
module_radlw_kgb04_nmmb of radlw_datatb_nmmb.f (that module has no forref, so taugb04 cannot use it).
Data statements are column-major Fortran; this parser fills them the Fortran way.

Output (C-order, Fortran A(i,j) -> arr[j-1, i-1]):
  data/rrtm_lw/kgb04.npz : absa [585, 14], absb [1175, 14], selfref [10, 14], forref [4, 14],
                           fracrefa [9, 14], fracrefb [5, 14]     (index [ms, ig], ig fastest in Fortran)
  data/rrtm_lw/ref.npz   : chi_mls [59, 7] (chi_mls(gas, ref) -> [ref-1, gas-1]), preflog [59], tref [59]
Usage: python3 extract_rrtm_lw_kgb04.py <radlw_datatb.f> <out_dir>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
lines = SRC.read_text(errors="replace").splitlines()


def clean(block):
    """Drop comment lines and join Fortran continuations (trailing '&' and leading '&')."""
    out = []
    for ln in block:
        s = ln
        if s.lstrip().startswith("!"):
            continue
        if "!" in s:
            s = s[: s.index("!")]
        out.append(s)
    text = ""
    for s in out:
        s = s.rstrip()
        if text.rstrip().endswith("&"):
            text = text.rstrip()[:-1]
            s = s.lstrip()
            if s.startswith("&"):
                s = s[1:]
        text += " " + s + "\n"
    return text


def module_block(name):
    lo = next(i for i, l in enumerate(lines) if re.match(rf"\s*module\s+{name}\s*!?\s*$", l, re.I))
    hi = next(i for i in range(lo, len(lines)) if re.match(rf"\s*end\s+module\s+{name}\b", lines[i], re.I))
    return lines[lo:hi + 1]


def fortran_indices(spec, dims):
    """Tuples (i, j, ...) in Fortran column-major order (first subscript fastest) for a subscript spec."""
    subs = [s.strip() for s in spec.split(",")] if spec is not None else [":"] * len(dims)
    if len(subs) == 1 and len(dims) == 2:
        subs = subs + [":"]
    axes = []
    for s, d in zip(subs, dims):
        if s == ":":
            axes.append(list(range(1, d + 1)))
        elif ":" in s:
            a, b = s.split(":")
            axes.append(list(range(int(a), int(b) + 1)))
        else:
            axes.append([int(s)])
    # build with the last axis outermost so the first axis varies fastest
    combos = [()]
    for ax in reversed(axes):
        combos = [(v,) + c for c in combos for v in ax] if False else [c + (v,) for v in ax for c in combos]
    combos = [tuple(reversed(c)) for c in combos]
    return combos


def parse_module(block, arrays):
    """arrays: {name: Fortran dims}. Returns {name: C-order array with reversed dims}."""
    text = clean(block)
    out = {n: np.full(tuple(reversed(d)), np.nan, dtype=np.float64) for n, d in arrays.items()}
    count = {n: 0 for n in arrays}
    pat = re.compile(r"\bdata\s+(\w+)\s*(\(([^)]*)\))?\s*/([^/]*)/", re.I)
    for m in pat.finditer(text):
        name = m.group(1).lower()
        if name not in arrays:
            continue
        dims = arrays[name]
        idx = fortran_indices(m.group(3), dims)
        vals = [v.strip() for v in m.group(4).replace("\n", " ").split(",") if v.strip()]
        vals = [float(v.replace("d", "e").replace("D", "e")) for v in vals]
        assert len(vals) == len(idx), (name, m.group(2), len(vals), len(idx))
        for t, v in zip(idx, vals):
            # Fortran (i, j) -> C [j-1, i-1]
            out[name][tuple(reversed([x - 1 for x in t]))] = v
        count[name] += len(vals)
    for n, d in arrays.items():
        assert not np.isnan(out[n]).any(), f"{n}: unfilled entries"
        print(f"{n}: {count[n]} values, shape {out[n].shape}")
    return out


kgb04 = parse_module(module_block("module_radlw_kgb04"), {
    "absa": (14, 585), "absb": (14, 1175), "selfref": (14, 10), "forref": (14, 4),
    "fracrefa": (14, 9), "fracrefb": (14, 5),
})
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "kgb04.npz", **{k: v.astype(np.float32) for k, v in kgb04.items()})

ref = parse_module(module_block("module_radlw_ref"), {"chi_mls": (7, 59), "preflog": (59,), "tref": (59,)})
np.savez(OUT / "ref.npz", chi_mls=ref["chi_mls"].astype(np.float32),
         preflog=ref["preflog"].astype(np.float32), tref=ref["tref"].astype(np.float32))
print("wrote", OUT / "kgb04.npz", OUT / "ref.npz")
