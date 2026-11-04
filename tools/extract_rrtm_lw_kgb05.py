"""Extract the band-5 LW k-tables that NMMB taugb05 binds to by name (module_radlw_kgb05).

Source: nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_kgb05 (lines 15209-21676) and
module_radlw_ref (lines 733-905, chi_mls). taugb05 (radlw_main_nmmb.f:4341) has `use module_radlw_kgb05`;
that name exists only in radlw_datatb.f, not in radlw_datatb_nmmb.f (which defines module_radlw_kgb05_nmmb
with a different layout). The GFS-named module is therefore the one the NAM binary reads.

Writes an npz with arrays in FORTRAN SHAPE (numpy index [i-1, j-1, ...] = Fortran (i, j, ...)):
  absa   (16, 585)   absa(ig, id)
  absb   (16, 1175)  absb(ig, id)
  selfref(16, 10)    selfref(ig, is)
  forref (16, 4)     forref(ig, if)
  fracrefa (16, 9)   fracrefa(ig, jp)
  fracrefb (16, 5)   fracrefb(ig, jp)
  ka_mo3 (16, 9, 19) ka_mo3(ig, jmo3, indm)
  ccl4   (16,)       ccl4(ig)
  chi_mls (7, 59)    chi_mls(i, j)
Every element of every array must be assigned by exactly one DATA value (checked).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
lines = SRC.read_text(errors="replace").splitlines()


def block(start_pat, end_pat):
    lo = next(i for i, l in enumerate(lines) if re.match(start_pat, l, re.I))
    hi = next(i for i in range(lo + 1, len(lines)) if re.match(end_pat, lines[i], re.I))
    body = []
    for l in lines[lo:hi + 1]:
        l = l.split("!", 1)[0]
        body.append(l.replace("&", " "))
    return " ".join(body)


def parse_data(text, dims):
    """dims: {name: tuple of Fortran extents}. Returns {name: ndarray in Fortran shape}."""
    out = {n: np.zeros(d, dtype=np.float64) for n, d in dims.items()}
    seen = {n: np.zeros(d, dtype=bool) for n, d in dims.items()}
    pat = re.compile(r"\bdata\s+(\w+)\s*\(([^)]*)\)\s*/([^/]*)/", re.I)
    for m in pat.finditer(text):
        name = m.group(1).lower()
        if name not in dims:
            continue
        ext = dims[name]
        parts = [p.replace(" ", "") for p in m.group(2).split(",")]
        assert len(parts) == len(ext), (name, parts)
        ranges = []
        for p, e in zip(parts, ext):
            if p == ":":
                ranges.append(list(range(1, e + 1)))
            elif ":" in p:
                a, b = p.split(":")
                ranges.append(list(range(int(a), int(b) + 1)))
            else:
                ranges.append([int(p)])
        vals = []
        for tok in m.group(3).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "*" in tok:
                rep, v = tok.split("*")
                vals += [float(v.lower().replace("d", "e"))] * int(rep)
            else:
                vals.append(float(tok.lower().replace("d", "e")))
        count = int(np.prod([len(r) for r in ranges]))
        assert len(vals) == count, (name, m.group(2), len(vals), count)
        k = 0
        # Fortran order: first subscript varies fastest
        for combo in itertools.product(*ranges[::-1]):
            idx = tuple(i - 1 for i in combo[::-1])
            assert not seen[name][idx], ("duplicate assignment", name, idx)
            seen[name][idx] = True
            out[name][idx] = vals[k]
            k += 1
    for n in dims:
        assert seen[n].all(), ("unassigned elements", n, int(seen[n].size - seen[n].sum()))
    return out


kgb = block(r"^\s+module module_radlw_kgb05\s", r"^\s+end module module_radlw_kgb05")
tabs = parse_data(kgb, {
    "absa": (16, 585), "absb": (16, 1175), "selfref": (16, 10), "forref": (16, 4),
    "fracrefa": (16, 9), "fracrefb": (16, 5), "ka_mo3": (16, 9, 19), "ccl4": (16,),
})
ref = block(r"^\s+module module_radlw_ref\s", r"^\s+end module module_radlw_ref")
chi = parse_data(ref, {"chi_mls": (7, 59)})
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **tabs, chi_mls=chi["chi_mls"])
print("wrote", OUT)
for n, a in tabs.items():
    print(n, a.shape, "min", float(a.min()), "max", float(a.max()))
print("absa(1,1)=", tabs["absa"][0, 0], "absa(16,585)=", tabs["absa"][15, 584])
print("absb(16,1175)=", tabs["absb"][15, 1174], "ka_mo3(1,1,1)=", tabs["ka_mo3"][0, 0, 0])
print("chi_mls(1,5)=", chi["chi_mls"][0, 4], "chi_mls(2,5)=", chi["chi_mls"][1, 4])
