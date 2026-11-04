"""Extract the LW band-1 k-tables (module_radlw_kgb01) from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_kgb01 (original RRTM file).
NG01 = 10 here: the NAM build links module_radlw_parameters (radlw_param.f, NG01=10) into
radlw_main_nmmb.f TAUGB01; the _nmmb parameter module (NG01=08) is not used by that routine.
Why this file and not radlw_datatb_nmmb.f: radlw_main_nmmb.f TAUGB01 says `use module_radlw_kgb01`,
and the only module of that name in the NAM build is in radlw_datatb.f. The _nmmb module is
named module_radlw_kgb01_nmmb, has a different layout, and has no ka_mn2/kb_mn2.

Writes namgpu/data/rrtm_lw/kgb01.npz. Each 2-D array is stored as numpy shape (n_index, NG01) in C
order, so element [ind-1, ig-1] is Fortran X(ig, ind). Flat float32 memory is then the Fortran
column-major memory with ig fastest.
  absa [65, 10]  absb [235, 10]  selfref [10, 10] forref [4, 10]
  ka_mn2 [19, 10] kb_mn2 [19, 10] fracrefa [10]    fracrefb [10]

usage: python -I extract_rrtm_lw_kgb01.py <radlw_datatb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NG01, MSA01, MSB01, MSF01, MFR01, MMN01 = 10, 65, 235, 10, 4, 19

lines = SRC.read_text(errors="replace").splitlines()
lo = next(i for i, l in enumerate(lines)
          if re.match(r"^\s*module\s+module_radlw_kgb01\s*!?\s*$", l, re.I))
hi = next(i for i in range(lo, len(lines))
          if re.match(r"^\s*end\s+module\s+module_radlw_kgb01", lines[i], re.I))

seq = {}      # name -> floats in source order (DATA statements for one array appear in order)
cur = None    # name of the DATA statement being read
for raw in lines[lo:hi]:
    text = raw.split("!", 1)[0]
    if cur is None:
        m = re.match(r"^\s*data\s+(\w+)", text, re.I)
        if not m:
            continue
        cur = m.group(1).lower()
        seq.setdefault(cur, [])
        text = text[text.index("/", m.end()) + 1:]     # opening slash
    done = "/" in text
    body = text.split("/", 1)[0]
    for tok in body.replace("&", " ").split(","):
        tok = tok.strip().replace("d", "e").replace("D", "e")
        if tok:
            seq[cur].append(float(tok))
    if done:
        cur = None


def fort2d(name, nind):
    # Fortran X(NG01, nind), ig fastest: flat index (ind-1)*NG01 + (ig-1) -> numpy [ind-1, ig-1]
    vals = np.array(seq[name], dtype=np.float64)
    assert vals.size == NG01 * nind, (name, vals.size, NG01 * nind)
    return vals.reshape(nind, NG01)


out = dict(
    absa=fort2d("absa", MSA01).astype(np.float32),
    absb=fort2d("absb", MSB01).astype(np.float32),
    selfref=fort2d("selfref", MSF01).astype(np.float32),
    forref=fort2d("forref", MFR01).astype(np.float32),
    ka_mn2=fort2d("ka_mn2", MMN01).astype(np.float32),
    kb_mn2=fort2d("kb_mn2", MMN01).astype(np.float32),
    fracrefa=np.array(seq["fracrefa"], dtype=np.float32),
    fracrefb=np.array(seq["fracrefb"], dtype=np.float32),
)
assert out["fracrefa"].size == NG01 and out["fracrefb"].size == NG01
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
for k, v in out.items():
    print(k, v.shape, v.dtype, float(v.min()), float(v.max()))
print("wrote", OUT)
