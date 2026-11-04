"""Extract the LW cloud-property coefficient tables from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_cldprlw (lines 910-1400).
Writes namgpu/data/rrtm_lw_cldprlw.npz with C-order arrays laid out as CONTRACT section 1 says:
  absliq1 [NBANDS=16, 58]   absliq1(index, ib)
  absice0 [2]               absice0(m)
  absice1 [5, 2]            absice1(m, ia)  (ia = 1..5, m = 1..2)
  absice2 [16, 43]          absice2(index, ib)
  absice3 [16, 46]          absice3(index, ib)
  ipat    [16]              ipat(ib)
"""
import re
import sys
from pathlib import Path

import numpy as np

from f32_literal import f32_literal

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
text = SRC.read_text(errors="replace").splitlines()
lo = next(i for i, l in enumerate(text) if l.lower().startswith("      module module_radlw_cldprlw"))
hi = next(i for i in range(lo, len(text)) if text[i].lower().startswith("      end module module_radlw_cldprlw"))
body = text[lo:hi]

stmts = {}
i = 0
while i < len(body):
    line = body[i].split("!")[0] if not body[i].lstrip().startswith("!") else ""
    m = re.match(r"\s*data\s+(\w+)\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/", line, re.I)
    m2 = re.match(r"\s*data\s+(\w+)\s*/", line, re.I)
    if m or m2:
        name = (m or m2).group(1).lower()
        col = int((m or [0, 0])[2]) if m else 0
        buf = line.split("/", 1)[1] if "/" in line else ""
        # collect until closing slash
        acc = buf
        while "/" not in acc.replace("/", "", 0) or acc.count("/") < 1:
            i += 1
            nxt = body[i].split("!")[0]
            acc += nxt.replace("&", " ")
        vals_txt = acc.split("/")[0]
        vals = [f32_literal(v) for v in vals_txt.replace("&", " ").split(",") if v.strip()]
        stmts.setdefault(name, {})
        stmts[name].setdefault(col, []).extend(vals)
    i += 1

def cols(name, n, ncol):
    out = np.zeros((ncol, n), dtype=np.float64)
    for c in range(1, ncol + 1):
        v = stmts[name][c]
        assert len(v) == n, (name, c, len(v))
        out[c - 1, :] = v
    return out

absliq1 = cols("absliq1", 58, 16)
absice2 = cols("absice2", 43, 16)
absice3 = cols("absice3", 46, 16)
absice1_flat = stmts["absice1"][0]
absice0 = np.array(stmts["absice0"][0])
assert len(absice1_flat) == 10 and len(absice0) == 2
absice1 = np.array(absice1_flat).reshape((2, 5), order="F").T          # [ia, m]
ipat = np.array([1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int32)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, absliq1=absliq1.astype(np.float32), absice0=absice0.astype(np.float32),
         absice1=absice1.astype(np.float32), absice2=absice2.astype(np.float32),
         absice3=absice3.astype(np.float32), ipat=ipat)
print("wrote", OUT, absliq1.shape, absice2.shape, absice3.shape, absice1.shape, absice0.shape)
print("absliq1(1,1)=", absliq1[0,0], "absliq1(58,16)=", absliq1[15,57], "absice2(43,16)=", absice2[15,42], "absice3(46,16)=", absice3[15,45], "absice1 first", absice1[0])
