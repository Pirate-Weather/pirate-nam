"""Extract the RRTM LW Planck-flux table totplnk from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radlw_datatb.f, module_radlw_avplank (lines 58-728), the module
radlw_main_nmmb.f links. (radlw_datatb_nmmb.f's module_radlw_avplank_nmmb is a different table.)
Writes namgpu/data/rrtm_lw_avplank.npz:
  totplnk [NBANDS=16, NPLNK=181] float32, totplnk[b, k] = Fortran totplnk(k+1, b+1)
  (CONTRACT section 1: Fortran (i,j) -> CuPy [j, i]; here totplnk(NPLNK, NBANDS) -> [NBANDS, NPLNK]).
"""
import re
import sys
from pathlib import Path

import numpy as np

NPLNK, NBANDS = 181, 16
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LO, HI = 58, 728

lines = SRC.read_text(errors="replace").splitlines()
body = lines[LO - 1:HI]
assert body[0].lower().startswith("      module module_radlw_avplank"), body[0]
assert body[-1].lower().startswith("      end module module_radlw_avplank"), body[-1]

# Drop comments and continuation markers; data statements may span lines.
flat = "\n".join(l.split("!")[0].replace("&", " ") for l in body)
stmts = re.findall(r"data\s+totplnk\s*\(\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*\)\s*/([^/]*)/",
                   flat, re.I)
assert len(stmts) == 4 * NBANDS, len(stmts)  # 1:50, 51:100, 101:150, 151:181 per band

tab = np.full((NBANDS, NPLNK), np.nan, dtype=np.float64)
count = np.zeros((NBANDS, NPLNK), dtype=np.int32)
for lo_s, hi_s, band_s, vals_s in stmts:
    lo, hi, band = int(lo_s), int(hi_s), int(band_s)
    vals = [float(v.strip().replace("D", "e").replace("d", "e"))
            for v in vals_s.split(",") if v.strip()]
    assert len(vals) == hi - lo + 1, (lo, hi, band, len(vals))
    assert 1 <= band <= NBANDS and 1 <= lo <= hi <= NPLNK
    tab[band - 1, lo - 1:hi] = vals
    count[band - 1, lo - 1:hi] += 1

assert (count == 1).all(), "every totplnk element must be set exactly once"
assert np.isfinite(tab).all() and (tab > 0).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, totplnk=tab.astype(np.float32))
print("wrote", OUT, tab.shape)
print("totplnk(1,1)=", tab[0, 0], "totplnk(181,1)=", tab[0, -1],
      "totplnk(1,16)=", tab[15, 0], "totplnk(181,16)=", tab[15, -1])
