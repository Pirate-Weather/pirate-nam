"""Extract module_radsw_ref preflog(59) and tref(59) from the NAM Fortran DATA statements.

Source: nam.v4.2.11 (83df4e13c1) phys/radsw_datatb.f, module_radsw_ref (lines 55-115). NOAA US
government work; the NOAA header of that file is the notice. Writes namgpu/data/rrtm_sw/ref.npz with
float32 arrays, the same single-precision values Fortran builds from its decimal literals.

usage: python -I tools/extract_rrtm_sw_ref.py <radsw_datatb.f> <out.npz>
"""
import re
import sys
from pathlib import Path

import numpy as np

NREF = 59
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def grab(text: str, name: str) -> np.ndarray:
    m = re.search(r"data\s+" + name + r"\(:\)\s*/(.*?)/", text, re.S | re.I)
    if m is None:
        raise SystemExit(f"DATA statement for {name} not found")
    body = m.group(1).replace("&", " ")
    vals = [float(v.replace("d", "e").replace("D", "e")) for v in NUM.findall(body)]
    if len(vals) != NREF:
        raise SystemExit(f"{name}: got {len(vals)} values, want {NREF}")
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text(errors="replace")
    preflog = grab(text, "preflog")
    tref = grab(text, "tref")
    if not np.isfinite(preflog).all() or not np.isfinite(tref).all():
        raise SystemExit("non-finite reference values")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, preflog=preflog, tref=tref)
    print("wrote", out, "preflog", preflog[[0, -1]].tolist(), "tref", tref[[0, -1]].tolist())


if __name__ == "__main__":
    main()
