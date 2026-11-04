"""Extract the RRTM band-10 k-tables from the NAM radlw_datatb.f source into an .npz.

    python -I tools/extract-kgb10.py <path to nam radlw_datatb.f> <out.npz>

The NAM radiation code (radlw_main_nmmb.f, taugb10) reads module_radlw_kgb10, which is defined in
radlw_datatb.f lines 27968-28421 (absa(NG10,MSA10), absb(NG10,MSB10), selfref(NG10,MSF10),
forref(NG10,MFR10), fracrefa(NG10), fracrefb(NG10)). The module of the same name in
radlw_datatb_nmmb.f is a different, unused layout and is not read here.

Fortran data statements fill column-major, so the g index (NG10, fastest) varies first. The flat
value list of each array is therefore reshaped C-order to (second dim, NG10): absa[ind, ig],
absb[ind, ig], selfref[inds, ig], forref[indf, ig], fracrefa[ig], fracrefb[ig].
"""
import re
import sys

import numpy as np

FIRST, LAST = 27968, 28421          # 1-based inclusive line range of module_radlw_kgb10
NG10 = 6
SHAPES = {                          # flat counts from the Fortran declarations
    "absa": (65, NG10), "absb": (235, NG10), "selfref": (10, NG10),
    "forref": (4, NG10), "fracrefa": (NG10,), "fracrefb": (NG10,),
}
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def main(src, out):
    lines = open(src, encoding="latin-1").read().splitlines()[FIRST - 1:LAST]
    text = "\n".join(l.split("!", 1)[0].replace("&", " ") for l in lines)
    vals = {k: [] for k in SHAPES}
    for m in re.finditer(r"\bdata\s+(\w+)\s*(\([^)]*\))?\s*/(.*?)/", text, re.S | re.I):
        name = m.group(1).lower()
        if name not in vals:
            raise SystemExit(f"unexpected data statement for {name}")
        body = m.group(3).replace("\n", " ")
        vals[name].extend(float(n.replace("d", "e").replace("D", "e")) for n in NUM.findall(body))
    arrays = {}
    for name, shape in SHAPES.items():
        n = int(np.prod(shape))
        if len(vals[name]) != n:
            raise SystemExit(f"{name}: found {len(vals[name])} values, expected {n}")
        arrays[name] = np.asarray(vals[name], dtype=np.float32).reshape(shape)
    np.savez(out, **arrays)
    for name in SHAPES:
        a = arrays[name]
        print(f"{name} {a.shape} min={a.min():.6g} max={a.max():.6g} first={a.ravel()[0]:.6g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
