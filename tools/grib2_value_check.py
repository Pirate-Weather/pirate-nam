"""Value check of every record-table row: an archived NAM awphys file and ours, decoded (gpuwm's grib2_dump) on the
same grid 218. For each table record with a source it prints min/max of both, the mean and RMS of ours minus NAM,
and the correlation. A record whose correlation is under 0.7, or whose mean difference is over half of NAM's own
spatial standard deviation, is flagged CHECK: that is how a mis-scaled or mis-levelled record shows (a forecast
that differs from NAM's only by weather keeps a high correlation and a small bias on a 12 km CONUS grid; 1 h
precipitation in the first hours sits near 0.9). Two identical constant fields (zero precipitation at 0 h) pass.

    python tools/grib2_value_check.py <archived awphysFF.grib2> <our nam.tHHz.awphysFF.tm00.grib2> [--table T]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.bridges import bridges_dir  # noqa: E402
from namgpu.io import products as P  # noqa: E402
from namgpu.products import spec  # noqa: E402


def key(r):
    return (r["key"], r.get("t1"), float(r.get("v1") or 0.0), r.get("t2") if r.get("t2") != 255 else None,
            None if r.get("t2") in (None, 255) else float(r.get("v2") or 0.0),
            r.get("stat") if r.get("pdt") == 8 else None)


def dump(path, idxs, work):
    d = os.path.join(work, os.path.basename(path))
    subprocess.run([str(bridges_dir() / "grib2_dump"), str(path), d, *map(str, idxs)], check=True,
                   capture_output=True)
    out = {}
    for line in open(os.path.join(d, "metadata.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        out[int(f[0])] = np.fromfile(os.path.join(d, f[-1]), "<f8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("ours", type=Path)
    ap.add_argument("fhr", nargs="?", help="ignored (kept for older callers)")
    ap.add_argument("--table", type=Path, default=None)
    a = ap.parse_args(argv)
    ref, mine = P.scan(a.reference), P.scan(a.ours)
    ref_at, mine_at = {}, {}
    for i, r in enumerate(ref):
        ref_at.setdefault(key(r), []).append(i)
    for i, r in enumerate(mine):
        mine_at.setdefault(key(r), []).append(i)
    occ = {}
    pairs = []
    for t in spec.read(a.table or spec.TABLE):
        k = spec.inventory_key(t)
        n = occ.get(k, 0)
        occ[k] = n + 1
        if t["source"] in ("", "-"):
            continue
        ia = ref_at.get(k, [])[n] if n < len(ref_at.get(k, [])) else None
        ib = mine_at.get(k, [])[n] if n < len(mine_at.get(k, [])) else None
        if ia is not None and ib is not None:
            pairs.append((t, ia, ib))
        elif ia is not None or ib is not None:
            print(f"rec {t['rec']} {t['name']} {t['level']}: NAM {'yes' if ia is not None else 'no'}, "
                  f"ours {'yes' if ib is not None else 'no'}")
    work = tempfile.mkdtemp(prefix="nam-vals-")
    flagged = 0
    try:
        da = dump(a.reference, sorted({p[1] for p in pairs}), work)
        db = dump(a.ours, sorted({p[2] for p in pairs}), work)
        print(f"{'rec':>5s} {'record':28s} {'NAM min':>10s} {'NAM max':>10s} {'ours min':>10s} {'ours max':>10s} "
              f"{'mean(o-N)':>10s} {'rms(o-N)':>9s} {'corr':>6s}")
        for t, ia, ib in pairs:
            x, y = da[ia], db[ib]
            ok = np.isfinite(x) & np.isfinite(y)
            d = y[ok] - x[ok]
            sx = float(np.std(x[ok]))
            corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if sx > 0 and np.std(y[ok]) > 0 else float("nan")
            bias = float(d.mean())
            same_const = sx == 0 and float(np.std(y[ok])) == 0 and abs(bias) == 0
            check = not same_const and (not (corr >= 0.7) or (sx > 0 and abs(bias) > 0.5 * sx))
            flagged += check
            lab = f"{t['name']} {t['level']}" + (f" {t['level2']}" if t["level2"] not in ("", "-") else "")
            print(f"{t['rec']:>5s} {lab[:28]:28s} {np.nanmin(x):10.4g} {np.nanmax(x):10.4g} {np.nanmin(y):10.4g} "
                  f"{np.nanmax(y):10.4g} {bias:10.4g} {np.sqrt((d * d).mean()):9.4g} {corr:6.3f}"
                  + ("  CHECK" if check else ""))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(pairs)} records compared, {flagged} flagged CHECK")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
