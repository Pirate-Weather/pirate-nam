"""Bitwise comparison of two runs' history: every wrfout file, every variable, every value.

    python tools/hist_compare.py <run A out> <run B out> [--json out.json] [--workers N]

Prints one line per hour (variables compared, variables differing, the first differing variable with its count, max
|diff| and global rows) and a total. Exit 0 only when both runs have the same hours and every variable is bitwise
identical. Hours are compared in parallel (one process per hour; the zlib decode is the cost). Used for
MULTICARD-PLAN.md 10.1 (one card unchanged) and 10.2 (1 vs N cards).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def compare_hour(args) -> dict:
    import netCDF4
    a, b, name = args
    with netCDF4.Dataset(Path(a) / name) as da, netCDF4.Dataset(Path(b) / name) as db:
        va, vb = set(da.variables), set(db.variables)
        h = {"file": name, "vars": len(va & vb), "only_a": sorted(va - vb), "only_b": sorted(vb - va), "diff": []}
        for v in sorted(va & vb):
            xa, xb = np.asarray(da[v][:]), np.asarray(db[v][:])
            if xa.shape != xb.shape:
                h["diff"].append({"var": v, "shape": [list(xa.shape), list(xb.shape)]})
                continue
            if xa.dtype.kind == "f":
                same = np.array_equal(xa.view(np.uint32 if xa.itemsize == 4 else np.uint64),
                                      xb.view(np.uint32 if xb.itemsize == 4 else np.uint64))
            else:
                same = np.array_equal(xa, xb)
            if not same:
                d = {"var": v, "n": int(np.count_nonzero(xa != xb))}
                if xa.dtype.kind == "f" and xa.ndim >= 2:
                    d["maxabs"] = float(np.nanmax(np.abs(xa.astype(np.float64) - xb.astype(np.float64))))
                    rows = np.nonzero(np.any((xa != xb).reshape(-1, xa.shape[-2], xa.shape[-1]), axis=(0, 2)))[0]
                    d["rows_1based"] = [int(rows[0]) + 1, int(rows[-1]) + 1] if len(rows) else []
                h["diff"].append(d)
    return h


def compare(a: Path, b: Path, workers: int = 8) -> dict:
    fa = sorted(p.name for p in a.glob("wrfout_d01_*"))
    fb = sorted(p.name for p in b.glob("wrfout_d01_*"))
    rec = {"a": str(a), "b": str(b), "hours_a": len(fa), "hours_b": len(fb), "only_a": sorted(set(fa) - set(fb)),
           "only_b": sorted(set(fb) - set(fa)), "hours": []}
    names = sorted(set(fa) & set(fb))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(names) or 1))) as ex:
        hours = list(ex.map(compare_hour, [(str(a), str(b), n) for n in names]))
    nbad = 0
    for h in hours:
        nbad += len(h["diff"]) + len(h["only_a"]) + len(h["only_b"])
        first = h["diff"][0] if h["diff"] else None
        print(f"{h['file']}: {h['vars']} vars, {len(h['diff'])} differ" + (f"; first {first}" if first else ""),
              flush=True)
    rec["hours"] = hours
    rec["bitwise"] = nbad == 0 and not rec["only_a"] and not rec["only_b"] and len(fa) > 0
    print(f"{len(hours)} hours compared: {'BITWISE IDENTICAL' if rec['bitwise'] else 'DIFFERENT'}", flush=True)
    return rec


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    r = compare(Path(sys.argv[1]), Path(sys.argv[2]), w)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(r, indent=1))
    sys.exit(0 if r["bitwise"] else 1)
