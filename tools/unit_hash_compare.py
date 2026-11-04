"""Compare per-unit owned-row hashes of a one-card run with an N-card run (NAMGPU_UNIT_HASH=1).

    python tools/unit_hash_compare.py <one-card run out> <N-card run out>

The one-card file is <out>/unit_hash_r0.npz (or <out>/rank0/...); the N-card files are <out>/rank<r>/unit_hash_r<r>.npz.
Each record is (step, sequence, unit/cadence[/Xn], field) -> per-row hashes of the owned rows. The N-card rows are put at
their global rows and compared with the one-card rows. Prints the first differing record (unit and field), its first
differing global row, and a count; exit 0 when every record matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load(path: Path):
    z = np.load(path)
    return list(z["keys"]), z["rows"], int(z["jts"])


def _find(out: Path, r: int) -> Path:
    for p in (out / f"rank{r}" / f"unit_hash_r{r}.npz", out / f"unit_hash_r{r}.npz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no unit_hash_r{r}.npz under {out}")


def main(a, b) -> int:
    a, b = Path(a), Path(b)
    k1, rows1, _ = _load(_find(a, 0))
    ranks = sorted(int(p.name[4:]) for p in b.glob("rank*") if p.is_dir() and p.name[4:].isdigit())
    if not ranks:
        ranks = [0]
    parts = [_load(_find(b, r)) for r in ranks]
    pidx = [{k: n for n, k in enumerate(kk)} for kk, _, _ in parts]
    idx1 = {k: n for n, k in enumerate(k1)}
    jmg = rows1.shape[1]
    nbad, first = 0, None
    keys = parts[0][0]
    for n, key in enumerate(keys):
        if key not in idx1:
            print(f"record only in the N-card run: {key}")
            nbad += 1
            continue
        full = np.zeros(jmg, dtype=np.uint64)
        for (kk, rows, jts), ix in zip(parts, pidx):
            v = rows[ix[key]]
            full[jts - 1:jts - 1 + v.shape[0]] = v
        ref = rows1[idx1[key]]
        bad = np.nonzero(full != ref)[0]
        if bad.size:
            nbad += 1
            if first is None:
                first = (key, int(bad[0]) + 1, int(bad.size))
    print(f"{len(keys)} records compared, {nbad} differ")
    if first:
        step, seq, unit, field = first[0].split("|")[0], first[0].split("|")[1], "|".join(first[0].split("|")[2:-1]), \
            first[0].split("|")[-1]
        print(f"FIRST DIFFERENCE: step {int(step)} record {int(seq)} after unit {unit}, field {field}, first global row "
              f"{first[1]} ({first[2]} rows differ)")
        return 1
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
