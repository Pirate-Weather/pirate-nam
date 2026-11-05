"""Two-way vs concurrent nest runs: what the feedback did to the parent, hour by hour.

    python tools/twoway_compare.py RUN_TWOWAY RUN_CONCURRENT [--out DIR]

For every common hourly history time: the 12 km parent's max |difference| (two-way minus concurrent) of T, Q, U, V and
MSLP inside the nest's update region and outside it, and per run the min MSLP over water (SM > 0.5) with its position
and the max 10 m wind within 150 km, for the parent and the nest. Writes <out>/twoway-compare.csv and .json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _var(nc, k):
    return np.asarray(nc.variables[k][0], dtype=np.float64)


def _minp(nc):
    lat, lon, p, sm = _var(nc, "XLAT"), _var(nc, "XLONG"), _var(nc, "MSLP"), _var(nc, "SM")
    ws = np.hypot(_var(nc, "U10"), _var(nc, "V10"))
    pw = np.where(sm > 0.5, p, np.inf)
    j, i = np.unravel_index(np.argmin(pw), pw.shape)
    r = 111.2 * np.hypot(lat - lat[j, i], (lon - lon[j, i]) * np.cos(np.radians(lat[j, i])))
    return round(float(p[j, i]) / 100.0, 1), round(float(lat[j, i]), 2), round(float(lon[j, i]), 2), \
        round(float(ws[r <= 150.0].max()), 1)


def main(argv=None) -> int:
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument("twoway", type=Path)
    ap.add_argument("concurrent", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.twoway
    rp = a.twoway / "twoway-receipt.json"
    if rp.is_file():
        reg = json.loads(rp.read_text())["coupler"]["update_region_parent_1based"]     # (I1, I2, J1, J2)
    else:                                       # a run still going: the same region from the nest configfile
        from namgpu import config
        from namgpu.nest import twoway as tw
        nc = config.load(next(a.twoway.glob("nam_configfile_*_in*")))
        pc = config.load(next(p for p in a.twoway.glob("nam_configfile_*") if "_in" not in p.name))
        reg = tw.update_points(int(nc.raw["i_parent_start"]), int(nc.raw["j_parent_start"]),
                               int(nc.raw["parent_child_space_ratio"]), nc.im, nc.jm, nc.lnsh, pc.im, pc.jm)[2]
    rows = []
    for f in sorted((a.twoway / "parent").glob("wrfout_d01_*")):
        g = a.concurrent / "parent" / f.name
        if not g.is_file():
            continue
        x, y = netCDF4.Dataset(f), netCDF4.Dataset(g)
        jm, im = _var(x, "MSLP").shape
        inside = np.zeros((jm, im), bool)
        inside[reg[2] - 1:reg[3], reg[0] - 1:reg[1]] = True
        row = {"valid": f.name.split("_d01_")[1]}
        for k in ("T", "Q", "U", "V", "MSLP"):
            if k not in x.variables:
                continue
            d = np.abs(np.asarray(x.variables[k][0], np.float64) - np.asarray(y.variables[k][0], np.float64))
            d = d.reshape(-1, jm, im)
            row[f"{k}_maxdiff_in"] = round(float(d[:, inside].max()), 4)
            row[f"{k}_maxdiff_out"] = round(float(d[:, ~inside].max()), 4)
        for tag, run in (("2way", a.twoway), ("conc", a.concurrent)):
            for dom in ("parent", "nest"):
                h = run / dom / f.name
                if h.is_file():
                    with netCDF4.Dataset(h) as nc:
                        p, la, lo, w = _minp(nc)
                    row.update({f"{tag}_{dom}_minmslp": p, f"{tag}_{dom}_lat": la, f"{tag}_{dom}_lon": lo,
                                f"{tag}_{dom}_wind10": w})
        x.close()
        y.close()
        rows.append(row)
    if not rows:
        raise SystemExit("no common parent history times")
    with open(out / "twoway-compare.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)
    (out / "twoway-compare.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print({k: v for k, v in r.items() if "maxdiff" in k or "minmslp" in k or k == "valid"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
