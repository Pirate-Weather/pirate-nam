"""Our f00 beside the start source's own analysis: 2 m temperature, MSLP and 850 hPa wind, drawn with the Rust
renderer (rw_wrfbatch --cf, render law) on the domain's own frame.

    python tools/source_f00_maps.py --hist runs/x/<history f00>.nc --source-file cases/<case>/raw/ifs/<f000>.grib2 \
        --source-label "IFS HRES 0.1 deg f000 (ECMWF open data, CC BY 4.0)" --out runs/x/f00-vs-source

The source panel is the source's own grid (no interpolation), cropped to the domain plus a margin; the model panel
is the history's own XLAT/XLONG. Both get the same colour table (namgpu/io/maps.py TABLE rows) and frame.
Titles: "WOOF-NAM <res> km | <field>  init ... valid ... (+00h)" and "<source label> | <field>  valid ...".
Host only (no card): the GRIB2 decode is the gpuwm grib2_dump bridge, the renders run in parallel processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# product -> source GRIB2 records (d, c, p, level_type, level_value) and how to make the CF field
SOURCE_RECORDS = {
    "t2m": [("t", (0, 0, 0, "103", 2.0))],
    "mslp": [("p", (0, 3, 0, "101", None))],
    "wind850": [("u", (0, 2, 2, "100", 85000.0)), ("v", (0, 2, 3, "100", 85000.0))],
}


def _source_fields(path: Path, bounds, margin=1.0):
    from namgpu.init.init_grib2_decode import _bridge, inventory
    inv = inventory(path)
    pick = {}
    for prod, recs in SOURCE_RECORDS.items():
        for key, (d, c, p, lt, lv) in recs:
            hits = [r for r in inv if r["dcp"] == (d, c, p) and str(r["level_type"]) == lt
                    and (lv is None or abs(float(r["level_value"]) - lv) < 0.5)]
            if len(hits) != 1:
                raise SystemExit(f"{path.name}: {prod} {key}: {len(hits)} records")
            pick[(prod, key)] = hits[0]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "d"
        subprocess.run([str(_bridge("grib2_dump")), str(path), str(out),
                        *sorted({str(r["index"]) for r in pick.values()})], check=True, capture_output=True)
        meta = {int(r["index"]): r for r in csv.DictReader(
            [l for l in (out / "metadata.tsv").read_text().splitlines() if not l.startswith("#")], delimiter="\t")}
        r0 = next(iter(pick.values()))
        nx, ny = r0["nx"], r0["ny"]
        lat1, lon1 = float(r0["lat1"]), float(r0["lon1"])
        dlat, dlon = 180.0 / (ny - 1), 360.0 / nx
        lats = lat1 - dlat * np.arange(ny)                         # stored north to south (scan 0x00)
        lons = ((lon1 + dlon * np.arange(nx)) + 180.0) % 360.0 - 180.0
        w, e, s, n = bounds
        jj = np.where((lats >= s - margin) & (lats <= n + margin))[0]
        ii = np.where((lons >= w - margin) & (lons <= e + margin))[0]
        ii = ii[np.argsort(lons[ii])]
        raw = {}
        for (prod, key), r in pick.items():
            a = np.fromfile(out / meta[r["index"]]["filename"], "<f8").reshape(ny, nx)
            raw[(prod, key)] = a[np.ix_(jj, ii)]
    lon2, lat2 = np.meshgrid(lons[ii], lats[jj])
    f = {"t2m": ((raw[("t2m", "t")] - 273.15) * 1.8 + 32.0, "degF"),
         "mslp": (raw[("mslp", "p")] / 100.0, "hPa"),
         "ws850": (np.hypot(raw[("wind850", "u")], raw[("wind850", "v")]), "m s-1")}
    return lat2, lon2, f


def main(argv=None):
    from namgpu.io import maps
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", type=Path, required=True)
    ap.add_argument("--source-file", type=Path, required=True)
    ap.add_argument("--source-label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--res-km", default="3")
    a = ap.parse_args(argv)
    valid, start, f = maps._read(a.hist, {"XLAT", "XLONG", "T2", "MSLP", "U", "V", "PRES"})
    prods = ["t2m", "mslp", "wind850"]
    ours, skipped = maps.fields_for(None, (valid, start, f, a.hist), prods)
    lat, lon = f["XLAT"], (f["XLONG"] + 180.0) % 360.0 - 180.0
    bounds = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    slat, slon, theirs = _source_fields(a.source_file, bounds)
    a.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cf-", dir=a.out))
    fh = int(round((valid - start).total_seconds() / 3600))
    when = f"init {start:%Y-%m-%d %HZ}  valid {valid:%Y-%m-%d %HZ} (+{fh:02d}h)"
    model_cf, src_cf = work / "model.nc", work / "source.nc"
    # the renderer's header is "<CF source> | <table title>", cut at ~95 characters: keep each half short
    maps._write_cf(model_cf, lat, lon, valid, ours, f"WOOF-NAM {a.res_km} km")
    maps._write_cf(src_cf, slat, slon, valid, theirs, a.source_label)
    renderer = maps._bridges() / "rw_wrfbatch"
    bstr = f"{bounds[0]:.3f},{bounds[1]:.3f},{bounds[2]:.3f},{bounds[3]:.3f}"
    jobs = []
    for prod in prods:
        var, row = maps.TABLE[prod]
        for tag, cf, title in (("woof-nam", model_cf, f"{row['title']}  {when}"),
                               ("source", src_cf, f"{row['title']}  valid {valid:%Y-%m-%d %HZ}")):
            if tag == "woof-nam" and var not in ours:
                continue
            t = dict(row, title=title)
            table = work / f"table_{prod}_{tag}.json"
            table.write_text(json.dumps([t]))
            jobs.append((prod, tag, cf, var, table))

    def one(job):
        prod, tag, cf, var, table = job
        od = work / f"r_{prod}_{tag}"
        cmd = [renderer, "--cf", cf, "--variable", var, "--out-dir", od, "--valid",
               valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", bstr, "--cf-table", table,
               "--width", "1100", "--height", "1000"]
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
        if p.returncode or not pngs:
            return f"FAILED {prod} {tag}: {(p.stderr or p.stdout).strip()[-300:]}"
        dst = a.out / f"{prod}_{tag}_{start:%Y%m%d%H}_f{fh:03d}.png"
        shutil.move(str(pngs[0]), dst)
        return str(dst)

    with ThreadPoolExecutor(len(jobs)) as ex:
        for r in ex.map(one, jobs):
            print(r)
    for p, why in skipped:
        print(f"skipped {p}: {why}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
