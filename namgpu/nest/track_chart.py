"""Track and intensity chart of storm-following (and static) nests against the best track (an analysis chart).

    python -m namgpu.nest.track_chart --run "WOOF-NAM moving=<run dir>" [--run "static 35.7N=<run dir>" ...] \
        --storm-id SID | --name AMPIL --season 2024 | (default: the storm nearest the first run's start centre) \
        --out <dir> [--cache <ibtracs cache>] [--every 1]

Left: the tracks on a lon/lat map (Natural Earth 10 m coastline from the renderer's basemap), best track in black
with its 6-hourly times; a moving run also draws each window's outline at the hour it took over, in its colour.
Right: track error (km, great circle to the best track at the same hour) and minimum sea-level pressure against
lead time. Numbers: <out>/track-errors.json (per run and lead: fix, error, min MSLP, best track) and the PNG
<out>/track-intensity.png.

Fix per history frame: the 20 km-smoothed MSLP minimum (scipy gaussian), refined by the pressure-deficit centroid
within 40 km; the minimum MSLP is the raw minimum within 60 km of the fix. When a run folder carries the follow
controller's fixes (<run>/../follow-state.json or --fixes), those are drawn instead and named so.
Drawn with Pillow (a package dependency): this is a chart of positions and numbers, not a weather field (the render
law's weather maps are the rw_wrfbatch sheets).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu.io import best_track as bt

UNIT = "track_chart"
COLOURS = ["#1f5fbf", "#d9730d", "#2e8b3d", "#9c3fb5", "#c23b3b", "#5a6f80"]
SMOOTH_KM, CENTROID_KM, MIN_KM = 20.0, 40.0, 60.0


# ------------------------------------------------------------------------------------------ fixes from history

def frame_fix(path: Path) -> dict | None:
    import netCDF4
    from scipy import ndimage
    with netCDF4.Dataset(path) as nc:
        t = datetime.strptime("".join(c.decode() for c in nc.variables["Times"][0]), "%Y-%m-%d_%H:%M:%S")
        s = datetime.strptime(nc.START_DATE, "%Y-%m-%d_%H:%M:%S")
        if "MSLP" not in nc.variables:
            return None
        p = np.asarray(nc.variables["MSLP"][0], np.float64) / 100.0
        lat = np.asarray(nc.variables["XLAT"][0], np.float64)
        lon = np.asarray(nc.variables["XLONG"][0], np.float64)
        dy = float(nc.DY) / 1000.0
        gen = int(nc.FOLLOW_GENERATION) if "FOLLOW_GENERATION" in nc.ncattrs() else None
        land = float(nc.FOLLOW_LAND_HOUR) if "FOLLOW_LAND_HOUR" in nc.ncattrs() else 0.0
    sm = ndimage.gaussian_filter(p, SMOOTH_KM / dy / 2.0, mode="nearest")
    j, i = np.unravel_index(np.argmin(sm), sm.shape)
    d = _dist(lat, lon, lat[j, i], lon[j, i])
    # pressure-deficit weights inside CENTROID_KM, against the 90th percentile of the surrounding 120 km
    w = np.where(d <= CENTROID_KM, np.maximum(np.percentile(sm[d <= 3 * CENTROID_KM], 90) - sm, 0.0), 0.0)
    if w.sum() > 0:
        x, y, z = _xyz(lat, lon)
        cx, cy, cz = (float((w * a).sum() / w.sum()) for a in (x, y, z))
        flat, flon = math.degrees(math.atan2(cz, math.hypot(cx, cy))), math.degrees(math.atan2(cy, cx))
    else:
        flat, flon = float(lat[j, i]), float(lon[j, i])
    near = _dist(lat, lon, flat, flon) <= MIN_KM
    edge = min(j, i, p.shape[0] - 1 - j, p.shape[1] - 1 - i)
    return {"time": t, "lead_h": round((t - s).total_seconds() / 3600.0, 3), "lat": flat, "lon": flon,
            "mslp_min": float(p[near].min()) if near.any() else float(p.min()), "edge_cells": int(edge),
            "generation": gen, "land_hour": land, "outline": _outline(lat, lon)}


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)


def _dist(lat, lon, la0, lo0):
    p1, p2 = np.radians(lat), math.radians(la0)
    dl = np.radians(((lon - lo0 + 180.0) % 360.0) - 180.0)
    c = np.sin(p1) * math.sin(p2) + np.cos(p1) * math.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1.0, 1.0))


def _outline(lat, lon, step=8):
    ring = [(lat[0, i], lon[0, i]) for i in range(0, lat.shape[1], step)]
    ring += [(lat[j, -1], lon[j, -1]) for j in range(0, lat.shape[0], step)]
    ring += [(lat[-1, i], lon[-1, i]) for i in range(lat.shape[1] - 1, -1, -step)]
    ring += [(lat[j, 0], lon[j, 0]) for j in range(lat.shape[0] - 1, -1, -step)]
    return [(float(a), float(b)) for a, b in ring]


def run_fixes(run: Path, every: int = 1) -> list[dict]:
    files = sorted(Path(run).glob("wrfout_d01_*"))
    out = []
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from namgpu import cores
    with ProcessPoolExecutor(max_workers=max(1, min(len(files), cores.usable() // 2, 24)),
                             mp_context=mp.get_context("spawn")) as ex:
        for f in ex.map(frame_fix, files):
            # whole hours only (a moving run also writes frames 15 and 30 min after each move)
            if f is not None and f["lead_h"] == int(f["lead_h"]) and int(f["lead_h"]) % every == 0:
                out.append(f)
    return sorted(out, key=lambda f: f["time"])


# ------------------------------------------------------------------------------------------ coastline

def coastline(bounds, basemap: Path | None):
    """Polylines (lists of (lon, lat)) of the Natural Earth 10 m coastline inside bounds (W, E, S, N; E may pass
    180), read straight from the renderer basemap's shapefile."""
    if basemap is None:
        return []
    shp = Path(basemap) / "natural_earth_10m" / "ne_10m_coastline.shp"
    if not shp.is_file():
        return []
    w, e, s, n = bounds
    data = shp.read_bytes()
    pos, lines = 100, []
    while pos + 8 <= len(data):
        _, clen = struct.unpack(">ii", data[pos:pos + 8])
        rec = data[pos + 8:pos + 8 + clen * 2]
        pos += 8 + clen * 2
        if len(rec) < 44 or struct.unpack("<i", rec[:4])[0] != 3:
            continue
        x0, y0, x1, y1 = struct.unpack("<4d", rec[4:36])
        if y1 < s or y0 > n:
            continue
        for sh in (0.0, 360.0, -360.0):
            if x1 + sh < w or x0 + sh > e:
                continue
            npart, npt = struct.unpack("<2i", rec[36:44])
            parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart])) + [npt]
            pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], "<f8").reshape(npt, 2)
            for a, b in zip(parts, parts[1:]):
                lines.append([(float(x) + sh, float(y)) for x, y in pts[a:b]])
    return lines


# ------------------------------------------------------------------------------------------ chart

def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw(runs: list[tuple[str, list[dict]]], track: dict, out_png: Path, title: str, basemap: Path | None) -> None:
    from PIL import Image, ImageDraw
    W, H = 2200, 1150
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_l, f_s = _font(30), _font(22), _font(18)
    d.text((40, 20), title, fill="black", font=f_t)
    # map panel bounds from every point (lon unwrapped around the first best-track point)
    c = track["points"][0]["lon"] if track["points"] else runs[0][1][0]["lon"]

    def unwrap(lo):
        return c + ((lo - c + 180.0) % 360.0) - 180.0
    t0 = min(f["time"] for _, fx in runs for f in fx)
    t1 = max(f["time"] for _, fx in runs for f in fx)
    bpts = [p for p in track["points"] if t0 - timedelta(hours=6) <= datetime.fromisoformat(p["time"]) <= t1 + timedelta(hours=6)]
    lats = [p["lat"] for p in bpts] + [a for _, fx in runs for f in fx for a, _ in f["outline"]]
    lons = [unwrap(p["lon"]) for p in bpts] + [unwrap(b) for _, fx in runs for f in fx for _, b in f["outline"]]
    s, n = min(lats) - 1.0, max(lats) + 1.0
    w, e = min(lons) - 1.0, max(lons) + 1.0
    px0, py0, pw, ph = 40, 80, 1180, 1030
    kx = math.cos(math.radians((s + n) / 2))
    sc = min(pw / ((e - w) * kx), ph / (n - s))
    ox = px0 + (pw - (e - w) * kx * sc) / 2
    oy = py0 + (ph - (n - s) * sc) / 2

    def xy(la, lo):
        return ox + (unwrap(lo) - w) * kx * sc, oy + (n - la) * sc
    d.rectangle([ox, oy, ox + (e - w) * kx * sc, oy + (n - s) * sc], outline="#888888", fill="#f4f8fb")
    for ln in coastline((w, e, s, n), basemap):
        pts = [xy(y, x) for x, y in ln if s - 1 <= y <= n + 1]
        if len(pts) > 1:
            d.line(pts, fill="#7a7a7a", width=2)
    # clip the coastline to the map box (the polylines run past it), then the box outline and the title
    mx1, my1 = ox + (e - w) * kx * sc, oy + (n - s) * sc
    for r in ((0, 0, 1260, oy - 1), (0, my1 + 1, 1260, H), (0, 0, ox - 1, H), (mx1 + 1, 0, 1260, H)):
        d.rectangle(r, fill="white")
    d.rectangle([ox, oy, mx1, my1], outline="#888888")
    d.text((40, 20), title, fill="black", font=f_t)
    # graticule labels
    for la in range(math.ceil(s), math.floor(n) + 1, 2):
        d.text((ox + 4, xy(la, w)[1] - 10), f"{abs(la)}{'N' if la >= 0 else 'S'}", fill="#666666", font=f_s)
    for lo in range(math.ceil(w), math.floor(e) + 1, 2):
        lw = ((lo + 180) % 360) - 180
        d.text((xy(s, lo)[0] - 14, oy + (n - s) * sc - 24), f"{abs(lw)}{'E' if lw >= 0 else 'W'}", fill="#666666",
               font=f_s)
    # each run's windows: the outline at the first frame of every window it had (a static nest has one)
    for k, (label, fx) in enumerate(runs):
        col = COLOURS[k % len(COLOURS)]
        seen = []
        for f in fx:
            key = (round(f["outline"][0][0], 3), round(f["outline"][0][1], 3))
            if key in seen:
                continue
            seen.append(key)
            ring = [xy(a, b) for a, b in f["outline"]]
            d.line(ring + [ring[0]], fill=col, width=2 if len(seen) > 1 or k == 0 else 1)
            d.text((ring[0][0] + 6, ring[0][1] - 24 - 22 * k), f"window from +{f.get('land_hour', 0.0):g} h", fill=col, font=f_s)
    # best track
    bp = [xy(p["lat"], p["lon"]) for p in bpts]
    if len(bp) > 1:
        d.line(bp, fill="black", width=4)
    for p in bpts:
        t = datetime.fromisoformat(p["time"])
        x, y = xy(p["lat"], p["lon"])
        if t.hour % 6 == 0:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill="black")
            if t.hour % 12 == 0:
                d.text((x + 9, y - 10), f"{t:%dT%HZ}", fill="black", font=f_s)
    for k, (label, fx) in enumerate(runs):
        col = COLOURS[k % len(COLOURS)]
        pts = [xy(f["lat"], f["lon"]) for f in fx]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        for f, (x, y) in zip(fx, pts):
            if int(round(f["lead_h"])) % 6 == 0:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
    # legend
    ly = oy + 10
    for k, (label, _) in enumerate([("best track " + track_label(track), None)] + list(runs)):
        col = "black" if k == 0 else COLOURS[(k - 1) % len(COLOURS)]
        d.line([ox + 14, ly + 12, ox + 54, ly + 12], fill=col, width=4)
        d.text((ox + 62, ly), label, fill="black", font=f_l)
        ly += 30
    # right panels: track error and min MSLP against lead
    errs = errors(runs, track)
    leads = sorted({e_["lead_h"] for r in errs.values() for e_ in r})
    if not leads:
        img.save(out_png)
        return
    L0, L1 = 0.0, max(leads)

    def panel(y0, hgt, key, ylab, best_key=None):
        x0, wid = 1300, 860
        vals = [e_[key] for r in errs.values() for e_ in r if e_.get(key) is not None]
        if best_key:
            vals += [e_[best_key] for r in errs.values() for e_ in r if e_.get(best_key) is not None]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        pad = max(1.0, 0.08 * (hi - lo))
        lo, hi = (0.0 if key == "error_km" else lo - pad), hi + pad
        d.rectangle([x0, y0, x0 + wid, y0 + hgt], outline="#888888")
        d.text((x0, y0 - 30), ylab, fill="black", font=f_l)

        def pxy(ld, v):
            return x0 + (ld - L0) / max(1e-6, L1 - L0) * wid, y0 + hgt - (v - lo) / max(1e-6, hi - lo) * hgt
        for k in range(5):
            v = lo + (hi - lo) * k / 4
            d.text((x0 - 70, pxy(L0, v)[1] - 10), f"{v:.0f}", fill="#555555", font=f_s)
        for ld in range(0, int(L1) + 1, 6):
            d.text((pxy(ld, lo)[0] - 10, y0 + hgt + 6), f"{ld}", fill="#555555", font=f_s)
        if best_key:
            b = [(e_["lead_h"], e_[best_key]) for e_ in next(iter(errs.values())) if e_.get(best_key) is not None]
            if len(b) > 1:
                d.line([pxy(*q) for q in b], fill="black", width=4)
        for k, (label, r) in enumerate(errs.items()):
            pts = [pxy(e_["lead_h"], e_[key]) for e_ in r if e_.get(key) is not None]
            if len(pts) > 1:
                d.line(pts, fill=COLOURS[k % len(COLOURS)], width=3)
    panel(120, 400, "error_km", "track error vs best track (km)")
    panel(640, 400, "mslp_min", "minimum sea-level pressure (hPa); best track black", best_key="best_pres_hpa")
    d.text((1300, 1080), "lead time (h)", fill="black", font=f_l)
    img.save(out_png)


def track_label(track: dict) -> str:
    ags = sorted({p["agency"] for p in track["points"]})
    return f"{track['name']} {track['season']} (IBTrACS {track['sid']}, {'/'.join(ags)})"


def errors(runs, track) -> dict:
    out = {}
    for label, fx in runs:
        r = []
        for f in fx:
            b = bt.interp(track, f["time"])
            row = {"time": f"{f['time']:%Y-%m-%dT%H:%MZ}", "lead_h": f["lead_h"], "lat": round(f["lat"], 3),
                   "lon": round(f["lon"], 3), "mslp_min": round(f["mslp_min"], 1), "generation": f.get("generation"),
                   "edge_cells": f["edge_cells"]}
            if b is not None:
                row.update(error_km=round(bt.distance_km(f["lat"], f["lon"], b["lat"], b["lon"]), 1),
                           best_lat=round(b["lat"], 3), best_lon=round(b["lon"], 3), best_pres_hpa=b.get("pres_hpa"),
                           best_agency=b.get("agency"))
            r.append(row)
        out[label] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=DIR (a folder of wrfout_d01_* files)")
    ap.add_argument("--storm-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "woof-nam" / "ibtracs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--every", type=int, default=1, help="fix every N hours (default 1)")
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        label, _, d = spec.partition("=")
        fx = run_fixes(Path(d), a.every)
        if not fx:
            raise SystemExit(f"{UNIT}: {d}: no history frames with MSLP")
        runs.append((label, fx))
    f0 = runs[0][1][0]
    tr = bt.resolve(a.cache, a.storm_id, a.name, a.season, near=(f0["lat"], f0["lon"]), at=f0["time"]) \
        if not (a.storm_id or a.name) else bt.resolve(a.cache, a.storm_id, a.name, a.season)
    try:
        from namgpu.io import maps
        basemap = maps.basemap_dir(maps.renderer())
    except (SystemExit, Exception):  # noqa: BLE001  -- no renderer: the chart is drawn without a coastline
        basemap = None
    a.out.mkdir(parents=True, exist_ok=True)
    title = a.title or (f"WOOF-NAM 3 km storm track and intensity: {track_label(tr)}, init "
                        f"{runs[0][1][0]['time'] - timedelta(hours=runs[0][1][0]['lead_h']):%Y-%m-%d %HZ}")
    draw(runs, tr, a.out / "track-intensity.png", title, basemap)
    errs = errors(runs, tr)
    summ = {lab: {"mean_error_km": round(float(np.mean([e["error_km"] for e in r if "error_km" in e])), 1)
                  if any("error_km" in e for e in r) else None,
                  "error_km_by_6h": {str(int(e["lead_h"])): e.get("error_km") for e in r if int(e["lead_h"]) % 6 == 0},
                  "mslp_min_by_6h": {str(int(e["lead_h"])): e["mslp_min"] for e in r if int(e["lead_h"]) % 6 == 0}}
            for lab, r in errs.items()}
    (a.out / "track-errors.json").write_text(json.dumps({"storm": {k: tr[k] for k in ("sid", "name", "season", "basin")},
                                                           "licence": tr["licence"], "summary": summ, "runs": errs},
                                                          indent=1))
    print(json.dumps(summ, indent=1))
    print(f"{UNIT}: {a.out / 'track-intensity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
