"""Best-track referee row for storm-following nests: IBTrACS v04r01 (NOAA NCEI, public domain).

    python -m namgpu.io.best_track --storm-id 2024225N22142 [--cache DIR]       # by IBTrACS SID
    python -m namgpu.io.best_track --name AMPIL --season 2024
    python -m namgpu.io.best_track --near 29.6,140.8 --time 2024081512          # the storm nearest a point

A row table, not a code path per agency or basin (the arbitrary acceptance test):
  BASINS   basin -> the geographic boxes it is looked for in (for --near) and the agency columns read, in
           order. Each time takes the first agency with a position there (the WMO RSMC first: JMA "TOKYO" for
           the western Pacific, RSMC Wellington and then BoM in the south Pacific, ...), and its pressure and
           wind from the same agency where it has them (else the WMO_* columns). The agency used at each time is
           kept in the output, so a track that changes agency (BoM -> Wellington) says so.
  FILE     one CSV per basin, ibtracs.<BASIN>.list.v04r01.csv (35-115 MB), fetched once into the cache with 16
           parallel byte ranges and reused while it is younger than CACHE_DAYS (NCEI rewrites it weekly).
Output: {"sid", "name", "season", "basin", "points": [{"time", "lat", "lon", "pres_hpa", "wind_kt", "agency",
"nature"}]} at the agency's own times, plus interp(track, t) for hourly comparisons (linear in time on lat,
lon and pressure; never past the ends).
Licence: IBTrACS is a NOAA NCEI product, US government work, public domain (Knapp et al. 2010, BAMS 91, 363-376;
IBTrACS v04r01, doi:10.25921/82ty-9e16).
Host work: the fetch is I/O in a thread pool; the parse reads only the lines carrying the storm's SID, name or
date (a byte search before the CSV parse), a few seconds on one core.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

UNIT = "best_track"
URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
       "v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv")
LICENCE = ("IBTrACS v04r01 (NOAA NCEI, doi:10.25921/82ty-9e16), US government work, public domain; "
           "agency positions as published by the WMO RSMCs/TCWCs")
CREDIT = "best track: IBTrACS v04r01 (NOAA NCEI)"
CACHE_DAYS = 7

# basin -> geographic boxes (lat0, lat1, lon0, lon1; lon -180..180) and the agency column prefixes, in order
BASINS = {
    "WP": {"boxes": [(0.0, 60.0, 100.0, 180.0)], "agencies": ["TOKYO", "WMO", "USA", "CMA", "HKO", "KMA"]},
    "SP": {"boxes": [(-60.0, 0.0, 135.0, 180.0), (-60.0, 0.0, -180.0, -120.0)],
           "agencies": ["WELLINGTON", "BOM", "NADI", "WMO", "USA"]},
    "SI": {"boxes": [(-60.0, 0.0, 10.0, 135.0)], "agencies": ["REUNION", "BOM", "WMO", "USA"]},
    "NI": {"boxes": [(0.0, 40.0, 40.0, 100.0)], "agencies": ["NEWDELHI", "WMO", "USA"]},
    "EP": {"boxes": [(0.0, 60.0, -180.0, -80.0)], "agencies": ["USA", "WMO"]},
    "NA": {"boxes": [(0.0, 70.0, -110.0, 10.0)], "agencies": ["USA", "WMO"]},
    "SA": {"boxes": [(-60.0, 0.0, -70.0, 10.0)], "agencies": ["USA", "WMO"]},
}


def _get(url: str, rng: str | None = None, timeout=300) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError:
            if k == 4:
                raise
            time.sleep(2.0 * (k + 1))
    raise AssertionError("unreachable")


def fetch_basin(basin: str, cache: Path, log=print) -> Path:
    """The basin CSV in the cache (fetched in 16 byte ranges when missing or older than CACHE_DAYS)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"ibtracs.{basin}.list.v04r01.csv"
    if dest.is_file() and time.time() - dest.stat().st_mtime < CACHE_DAYS * 86400:
        return dest
    url = URL.format(basin=basin)
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        n = int(r.headers["Content-Length"])
    step = -(-n // 16)
    ranges = [(a, min(n, a + step) - 1) for a in range(0, n, step)]
    with ThreadPoolExecutor(16) as ex:
        parts = list(ex.map(lambda r: _get(url, f"{r[0]}-{r[1]}"), ranges))
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(b"".join(parts))
    if tmp.stat().st_size != n:
        raise OSError(f"{UNIT}: {url}: got {tmp.stat().st_size} of {n} bytes")
    tmp.replace(dest)
    log(f"{UNIT}: {dest.name} {n / 1e6:.0f} MB in {time.perf_counter() - t0:.1f} s (16 ranges)")
    return dest


def basins_at(lat: float, lon: float) -> list[str]:
    lo = ((lon + 180.0) % 360.0) - 180.0
    return [b for b, row in BASINS.items() if any(a <= lat <= c and d <= lo <= e for a, c, d, e in row["boxes"])]


def _f(x):
    """A number from a CSV cell, or None (blank, NaN, or IBTrACS' negative no-value marks such as wind -1)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _pos(x):
    v = _f(x)
    return v if (v is not None and v > 0) else None


def _rows(path: Path, want: list[bytes]):
    """Header + the CSV rows whose line holds any of the byte strings in want (byte search, then csv)."""
    with open(path, "rb") as f:
        head = f.readline()
        f.readline()                                  # the units line
        keep = [ln for ln in f if any(w in ln for w in want)]
    rd = csv.reader(io.StringIO((head + b"".join(keep)).decode("utf-8", "replace")))
    cols = [c.strip() for c in next(rd)]
    return cols, [dict(zip(cols, (v.strip() for v in r))) for r in rd]


def track(path: Path, basin: str, sid: str | None = None, name: str | None = None,
          season: int | None = None) -> dict:
    """{"sid", "name", "season", "basin", "points": [...]} of one storm from a basin CSV."""
    if not sid and not name:
        raise ValueError(f"{UNIT}: give a SID or a name")
    want = [sid.encode()] if sid else [f",{name.upper()},".encode()]
    _, rows = _rows(path, want)
    rows = [r for r in rows if (sid and r["SID"] == sid) or
            (not sid and r["NAME"].upper() == name.upper() and (season is None or int(r["SEASON"]) == int(season)))]
    if not rows:
        raise SystemExit(f"{UNIT}: no IBTrACS {basin} storm {sid or name}{'' if season is None else f' {season}'}")
    sids = sorted({r["SID"] for r in rows})
    if len(sids) > 1:
        raise SystemExit(f"{UNIT}: {name} {season} matches several storms {sids}: give --storm-id")
    pts = []
    for r in rows:
        for a in BASINS[basin]["agencies"]:
            la, lo = _f(r.get(f"{a}_LAT")), _f(r.get(f"{a}_LON"))
            if la is None or lo is None:
                continue
            pres = _pos(r.get(f"{a}_PRES")) or _pos(r.get("WMO_PRES"))
            wind = _pos(r.get(f"{a}_WIND")) or _pos(r.get("WMO_WIND"))
            pts.append({"time": r["ISO_TIME"][:16].replace(" ", "T"), "lat": la, "lon": lo, "pres_hpa": pres,
                        "wind_kt": wind, "agency": a, "nature": r.get("NATURE")})
            break
    return {"sid": sids[0], "name": rows[0]["NAME"], "season": int(rows[0]["SEASON"]), "basin": basin,
            "points": pts, "licence": LICENCE, "credit": CREDIT}


def distance_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, c)))


def nearest_storm(path: Path, lat: float, lon: float, t: datetime, max_km: float = 500.0) -> str | None:
    """SID of the storm whose position within 3 h of time t is nearest (lat, lon), within max_km."""
    days = {(t + timedelta(hours=dh)).strftime("%Y-%m-%d").encode() for dh in (-3, 0, 3)}
    _, rows = _rows(path, sorted(days))
    best = None
    for r in rows:
        try:
            rt = datetime.strptime(r["ISO_TIME"][:16], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if abs((rt - t).total_seconds()) > 3 * 3600:
            continue
        la, lo = _f(r.get("LAT")), _f(r.get("LON"))
        if la is None or lo is None:
            continue
        d = distance_km(lat, lon, la, lo)
        if d <= max_km and (best is None or d < best[0]):
            best = (d, r["SID"])
    return best[1] if best else None


def interp(tr: dict, t: datetime) -> dict | None:
    """Best track at time t: linear between the two agency times around it (lon across the dateline); None
    outside the track or across a gap of more than 12 h."""
    pts = [(datetime.fromisoformat(p["time"]), p) for p in tr["points"]]
    for (ta, a), (tb, b) in zip(pts, pts[1:]):
        if ta <= t <= tb and (tb - ta) <= timedelta(hours=12):
            w = (t - ta).total_seconds() / max(1.0, (tb - ta).total_seconds())
            dlon = ((b["lon"] - a["lon"] + 180.0) % 360.0) - 180.0
            pa, pb = a.get("pres_hpa"), b.get("pres_hpa")
            pres = (pa + w * (pb - pa)) if (pa is not None and pb is not None) else (pa if w < 0.5 else pb)
            return {"time": t.isoformat(timespec="minutes"), "lat": a["lat"] + w * (b["lat"] - a["lat"]),
                    "lon": ((a["lon"] + w * dlon + 180.0) % 360.0) - 180.0, "pres_hpa": pres,
                    "agency": a["agency"] if w < 0.5 else b["agency"]}
    if pts and pts[-1][0] == t:
        return dict(pts[-1][1])
    return None


def resolve(cache: Path, storm_id: str | None = None, name: str | None = None, season: int | None = None,
            near: tuple[float, float] | None = None, at: datetime | None = None, log=print) -> dict:
    """One storm's track by SID, name + season, or the storm nearest a point at a time."""
    if storm_id:
        # SID = YYYYDDDHLLLNNN (year, day of year, hemisphere, lat and lon of the first fix): look in the basins
        # whose boxes hold that first point first, then in every basin
        try:
            hemi = storm_id[7]
            la = int(storm_id[8:10]) * (1 if hemi == "N" else -1)
            lo = int(storm_id[10:13])
            first = basins_at(la, lo)
        except (IndexError, ValueError):
            first = []
        for b in first + [b for b in BASINS if b not in first]:
            try:
                return track(fetch_basin(b, cache, log), b, sid=storm_id)
            except SystemExit:
                continue
        raise SystemExit(f"{UNIT}: SID {storm_id} is in no IBTrACS basin file")
    if near is not None:
        if at is None:
            raise ValueError(f"{UNIT}: near= needs at=")
        for b in basins_at(*near):
            p = fetch_basin(b, cache, log)
            sid = nearest_storm(p, near[0], near[1], at)
            if sid:
                return track(p, b, sid=sid)
        raise SystemExit(f"{UNIT}: no IBTrACS storm within 500 km of {near} at {at:%Y-%m-%d %H}Z")
    for b in BASINS:
        try:
            return track(fetch_basin(b, cache, log), b, name=name, season=season)
        except SystemExit:
            continue
    raise SystemExit(f"{UNIT}: no IBTrACS storm {name} {season}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--near", default=None, help="LAT,LON")
    ap.add_argument("--time", default=None, help="YYYYMMDDHH (with --near)")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "woof-nam" / "ibtracs")
    ap.add_argument("--out", type=Path, default=None, help="write the track JSON here")
    a = ap.parse_args(argv)
    near = tuple(float(x) for x in a.near.split(",")) if a.near else None
    at = datetime.strptime(a.time, "%Y%m%d%H") if a.time else None
    tr = resolve(a.cache, a.storm_id, a.name, a.season, near, at)
    if a.out:
        a.out.write_text(json.dumps(tr, indent=1))
    print(json.dumps({k: tr[k] for k in ("sid", "name", "season", "basin")}), f"{len(tr['points'])} points")
    for p in tr["points"]:
        print(p["time"], f"{p['lat']:7.2f} {p['lon']:8.2f}", p.get("pres_hpa"), p.get("wind_kt"), p["agency"],
              p.get("nature"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
