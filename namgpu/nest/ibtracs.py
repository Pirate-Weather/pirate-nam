"""IBTrACS v04r01 best tracks: the referee for storm-following nests, and the door's --storm-id start position.

Source: NOAA NCEI International Best Track Archive for Climate Stewardship v04r01 (public domain, US Government
work; Knapp et al. 2010, doi:10.1175/2009BAMS2755.1; data doi:10.25921/82ty-9e16). One CSV per basin
(ibtracs.<BASIN>.list.v04r01.csv), fetched once into the cache and read with the csv module (no pandas).

Rows are per storm and time. LAT/LON are IBTrACS's own combined position (the WMO agency's where it has one,
interpolated 3-hourly rows in between). The central pressure is WMO_PRES when the WMO agency wrote one at that
time, else the WMO agency's own column (TOKYO_PRES for RSMC Tokyo, WELLINGTON_PRES / BOM_PRES / NADI_PRES in the
South Pacific, ...), else USA_PRES; the column used is recorded per point.

Only the storm's start position enters a run (--storm-id); the rest of the track is the referee, read after.
"""
from __future__ import annotations

import csv
import os
import urllib.request
from datetime import datetime
from pathlib import Path

URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/"
       "access/csv/ibtracs.{basin}.list.v04r01.csv")
LICENCE = "NOAA NCEI IBTrACS v04r01, US Government work (public domain), doi:10.25921/82ty-9e16"
BASINS = ("WP", "SP", "SI", "NI", "EP", "NA", "SA")
# the WMO agency (IBTrACS WMO_AGENCY value) -> its own pressure column
AGENCY_PRES = {"tokyo": "TOKYO_PRES", "wellington": "WELLINGTON_PRES", "bom": "BOM_PRES", "nadi": "NADI_PRES",
               "reunion": "REUNION_PRES", "newdelhi": "NEWDELHI_PRES", "hurdat_atl": "USA_PRES",
               "hurdat_epa": "USA_PRES", "cphc": "USA_PRES", "atcf": "USA_PRES"}
PRES_FALLBACK = ("WELLINGTON_PRES", "TOKYO_PRES", "BOM_PRES", "NADI_PRES", "CMA_PRES", "HKO_PRES", "USA_PRES")


def cache_dir() -> Path:
    from namgpu import paths
    d = Path(os.environ.get("NAMGPU_IBTRACS_DIR", "")) if os.environ.get("NAMGPU_IBTRACS_DIR") else \
        Path(os.environ.get("NAMGPU_CASES", str(paths.work_root() / "cases"))) / "ibtracs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def basin_file(basin: str) -> Path:
    basin = basin.upper()
    if basin not in BASINS:
        raise SystemExit(f"IBTrACS basin {basin!r}: one of {BASINS}")
    f = cache_dir() / f"ibtracs.{basin}.list.v04r01.csv"
    if not f.is_file() or f.stat().st_size < 1_000_000:
        tmp = f.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(basin=basin), tmp)
        tmp.rename(f)
    return f


def _f(x):
    x = (x or "").strip()
    try:
        return float(x)
    except ValueError:
        return None


def basin_of_sid(sid: str) -> list[str]:
    """IBTrACS SIDs are YYYYDDDHLLLLL (genesis day, hemisphere, position): N -> WP/EP/NA/NI, S -> SP/SI/SA. The
    storm's basin is not in the SID, so try the hemisphere's basins in order of size."""
    if len(sid) != 13 or sid[7] not in "NS":
        raise SystemExit(f"--storm-id {sid!r}: an IBTrACS SID is 13 characters, e.g. 2024225N22135 (Ampil 2024)")
    return ["WP", "EP", "NA", "NI"] if sid[7] == "N" else ["SP", "SI", "SA"]


def track(sid: str, basins=None) -> dict:
    """{'sid', 'name', 'basin', 'points': [{'time', 'lat', 'lon', 'pres_hpa', 'pres_col', 'wind_kt', 'nature',
    'agency', 'track_type'}]} for one storm, every time IBTrACS lists (3-hourly)."""
    for b in (basins or basin_of_sid(sid)):
        f = basin_file(b)
        pts, name = [], None
        with open(f, newline="") as fh:
            r = csv.reader(fh)
            head = next(r)
            next(r)                          # the units row
            ix = {k: i for i, k in enumerate(head)}
            for row in r:
                if row[0] != sid:
                    if pts:
                        break                # rows of one storm are contiguous
                    continue
                name = row[ix["NAME"]]
                ag = row[ix["WMO_AGENCY"]].strip()
                p, col = _f(row[ix["WMO_PRES"]]), "WMO_PRES"
                if p is None and ag in AGENCY_PRES:
                    p, col = _f(row[ix[AGENCY_PRES[ag]]]), AGENCY_PRES[ag]
                if p is None:
                    for c in PRES_FALLBACK:
                        if c in ix and _f(row[ix[c]]) is not None:
                            p, col = _f(row[ix[c]]), c
                            break
                pts.append({"time": datetime.strptime(row[ix["ISO_TIME"]], "%Y-%m-%d %H:%M:%S").isoformat(),
                            "lat": _f(row[ix["LAT"]]), "lon": _f(row[ix["LON"]]), "pres_hpa": p,
                            "pres_col": col if p is not None else None, "wind_kt": _f(row[ix["WMO_WIND"]]),
                            "nature": row[ix["NATURE"]], "agency": ag or None,
                            "track_type": row[ix["TRACK_TYPE"]]})
        if pts:
            return {"sid": sid, "name": name, "basin": b, "licence": LICENCE, "points": pts}
    raise SystemExit(f"--storm-id {sid}: not in IBTrACS v04r01 basins {basins or basin_of_sid(sid)}")


def at(trk: dict, when: datetime) -> dict | None:
    """The best-track point at `when`, linear in time between the two listed points around it (positions on the
    sphere through lat/lon with the longitude unwrapped; pressure linear), or None outside the track."""
    pts = trk["points"]
    ts = [datetime.fromisoformat(p["time"]) for p in pts]
    for i in range(len(ts) - 1):
        if ts[i] <= when <= ts[i + 1]:
            a, b = pts[i], pts[i + 1]
            w = (when - ts[i]).total_seconds() / max(1.0, (ts[i + 1] - ts[i]).total_seconds())
            dl = ((b["lon"] - a["lon"] + 180.0) % 360.0) - 180.0
            lon = a["lon"] + w * dl
            pr = (a["pres_hpa"] + w * (b["pres_hpa"] - a["pres_hpa"])
                  if a["pres_hpa"] is not None and b["pres_hpa"] is not None else
                  (a["pres_hpa"] if w < 0.5 else b["pres_hpa"]))
            return {"time": when.isoformat(), "lat": a["lat"] + w * (b["lat"] - a["lat"]),
                    "lon": ((lon + 180.0) % 360.0) - 180.0, "pres_hpa": pr,
                    "exact": w in (0.0, 1.0), "nature": a["nature"] if w < 0.5 else b["nature"]}
    if ts and when == ts[-1]:
        return dict(pts[-1], exact=True)
    return None


def start_position(sid: str, when: datetime) -> tuple[float, float, dict]:
    trk = track(sid)
    p = at(trk, when)
    if p is None:
        raise SystemExit(f"--storm-id {sid} ({trk['name']}): no best-track point at {when:%Y-%m-%d %H}Z (track "
                         f"{trk['points'][0]['time']} .. {trk['points'][-1]['time']})")
    return p["lat"], p["lon"], {"sid": sid, "name": trk["name"], "basin": trk["basin"], "at_start": p,
                                "licence": LICENCE}
