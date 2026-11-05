"""Fetch one case's inputs from AWS NODD into cases/<case>/raw/ (parallel, byte-range subsets), for any domain.

    python -m namgpu.init.fetch_case --cycle 2026110412 --hours 48 [--config parm/nam_configfile_X]
        [--source gfs] [--boco-age 6] [--case-dir DIR] [--dry-run]

What is fetched is the FETCH_ROWS table: one row per source file family, with the archive's date range, its
path template per era, the record subset, the role (init / boco / referee / truth) and the region it covers.
A row is planned only when the cycle is inside its archive dates and the domain (the configfile's grid, every
H point) overlaps its region; anything else is skipped with the reason printed. Adding a source or a region
is a row here, never a new code path.

Rows today:
- gfs-f000   GFS 0.25 deg analysis (init), global, 2021-01-01 00Z on (the NOAA S3 archive start). Subset by
             .idx byte ranges to the records the start reads (pressure levels, surface, soil, snow, ice,
             canopy water). Written where sources.SOURCES["gfs"]["file"] expects it.
- gfs-boco   GFS 0.25 deg forecast leads every boco_interval_h covering the run, from the cycle boco_age_h
             older than the start (SOURCES["gfs"]: 6 h as operational NAM; 0 h for past cases), global.
- nam-*      NAM native B-grid analysis bgrd3d00 (the native start), native surface bgrdsfFF and the grid 218
             awphysFF picture referee: only where the domain overlaps the NAM 12 km grid, only for cycles
             before the NAM retirement (2026-11-03 12Z).
- mrms       MRMS MergedReflectivityQCComposite_00.50 (truth), the file nearest each hour, only where the
             domain overlaps MRMS CONUS (20-55 N, 130-60 W).
All requests run in a thread pool (I/O bound); nothing here is a single-core stage.

Any other init source with a "fetch" entry in sources.SOURCES (ECMWF open data IFS 0.1 / 0.25 deg):

    python -m namgpu.init.fetch_case --cycle 2026110312 --hours 48 --source ifs

fetches that row's start and boundary leads (every 3 h, the row's own cycle) into cases/<case>/raw/, by the row's
record selection on the ECMWF jsonl .index beside each object (byte ranges, adjacent records merged), its hosts in
order (a host that refuses the index or an object is skipped), plus the row's donor fields (GFS f000 ICEC by .idx).
One output file per lead: the selected records of the lead's objects concatenated, and the row's f000_only
records (the surface geopotential z, published at step 0 only) appended to every later lead. Prints the bytes.
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

NAM = "https://noaa-nam-pds.s3.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MRMS = "https://noaa-mrms-pds.s3.amazonaws.com"
MRMS_MAX_OFFSET_S = 600       # MRMS writes every 2 min; a file more than 10 min off is not that hour
ROOT = Path(__file__).resolve().parents[2]

GFS_VARS = re.compile(
    r":(HGT|TMP|SPFH|RH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR|GRLE|O3MR):(\d+(\.\d+)? mb):"
    r"|:(PRES|HGT|TMP|PRMSL|LAND|ICEC|WEASD|SNOD|TSOIL|SOILW):(surface|mean sea level|"
    r"0-0.1 m below ground|0.1-0.4 m below ground|0.4-1 m below ground|1-2 m below ground):"
    r"|:(TMP|SPFH|RH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# Boundary leads: only what init-gfs-bocos reads (gfs_start PL_ORDER levels, SFC_COLUMNS) plus the 2 m, 10 m and
# PRMSL records the source-forecast referee panels draw. Was GFS_VARS: O3MR, GRLE, soil, snow, ice and land are
# about 1/6 of the bytes and are never read from a boundary file.
# Isobaric levels at p >= 1 hPa only ("[1-9]\d* mb"): gfs_start.decode_source keeps p >= 1 hPa, so the 8 levels
# above it (0.01-0.7 mb) were downloaded in every boundary file and dropped (the door lane's 1b6435d pattern,
# 259 -> 183 MB per file with the GRLE/O3MR cut, was lost in the nest/all merge).
GFS_BOCO_VARS = re.compile(
    r":(HGT|TMP|SPFH|RH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR):([1-9]\d*(\.\d+)?) mb:"
    r"|:(PRES|HGT|TMP):surface:|:PRMSL:mean sea level:"
    r"|:(TMP|SPFH|RH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# The start also reads canopy water (gfs_start.SFC CNWAT), which only the analysis carries.
GFS_INIT_VARS = re.compile(GFS_VARS.pattern + r"|:CNWAT:surface:")
NAM_HOURLY = re.compile(
    r":(REFC):entire atmosphere|:APCP:surface:|:TMP:2 m above ground:|:DPT:2 m above ground:"
    r"|:(UGRD|VGRD):10 m above ground:|:PRMSL:mean sea level:|:MSLET:mean sea level:")
PATTERNS = {"gfs_init": GFS_INIT_VARS, "gfs_boco": GFS_BOCO_VARS, "nam_hourly": NAM_HOURLY}

# Regions a row covers. "global": everywhere. "bbox": geographic lat/lon box (degrees, lon -180..180).
# "rotated": a rotated lat-lon grid given by its centre (tph0d, tlm0d) and wbd/sbd, tested in its own rotated
# coordinates, so the NAM 12 km footprint is exact.
REGIONS = {
    "global": {"kind": "global"},
    # The NAM 12 km parent grid (parm/nam_configfile tph0d/tlm0d/wbd/sbd). The keys are written here, not read
    # through config.load: with NAMGPU_CONFIGFILE set, loading the default configfile returns the run's own
    # domain, so every nest "overlapped" NAM and fetched its archive (Gabrielle case, 2026-11-04: 1 GB of bgrdsf
    # for a New Zealand domain).
    "nam12": {"kind": "rotated", "tph0d": 54.0, "tlm0d": -106.0, "wbd": -60.039, "sbd": -45.036},
    # NCEP grid 218 (Lambert, 614 x 428, 12.19 N 133.46 W .. 57.29 N 49.39 W corners): its lat/lon hull.
    "grid218": {"kind": "bbox", "lat": (12.0, 61.5), "lon": (-153.0, -49.0)},
    "mrms_conus": {"kind": "bbox", "lat": (20.0, 55.0), "lon": (-130.0, -60.0)},
}

# Path eras: the GFS bucket moved files under atmos/ with GFS v16 (first cycle 2021-03-22 12Z); before it the
# files sit directly under the cycle folder. The bucket's archive begins 2021-01-01 00Z.
GFS_ERAS = (("2021-01-01T00", "gfs.{ymd}/{hh}/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"),
            ("2021-03-22T12", "gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"))
NAM_ERAS = (("2017-01-01T00", "nam.{ymd}/nam.t{hh}z.{stem}{lead:02d}.tm00.grib2"),)
NAM_RETIRED = "2026-11-03T12"

# One row per file family. cycle: "start" or "boco" (the boundary cycle, boco_age_h older). leads: "f000",
# "boco" (boco_interval_h steps covering the run), "every3" (0..hours by 3), "hourly36" (1..min(hours, 36),
# not multiples of 3: NAM writes hourly awphys only to 36 h). kind: whole | subset (records by .idx) | mrms.
# dest is relative to raw/. role init/boco rows are required when their source is the run's source. licence: the
# data's licence and host, beside the row (permissive data only; nothing NC-licensed is fetched).
FETCH_ROWS = (
    {"name": "gfs-f000", "licence": "NOAA/NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds",
     "source": "gfs", "role": "init", "host": GFS, "eras": GFS_ERAS, "cycle": "start",
     "leads": "f000", "kind": "subset", "records": "gfs_init", "region": "global",
     "dest": "gfs/gfs.t{hh}z.pgrb2.0p25.f000"},
    {"name": "gfs-boco", "licence": "NOAA/NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds",
     "source": "gfs", "role": "boco", "host": GFS, "eras": GFS_ERAS, "cycle": "boco",
     "leads": "boco", "kind": "subset", "records": "gfs_boco", "region": "global",
     "dest": "gfs/gfs.t{hh}z.f{lead:03d}.grib2"},
    {"name": "nam-bgrd3d00", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "init", "host": NAM, "eras": NAM_ERAS,
     "stem": "bgrd3d", "cycle": "start", "leads": "f000", "kind": "whole", "region": "nam12",
     "until": NAM_RETIRED, "dest": "nam/bgrd3d00.grib2"},
    {"name": "nam-bgrdsf", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_ERAS,
     "stem": "bgrdsf", "cycle": "start", "leads": "every3", "kind": "whole", "region": "nam12",
     "until": NAM_RETIRED, "dest": "nam/bgrdsf{lead:02d}.grib2"},
    {"name": "nam-awphys", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_ERAS,
     "stem": "awphys", "cycle": "start", "leads": "every3", "kind": "whole", "region": "grid218",
     "until": NAM_RETIRED, "dest": "nam/awphys{lead:02d}.grib2"},
    {"name": "nam-awphys-hourly", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_ERAS,
     "stem": "awphys", "cycle": "start", "leads": "hourly36", "kind": "subset", "records": "nam_hourly",
     "region": "grid218", "until": NAM_RETIRED, "dest": "nam/awphys{lead:02d}.sub.grib2"},
    {"name": "mrms", "licence": "NOAA MRMS, US government work (public domain), AWS NODD noaa-mrms-pds",
     "source": "mrms", "role": "truth", "host": MRMS, "cycle": "start", "leads": "hourly",
     "kind": "mrms", "region": "mrms_conus", "since": "2020-10-14T00", "dest": "mrms"},
)


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (OSError, urllib.error.URLError):
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def _get(url: str, rng: str | None = None, timeout=120) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})

    def once():
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    return _retry(once)


def _download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    def once():
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=8 << 20)
    _retry(once)
    tmp.rename(dest)
    return f"got {dest.name} {dest.stat().st_size / 1e6:.0f} MB"


def _subset(url: str, dest: Path, pattern: re.Pattern) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name}"
    idx = _get(url + ".idx").decode().splitlines()
    rows = [l.split(":") for l in idx if l.strip()]
    offs = [int(r[1]) for r in rows]
    want = [k for k, l in enumerate(idx) if pattern.search(":" + ":".join(l.split(":")[3:]) + ":")]
    if not want:
        # An empty match would write a zero-byte dest that the st_size > 0 skip check refetches
        # on every run, so it is an error, not a file.
        raise ValueError(f"{dest.name}: no idx records of {url} match the pattern ({len(idx)} records)")
    tmp = dest.with_suffix(".part")
    # Sub-messages (idx "329.1", "329.2") share one GRIB2 message and one byte offset: fetch each message
    # once, ending before the next larger offset. (Ending at offs[k + 1] - 1 made an inverted range for the
    # first sub-message, which S3 ignores and answers with the whole file.)
    starts = sorted({offs[k] for k in want})
    ranges = []                                    # [start, end] byte ranges; end None = to end of file
    for s in starts:
        nxt = next((o for o in offs if o > s), None)
        end = nxt - 1 if nxt is not None else None
        if ranges and ranges[-1][1] is not None and ranges[-1][1] + 1 == s:
            ranges[-1][1] = end                    # adjacent records: one request (same bytes, ~10x fewer GETs)
        else:
            ranges.append([s, end])
    # each range is a new connection; one at a time they ran at ~2 MB/s (118 GETs, 129 s for a GFS file),
    # so the ranges of one file are fetched in parallel and written in order (same bytes)
    with ThreadPoolExecutor(min(8, len(ranges))) as ex, open(tmp, "wb") as f:
        for chunk in ex.map(lambda r: _get(url, f"{r[0]}-{r[1] if r[1] is not None else ''}"), ranges):
            f.write(chunk)
    tmp.rename(dest)
    return f"got {dest.name} {len(want)} recs {dest.stat().st_size / 1e6:.0f} MB"


@lru_cache(maxsize=None)
def _list_mrms_day(day: str) -> tuple[str, ...]:
    """Every key under one UTC day prefix, following continuation tokens (MRMS has ~720 a day). Cached: each
    hour lists its own day and both neighbours, so a 48 h case asked for the same three days ~49 times."""
    prefix = f"CONUS/MergedReflectivityQCComposite_00.50/{day}/"
    keys, token = [], None
    while True:
        q = f"list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            q += "&continuation-token=" + urllib.parse.quote(token, safe="")
        xml = _get(f"{MRMS}/?{q}").decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml or not m:
            return tuple(keys)
        token = m.group(1)


def _mrms_hour(valid: datetime, dest_dir: Path) -> str:
    # The nearest file can sit in the previous or next UTC day's prefix (valid times near 00Z),
    # so list valid-1d, valid and valid+1d and pick the nearest across all three.
    dest = dest_dir / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
    if dest.exists():
        return f"have {dest.name}"
    keys = []
    for dd in (-1, 0, 1):
        keys += _list_mrms_day((valid + timedelta(days=dd)).strftime("%Y%m%d"))
    best, bestdt = None, None
    for k in keys:
        m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", k)
        if not m:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        d = abs((t - valid).total_seconds())
        if bestdt is None or d < bestdt:
            best, bestdt = k, d
    if best is None:
        return f"MRMS none for {valid}"
    if bestdt > MRMS_MAX_OFFSET_S:
        # A live case asks for valid times that have not happened yet; the nearest key is then the newest file,
        # hours early, and writing it under this hour would show the wrong truth (live case 2026-11-04 12Z: 19Z-12Z
        # next day all got 18:36Z). No file, so a later fetch picks the real one up.
        return f"MRMS none within {MRMS_MAX_OFFSET_S} s of {valid} (nearest {best.rsplit('_', 1)[-1]})"
    dest.write_bytes(gzip.decompress(_get(f"{MRMS}/{best}")))
    return f"got {dest.name} ({best.rsplit('_', 1)[-1]})"


# ---------------------------------------------------------------------------------------------------------
# Domain footprint and region tests
# ---------------------------------------------------------------------------------------------------------
def domain_points(cfg=None, stride: int = 4):
    """Geographic (lat, lon) degrees of the configured grid's H points (every stride-th row/column plus the
    whole boundary ring), lon in -180..180. cfg None = the run's configfile (NAMGPU_CONFIGFILE)."""
    from namgpu import config, grid as gridmod
    cfg = cfg or config.load()
    g = gridmod.build(cfg)
    lat, lon = np.degrees(g.glat_h), np.degrees(g.glon_h)
    m = np.zeros(lat.shape, bool)
    m[::stride, ::stride] = True
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return lat[m], lon[m]


def _geo_to_rotated(lat, lon, tph0d, tlm0d):
    """Geographic -> rotated lat-lon (degrees) for the rotated pole of namgpu.grid (inverse of
    rotated_to_geographic)."""
    la, lo = np.radians(lat), np.radians(lon)
    p0, l0 = np.radians(tph0d), np.radians(tlm0d)
    x = np.cos(la) * np.cos(lo - l0)
    y = np.cos(la) * np.sin(lo - l0)
    z = np.sin(la)
    xr = np.cos(p0) * x + np.sin(p0) * z
    zr = -np.sin(p0) * x + np.cos(p0) * z
    return np.degrees(np.arcsin(np.clip(zr, -1.0, 1.0))), np.degrees(np.arctan2(y, xr))


def overlaps(region: str, lat, lon) -> bool:
    r = REGIONS[region]
    if r["kind"] == "global":
        return True
    if r["kind"] == "bbox":
        (a, b), (c, d) = r["lat"], r["lon"]
        lo = (np.asarray(lon) + 180.0) % 360.0 - 180.0
        return bool(np.any((lat >= a) & (lat <= b) & (lo >= c) & (lo <= d)))
    if r["kind"] == "rotated":
        tph, tlm = _geo_to_rotated(lat, lon, r["tph0d"], r["tlm0d"])
        return bool(np.any((tlm >= r["wbd"]) & (tlm <= -r["wbd"]) & (tph >= r["sbd"]) & (tph <= -r["sbd"])))
    raise ValueError(f"fetch-case: region {region!r} kind {r['kind']!r}")


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H")


def _era_path(row, cyc: datetime, lead: int) -> str:
    tmpl = None
    for since, t in row["eras"]:
        if cyc >= _t(since):
            tmpl = t
    if tmpl is None:
        raise ValueError(f"fetch-case: {row['name']} has no archive before {row['eras'][0][0]} (cycle {cyc:%Y-%m-%d %HZ})")
    return tmpl.format(ymd=f"{cyc:%Y%m%d}", hh=f"{cyc:%H}", lead=lead, stem=row.get("stem", ""))


def _leads(row, hours: int, age: int, interval: int):
    spec = row["leads"]
    if spec == "f000":
        return [0]
    if spec == "boco":
        return list(range(age, hours + age + 1, interval))
    if spec == "every3":
        return list(range(0, hours + 1, 3))
    if spec == "hourly36":
        return [f for f in range(1, min(hours, 36) + 1) if f % 3]
    if spec == "hourly":
        return list(range(0, hours + 1))
    raise ValueError(f"fetch-case: {row['name']} leads {spec!r}")


def plan(cycle: datetime, hours: int, raw: Path, cfg=None, source: str = "gfs", age_h: int | None = None,
         rows=FETCH_ROWS):
    """Jobs [(fn, *args, required)] for one case on one domain, plus the skipped rows [(name, reason)]."""
    from namgpu.init import sources as srcmod
    srow = srcmod.SOURCES["gfs"]
    age = srcmod.boco_age_h(srow, cfg, age_h)
    interval = int(srow.get("boco_interval_h", 3))
    lat, lon = domain_points(cfg)
    jobs, skipped = [], []
    for row in rows:
        if "until" in row and cycle >= _t(row["until"]):
            skipped.append((row["name"], f"archive ends {row['until']} (cycle {cycle:%Y-%m-%d %HZ})"))
            continue
        if "since" in row and cycle < _t(row["since"]):
            skipped.append((row["name"], f"archive starts {row['since']}"))
            continue
        if not overlaps(row["region"], lat, lon):
            skipped.append((row["name"], f"domain outside region {row['region']}"))
            continue
        # a native (NAM analysis) start takes the GFS boundaries, as operational NAM (CONTRACT section 5)
        required = (row["role"] == "init" and row["source"] == source) or \
                   (row["role"] == "boco" and (row["source"] == source or (source == "native" and row["source"] == "gfs")))
        cyc = cycle - timedelta(hours=age) if row["cycle"] == "boco" else cycle
        if row["kind"] == "mrms":
            for f in _leads(row, hours, age, interval):
                jobs.append((_mrms_hour, cycle + timedelta(hours=f), raw / row["dest"], required))
            continue
        for f in _leads(row, hours, age, interval):
            url = f"{row['host']}/{_era_path(row, cyc, f)}"
            dest = raw / row["dest"].format(hh=f"{cyc:%H}", lead=f)
            if row["kind"] == "whole":
                jobs.append((_download, url, dest, required))
            else:
                jobs.append((_subset, url, dest, PATTERNS[row["records"]], required))
    return jobs, skipped


def fetch(cycle: datetime, case_dir: Path, hours: int = 48, workers: int = 16, cfg=None, source: str = "gfs",
          age_h: int | None = None, log=print, rows=FETCH_ROWS) -> int:
    """Fetch every planned input of one case. Returns the number of failed REQUIRED requests (0 = the run's
    init and boundary files are complete); failed referee/truth requests are printed and not counted."""
    raw = Path(case_dir) / "raw"
    from namgpu.init import sources as srcmod
    if "fetch" in srcmod.SOURCES.get(source, {}):
        # ifs, ifs-0p25: the row's own fetch entry (ECMWF jsonl .index byte ranges, the GFS ICEC donor)
        raw.mkdir(parents=True, exist_ok=True)
        sjobs = plan_source(source, cycle, hours, raw)
        t0 = time.perf_counter()
        bad = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fn, *args): next(x for x in args if isinstance(x, Path)) for fn, *args in sjobs}
            for fu in as_completed(futs):
                try:
                    log(fu.result())
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    log(f"FAILED {type(e).__name__}: {e}")
        total = sum(p.stat().st_size for p in futs.values() if p.exists())
        log(f"fetch-case: {source} {len(sjobs)} files, {total / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s, "
            f"{bad} failed")
        return bad
    jobs, skipped = plan(cycle, hours, raw, cfg, source, age_h, rows)
    for name, why in skipped:
        log(f"fetch-case: skip {name}: {why}")
    for d in {Path(j[2]).parent if j[0] is not _mrms_hour else Path(j[2]) for j in jobs}:
        d.mkdir(parents=True, exist_ok=True)
    bad = soft = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *args[:-1]): args[-1] for fn, *args in jobs}
        for fu in as_completed(futs):
            try:
                log(fu.result())
            except Exception as e:  # noqa: BLE001
                if futs[fu]:
                    bad += 1
                else:
                    soft += 1
                log(f"FAILED{'' if futs[fu] else ' (optional)'} {type(e).__name__}: {e}")
    log(f"fetch-case: {len(jobs)} requests in {time.perf_counter() - t0:.0f} s, {bad} required failed, "
        f"{soft} optional failed")
    return bad


def _jsonl_index(url: str) -> list[dict]:
    import json
    return [json.loads(l) for l in _get(url).decode().splitlines() if l.strip()]


def _jsonl_pick(rows: list[dict], select: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for sel in select:
            if r.get("levtype") == sel["levtype"] and r.get("param") in sel["param"]:
                out.append(r)
                break
    return out


def _ranges(rows: list[dict]) -> list[tuple[int, int]]:
    """Byte ranges [start, end] of the rows, adjacent records merged into one request."""
    spans = sorted((int(r["_offset"]), int(r["_offset"]) + int(r["_length"]) - 1) for r in rows)
    out = []
    for a, b in spans:
        if out and a == out[-1][1] + 1:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _era(fetch: dict, cycle: datetime) -> dict:
    t = cycle.strftime("%Y-%m-%dT%H")
    for e in fetch["eras"]:
        if t >= e["from"] and (e["until"] is None or t < e["until"]) and cycle.hour in e["cycles"]:
            return e
    raise ValueError(f"fetch_case: no publication era of {fetch['route']} holds the {t}Z cycle")


def _object_path(tmpl: str, cycle: datetime, lead: int) -> str:
    return tmpl.format(ymd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H"), lead=lead)


def _jsonl_lead(fetch: dict, cycle: datetime, lead: int, dest: Path) -> str:
    """One lead: every object of the era, its selected records (and the f000_only records of f000 for a later
    lead), concatenated into dest. Hosts are tried in order per lead."""
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name} {dest.stat().st_size / 1e6:.0f} MB"
    era = _era(fetch, cycle)
    errors = []
    for host in fetch["hosts"]:
        try:
            parts = []
            for tmpl in era["files"]:
                obj = f"{host}/{_object_path(tmpl, cycle, lead)}"
                rows = _jsonl_index(obj.rsplit(".grib2", 1)[0] + ".index")
                want = _jsonl_pick(rows, fetch["select"])
                if lead and fetch.get("f000_only"):
                    o0 = f"{host}/{_object_path(tmpl, cycle, 0)}"
                    r0 = _jsonl_pick(_jsonl_index(o0.rsplit(".grib2", 1)[0] + ".index"), fetch["f000_only"])
                    parts += [(o0, rg) for rg in _ranges(r0)]
                parts += [(obj, rg) for rg in _ranges(want)]
            if not parts:
                raise ValueError("no records selected")
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for url, (a, b) in parts:
                    f.write(_get(url, f"{a}-{b}", timeout=300))
            tmp.rename(dest)
            return f"got {dest.name} {len(parts)} ranges {dest.stat().st_size / 1e6:.0f} MB from {host.split('/')[2]}"
        except (OSError, urllib.error.URLError, ValueError) as e:
            errors.append(f"{host.split('/')[2]}: {type(e).__name__} {e}")
    raise OSError(f"{dest.name}: every host failed: {errors}")


def plan_source(name: str, cycle: datetime, hours: int, raw: Path):
    """Jobs for an init source row with a "fetch" entry: its start and boundary leads plus its donor fields."""
    from namgpu.init.sources import SOURCES
    row = SOURCES[name]
    fetch = row["fetch"]
    if fetch["kind"] != "ecmwf-jsonl":
        raise ValueError(f"fetch_case: source {name!r} fetch kind {fetch['kind']!r} has no fetcher")
    last = max(s for s, _ in fetch["steps"])
    if hours > last:
        raise ValueError(f"fetch_case: source {name!r} publishes 3-hourly leads to {last} h, not {hours} h")
    b = row["boundary"]
    jobs = []
    for lead in range(0, hours + 1, int(b["interval_h"])):
        dest = raw.parent / b["file"].format(bcyc=cycle.hour, lead=lead)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((_jsonl_lead, fetch, cycle, lead, dest))
    for d in fetch.get("donors", []):
        dest = raw.parent / d["file"].format(hh=cycle.strftime("%H"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = d["url"].format(ymd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H"))
        jobs.append((_subset, url, dest, re.compile(re.escape(d["idx_match"]))))
    return jobs


def main_source(a) -> int:
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    from namgpu import paths
    raw = (getattr(a, "case_dir", None) or paths.work_root() / "cases" / cycle.strftime("%Y-%m-%d_%Hz")) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = plan_source(a.source, cycle, a.hours, raw)
    t0 = time.perf_counter()
    bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fn, *args): next(x for x in args if isinstance(x, Path)) for fn, *args in jobs}
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
    total = sum(p.stat().st_size for p in futs.values() if p.exists())
    wall = time.perf_counter() - t0
    print(f"fetch {a.source} {a.cycle} +{a.hours} h: {len(jobs)} files, {total / 1e6:.0f} MB in {wall:.0f} s "
          f"({total / 1e6 / max(wall, 1e-3):.0f} MB/s, {a.workers} threads), {bad} failed")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--config", type=Path, default=None, help="domain configfile (default NAMGPU_CONFIGFILE)")
    ap.add_argument("--source", default="gfs", help="init source row (its init/boco rows are required; ifs, ifs-0p25 "
                    "fetch through the row's own fetch entry)")
    ap.add_argument("--boco-age", type=int, default=None, help="boundary cycle age, h (default: the row's)")
    ap.add_argument("--case-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from namgpu.init import sources as srcmod
    if "fetch" in srcmod.SOURCES.get(a.source, {}):
        return main_source(a)          # ifs, ifs-0p25: the row's own fetch entry (jsonl .index)
    from namgpu import config, paths
    cfg = config.load(a.config) if a.config else config.load()
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    case = a.case_dir or paths.work_root() / "cases" / cycle.strftime("%Y-%m-%d_%Hz")
    if a.dry_run:
        jobs, skipped = plan(cycle, a.hours, case / "raw", cfg, a.source, a.boco_age)
        for name, why in skipped:
            print(f"skip {name}: {why}")
        for fn, *args in jobs:
            print(fn.__name__, *(str(x) for x in args[:2]), "required" if args[-1] else "optional")
        return 0
    bad = fetch(cycle, case, a.hours, a.workers, cfg, a.source, a.boco_age)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
