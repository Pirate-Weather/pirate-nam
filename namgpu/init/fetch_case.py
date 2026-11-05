"""Fetch one case's inputs from AWS NODD into cases/<case>/raw/ (parallel, byte-range subsets).

    python -m namgpu.init.fetch_case --cycle 2025031412 --hours 48

What it gets:
- NAM native B-grid analysis (initial state): nam.tHHz.bgrd3d00.tm00.grib2 (954x835, 60 hybrid
  levels, GDT 3.32769), whole file.  Plus bgrdsf00..HH every 3 h (native-grid surface, referee).
- NAM 12 km grid 218 forecast of the same cycle (picture referee): awphysFF every 3 h, whole files,
  plus REFC/APCP/TMP 2 m/UGRD/VGRD 10 m/PRMSL records hourly to 36 h (byte ranges from .idx).
- GFS 0.25 deg boundaries, the 6-h-old cycle as operational NAM uses it: gfs.tHHz.pgrb2.0p25.fFFF
  every 3 h covering the forecast, subset to the fields the boundary init needs.
- MRMS MergedReflectivityQCComposite_00.50 (truth): the file nearest each hour.
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
from pathlib import Path

NAM = "https://noaa-nam-pds.s3.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MRMS = "https://noaa-mrms-pds.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parents[2]

GFS_VARS = re.compile(
    r":(HGT|TMP|SPFH|RH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR|GRLE|O3MR):(\d+(\.\d+)? mb):"
    r"|:(PRES|HGT|TMP|PRMSL|LAND|ICEC|WEASD|SNOD|TSOIL|SOILW):(surface|mean sea level|"
    r"0-0.1 m below ground|0.1-0.4 m below ground|0.4-1 m below ground|1-2 m below ground):"
    r"|:(TMP|SPFH|RH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
NAM_HOURLY = re.compile(
    r":(REFC):entire atmosphere|:APCP:surface:|:TMP:2 m above ground:|:DPT:2 m above ground:"
    r"|:(UGRD|VGRD):10 m above ground:|:PRMSL:mean sea level:|:MSLET:mean sea level:")


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


def _list_mrms_day(day: str) -> list[str]:
    """Every key under one UTC day prefix, following continuation tokens (MRMS has ~720 a day)."""
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
            return keys
        token = m.group(1)


def _mrms_hour(valid: datetime, dest_dir: Path) -> str:
    # The nearest file can sit in the previous or next UTC day's prefix (valid times near 00Z),
    # so list valid-1d, valid and valid+1d and pick the nearest across all three.
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
    dest = dest_dir / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
    if dest.exists():
        return f"have {dest.name}"
    dest.write_bytes(gzip.decompress(_get(f"{MRMS}/{best}")))
    return f"got {dest.name} ({best.rsplit('_', 1)[-1]})"


def plan(cycle: datetime, hours: int, raw: Path, nam: bool = True):
    """nam=False: GFS only (a GFS start; cycles after the NAM's 2026-11-03 retirement have no NAM files)."""
    d, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    if not nam:
        return _gfs_jobs(cycle, hours, raw)
    jobs = []
    nam = f"{NAM}/nam.{d}/nam.t{hh}z"
    jobs.append((_download, f"{nam}.bgrd3d00.tm00.grib2", raw / "nam" / "bgrd3d00.grib2"))
    for f in range(0, hours + 1, 3):
        jobs.append((_download, f"{nam}.bgrdsf{f:02d}.tm00.grib2", raw / "nam" / f"bgrdsf{f:02d}.grib2"))
        jobs.append((_download, f"{nam}.awphys{f:02d}.tm00.grib2", raw / "nam" / f"awphys{f:02d}.grib2"))
    # Hourly picture-referee subsets stop at 36 h: NAM 12 km writes hourly awphys output only to
    # 36 h (3-hourly after), so this is a limit of the 48 h default, not a bug.
    for f in range(1, min(hours, 36) + 1):
        if f % 3:
            jobs.append((_subset, f"{nam}.awphys{f:02d}.tm00.grib2",
                         raw / "nam" / f"awphys{f:02d}.sub.grib2", NAM_HOURLY))
    return jobs + _gfs_jobs(cycle, hours, raw)


def _gfs_jobs(cycle: datetime, hours: int, raw: Path):
    jobs = []
    gcyc = cycle - timedelta(hours=6)
    gd, gh = gcyc.strftime("%Y%m%d"), gcyc.strftime("%H")
    for f in range(6, hours + 7, 3):
        url = f"{GFS}/gfs.{gd}/{gh}/atmos/gfs.t{gh}z.pgrb2.0p25.f{f:03d}"
        jobs.append((_subset, url, raw / "gfs" / f"gfs.t{gh}z.f{f:03d}.grib2", GFS_VARS))
    return jobs


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
    raw = ROOT / "cases" / cycle.strftime("%Y-%m-%d_%Hz") / "raw"
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
    ap.add_argument("--source", default=None,
                    help="an init source row with a fetch entry (ifs, ifs-0p25); default: the NAM case set")
    a = ap.parse_args(argv)
    if a.source:
        return main_source(a)
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    from namgpu import paths
    case = paths.work_root() / "cases" / cycle.strftime("%Y-%m-%d_%Hz")
    raw = case / "raw"
    for sub in ("nam", "gfs", "mrms"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    jobs = plan(cycle, a.hours, raw)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn, *args) for fn, *args in jobs]
        futs += [ex.submit(_mrms_hour, cycle + timedelta(hours=h), raw / "mrms")
                 for h in range(0, a.hours + 1)]
        bad = 0
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
