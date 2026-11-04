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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    case = ROOT / "cases" / cycle.strftime("%Y-%m-%d_%Hz")
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
