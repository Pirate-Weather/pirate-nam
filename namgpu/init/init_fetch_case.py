"""init-fetch-case: fetch one case's raw inputs from AWS NODD into cases/<case>/raw/.

Unit: init-fetch-case (group init, mode new, no Fortran counterpart, no GPU work).
Inputs:  ctx.start (datetime of the cycle, 00/06/12/18 UTC), ctx.case_dir (cases/<case>),
         optional hours (default 48). Without ctx, pass cycle and case-dir explicitly.
Outputs: raw/nam/  NAM native B-grid analysis bgrd3d00 (initial state), bgrdsfFF (native surface
                   referee), awphysFF / awphysFF.sub (grid 218 picture referee)
         raw/gfs/  GFS 0.25 deg pressure-level subsets f006..f(hours+6) every 3 h (boundaries,
                   the 6-h-old cycle)
         raw/mrms/ MRMS MergedReflectivityQCComposite_00.50, nearest file per hour (truth)
Kernel:  none (pure I/O: no kernel is expected, and UNITS.json records kernel ""). The unit moves
         bytes from S3/HTTP to disk; nothing is computed on a GPU or on a CPU step path. Fetches
         are I/O bound and run in a thread pool.
Limit:   hourly awphys.sub subsets cover f=1..36 (NAM's own hourly output limit). Runs longer than
         36 h keep the 3-hourly awphys files only, and fetch() says so.
Reuses:  namgpu/init/fetch_case.py (plan, _mrms_hour; _download/_subset via plan). Existing files
         are skipped ("have"), so a rerun fetches only what is missing.
Run:     python -m namgpu.init.init_fetch_case --cycle 2025031412 --hours 48 --case-dir cases/2025-03-14_12z
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from namgpu.init.fetch_case import NAM, _mrms_hour, plan, plan_source

UNIT = "init-fetch-case"
DEFAULT_HOURS = 48
DEFAULT_WORKERS = 16


def _nam_cycle_exists(cycle: datetime) -> bool:
    """True if NODD holds this cycle's NAM native analysis (or the check itself could not be made)."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    url = f"{NAM}/nam.{cycle:%Y%m%d}/nam.t{cycle:%H}z.bgrd3d00.tm00.grib2"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30):
            return True
    except urllib.error.HTTPError as e:
        return e.code not in (403, 404)
    except OSError:
        return True


def fetch(cycle: datetime, case_dir: Path, hours: int = DEFAULT_HOURS, workers: int = DEFAULT_WORKERS,
          gfs_only: bool = False, source: str | None = None) -> int:
    """Fetch every input of one case. Returns the number of failed requests (0 = complete).
    source: an init source row with its own fetch entry (sources.SOURCES: ifs, ifs-0p25): its start and boundary
    leads, its donor fields and MRMS, instead of the NAM/GFS set."""
    if source:
        return _fetch_source(cycle, Path(case_dir), hours, workers, source)
    if not gfs_only and not _nam_cycle_exists(cycle):
        # NCEP retired the NAM on 2026-11-03 12Z: later cycles (and gaps in the archive) have no NAM files
        print(f"{UNIT}: no NAM files for cycle {cycle:%Y%m%d%H} on NODD; fetching GFS and MRMS only "
              "(a GFS start: --source gfs)", flush=True)
        gfs_only = True
    raw = Path(case_dir) / "raw"
    for sub in ("nam", "gfs", "mrms"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    if hours > 36 and not gfs_only:
        print(f"{UNIT}: note, hourly picture-referee subsets stop at 36 h (NAM hourly limit); "
              f"f{36 + 1:03d}..f{hours:03d} keep 3-hourly awphys only", flush=True)
    jobs = plan(cycle, hours, raw, nam=not gfs_only)
    bad = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, *args) for fn, *args in jobs]
        futs += [ex.submit(_mrms_hour, cycle + timedelta(hours=h), raw / "mrms")
                 for h in range(0, hours + 1)]
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
    print(f"{UNIT}: done, {bad} failed", flush=True)
    return bad


def _fetch_source(cycle: datetime, case_dir: Path, hours: int, workers: int, source: str) -> int:
    from namgpu.init.sources import SOURCES  # noqa: PLC0415
    row = SOURCES.get(source)
    if row is None or "fetch" not in row:
        have = sorted(k for k, r in SOURCES.items() if "fetch" in r)
        raise SystemExit(f"{UNIT}: --source {source!r} has no fetch entry (sources with one: {', '.join(have)}; "
                         "native and gfs are the default set)")
    if row.get("valid_from") and cycle.strftime("%Y-%m-%dT%H") < row["valid_from"]:
        alt = f"; use --source {row['before_use']}" if row.get("before_use") else ""
        raise SystemExit(f"{UNIT}: source {source!r} is published from {row['valid_from']}Z, "
                         f"not {cycle:%Y-%m-%d %H}Z{alt}")
    raw = Path(case_dir) / "raw"
    (raw / "mrms").mkdir(parents=True, exist_ok=True)
    jobs = plan_source(source, cycle, hours, raw)
    bad = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, *args) for fn, *args in jobs]
        futs += [ex.submit(_mrms_hour, cycle + timedelta(hours=h), raw / "mrms") for h in range(0, hours + 1)]
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
    print(f"{UNIT}: {source} done, {bad} failed ({row.get('attribution', '')})", flush=True)
    return bad


def run(ctx, hours: int = DEFAULT_HOURS, workers: int = DEFAULT_WORKERS) -> int:
    """Driver entry point: ctx.start is the cycle, ctx.case_dir the case folder."""
    if ctx.start is None or ctx.case_dir is None:
        raise ValueError(f"{UNIT}: ctx.start and ctx.case_dir must be set")
    return fetch(ctx.start, Path(ctx.case_dir), hours, workers)


def main(argv=None) -> int:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Download one case's inputs from NOAA's open data on AWS (no account needed) into "
                    "<case-dir>/raw/: the NAM native analysis and the archived NAM forecast (cycles before "
                    "the NAM's retirement, 2026-11-03 12Z), the GFS boundaries and MRMS reflectivity. "
                    "Files already there are skipped. A 24 h case is about 3.7 GB.",
        epilog="examples:\n"
               "  woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z\n"
               "  woof-nam fetch --cycle 2026110412 --hours 24 --case-dir cases/2026-11-04_12z --gfs-only\n"
               "  woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z --source ifs-0p25\n"
               "Name the case folder YYYY-MM-DD_HHz: woof-nam run reads the start time from it.",
        formatter_class=HelpFormatter)
    ap.add_argument("--cycle", required=True, help="start cycle, YYYYMMDDHH (00, 06, 12 or 18 UTC)")
    ap.add_argument("--hours", type=int, default=DEFAULT_HOURS,
                    help=f"forecast length to fetch inputs for, in hours (default {DEFAULT_HOURS})")
    ap.add_argument("--case-dir", required=True, help="case folder, e.g. cases/2025-03-14_12z")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel fetches (default {DEFAULT_WORKERS})")
    ap.add_argument("--gfs-only", action="store_true",
                    help="GFS boundaries/start and MRMS only, no NAM files (for --source gfs; any cycle, "
                         "including after the NAM's 2026-11-03 retirement)")
    ap.add_argument("--source", default=None,
                    help="fetch another init source's start and boundaries instead: ifs (ECMWF open data IFS 0.1 deg, "
                         "cycles from 2026-10-20 12Z) or ifs-0p25 (0.25 deg, from 2024-03-18 12Z); CC BY 4.0")
    a = ap.parse_args(argv)
    bad = fetch(datetime.strptime(a.cycle, "%Y%m%d%H"), Path(a.case_dir), a.hours, a.workers, a.gfs_only, a.source)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
