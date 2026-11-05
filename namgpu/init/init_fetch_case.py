"""init-fetch-case: fetch one case's raw inputs from AWS NODD into cases/<case>/raw/, for any domain.

Unit: init-fetch-case (group init, mode new, no Fortran counterpart, no GPU work).
Inputs:  ctx.start (datetime of the cycle, 00/06/12/18 UTC), ctx.case_dir (cases/<case>), ctx.cfg (the
         domain: its grid decides which regional rows apply), the init source (ctx.init_source or
         ctx.cfg.raw["init_source"], default gfs here), optional hours (default 48). Without ctx, pass
         cycle and case-dir explicitly.
Outputs: the rows of fetch_case.FETCH_ROWS that apply to this cycle and domain:
         raw/gfs/  GFS 0.25 deg analysis subset (the start) and forecast-lead subsets every 3 h covering the
                   run from the cycle boco_age_h older (boundaries; 6 h as operational NAM, 0 h for past
                   cases), anywhere on Earth, any cycle from 2021-01-01 00Z
         raw/nam/  NAM native analysis bgrd3d00, bgrdsfFF and grid 218 awphysFF(.sub), only where the domain
                   overlaps those grids and before the NAM retirement (2026-11-03 12Z)
         raw/mrms/ MRMS composite reflectivity nearest each hour, only where the domain overlaps MRMS CONUS
Kernel:  none (pure I/O: no kernel is expected, and UNITS.json records kernel ""). Fetches are I/O bound
         and run in a thread pool.
Limit:   hourly awphys.sub subsets cover f=1..36 (NAM's own hourly output limit).
Reuses:  namgpu/init/fetch_case.py (plan, fetch). Existing files are skipped ("have"), so a rerun fetches
         only what is missing. Failed referee/truth requests are reported, not counted.
Run:     python -m namgpu.init.init_fetch_case --cycle 2025031412 --hours 48 --case-dir cases/2025-03-14_12z
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from namgpu.init import fetch_case
from namgpu.init.fetch_case import NAM, _mrms_hour, plan_source

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
          gfs_only: bool = False, source: str | None = None, cfg=None, age_h: int | None = None,
          inputs_only: bool = False) -> int:
    """Fetch every input of one case for the configured domain (fetch_case.FETCH_ROWS). Returns the number of
    failed required requests (0 = complete).
    hours: the forecast length. The boundaries cover exactly what a run of that length reads (domains.bc_hours:
    at least 6 h, in 3-hourly steps); the referee and truth rows (archived NAM forecast, MRMS) cover 0..hours.
    source None: the NAM set (native start) when NODD holds this cycle's NAM files, else GFS only.
    source with its own fetch entry (sources.SOURCES: ifs, ifs-0p25): its start and boundary leads, its donor
    fields and MRMS. gfs_only: no NAM rows (any cycle, including after the NAM's 2026-11-03 retirement).
    inputs_only: only the rows a forecast reads (roles init and boco): no referee or truth files (MRMS, the
    archived NAM forecast and native surface files), which only the side-by-side comparison uses."""
    if source and source not in ("gfs", "native"):
        return _fetch_source(cycle, Path(case_dir), hours, workers, source)
    import json
    import time
    t0 = time.perf_counter()
    if source is None:
        source = "gfs" if gfs_only else "native"
    if source == "native" and not _nam_cycle_exists(cycle):
        # NCEP retired the NAM on 2026-11-03 12Z: later cycles (and gaps in the archive) have no NAM files
        print(f"{UNIT}: no NAM files for cycle {cycle:%Y%m%d%H} on NODD; fetching GFS and MRMS only "
              "(a GFS start: --source gfs)", flush=True)
        source, gfs_only = "gfs", True
    rows = tuple(r for r in fetch_case.FETCH_ROWS if not (gfs_only and r["name"].startswith("nam-"))
                 and not (inputs_only and r["role"] in ("referee", "truth")))
    if hours > 36 and not gfs_only and not inputs_only:
        print(f"{UNIT}: note, hourly NAM picture-referee subsets stop at 36 h (NAM hourly limit)", flush=True)
    bad = fetch_case.fetch(cycle, Path(case_dir), hours, workers, cfg=cfg, source=source, age_h=age_h,
                           log=lambda m: print(m, flush=True), rows=rows)
    secs = round(time.perf_counter() - t0, 1)
    raw = Path(case_dir) / "raw"
    nbytes = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file()) if raw.is_dir() else 0
    print(f"{UNIT}: done, {bad} failed, {secs} s", flush=True)
    (Path(case_dir) / "fetch-receipt.json").write_text(json.dumps(
        {"cycle": f"{cycle:%Y%m%d%H}", "hours": hours, "source": source, "gfs_only": gfs_only,
         "inputs_only": inputs_only, "rows": [r["name"] for r in rows], "failed": bad, "seconds": secs,
         "raw_bytes": nbytes, "parallelism": f"{workers} threads (I/O)"}, indent=1))
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
    """Driver entry point: ctx.start is the cycle, ctx.case_dir the case folder, ctx.cfg the domain."""
    if ctx.start is None or ctx.case_dir is None:
        raise ValueError(f"{UNIT}: ctx.start and ctx.case_dir must be set")
    cfg = getattr(ctx, "cfg", None)
    src = getattr(ctx, "init_source", None) or (cfg.raw.get("init_source") if cfg is not None else None) or "gfs"
    # the run's own length when the driver knows it (a 6 h run fetched 48 h of inputs before 0.1.1)
    hours = int(getattr(ctx, "forecast_hours", None) or hours)
    return fetch(ctx.start, Path(ctx.case_dir), hours, workers, cfg=cfg, source=str(src).lower())


def main(argv=None) -> int:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Download one case's inputs from NOAA's open data on AWS (no account needed) into "
                    "<case-dir>/raw/: the NAM native analysis and the archived NAM forecast (cycles before "
                    "the NAM's retirement, 2026-11-03 12Z), the GFS start and boundaries and MRMS reflectivity, "
                    "each only where the domain overlaps it. Files already there are skipped. A 24 h case is "
                    "about 3.7 GB.",
        epilog="examples:\n"
               "  woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z\n"
               "  woof-nam fetch --cycle 2026110412 --hours 24 --case-dir cases/2026-11-04_12z --gfs-only\n"
               "  woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z --source ifs-0p25\n"
               "Name the case folder YYYY-MM-DD_HHz: woof-nam run reads the start time from it. Any domain: "
               "woof-nam nest fetches for its own grid.",
        formatter_class=HelpFormatter)
    ap.add_argument("--cycle", required=True, help="start cycle, YYYYMMDDHH (00, 06, 12 or 18 UTC)")
    ap.add_argument("--hours", type=int, default=DEFAULT_HOURS,
                    help=f"forecast length to fetch inputs for, in hours (default {DEFAULT_HOURS})")
    ap.add_argument("--case-dir", required=True, help="case folder, e.g. cases/2025-03-14_12z")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel fetches (default {DEFAULT_WORKERS})")
    ap.add_argument("--inputs-only", action="store_true",
                    help="only what a forecast reads (the NAM analysis and the GFS boundaries): no MRMS and no "
                         "archived NAM forecast, which only the side-by-side comparison uses")
    ap.add_argument("--gfs-only", action="store_true",
                    help="GFS boundaries/start and MRMS only, no NAM files (for --source gfs; any cycle, "
                         "including after the NAM's 2026-11-03 retirement)")
    ap.add_argument("--source", default=None,
                    help="native (default: the NAM set when the cycle has it), gfs, or another init source's start "
                         "and boundaries: ifs (ECMWF open data IFS 0.1 deg, cycles from 2026-10-20 12Z) or ifs-0p25 "
                         "(0.25 deg, from 2025-03-14 12Z); CC BY 4.0")
    ap.add_argument("--config", type=Path, default=None,
                    help="domain configfile (default NAMGPU_CONFIGFILE, else the NAM 12 km grid)")
    ap.add_argument("--boco-age", type=int, default=None,
                    help="boundary cycle age in hours (default: the source row's, 6 h as operational NAM)")
    a = ap.parse_args(argv)
    from namgpu import config
    cfg = config.load(a.config) if a.config else config.load()
    bad = fetch(datetime.strptime(a.cycle, "%Y%m%d%H"), Path(a.case_dir), a.hours, a.workers, a.gfs_only, a.source,
                cfg=cfg, age_h=a.boco_age, inputs_only=a.inputs_only)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
