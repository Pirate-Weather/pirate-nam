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

from namgpu.init.fetch_case import _mrms_hour, plan

UNIT = "init-fetch-case"
DEFAULT_HOURS = 48
DEFAULT_WORKERS = 16


def fetch(cycle: datetime, case_dir: Path, hours: int = DEFAULT_HOURS, workers: int = DEFAULT_WORKERS) -> int:
    """Fetch every input of one case. Returns the number of failed requests (0 = complete)."""
    raw = Path(case_dir) / "raw"
    for sub in ("nam", "gfs", "mrms"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    if hours > 36:
        print(f"{UNIT}: note, hourly picture-referee subsets stop at 36 h (NAM hourly limit); "
              f"f{36 + 1:03d}..f{hours:03d} keep 3-hourly awphys only", flush=True)
    jobs = plan(cycle, hours, raw)
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


def run(ctx, hours: int = DEFAULT_HOURS, workers: int = DEFAULT_WORKERS) -> int:
    """Driver entry point: ctx.start is the cycle, ctx.case_dir the case folder."""
    if ctx.start is None or ctx.case_dir is None:
        raise ValueError(f"{UNIT}: ctx.start and ctx.case_dir must be set")
    return fetch(ctx.start, Path(ctx.case_dir), hours, workers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    ap.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    a = ap.parse_args(argv)
    bad = fetch(datetime.strptime(a.cycle, "%Y%m%d%H"), Path(a.case_dir), a.hours, a.workers)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
