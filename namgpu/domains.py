"""WOOF-NAM domains: the table parm/domains.json, read by `woof-nam run --domain <name>` (and init, static).

A domain is data: a configfile in parm/ and a row in parm/domains.json (the start sources it takes, how its static
surface is made, whether a multi-card start builds the initial state in CUDA managed memory). Selecting one sets
NAMGPU_CONFIGFILE (namgpu/config.py), which moves every default config.load() of this process and of its children
(init save, ranks, history writers) to that grid. No code path is per domain.

Static surface (ensure_static), by the row's "static" kind:
  nam-analysis  the NAM 12 km grid: the case's own archived analysis (raw/nam/bgrd3d00.grib2) when it has one,
                else static/nam12_static.npz under the work root, else that file is built once from the archived
                NAM analysis of the same month and day in the latest archived year (one file from NOAA's open
                data; the NAM's last cycle is 2026-11-03 06Z).
  geog          any other grid: built on the GPU from WPS_GEOG by init-static-build into the configfile's
                static_npz, and rebuilt when the start month differs from the one it was built for (greenness
                and albedo are monthly).
Both builds run in a child process pinned to one card, so the calling process keeps no CUDA context.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from namgpu import paths

TABLE = paths.PARM / "domains.json"
LAST_NAM_CYCLE = datetime(2026, 11, 3, 6)      # NCEP retired the NAM at 2026-11-03 12Z (SCN 26-47)


def table() -> dict:
    return json.loads(TABLE.read_text())


def names() -> list[str]:
    return list(table()["domains"])


def get(name: str | None) -> dict:
    t = table()
    name = name or t["default"]
    if name not in t["domains"]:
        raise SystemExit(f"woof-nam: unknown domain {name!r}; domains: {', '.join(t['domains'])}")
    return dict(t["domains"][name], name=name, configfile_path=str(paths.PARM / t["domains"][name]["configfile"]))


def select(name: str | None, source: str | None = None) -> dict | None:
    """Make `name` this process's domain (and its children's). None leaves NAMGPU_CONFIGFILE as it is."""
    if name is None:
        return None
    row = get(name)
    if source is not None and source not in row["sources"]:
        raise SystemExit(f"woof-nam: domain {name} starts from {' or '.join(row['sources'])}, not {source!r}")
    from namgpu import config
    os.environ[config.CONFIG_ENV] = row["configfile_path"]
    return row


def current() -> dict:
    """The row of the configfile this process reads now (NAMGPU_CONFIGFILE or the default)."""
    from namgpu import config
    cf = config.configfile_path(None).resolve()
    for n in names():
        row = get(n)
        if Path(row["configfile_path"]).resolve() == cf:
            return row
    return {"name": None, "configfile": str(cf), "sources": [], "static": "geog", "managed_init": False,
            "configfile_path": str(cf)}


def archive_cycle(start: datetime) -> datetime:
    """The archived NAM 00Z analysis of the same month and day, in the latest year the archive holds."""
    for k in range(0, 12):
        day = min(start.day, 28) if start.month == 2 else start.day
        c = datetime(start.year - k, start.month, day)
        if c <= LAST_NAM_CYCLE:
            return c
    raise SystemExit(f"woof-nam: no archived NAM analysis for {start:%m-%d}")


def built_meta(npz: Path) -> dict:
    import numpy as np
    try:
        with np.load(npz) as z:
            return json.loads(str(z["META"])) if "META" in z.files else {}
    except Exception:  # noqa: BLE001  (unreadable: rebuild)
        return {}


def built_month(npz: Path) -> int | None:
    """Month of the start date a static file was built for (its META "valid"), None if it does not say."""
    import numpy as np
    try:
        with np.load(npz) as z:
            if "META" not in z.files:
                return None
            v = json.loads(str(z["META"])).get("valid")
        return datetime.fromisoformat(v).month if v else None
    except Exception:  # noqa: BLE001  (unreadable: rebuild)
        return -1


def _child(mod: str, argv: list, card: str | None) -> int:
    env = dict(os.environ)
    if card is not None:
        env["CUDA_VISIBLE_DEVICES"] = card
    cmd = [sys.executable, "-c", "import sys; sys.path.insert(0, %r); from %s import main; "
           "sys.exit(main(sys.argv[1:]))" % (str(paths.PKG.parent), mod)] + argv
    return subprocess.call(cmd, env=env)


def ensure_static(case_dir: Path, start: datetime, source: str, card: str | None = None, log=print) -> dict:
    """Make sure the current domain has its static surface for this start. Returns a receipt row
    {"domain", "kind", "file", "action": "case analysis"|"cached"|"built", "seconds"}."""
    from namgpu import config
    row = current()
    t0 = time.perf_counter()
    rec = {"domain": row["name"], "kind": row["static"]}
    if row["static"] == "nam-analysis":
        own = Path(case_dir) / "raw" / "nam" / "bgrd3d00.grib2"
        from namgpu.init import gfs_start
        if source == "native" or own.is_file():
            rec.update(action="case analysis", file=str(own))
        elif gfs_start.STATIC_NPZ.is_file():
            rec.update(action="cached", file=str(gfs_start.STATIC_NPZ))
        else:
            cyc = archive_cycle(start)
            src = paths.work_root() / "static" / "src" / f"nam.{cyc:%Y%m%d%H}.bgrd3d00.grib2"
            src.parent.mkdir(parents=True, exist_ok=True)
            from namgpu.init.fetch_case import NAM, _download
            got = _download(f"{NAM}/nam.{cyc:%Y%m%d}/nam.t{cyc:%H}z.bgrd3d00.tm00.grib2", src)
            log(f"static: {gfs_start.STATIC_NPZ.name} from the archived NAM analysis {cyc:%Y-%m-%d %HZ} ({got})")
            rc = _child("namgpu.init.gfs_start", [str(src), "--out", str(gfs_start.STATIC_NPZ)], card)
            if rc != 0 or not gfs_start.STATIC_NPZ.is_file():
                raise SystemExit(f"woof-nam: static build failed (exit {rc})")
            src.unlink(missing_ok=True)
            rec.update(action="built", file=str(gfs_start.STATIC_NPZ), from_cycle=f"{cyc:%Y%m%d%H}")
    else:
        npz = config.static_npz(config.load())
        if npz is None:
            raise SystemExit(f"woof-nam: configfile {row['configfile_path']} has no static_npz key")
        from namgpu.init.static_build import soil_set
        month = built_month(npz) if npz.is_file() else None
        want = soil_set()
        have = built_meta(npz).get("soil_set", "wrf") if npz.is_file() else None
        if npz.is_file() and month == start.month and have == want:
            rec.update(action="cached", file=str(npz), soil_set=have)
        else:
            why = ("missing" if not npz.is_file() else
                   f"built for month {month}, start is month {start.month}" if month != start.month else
                   f"built from soil set {have}, selected {want}")
            log(f"static: building {npz.name} from WPS_GEOG for {start:%Y-%m-%d} ({why})")
            part = npz.with_name(npz.stem + ".part.npz")
            rc = _child("namgpu.init.static_build", ["--config", row["configfile_path"], "--date",
                                                     f"{start:%Y-%m-%dT%H}", "--out", str(part)], card)
            if rc != 0 or not part.is_file():
                raise SystemExit(f"woof-nam: static build failed (exit {rc})")
            os.replace(part, npz)
            rec.update(action="built", file=str(npz), soil_set=want)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def bc_hours(hours: float) -> int:
    """Hours of lateral boundaries a run of `hours` needs: at least 6, rounded up to the 3-hourly GFS steps
    (boco.NNNN starts every 3 h; a run of h hours reads the files that start before h)."""
    h = max(6, math.ceil(float(hours)))
    return 3 * math.ceil(h / 3)


def start_of(case_dir: Path) -> datetime:
    """The start time from the case folder's name (YYYY-MM-DD_HHz). A folder named otherwise stops in one line."""
    name = Path(case_dir).name
    try:
        return datetime.strptime(name[:14], "%Y-%m-%d_%Hz")
    except ValueError:
        raise SystemExit(f"woof-nam: case folder {name!r}: the name must be the start time, YYYY-MM-DD_HHz (e.g. "
                         f"cases/2025-03-14_12z); rename the folder or fetch into one named so") from None
