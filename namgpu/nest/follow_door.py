"""The door's --follow: a storm-following WOOF-NAM nest from the one `woof-nam nest` command.

    woof-nam nest --follow tc --storm 31.5N,140.8E --size-km 1000 --res-km 3 --source gfs \
        --start 2024081512 --hours 48 --out runs/ampil-follow
    woof-nam nest --follow tc --storm-id 2024225N22135 ...            (IBTrACS v04r01 start position)
    woof-nam nest --follow low --storm 34.2S,176.0E ...
    woof-nam nest --follow path --follow-config '{"path": [["2024-08-15T12", 31.5, 140.8], ...]}' ...

Without --follow the door is the static nest, byte for byte (nothing here is imported).

What --follow changes:
  - the first window is centred on the storm (--center, when given, overrides it);
  - the forecast runs as segments on windows that shift by whole cells in the nest's own rotated frame (pole fixed at
    the first centre, namgpu/nest/move.py); the tracker reads each closed hourly frame (namgpu/nest/track.py), the
    controller (namgpu/nest/follow.py) decides moves at least one frame ahead and lands them only at landable times
    (follow.SOURCE_LANDING), and the preparer (namgpu/nest/prepare.py) builds the next window's static, start and
    boundaries while the card integrates;
  - the receipt gets a "follow" block (fixes, moves, shock numbers, walls) and the sheets follow the window.

Targets are table rows (track.TARGETS); landable times are source rows (follow.SOURCE_LANDING): a new target or source is a
row, not a code path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# The tables are the tracker's: targets are namgpu/nest/track.py TARGETS rows (the fixer, its knobs and the core
# margin) and landable times are namgpu/nest/follow.py SOURCE_LANDING rows. A new target or source is a row there.


def _targets() -> dict:
    from namgpu.nest import track
    return track.TARGETS


def _sources() -> dict:
    from namgpu.nest import follow
    return follow.SOURCE_LANDING


def add_args(n) -> None:
    """The --follow options on the `nest` subparser (namgpu/nest/cli.py build_parser)."""
    n.add_argument("--follow", default=None, help="follow a feature: " + " | ".join(_targets())
                   + " (the window moves with it; without --follow the nest is static)")
    n.add_argument("--storm", default=None, help="--follow: the feature's position at the start, LAT,LON")
    n.add_argument("--storm-id", default=None, help="--follow: an IBTrACS v04r01 SID (e.g. 2024225N22135); the start "
                   "position is its best-track point at --start")
    n.add_argument("--check-minutes", type=float, default=60.0, help="--follow: how often the tracker reads a "
                   "closed history frame (a multiple of the 60 min history interval)")
    n.add_argument("--follow-config", default=None, help="--follow: JSON (inline or a file) overriding the target "
                   "row and the policy (core_margin_km, max_speed_kmh, path, ...)")


def _json_arg(text):
    if text is None:
        return {}
    p = Path(text)
    s = p.read_text() if p.is_file() else text
    try:
        v = json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--follow-config: not JSON and not a file ({e})")
    if not isinstance(v, dict):
        raise SystemExit("--follow-config: want a JSON object")
    return v


def resolve(a, cycle: datetime, parse_center) -> dict:
    """Check the --follow options against the tables and settle the start position. Returns the follow spec the
    controller is driven with; raises SystemExit naming the breakage for anything it cannot run."""
    if a.follow is None:
        for k in ("storm", "storm_id", "follow_config"):
            if getattr(a, k):
                raise SystemExit(f"--{k.replace('_', '-')} is a --follow option; add --follow "
                                 f"{'|'.join(_targets())}")
        return {}
    from namgpu.nest import track
    row = dict(track.target_row(a.follow))
    row.pop("needs", None)
    row["needs_start"] = row["mode"] != "path"
    cfg = _json_arg(a.follow_config)
    row.update({k: v for k, v in cfg.items() if k not in ("path",)})
    if a.check_minutes <= 0 or a.check_minutes % 60:
        raise SystemExit(f"--check-minutes {a.check_minutes:g}: the tracker reads closed hourly history frames, so "
                         f"a check is a whole number of hours (60, 120, ...)")
    if a.cards > 1:
        raise SystemExit("--follow with --cards > 1: a move hands the state over through the restart set "
                         "(namgpu/restart.py), which is written by one card")
    src = a.source
    storm, meta = None, {}
    if a.storm and a.storm_id:
        raise SystemExit("--storm and --storm-id: give one (the start position)")
    if a.storm:
        storm = parse_center(a.storm)
        meta = {"given": a.storm}
    elif a.storm_id:
        from namgpu.nest import ibtracs
        lat, lon, meta = ibtracs.start_position(a.storm_id, cycle)
        storm = (lat, lon)
    path = cfg.get("path")
    if a.follow == "path":
        if not path or len(path) < 2:
            raise SystemExit("--follow path needs --follow-config '{\"path\": [[\"YYYY-MM-DDTHH\", lat, lon], ...]}' "
                             "with at least two points")
        try:
            path = [[datetime.fromisoformat(str(t)).isoformat(), float(la), float(lo)] for t, la, lo in path]
        except (TypeError, ValueError) as e:
            raise SystemExit(f"--follow-config path: each point is [time, lat, lon] ({e})")
        if storm is None:
            storm = (path[0][1], path[0][2])
    if row["needs_start"] and storm is None:
        raise SystemExit(f"--follow {a.follow} needs the feature's start position: --storm LAT,LON or --storm-id "
                         f"SID (the only outside input; the tracker reads model state from there on)")
    return {"target": a.follow, "row": row, "storm_start": list(storm), "storm_meta": meta, "path": path,
            "check_minutes": a.check_minutes, "config": cfg, "source": src}


def landable(source: str, spec: dict) -> dict:
    """The source's landable row; a source with no row cannot carry a moving nest (no full state at a move)."""
    rows = _sources()
    if source not in rows:
        raise SystemExit(f"--follow with --source {source}: no row in namgpu/nest/follow.py SOURCE_LANDING (rows: "
                         f"{sorted(rows)}): a move needs a full start state from the source at the landing time")
    return rows[source]


def center_from(spec: dict, a, parse_center):
    """The first window's centre: --center when given, else the storm's start position."""
    if a.center:
        return parse_center(a.center)
    return float(spec["storm_start"][0]), float(spec["storm_start"][1])


