"""CPU checks of the door's --follow (no GPU, no network): python tests/test_follow_door.py or pytest.

  - the static door's arguments parse to the same values as before --follow existed (option off);
  - --follow options are refused by name when they cannot run (bad target, no start position, a check that is not
    whole hours, several cards, a follow option without --follow, a source with no landable row);
  - the first window is centred on the storm unless --center is given;
  - the path target reads its points.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.nest import cli, follow_door  # noqa: E402

BASE = ["nest", "--center", "35.7N,139.7E", "--size-km", "1000", "--res-km", "3", "--source", "gfs",
        "--start", "2024081512", "--hours", "48", "--out", "/tmp/x"]
CYCLE = datetime(2024, 8, 15, 12)


def _refused(argv, want):
    a = cli.build_parser().parse_args(argv)
    try:
        follow_door.resolve(a, CYCLE, cli.parse_center)
    except SystemExit as e:
        assert want in str(e), (argv, str(e))
        return
    raise AssertionError(f"not refused: {argv}")


def test_option_off_namespace():
    a = cli.build_parser().parse_args(BASE)
    assert a.follow is None and a.storm is None and a.storm_id is None and a.follow_config is None
    assert follow_door.resolve(a, CYCLE, cli.parse_center) == {}
    # every option the static door had keeps its value
    want = {"center": "35.7N,139.7E", "size_km": 1000.0, "res_km": 3.0, "source": "gfs", "start": "2024081512",
            "hours": 48.0, "cards": 1, "boco_age_h": None, "no_maps": False, "no_sheets": False, "dry_run": False}
    for k, v in want.items():
        assert getattr(a, k) == v, (k, getattr(a, k), v)


def test_center_still_required_without_follow():
    argv = [x for x in BASE if x not in ("--center", "35.7N,139.7E")]
    with contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            cli.main(argv)
        except SystemExit as e:
            assert e.code == 2
            assert "--center" in err.getvalue()
            return
    raise AssertionError("no --center and no --follow was not refused")


def test_refusals():
    _refused(BASE + ["--follow", "hurricane", "--storm", "31.5,140.8"], "TARGETS")
    _refused(BASE + ["--follow", "tc"], "start position")
    _refused(BASE + ["--follow", "tc", "--storm", "31.5,140.8", "--check-minutes", "30"], "whole number of hours")
    _refused(BASE + ["--follow", "tc", "--storm", "31.5,140.8", "--cards", "2"], "restart set")
    _refused(BASE + ["--storm", "31.5,140.8"], "--follow")
    _refused(BASE + ["--follow", "tc", "--storm", "31.5,140.8", "--storm-id", "2024225N22135"], "give one")
    _refused(BASE + ["--follow", "path"], "--follow path needs")
    _refused(BASE + ["--follow", "tc", "--storm", "31.5,140.8", "--follow-config", "{not json"], "not JSON")
    try:
        follow_door.landable("era5", {})
    except SystemExit as e:
        assert "SOURCE_LANDING" in str(e)
    else:
        raise AssertionError("a source with no landable row was not refused")


def test_storm_centres_the_window():
    argv = [x for x in BASE if x not in ("--center", "35.7N,139.7E")] + ["--follow", "tc", "--storm", "31.5N,140.8E"]
    a = cli.build_parser().parse_args(argv)
    spec = follow_door.resolve(a, CYCLE, cli.parse_center)
    assert spec["storm_start"] == [31.5, 140.8] and spec["target"] == "tc"
    assert follow_door.center_from(spec, a, cli.parse_center) == (31.5, 140.8)
    a2 = cli.build_parser().parse_args(BASE + ["--follow", "tc", "--storm", "31.5N,140.8E"])
    assert follow_door.center_from(follow_door.resolve(a2, CYCLE, cli.parse_center), a2, cli.parse_center) == \
        (35.7, 139.7)
    assert spec["row"]["core_margin_km"] == 150.0
    a3 = cli.build_parser().parse_args(argv + ["--follow-config", '{"core_margin_km": 220}'])
    assert follow_door.resolve(a3, CYCLE, cli.parse_center)["row"]["core_margin_km"] == 220


def test_path_target():
    argv = BASE + ["--follow", "path", "--follow-config",
                   '{"path": [["2024-08-15T12", 31.5, 140.8], ["2024-08-17T12", 37.7, 147.7]]}']
    a = cli.build_parser().parse_args(argv)
    spec = follow_door.resolve(a, CYCLE, cli.parse_center)
    assert spec["storm_start"] == [31.5, 140.8] and len(spec["path"]) == 2
    assert spec["path"][1][0] == "2024-08-17T12:00:00"


def test_landable_rows():
    assert follow_door.landable("gfs", {})["every_h"] == 3
    assert follow_door.landable("parent", {})["every_h"] == 1


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"ok: {n} tests", flush=True)
    os._exit(0)
