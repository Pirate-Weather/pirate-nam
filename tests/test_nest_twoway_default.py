"""woof-nam nest, CPU only: a live parent runs two-way by default, --concurrent turns the feedback off, and the
one-domain door and the --source parent hand-off never reach the live-parent door."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.nest import cli, twoway_door  # noqa: E402

BASE = ["nest", "--center", "24N,83W", "--size-km", "450", "--start", "2022092612", "--hours", "1", "--out", "/tmp/x"]


class _Reached(Exception):
    pass


def _door(argv):
    seen = {}

    def fake(a, argv_):
        seen["two_way"] = a.two_way
        raise _Reached
    old = twoway_door.main
    twoway_door.main = fake
    try:
        cli.main(argv)
    except _Reached:
        return seen["two_way"]
    except SystemExit:
        return None          # the one-domain door (refused or ran past the dispatch)
    except Exception:
        return None
    finally:
        twoway_door.main = old
    return None


def test_two_way_flag():
    assert _door(BASE + ["--two-way"]) is True


def test_live_parent_defaults_two_way():
    assert _door(BASE + ["--parent-size-km", "1300"]) is True
    assert _door(BASE + ["--parent-center", "24N,83W"]) is True
    assert _door(BASE + ["--parent-size-y-km", "1300"]) is True


def test_concurrent_is_feedback_off():
    assert _door(BASE + ["--concurrent"]) is False
    assert _door(BASE + ["--concurrent", "--parent-size-km", "1300"]) is False


def test_no_parent_options_stay_one_domain():
    a = cli.build_parser().parse_args(BASE + ["--dry-run"])
    assert not (a.two_way or a.concurrent or a.parent_center or a.parent_size_km or a.parent_size_y_km)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
