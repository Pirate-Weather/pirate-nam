"""CPU checks of a moving-nest segment's side of control.json (namgpu/nest/segment.py) and of the door's window
arithmetic against the move step's configfile (no GPU): python tests/test_follow_segment.py or pytest.

  - landable steps: every landing_every_s of the run, after the segment's first step, before the run's end;
  - a committed stop stops at exactly that step, a committed continue goes on, no decision continues after wait_s;
  - a control.json that is missing is refused by name;
  - the door's history view links each valid time to the newest generation that wrote it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.nest import follow_run, segment  # noqa: E402


def _control(d: Path, landings: dict) -> Path:
    p = d / "control.json"
    p.write_text(json.dumps({"schema": "namgpu.follow-control/v1", "hours": 12, "landing_every_s": 10800.0,
                             "landings": landings}))
    return p


def test_landing_steps():
    with tempfile.TemporaryDirectory() as t:
        fs = segment.FollowStop(_control(Path(t), {}), 6.25, wait_s=0.0)
        assert fs.k == 1728
        assert fs.landing_steps(0, 6912) == [1728, 3456, 5184]
        assert fs.landing_steps(3456, 6912) == [5184]
        assert fs.is_landing(3456) and not fs.is_landing(3455) and not fs.is_landing(6912) and not fs.is_landing(0)


def test_decisions():
    with tempfile.TemporaryDirectory() as t:
        p = _control(Path(t), {"10800": {"action": "continue"}, "21600": {"action": "stop", "stop_step": 3456}})
        fs = segment.FollowStop(p, 6.25, wait_s=0.2, poll_s=0.05)
        assert fs.decide(1728, log=lambda m: None) is False
        assert fs.decide(3456, log=lambda m: None) is True
        msgs = []
        assert fs.decide(5184, log=msgs.append) is False          # no decision: continue after wait_s
        assert msgs and "no decision" in msgs[0]
        r = fs.receipt()
        assert [x["action"] for x in r["landings"]] == ["continue", "stop", "continue (no decision)"]


def test_missing_control_refused():
    with tempfile.TemporaryDirectory() as t:
        try:
            segment.FollowStop(Path(t) / "control.json", 6.25)
        except SystemExit as e:
            assert "control.json" in str(e)
        else:
            raise AssertionError("a missing control.json was not refused")


def test_history_view():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        for g, names in ((0, ["wrfout_d01_2024-08-15_12:00:00", "wrfout_d01_2024-08-15_18:00:00"]),
                         (1, ["wrfout_d01_2024-08-15_18:15:00", "wrfout_d01_2024-08-15_19:00:00"])):
            (out / f"seg-{g}").mkdir()
            for n in names:
                (out / f"seg-{g}" / n).write_text(str(g))
        v = follow_run.history_view(out, out / "run")
        assert v["frames"] == 4 and v["by_generation"] == {"seg-0": 2, "seg-1": 2}
        assert (out / "run" / "wrfout_d01_2024-08-15_19:00:00").read_text() == "1"
        assert (out / "run" / "wrfout_d01_2024-08-15_18:00:00").read_text() == "0"


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"ok: {n} tests", flush=True)
    os._exit(0)
