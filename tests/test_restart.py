"""CPU checks of restart options (no GPU, no network).

Run: python tests/test_restart.py   (or pytest). The bit-for-bit restart proof is a real run (README "Restarts").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import restart  # noqa: E402


def test_parse_hours():
    assert restart.parse_hours("1h") == 1.0
    assert restart.parse_hours("90m") == 1.5
    assert restart.parse_hours("3") == 3.0
    assert restart.parse_hours("0") == 0.0 and restart.parse_hours("off") == 0.0
    assert restart.steps_for(1.0, 25.0) == 144
    try:
        restart.parse_hours("hourly")
    except ValueError:
        pass
    else:
        raise AssertionError("a bad duration must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
