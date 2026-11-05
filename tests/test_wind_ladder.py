"""The wind maps' ladder past 60 kt: a wind frame over 60 kt gets the 0-140 kt scale, one at or under 60 kt keeps the
renderer's own ladder (no scale in its row, so its map does not change).

Origin: the renderer's 10 m wind ladder ends at 60 kt, so hurricane cores were one flat top colour (Helene 3 km nest
f57, 2026-11-05). CPU only, no gpuwm needed. Run: python tests/test_wind_ladder.py
"""
import os
import sys
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from namgpu.io import maps  # noqa: E402


def main() -> int:
    sc = maps.wind_hi_scale()
    lv, col = sc["levels"], sc["colors"]
    assert (lv[0], lv[24], lv[-1], len(lv)) == (0.0, 60.0, 140.0, 57), lv
    assert all(abs((b - a) - 2.5) < 1e-9 for a, b in zip(lv, lv[1:])), "edges are not evenly spaced"
    assert len(col) == len(lv) - 1, (len(col), len(lv))
    assert col[0] == [255, 255, 255] and col[-1] == [0, 0, 0], (col[0], col[-1])
    assert not ({tuple(c) for c in col[:24]} & {tuple(c) for c in col[25:]}), "a colour past 60 kt repeats a lower one"
    under = np.array([[1.0, np.nan], [2.0, 30.0]])                  # 58.3 kt
    over = np.array([[1.0, np.nan], [2.0, 31.0]])                   # 60.3 kt
    assert maps.wind_row_scale("wind10", under) is None
    assert maps.wind_row_scale("wind10", over) == sc
    assert maps.wind_row_scale("wind850", over) == sc
    assert maps.wind_row_scale("t2m", np.full((2, 2), 400.0)) is None
    assert maps.wind_row_scale("wind10", np.full((2, 2), np.nan)) is None
    row = maps.cf_row("wind10", "t", None, sc)
    assert row["scale"] == sc and "scale" not in maps.TABLE["wind10"][1]
    assert "scale" not in maps.cf_row("wind10", "t")
    print("test_wind_ladder: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
