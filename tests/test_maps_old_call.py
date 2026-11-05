"""maps._prep_frame keeps its 0.1.0 call: the earlier frame as one path (or None) instead of the {hours back: path}
dict, and the 5-value return (cf, valid, start, CF variables written, skipped). The dict call keeps its 7 values.

Origin: the hosted service worker makes the 0.1.0 call; 0.1.1.dev6 took only the dict, so a worker on it had
no maps (2026-11-05). A dry window in the 0.1.0 call is a skip, not zeros gpuwm's renderer paints domain-wide.
CPU only, numpy only (the history read and the CF write are stubbed). Run: python tests/test_maps_old_call.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from namgpu.io import maps  # noqa: E402

START = datetime(2023, 3, 31, 12)
RAIN = {}                                                   # path name -> RAINNC
WRITTEN = {}                                                # cf path -> source label


def _read(path, need):
    h = int(Path(path).name.split("_f")[1])
    ny, nx = 4, 5
    lat, lon = np.meshgrid(np.linspace(30, 40, ny), np.linspace(-100, -90, nx), indexing="ij")
    f = {"XLAT": lat, "XLONG": lon, "T2": np.full((ny, nx), 290.0), "RAINC": np.zeros((ny, nx)),
         "RAINNC": RAIN.get(Path(path).name, np.zeros((ny, nx)))}
    return START + timedelta(hours=h), START, {k: v for k, v in f.items() if k in need | {"XLAT", "XLONG"}}


def _write_cf(path, lat, lon, valid, fields, label):
    WRITTEN[str(path)] = (label, sorted(fields))


def main() -> int:
    maps._read, maps._write_cf = _read, _write_cf
    work = tempfile.mkdtemp()
    need = sorted({"XLAT", "XLONG", "T2", "RAINNC", "RAINC"})
    prods = ["t2m", "qpf_1h"]
    wet = np.zeros((4, 5))
    wet[1, 2] = 5.0                                          # 5 mm in the hour at one point
    RAIN["h_f001"] = wet

    # the 0.1.0 call with the frame one hour back: 5 values, both maps, the old source line, no table written
    r = maps._prep_frame(("h_f001", "h_f000", need, prods, work, "WOOF-NAM"))
    assert len(r) == 5, r
    cf, valid, start, have, sk = r
    assert (valid, start) == (START + timedelta(hours=1), START), r
    assert have == ["qpf1h", "t2m"] and sk == [], r
    assert WRITTEN[cf][0] == "WOOF-NAM, init 2023-03-31 12Z +1 h", WRITTEN[cf]
    assert not list(Path(work).glob("cf-table_*.json")), "the 0.1.0 call writes no table"

    # the 0.1.0 call with no earlier frame: 5 values, the rain map skipped
    cf, valid, start, have, sk = maps._prep_frame(("h_f001", None, need, prods, work, "WOOF-NAM"))
    assert have == ["t2m"] and [s[0] for s in sk] == ["qpf_1h"], (have, sk)

    # a dry hour in the 0.1.0 call: skipped, not a field of zeros
    RAIN["h_f002"] = wet
    cf, valid, start, have, sk = maps._prep_frame(("h_f002", "h_f001", need, prods, work, "WOOF-NAM"))
    assert have == ["t2m"] and [s[0] for s in sk] == ["qpf_1h"] and "dry" in sk[0][1], (have, sk)

    # the dict call: 7 values (table, frame bounds); a wet hour draws, a dry hour is skipped too (gpuwm's renderers
    # paint a field of zeros domain-wide in the first colour; only the tree's patched renderer reads mask_below)
    r = maps._prep_frame(("h_f001", {1: "h_f000"}, need, prods, work, "WOOF-NAM"))
    assert len(r) == 7 and r[3] == ["qpf1h", "t2m"] and Path(r[5]).is_file() and len(r[6]) == 4, r
    r = maps._prep_frame(("h_f002", {1: "h_f001"}, need, prods, work, "WOOF-NAM"))
    assert len(r) == 7 and r[3] == ["t2m"] and [s[0] for s in r[4]] == ["qpf_1h"] and "dry" in r[4][0][1], r
    print("test_maps_old_call: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
