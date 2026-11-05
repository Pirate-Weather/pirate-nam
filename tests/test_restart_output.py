"""CPU checks of restarts, history field selection and the product index (no GPU, no network).

Run: python tests/test_restart_output.py   (or pytest). The bit-for-bit restart proof is a real run (README "Restarts").
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import restart  # noqa: E402
from namgpu.io import history_select, products  # noqa: E402


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


def test_history_presets():
    fields = {n: None for n in history_select.KNOWN}
    sel = history_select.Selection("products", restart.steps_for(3.0, 25.0))
    assert sel.pick(0, fields) == fields                      # hour 0: full
    assert sel.pick(3 * 144, fields) == fields                # every 3 h: full
    hour1 = sel.pick(144, fields)
    # the awphys GRIB2 reads W (VVEL, DZDT) and the physics carriers every hour
    assert {"T", "QVAPOR", "PRES", "PH", "U", "V", "W", "COMPOSITE", "T2", "RAINNC", "TWBS", "T02MAX",
            "TKE_PBL"} <= set(hour1)
    listed = history_select.Selection("T2,Q2", restart.steps_for(3.0, 25.0)).pick(144, fields)
    assert set(listed) == {"T2", "Q2"}
    assert set(history_select.Selection("full", 0).pick(144, fields)) == set(fields)
    assert set(history_select.Selection("T2,Q2", 0).pick(144, fields)) == {"T2", "Q2"}
    assert "TSLB" in history_select.Selection("T2,Q2", 0).pick(144, dict(fields, TSLB=None))   # added sets stay
    for preset in ("products", "state"):
        assert set(history_select.PRESETS[preset]) <= set(history_select.KNOWN), preset
    try:
        history_select.parse("T2,NOT_A_FIELD")
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown field must stop the run before the init")


def test_maps_and_side_by_side_read_only_products_fields():
    """The products preset carries everything woof-nam's maps and side-by-side panels read."""
    from namgpu.io import io_referee_panels, maps
    need = set().union(*maps.NEEDS.values()) | set(io_referee_panels.NAM_GPU_NEEDS) - {"XLAT", "XLONG"}
    assert need <= set(history_select.PRESETS["products"]), need - set(history_select.PRESETS["products"])
    assert set(products.NEED) <= set(history_select.PRESETS["products"])


def test_idx_lines():
    ref = datetime(2025, 3, 14, 12)
    recs = [
        {"msg": 1, "sub": None, "offset": 0, "key": "0.3.1", "ref": ref, "unit": 1, "ftime": 6, "pdt": 0,
         "t1": 101, "v1": 0.0, "t2": 255, "v2": None},
        {"msg": 2, "sub": 1, "offset": 512, "key": "0.2.2", "ref": ref, "unit": 1, "ftime": 6, "pdt": 0,
         "t1": 100, "v1": 50000.0, "t2": 255, "v2": None},
        {"msg": 3, "sub": None, "offset": 900, "key": "0.1.8", "ref": ref, "unit": 1, "ftime": 0, "pdt": 8,
         "stat": 1, "range_unit": 1, "range_len": 6, "t1": 1, "v1": 0.0, "t2": 255, "v2": None},
        {"msg": 4, "sub": None, "offset": 1200, "key": "0.7.6", "ref": ref, "unit": 1, "ftime": 6, "pdt": 0,
         "t1": 108, "v1": 18000.0, "t2": 108, "v2": 0.0},
    ]
    assert products.idx_lines(recs) == [
        "1:0:d=2025031412:PRMSL:mean sea level:6 hour fcst:",
        "2.1:512:d=2025031412:UGRD:500 mb:6 hour fcst:",
        "3:900:d=2025031412:APCP:surface:0-6 hour acc fcst:",
        "4:1200:d=2025031412:CAPE:180-0 mb above ground:6 hour fcst:",
    ]


def test_nam_levels():
    assert products.NAM_LEVELS_HPA[0] == 50 and products.NAM_LEVELS_HPA[-1] == 1000
    assert len(products.NAM_LEVELS_HPA) == 39


def test_set_leaves_out_full_path_scratch():
    """A full-memory-path set leaves out what the fit path has released at the hour (restart.transient): step,
    history, unit and init-only scratch, the RRTM chunk context and driver-prep tables, the init-only terrain."""
    import types
    from namgpu import state as S
    ctx = types.SimpleNamespace(tables={"rrtm_call:chunkctx": object(), "rrtm_prep:out": {}, "rrtm_prep:KTOP": 1,
                                        "rrtm_prep:vec:SGML": 2, "hdiff_x": 3, "gwd_dem": 4, "co2_table": 5})
    lo = restart.transient(ctx)
    assert set(lo["tables"]) == {"rrtm_call:chunkctx", "rrtm_prep:out", "rrtm_prep:KTOP", "hdiff_x", "gwd_dem"}
    assert set(S.STEP_SCRATCH) | set(S.HISTORY_SCRATCH) | set(S.INIT_ONLY) <= set(lo["fields"])
    assert "history_fields" in lo["attrs"]


def test_unsaveable_entries_are_left_unset():
    """An entry the walker cannot save is left out of its dict (not a _Skipped placeholder a resumed unit would
    read: `--resume` died on "'_Skipped' object has no attribute 'w'"), and old sets' placeholders are dropped."""
    import threading
    from namgpu.init import slab_init
    sv = slab_init._Saver()
    out = sv.walk({"keep": 1, "lock": threading.Lock(), "inner": {"lock": threading.Lock(), "n": 2}}, "tables")
    assert out == {"keep": 1, "inner": {"n": 2}}, out
    assert len(sv.skipped) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
