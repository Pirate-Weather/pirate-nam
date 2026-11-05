"""CPU checks of the ERA5 start source row (no GPU, no network).

Run: python tests/test_era5_source.py   (or pytest tests/test_era5_source.py). CI runs it.
  - era5 is a SOURCES row and a record table on the shared isobaric path (no new code path): every field the
    start, the land surface and the boundaries read is mapped to a gpuwm regular-forcing field or derived;
  - the soil remap from ERA5's four layers to Noah's four keeps every target layer whole;
  - the fetch plan: one gpuwm ARCO-ERA5 job per boundary time into the row's per-time folder, the domain's
    latitude band with every longitude; an hour past the store's last published day is refused with that day.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from namgpu.init import fetch_case, source_records as srec, sources  # noqa: E402

# what gfs_start.stage, columns() and the boundaries read from a source's surface fields
STAGE_SFC = ("PRES", "HGT", "TSK", "T2", "RH2", "Q2", "U10", "V10", "TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4",
             "SOILW1", "SOILW2", "SOILW3", "SOILW4", "CNWAT", "WEASD", "SNOD", "ICEC", "LANDSRC")
# gpuwm.ingest.regular_netcdf.REGULAR_FIELD_UNITS keys that gpuwm.era5_arco writes (gpuwm 2.9.11)
GPUWM_ARCO_FIELDS = {"Z", "T", "U", "V", "SPFH", "PRES", "Q2", "U10", "V10", "T2", "D2", "LANDSEA", "SOILGEO",
                     "PSFC", "PMSL", "SKINTEMP", "SEAICE", "SST", "SNOW_EC", "ST000007", "ST007028", "ST028100",
                     "ST100289", "SM000007", "SM007028", "SM028100", "SM100289", "LAKE_WATER_TEMP",
                     "LAKE_ICE_TEMP", "LAKE_ICE_DEPTH"}


def _derived(rec):
    out = set()
    for o, _, _ in rec["sfc_derive"]:
        out |= set(o if isinstance(o, tuple) else (o,))
    return out


def test_era5_row_on_the_shared_path():
    row = sources.SOURCES["era5"]
    assert row["kind"] == "isobaric" and row["gdt"] == 0 and row["records"] == "era5"
    assert row["licence"] == "CC-BY-4.0" and "Copernicus" in row["attribution"]
    assert row["boundary"]["age_h"] == 0 and row["boundary"]["interval_h"] == 3   # NAM boco interval, 10800 s
    assert row["file"].format(cyc=12) == row["boundary"]["file"].format(bcyc=12, lead=0)
    rec = srec.table(row)
    assert rec is srec.ERA5 and rec["container"] == "regular-netcdf" and not rec["donors"]


def test_era5_record_table_complete():
    rec = srec.ERA5
    assert set(rec["pl"].values()) <= GPUWM_ARCO_FIELDS
    assert set(rec["sfc"].values()) <= GPUWM_ARCO_FIELDS
    have = set(rec["sfc"]) | _derived(rec)
    missing = [k for k in STAGE_SFC if k not in have]
    assert not missing, missing
    # derive inputs exist before use (decoded or an earlier output)
    seen = set(rec["sfc"])
    for o, op, args in rec["sfc_derive"]:
        names = [a for a in (args[0] if op == "layer_remap" else args) if isinstance(a, str)]
        assert set(names) <= seen, (o, names)
        seen |= set(o if isinstance(o, tuple) else (o,))
    # boundary times decode only the columns; their derived outputs come from those columns
    assert set(rec["sfc_columns"]) <= set(rec["sfc"])
    for o, op, args in rec["sfc_derive"]:
        outs = o if isinstance(o, tuple) else (o,)
        if any(x in rec["derive_columns"] for x in outs):
            assert {a for a in args if isinstance(a, str)} <= set(rec["sfc_columns"]) | {"HGT", "Q2"}, o
    # pl: HGT from geopotential before RH from q (the vinterp reads gpm and RH)
    assert list(rec["pl_derive"]) == ["HGT", "RH"]
    assert rec["pl_derive"]["HGT"] == ("scale", ("HGT", 1.0 / srec.G0))


def test_soil_layers_remap_whole():
    w = srec.conservative_overlap_weights(srec.IFS_SOIL_LAYERS, srec.NOAH_SOIL_LAYERS)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w[0], [0.7, 0.3, 0.0, 0.0])          # Noah 0-10 cm = 70 % ERA5 0-7 + 30 % 7-28


def test_era5_fetch_plan(tmp=None):
    real_get = fetch_case._get
    stop = {"valid_time_stop": "2026-06-30", "valid_time_stop_era5t": "2026-10-29"}
    fetch_case._get = lambda url, rng=None, timeout=120: json.dumps(stop).encode()
    try:
        raw = Path("/nonexistent-case/raw")
        made = []
        real_mkdir = Path.mkdir
        Path.mkdir = lambda self, *a, **k: made.append(self)
        try:
            jobs = fetch_case.plan_source("era5", datetime(2025, 3, 14, 12), 24, raw)
        finally:
            Path.mkdir = real_mkdir
        assert len(jobs) == 9
        fn, fetch, valid, area, dest = jobs[3]
        assert fn is fetch_case._arco_time and valid == datetime(2025, 3, 14, 21)
        assert dest == raw.parent / "raw/era5/t12z.f009/era5-combined.nc"
        south, west, north, east = (float(x) for x in area.split(","))
        assert (west, east) == (-180.0, 180.0) and north == 90.0 and -11.0 <= south <= -9.0   # NAM 12 km: -7.5 N
        try:
            fetch_case.plan_source("era5", datetime(2026, 10, 29, 12), 24, raw)
        except ValueError as e:
            assert "2026-10-29 23Z" in str(e), e
        else:
            raise AssertionError("an hour past the store's last day was not refused")
    finally:
        fetch_case._get = real_get


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("test_era5_source: PASS")
