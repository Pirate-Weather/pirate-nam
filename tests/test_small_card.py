"""The 0.1.2 defects found on an 8 GB card (RTX 3060 Ti), CPU only (no CUDA, no nvidia-smi): one test per defect.

1. radiation in 2-row chunks hung (rrtm_chunk.windows never ended for w <= 2) and the door's row ladder handed out 2;
2. the door's memory estimate was ~2.5 GiB above the measured peak on that card;
3. `woof-nam run` had no memory check (12 km on 8 GB: a raw OutOfMemoryError);
4. `--source era5` had no row in the nest door's FETCH table;
5. `woof-nam products` on a nest history: IndexError on every hour;
6. no `woof-nam doctor`; a case folder not named YYYY-MM-DD_HHz gave a strptime traceback; the receipt's VRAM read
   0.0 GiB in a container whose PID namespace hides the card's process ids.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from namgpu.nest import cli, domain

# the 3060 Ti's nests (centre 38.5N 91W, 3 km), as the door printed them: (size_km, size_y_km) -> (im, jm)
NESTS = {"99x100": (345.0, 297.0, 99, 100), "328x330": (1148.0, 989.0, 328, 330), "360x359": (1260.0, 1076.0, 360, 359),
         "378x377": (1323.0, 1129.0, 378, 377)}
FREE_3060TI = 7.64          # GiB free on the idle card (the door's own reading)


def nest(key):
    sx, sy, im, jm = NESTS[key]
    g = domain.nest_grid(38.5, -91.0, sx, 3, sy)
    assert (g["im"], g["jm"]) == (im, jm), (key, g["im"], g["jm"])
    return g


@pytest.fixture
def card_3060ti(monkeypatch):
    monkeypatch.setattr(domain, "card_total_gib", lambda index=None: 8.0)
    monkeypatch.setattr(domain, "card_name", lambda index=None: "NVIDIA GeForce RTX 3060 Ti")
    monkeypatch.setattr(domain, "free_gib", lambda index=None: FREE_3060TI)
    monkeypatch.delenv("NAMGPU_MEMORY_PATH", raising=False)
    monkeypatch.delenv("NAMGPU_SCRATCH_ARENA", raising=False)


# ---- 1. rows-2 hang
def test_windows_terminates_for_every_row_count():
    # in a child with a timeout: 0.1.2's windows(377, 2) never returned (the list grew until the host ran out)
    code = textwrap.dedent("""
        from namgpu.physics.rrtm.rrtm_chunk import windows
        for jm in (5, 10, 101, 359, 377, 835):
            for w in range(1, 40):
                try:
                    out = windows(jm, w)
                except ValueError as e:
                    assert w < 3 and w < jm and "at least 3" in str(e), (jm, w, e)
                    continue
                assert w >= 3 or w >= jm, (jm, w)
                owned = [r for (_, _, a, b) in out for r in range(a, b)]
                assert owned == list(range(jm)), (jm, w)
                assert all(w1 - w0 == min(w, jm) and w0 <= max(a - 1, 0) and b <= w1 for (w0, w1, a, b) in out)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def test_door_ladder_never_hands_out_fewer_than_4_rows(card_3060ti):
    assert min(domain.RRTM_ROW_CHOICES) >= 4
    for n in range(300, 520, 7):
        pf = domain.preflight({"im": n, "jm": n})
        assert (not pf["ok"]) or pf["pick"]["rrtm_rows"] >= 4, (n, pf)


def test_nest_door_refuses_rrtm_rows_2(tmp_path, capsys):
    rc = cli.main(["nest", "--center", "38.5N,91W", "--size-km", "300", "--res-km", "3", "--start", "2025031412",
                   "--hours", "1", "--rrtm-rows", "2", "--out", str(tmp_path / "n"), "--dry-run"])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 2 and len(err) == 1 and "--rrtm-rows 2" in err[0], err


# ---- 2. estimate vs the measured peaks
def test_estimate_within_10_percent_of_the_3060ti_peaks(card_3060ti):
    for key, rows, top in (("328x330", 8, 1.10), ("360x359", 4, 1.10), ("99x100", 32, 1.10), ("378x377", 16, 1.20)):
        e = domain.vram_estimate(nest(key), rows)
        peak = domain.MEASURED_PEAK_GIB[f"3060ti-{key}-r{rows}"]
        assert e["memory_path"] == "fit"
        assert e["total_gib"] >= peak, (key, e, peak)
        assert e["total_gib"] <= top * peak, (key, e, peak)
        # the pool cap holds the measured pool (peak less local memory and context)
        assert e["pool_limit_gib"] >= peak - e["lmem_gib"] - e["fixed_gib"], (key, e, peak)


def test_8gb_card_admits_the_nests_it_runs(card_3060ti):
    # 0.1.2 refused 415 x 414 ("needs about 7.73 GiB at rows 2") and gave 360 x 359 rows 4 (142.8 ms/step vs 71.0)
    pf = domain.preflight({"im": 415, "jm": 414})
    assert pf["ok"] and pf["pick"]["rrtm_rows"] >= 8, pf
    pf = domain.preflight(nest("360x359"))
    assert pf["ok"] and pf["pick"]["rrtm_rows"] >= 8, pf
    pf = domain.preflight(nest("378x377"))
    assert pf["ok"] and pf["pick"]["rrtm_rows"] >= 4, pf


# ---- 3. woof-nam run memory check
def test_run_refuses_12km_on_an_8gb_card(card_3060ti, monkeypatch, capsys):
    from argparse import Namespace

    from namgpu import run_forecast
    monkeypatch.delenv("NAMGPU_PREFLIGHT_DONE", raising=False)
    monkeypatch.delenv("NAMGPU_PREFLIGHT", raising=False)
    monkeypatch.delenv("NAMGPU_RRTM_ROWS", raising=False)
    monkeypatch.delenv("NAMGPU_CONFIGFILE", raising=False)
    monkeypatch.setattr(domain, "sm_threads", lambda index=None: 38 * 1536)
    rc = run_forecast.memory_check(Namespace(cards=1, card_list=""))
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 3 and len(err) == 1, err
    assert "954 x 835" in err[0] and "8.0 GiB" in err[0] and "7.64 GiB free" in err[0], err
    # a 96 GB card: no refusal
    monkeypatch.setattr(domain, "free_gib", lambda index=None: 94.0)
    monkeypatch.setattr(domain, "card_total_gib", lambda index=None: 95.6)
    assert run_forecast.memory_check(Namespace(cards=1, card_list="")) == 0
    # NAMGPU_RRTM_ROWS 2 is refused before the run starts
    monkeypatch.setenv("NAMGPU_RRTM_ROWS", "2")
    assert run_forecast.memory_check(Namespace(cards=1, card_list="")) == 3


# ---- 4. ERA5 on the nest door
def test_era5_is_a_nest_source_with_the_nest_band(tmp_path):
    fn, lic = cli.resolve_fetch("era5")
    assert fn is cli.fetch_rows and "CC BY 4.0" in lic
    from namgpu import config
    from namgpu.init import fetch_case
    g = domain.nest_grid(60.0, 10.0, 300.0, 3)
    cfgfile = tmp_path / "nam_configfile"
    domain.write_configfile(cfgfile, g, "t", tmp_path / "static.npz")
    south, _, north, _ = (float(x) for x in fetch_case.domain_band(1.0, config.load(cfgfile)).split(","))
    assert 55.0 <= south < 60.0 < north <= 65.0, (south, north)      # the nest's band, not the CONUS grid's


# ---- 5. products on a nest history
def _history(path: Path, nx: int, ny: int):
    from netCDF4 import Dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as d:
        d.createDimension("west_east", nx)
        d.createDimension("south_north", ny)


def test_products_refuses_a_nest_history(tmp_path, capsys, monkeypatch):
    from namgpu.io import products
    monkeypatch.delenv("NAMGPU_CONFIGFILE", raising=False)
    _history(tmp_path / "nest" / "wrfout_d01_2025-03-14_12:00:00", 328, 330)
    _history(tmp_path / "c12" / "wrfout_d01_2025-03-14_12:00:00", 954, 835)
    assert products.grid_refusal(tmp_path / "c12") is None
    why = products.grid_refusal(tmp_path / "nest")
    assert why and "328 x 330" in why and "12 km CONUS" in why, why
    assert products.main(["--run", str(tmp_path / "nest")]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("woof-nam products: refused:"), err


# ---- 6. doctor, case folder name, container VRAM sampling
def test_doctor_is_a_command_and_reads_the_card(card_3060ti, monkeypatch):
    from namgpu import cli as top
    from namgpu import doctor
    assert top.COMMANDS["doctor"][0] == "namgpu.doctor"
    monkeypatch.setattr(domain, "sm_threads", lambda index=None: 38 * 1536)
    lines = doctor.what_fits({"index": "0", "free_gib": FREE_3060TI})
    assert lines[0].startswith("12 km CONUS") and ": no," in lines[0], lines
    assert "333 x 333 (woof-nam nest): yes" in lines[2], lines
    n = int(lines[3].split("about ")[1].split(" x ")[0])
    assert n >= 400, lines


def test_case_folder_name_is_one_line():
    from namgpu import domains
    with pytest.raises(SystemExit) as e:
        domains.start_of(Path("cases/c12k"))
    assert "YYYY-MM-DD_HHz" in str(e.value) and "\n" not in str(e.value)
    assert domains.start_of(Path("cases/2025-03-14_12z")).hour == 12


def test_sampler_reads_the_whole_card_when_pids_are_hidden(monkeypatch):
    class R:
        def __init__(self, out):
            self.stdout = out

    def fake(cmd, **kw):
        q = " ".join(cmd)
        if "compute-apps" in q:
            return R("4242424, 4100\n")              # the host's pid, not one in this container
        if "memory.used" in q:
            return R("4120\n")
        if "utilization.gpu" in q:
            return R("97\n")
        if "pmon" in q:
            return R("# gpu pid type sm mem enc dec command\n    0 4242424 C 97 40 - - python\n")
        raise AssertionError(q)
    monkeypatch.setattr(cli.subprocess, "run", fake)
    assert cli._gpu_mib([1234], base_mib=20.0) == 4100.0
    assert cli._gpu_util([1234]) == 97.0

    def same_ns(cmd, **kw):                          # a host that sees its own pids: per-process, as before
        q = " ".join(cmd)
        if "compute-apps" in q:
            return R(f"1, 500\n")                    # pid 1 exists here, and is not ours
        if "pmon" in q:
            return R("    0 1 C 50 10 - - other\n")
        raise AssertionError(q)
    monkeypatch.setattr(cli.subprocess, "run", same_ns)
    assert cli._gpu_mib([1234]) == 0.0
    assert cli._gpu_util([1234]) == 0.0
