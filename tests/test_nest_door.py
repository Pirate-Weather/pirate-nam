"""woof-nam nest door, CPU only (no CUDA, no nvidia-smi): the pool cap follows the memory path the forecast takes,
and --parent-run takes the parent door's --out folder or its run/ folder."""
from __future__ import annotations

import pytest

from namgpu.nest import cli, domain

# the Milton Gulf parent (2026-11-05): 12 km, 3000 x 2600 km at 24.5N 87.5W
MILTON_PARENT = dict(lat=24.5, lon=-87.5, size_km=3000.0, res_km=12, size_y_km=2600.0)


@pytest.fixture
def card(monkeypatch):
    """A card of the given total memory with plenty free; nvidia-smi is never called."""
    def use(total_gib, sms=188):
        monkeypatch.setattr(domain, "card_total_gib", lambda index=None: total_gib)
        monkeypatch.setattr(domain, "sm_threads", lambda index=None: sms * 1536)
        monkeypatch.setattr(domain, "card_name", lambda index=None: f"test card {total_gib} GiB")
    monkeypatch.delenv("NAMGPU_MEMORY_PATH", raising=False)
    monkeypatch.delenv("NAMGPU_SCRATCH_ARENA", raising=False)
    return use


def test_milton_parent_grid():
    g = domain.nest_grid(**MILTON_PARENT)
    assert (g["im"], g["jm"], g["lm"]) == (215, 217, 60)


def test_cap_96gb_card_takes_the_full_path(card):
    # 96 GB card: the forecast auto-selects the full memory path (state.FIT_BELOW_GIB = 64), so the cap must be the
    # full path's. The old single model capped this parent at 12.3 GB and it died at the start of its forecast
    # needing 0.9 GB on top of 12.1 GB in use (Milton nest box, 2026-11-05).
    card(95.6)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=95.0)
    assert pf["ok"]
    e = pf["pick"]
    assert e["memory_path"] == "full"
    assert e["pool_limit_bytes"] > 13.0e9          # above what the parent had in use when it died (12.1 + 0.9 GB)
    assert e["pool_limit_gib"] >= domain.MEASURED_POOL_GIB["milton-parent-full-r32"]


def test_cap_32gb_card_takes_the_fit_path(card):
    card(31.8, sms=170)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=31.0)
    assert pf["ok"]
    e = pf["pick"]
    assert e["memory_path"] == "fit"
    assert e["pool_limit_gib"] >= domain.MEASURED_POOL_GIB["milton-parent-fit-r32"]
    full = domain.vram_estimate(g, 32, path="full", index=None)
    assert full["pool_limit_bytes"] > e["pool_limit_bytes"]     # the full path holds every array for the whole run


def test_memory_path_setting_wins(card, monkeypatch):
    card(95.6)
    monkeypatch.setenv("NAMGPU_MEMORY_PATH", "fit")
    assert domain.memory_path()["path"] == "fit"
    monkeypatch.setenv("NAMGPU_MEMORY_PATH", "full")
    card(31.8)
    assert domain.memory_path()["path"] == "full"
    monkeypatch.delenv("NAMGPU_MEMORY_PATH")
    monkeypatch.setenv("NAMGPU_SCRATCH_ARENA", "0")      # the older spelling, as state.memory_path reads it
    assert domain.memory_path()["path"] == "full"


def test_no_card_size_no_cap(card):
    # nvidia-smi gave no total: the door cannot tell which path the forecast takes, so it sets no cap
    card(0.0)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=95.0)
    assert pf["pick"]["pool_limit_bytes"] is None


def _hist(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "wrfout_d01_2024-10-07_00:00:00").write_bytes(b"")


def test_parent_run_door_out_folder(tmp_path):
    out = tmp_path / "parent12"
    _hist(out / "run")
    assert cli.resolve_parent_run(out) == (out / "run").resolve()


def test_parent_run_history_folder(tmp_path):
    out = tmp_path / "parent12"
    _hist(out / "run")
    assert cli.resolve_parent_run(out / "run") == (out / "run").resolve()
    flat = tmp_path / "flat"
    _hist(flat)
    assert cli.resolve_parent_run(flat) == flat.resolve()


def test_parent_run_refusal_names_folders(tmp_path):
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        cli.resolve_parent_run(empty)
    msg = str(e.value)
    assert str(empty) in msg and str(empty / "run") in msg and "wrfout_d01_" in msg
    with pytest.raises(SystemExit):
        cli.resolve_parent_run(None)
