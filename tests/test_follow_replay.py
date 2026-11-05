"""woof-nam nest --follow: the real controller (namgpu.nest.follow.Controller) on a simulated clock.

Every closed hourly frame is a synthetic vortex at the IBTrACS v04r01 best-track position (NOAA NCEI, public
domain; tests/data/ibtracs_*.json) on the current window's own grid points, so a storm outside the window cannot be
fixed. The checks are the done rule: every hour the best-track centre is inside the window's safe box (the
interior inside the lnsh boundary rows, less the target's core margin), nothing raises, and the move list (times,
centres, di/dj, planned or forced) is printed and saved (replay.json beside follow-state.json and control.json).

  Ampil (JMA, RSMC Tokyo) from 2024-08-15 12Z and Gabrielle (RSMC Wellington, BoM) from 2023-02-13 00Z, 48 h each,
  GFS landings every 3 h; Ampil again with our own 12 km parent (landings every hour); a resume from the saved state
  half way gives the same moves as the unbroken run.

Run alone (prints the move lists): python tests/test_follow_replay.py [OUT_DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.nest import follow, track  # noqa: E402

DATA = ROOT / "tests" / "data"
CASES = {"ampil": DATA / "ibtracs_ampil_2024.json", "gabrielle": DATA / "ibtracs_gabrielle_2023.json"}


def _check(res: dict):
    assert res["truth"], "no frames"
    assert len(res["truth"]) == int(res["hours"]) + 1, "a frame is missing"
    bad = [r for r in res["truth"] if r["room_km"] < 0 or r.get("room_new_km", 0.0) < 0]
    assert not bad, f"best-track centre outside the safe box: {bad[:3]}"
    assert res["moves"], "the storm moved more than a window's safe half-width but nothing moved"
    for m in res["moves"]:
        assert m["kind"] in ("planned", "forced", "forced-ready")
        assert m["at_s"] % res["land_every_s"] == 0, f"a move landed off a landable time: {m}"
        assert m["di"] or m["dj"]
    errs = [r["fix_error_km"] for r in res["truth"] if "fix_error_km" in r]
    assert len(errs) == len(res["truth"]), "a frame had no fix"
    assert max(errs) < 15.0, f"the TC fixer is {max(errs)} km off the synthetic vortex"


@pytest.mark.parametrize("name", sorted(CASES))
def test_replay_gfs_landings(name, tmp_path):
    res = follow.replay(follow.load_track(CASES[name]), hours=48.0, out=tmp_path / name, log=lambda *_: None)
    follow.print_moves(res)
    _check(res)
    saved = json.loads((tmp_path / name / "replay.json").read_text())
    assert saved["moves"] == json.loads(json.dumps(res["moves"], default=float))
    control = json.loads((tmp_path / name / "control.json").read_text())
    assert control["schema"] == follow.SCHEMA_CONTROL
    stops = [v for v in control["landings"].values() if v["action"] == "stop"]
    assert len(stops) == len(res["moves"])


def test_replay_parent_hourly(tmp_path):
    res = follow.replay(follow.load_track(CASES["ampil"]), hours=48.0, source="parent", prep_s=3600.0,
                        out=tmp_path / "ampil-parent", log=lambda *_: None)
    follow.print_moves(res)
    _check(res)
    assert res["land_every_s"] == 3600.0


def test_resume_gives_the_same_moves(tmp_path):
    tr = follow.load_track(CASES["ampil"])
    whole = follow.replay(tr, out=tmp_path / "whole", log=lambda *_: None)
    follow.replay(tr, out=tmp_path / "split", stop_after_s=20 * 3600.0, log=lambda *_: None)
    rest = follow.replay(tr, out=tmp_path / "split", resume_from=tmp_path / "split" / "follow-state.json",
                         log=lambda *_: None)
    assert [(m["at_s"], m["to"]) for m in rest["moves"]] == [(m["at_s"], m["to"]) for m in whole["moves"]]


def test_window_geometry():
    from namgpu.nest import domain
    g = domain.nest_grid(31.6, 140.8, 1000.0, 3)
    w = follow.Window.from_grid(g, lnsh=5)
    assert (w.im, w.jm) == (286, 334)
    assert abs(w.wbd - g["wbd"]) < 1e-9 and abs(w.sbd - g["sbd"]) < 1e-9
    lat, lon = w.centre()
    assert abs(lat - 31.6) < 1e-9 and abs(lon - 140.8) < 1e-9
    s = w.at(12, -7)
    assert abs((s.wbd - w.wbd) - 12 * w.dlmd) < 1e-12 and abs((s.sbd - w.sbd) + 7 * w.dphd) < 1e-12
    assert s.offset_for(*s.centre()) == (12, -7)
    # the interior half-width: (285/2 - 5) cells of 0.0315 deg in x, (333/2 - 5) of 0.027 deg in y
    assert 480.0 < w.interior_half_km() < 486.0
    assert w.edge_km(lat, lon) == pytest.approx(w.interior_half_km(), rel=1e-6)
    # a point just outside the east interior edge
    elat, elon = follow.to_geographic(w.half_x - 4 * w.dlmd, 0.0, w.tph0d, w.tlm0d)
    assert w.edge_km(float(elat), float(elon)) < 0
    far = w.at(0, int(46.0 / w.dphd))
    assert far.refusal() and "rotated latitude" in far.refusal()


def test_dateline_window():
    """Gabrielle's windows cross 180: rotated coordinates are continuous there."""
    from namgpu.nest import domain
    w = follow.Window.from_grid(domain.nest_grid(-37.8, 179.9, 1000.0, 3))
    assert w.edge_km(-37.8, -179.6) > 400.0
    assert w.offset_for(-37.8, 179.9) == (0, 0)


def test_refuse_history_preset():
    track.refuse_history("products", "tc")
    track.refuse_history("full", "object:uh")
    with pytest.raises(SystemExit, match="drops MSLP or PSFC\\+HGT\\+T2"):
        track.refuse_history("state", "tc")
    with pytest.raises(SystemExit, match="UPHLMAX25"):
        track.refuse_history("T2,U10,V10,COMPOSITE", "object:uh")
    track.refuse_history("T2,PSFC,HGT,PRES,U,V", "low")
    with pytest.raises(SystemExit, match="no row"):
        track.target_row("hail")


def _wrfout(path: Path, top_first: bool, with_mslp=True):
    import netCDF4
    from namgpu.nest import domain
    w = follow.Window.from_grid(domain.nest_grid(31.6, 140.8, 600.0, 3))
    lat, lon = w.grid(1)
    ny, nx = lat.shape
    x, y = track.project(31.5, 140.7, lat, lon)
    r = np.maximum(np.hypot(x, y), 1e-3)
    mslp = 95000.0 + 6000.0 * np.exp(-(40.0 / r) ** 1.4)
    vt = np.where(r < 40.0, 40.0 * r / 40.0, 40.0 * (40.0 / r) ** 0.6)
    u2, v2 = -vt * y / r, vt * x / r
    levels = np.array([100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 30000.0, 20000.0, 10000.0])
    pres = np.broadcast_to(levels[:, None, None], (levels.size, ny, nx)).copy()
    fac = np.array([1.0, 1.0, 1.0, 0.85, 0.3, 0.0, 0.0, 0.0])[:, None, None]
    u3, v3 = fac * u2, fac * v2
    if top_first:
        pres, u3, v3 = pres[::-1], u3[::-1], v3[::-1]
    with netCDF4.Dataset(str(path), "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("bottom_top", levels.size)
        nc.createDimension("south_north", ny)
        nc.createDimension("west_east", nx)
        two = dict(XLAT=lat, XLONG=lon, PSFC=mslp, HGT=np.zeros_like(lat), T2=np.full_like(lat, 300.0),
                   U10=0.8 * u2, V10=0.8 * v2)
        if with_mslp:
            two["MSLP"] = mslp
        for k, a in two.items():
            nc.createVariable(k, "f4", ("Time", "south_north", "west_east"))[0] = a
        for k, a in dict(PRES=pres, U=u3, V=v3).items():
            nc.createVariable(k, "f4", ("Time", "bottom_top", "south_north", "west_east"))[0] = a


@pytest.mark.parametrize("top_first", [False, True])
def test_history_adapter(tmp_path, top_first):
    p = tmp_path / "wrfout_d01_2024-08-15_12:00:00"
    _wrfout(p, top_first)
    f = track.frame_fields(p, "tc", guess=(31.6, 140.8), radius_km=700.0)
    assert f.u_p is not None and 85000.0 in f.u_p and np.isfinite(f.u_p[85000.0]).sum() > 1000
    fx = track.TCFixer().fix(0.0, f, (31.6, 140.8), 150.0)
    assert fx.ok and "vort850" in fx.reason
    mlat, mlon, mp = track.mslp_minimum(f)
    assert float(track.distance_km(fx.lat, fx.lon, 31.5, 140.7)) < 10.0
    assert float(track.distance_km(fx.lat, fx.lon, mlat, mlon)) < 10.0 and mp < 951.0
    assert track.frame_time(p).hour == 12


def test_history_adapter_refuses_missing(tmp_path):
    import netCDF4
    p = tmp_path / "wrfout_d01_2024-08-15_13:00:00"
    _wrfout(p, False)
    with netCDF4.Dataset(str(p), "a") as nc:
        nc.renameVariable("U", "U_GONE")
    with pytest.raises(SystemExit, match="has no U"):
        track.frame_fields(p, "tc")


def test_live_run_refuses_without_its_pieces(tmp_path):
    """The live controller needs the preparer's preparer and the move step's segment start; without them it says so by name
    before anything starts."""
    import importlib.util
    if importlib.util.find_spec("namgpu.nest.prepare") and importlib.util.find_spec("namgpu.nest.move"):
        pytest.skip("both pieces are in this tree")
    with pytest.raises(SystemExit, match="is not in this tree"):
        follow.main(["run", "--out", str(tmp_path), "--storm", "31.6N,140.8E", "--start", "2024081512"])


def test_control_stop_step(tmp_path):
    res = follow.replay(follow.load_track(CASES["gabrielle"]), out=tmp_path / "g", log=lambda *_: None)
    control = json.loads((tmp_path / "g" / "control.json").read_text())
    for v in control["landings"].values():
        if v["action"] == "stop":
            assert v["stop_step"] * 6.25 == v["stop_seconds"]
    assert res["moves"]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "follow-replay"
    runs = [("ampil", "gfs", 4 * 3600.0), ("gabrielle", "gfs", 4 * 3600.0), ("ampil", "parent", 3600.0)]
    summary = []
    for name, source, prep in runs:
        res = follow.replay(follow.load_track(CASES[name]), source=source, prep_s=prep,
                            out=out / f"{name}-{source}", log=lambda *_: None)
        follow.print_moves(res)
        _check(res)
        summary.append(dict(case=f"{name}-{source}", moves=res["moves"], min_room_km=res["min_room_km"],
                            holds=res["holds"], max_fix_error_km=max(r["fix_error_km"] for r in res["truth"])))
    (out / "moves.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"saved {out / 'moves.json'}")
