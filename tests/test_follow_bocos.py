"""Storm-following nests: window geometry, accumulation masks across a move, the static mosaic, the
follow-mode fetch plan and the IBTrACS row. CPU only, no network."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from namgpu.nest import domain, window


def ref_grid():
    return domain.nest_grid(31.6, 140.8, 333, 3)


def test_window_shift_is_whole_cells_on_the_fixed_pole():
    g = ref_grid()
    w = window.window_grid(g, 10, 30)
    assert w["tph0d"] == g["tph0d"] and w["tlm0d"] == g["tlm0d"]
    assert w["wbd"] == pytest.approx(g["wbd"] + 10 * g["dlmd"]) and w["sbd"] == pytest.approx(g["sbd"] + 30 * g["dphd"])
    assert window.offset_for(g, *w["center_geo"]) == (10, 30)
    with pytest.raises(SystemExit, match="envelope"):
        window.window_grid(g, 0, int(46 / g["dphd"]))


def test_window_configfile_reads_back_its_spacing(tmp_path):
    g = ref_grid()
    w = window.window_grid(g, -7, 12)
    cfg = tmp_path / "nam_configfile_w"
    window.write_configfile(cfg, w, "w", tmp_path / "s.npz", follow_meta=tmp_path / "GEN.json")
    from namgpu import config
    c = config.load(cfg)
    assert c.dlmd == pytest.approx(g["dlmd"]) and c.dphd == pytest.approx(g["dphd"])
    assert c.raw["window_i0"] == -7 and c.raw["window_j0"] == 12 and (c.win_i0, c.win_j0) == (-7, 12)
    assert c.ref_wbd == pytest.approx(g["wbd"]) and str(c.raw["follow_meta"]).endswith("GEN.json")


def test_symmetric_configfile_spacing_unchanged():
    from namgpu import config
    c = config.load()
    assert c.dlmd == -2.0 * c.wbd / (c.im - 1)


def test_overlap_slices():
    (ra, ca), (rb, cb) = window.overlap(96, 112, (0, 0), (10, 30))
    a = np.arange(112 * 96).reshape(112, 96)
    b = np.full_like(a, -1)
    b[rb, cb] = a[ra, ca]
    assert b[0, 0] == a[30, 10] and (b[rb, cb] >= 0).all()
    assert window.overlap(96, 112, (0, 0), (96, 0)) is None


def fol(i0, j0, land):
    return {"generation": 0, "i0": i0, "j0": j0, "land_hour": land, "ref_grid": {}, "moves": []}


def test_accumulation_masks_the_strip_of_a_later_move():
    jm, im = 112, 96
    prev = np.ones((jm, im))
    gens = [(0.0, (0, 0)), (3.0, (10, 30)), (6.0, (15, 60))]
    # frame 3 (window 0) to frame 9 (window 2): only points inside windows 1 and 2 count
    out = window.accum_prev_on(fol(15, 60, 6.0), fol(0, 0, 0.0), prev, 3.0, 9.0, gens)
    assert np.isnan(out[-30:, :]).all()           # window 2's northern 30 rows were not in window 1
    assert np.isnan(out[:, -5:]).all()            # nor its eastern 5 columns
    inside = out[:jm - 30, :im - 5]
    assert np.isfinite(inside).all()
    # points of window 1 that were not in window 0 entered at +3 h: accumulators started at 0 there
    assert (inside[jm - 60:jm - 30, :] == 0.0).all() and (inside[:jm - 60, :im - 15] == 1.0).all()


def test_accumulation_generation0_without_attributes():
    out = window.accum_prev_on(fol(10, 30, 3.0), None, np.ones((112, 96)), 2.0, 4.0,
                               [(0.0, (0, 0)), (3.0, (10, 30))])
    assert np.isnan(out).any() and np.nanmax(out) == 1.0


def test_accumulation_static_run_unchanged():
    a = np.arange(6.0).reshape(2, 3)
    assert window.accum_prev_on(None, None, a, 0, 1) is a


def test_gens_from_frames_orders_windows():
    g = window.gens_from_frames([fol(15, 60, 6.0), fol(0, 0, 0.0), None, fol(10, 30, 3.0), fol(10, 30, 3.0)])
    assert g == [(0.0, (0, 0)), (3.0, (10, 30)), (6.0, (15, 60))]


def test_frame_union_across_the_dateline():
    u = window.frame_union([(170.0, 179.0, -40.0, -35.0), (175.0, 185.0, -42.0, -37.0)])
    assert u == (170.0, 185.0, -42.0, -35.0)


def test_static_mosaic_keeps_carried_cells(tmp_path):
    from namgpu.nest import prepare
    jm, im = 8, 6
    g0 = {"HGT_SFC": np.zeros((jm, im), np.float32), "LAND": np.ones((jm, im), np.float32)}
    np.savez(tmp_path / "s0.npz", **g0)
    (tmp_path / "follow").mkdir()
    ref = {"grid": {"im": im, "jm": jm}, "static": str(tmp_path / "s0.npz")}
    fresh = {"HGT_SFC": np.full((jm, im), 5.0, np.float32), "LAND": np.ones((jm, im), np.float32)}
    m1, r1 = prepare.stamp_static(tmp_path, ref, fresh, 2, 3)
    assert (m1["HGT_SFC"][:5, :4] == 0).all() and (m1["HGT_SFC"][5:, :] == 5).all() and (m1["HGT_SFC"][:, 4:] == 5).all()
    assert r1["overlap_mismatch_before_stamp"]["HGT_SFC"] == 20 and r1["cells_new_to_run"] == jm * im - 20
    # a second window, prepared in any order, agrees with the first on their overlap
    fresh2 = {"HGT_SFC": np.full((jm, im), 9.0, np.float32), "LAND": np.ones((jm, im), np.float32)}
    m2, _ = prepare.stamp_static(tmp_path, ref, fresh2, 3, 4)
    assert (m2["HGT_SFC"][:7, :5] == m1["HGT_SFC"][1:, 1:]).all()


def test_follow_fetch_plan_fetches_start_sets_at_leads(tmp_path):
    from namgpu.init import fetch_case
    cfg_path = tmp_path / "cfg"
    domain.write_configfile(cfg_path, ref_grid(), "t", tmp_path / "s.npz")
    from namgpu import config
    cfg = config.load(cfg_path)
    jobs, _ = fetch_case.plan(datetime(2024, 8, 15, 12), 12, tmp_path / "raw", cfg, "gfs", 0, follow=True)
    lead = [j for j in jobs if j[0] is fetch_case._follow_lead]
    assert [Path(j[3]).name for j in lead] == [f"gfs.t12z.f{h:03d}.start.grib2" for h in (3, 6, 9, 12)]
    off, _ = fetch_case.plan(datetime(2024, 8, 15, 12), 12, tmp_path / "raw", cfg, "gfs", 0)
    assert not [j for j in off if j[0] is fetch_case._follow_lead]


def test_best_track_rows(tmp_path):
    from namgpu.io import best_track as bt
    hdr = "SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,TOKYO_LAT,TOKYO_LON,TOKYO_PRES,TOKYO_WIND\n"
    rows = ["2024225N22135,2024,1,WP,MM,AMPIL,2024-08-15 12:00:00,TS,31.6,140.8,85,950,31.6,140.8,950,85",
            "2024225N22135,2024,1,WP,MM,AMPIL,2024-08-15 18:00:00,TS,32.6,140.8,85,950,32.6,140.8,948,-1"]
    p = tmp_path / "ibtracs.WP.list.v04r01.csv"
    p.write_text(hdr + "units\n" + "\n".join(rows) + "\n")
    tr = bt.track(p, "WP", name="AMPIL", season=2024)
    assert tr["sid"] == "2024225N22135" and len(tr["points"]) == 2 and tr["points"][1]["wind_kt"] == 85
    mid = bt.interp(tr, datetime(2024, 8, 15, 15))
    assert mid["lat"] == pytest.approx(32.1) and mid["pres_hpa"] == pytest.approx(949.0)
    assert bt.nearest_storm(p, 31.0, 141.0, datetime(2024, 8, 15, 12)) == "2024225N22135"
