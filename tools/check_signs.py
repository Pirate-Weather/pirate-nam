"""Sign and geometry checks for a nest anywhere (lane init-gfs-anywhere), per domain configfile:
- Coriolis: init-grid-consts f (V points) against 2 Omega sin(geographic lat of the V point): sign and value.
- Wind rotation: init-gfs-wind-rotate (earth -> grid at V) then io-wind-rotate's V-point vectors back
  (grid -> earth) on a uniform westerly and a uniform southerly: the round trip and, at the centre, grid
  axes = east/north (the rotated equator runs through the centre).
- Source window: gfs_start.source_window for the domain's points: rows, columns, wrap, periodic.
- Interpolation across 0/360 and 180: init-gfs-hinterp on a synthetic global field f = lon (sin/cos form,
  so it is smooth across every meridian) through the window: max error against the exact value.
    python tools/check_signs.py CONFIG [CONFIG ...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one(cfgpath):
    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.init import gfs_start, init_gfs_hinterp as hin, init_gfs_wind_rotate as wr, init_grid_consts
    from namgpu.io import io_wind_rotate as iwr
    cfg = config.load(Path(cfgpath))
    g = gridmod.build(cfg)
    st = statemod.State(g, names=("GLAT", "GLON", "VLAT", "VLON"))
    ctx = SimpleNamespace(cfg=cfg, grid=g, st=st, stream=None, tables={}, first_step=True, case_dir=ROOT / "cases" / "none", analysis=None, start=None)
    from namgpu.init import init_grid_geo
    init_grid_geo.run(ctx)
    out = {"config": Path(cfgpath).name, "centre": [cfg.tph0d, cfg.tlm0d]}
    try:
        init_grid_consts.run(ctx)
        f = cp.asnumpy(ctx.tables["F_V"]).astype(np.float64)
        want = 2.0 * 7.292e-5 * np.sin(g.glat_v)
        out["coriolis"] = {"f_centre": float(f[g.jm // 2, g.im // 2]), "sign_ok": bool(np.all(np.sign(f) == np.sign(want))),
                           "max_rel_err": float(np.max(np.abs(f - want)) / np.max(np.abs(want)))}
    except Exception as e:  # noqa: BLE001
        out["coriolis"] = f"FAILED {type(e).__name__}: {e}"
    ce_e, ce_n, cn_e, cn_n = (cp.asnumpy(a) for a in wr._coefficients(ctx))
    he_e, he_n, hn_e, hn_n = (cp.asnumpy(a) for a in iwr._coefficients(ctx))   # io-wind-rotate V vectors
    rt = {}
    for name, (ue, vn) in {"westerly": (10.0, 0.0), "southerly": (0.0, 10.0)}.items():
        ug = ue * ce_e + vn * ce_n          # earth -> grid (init-gfs-wind-rotate at V)
        vg = ue * cn_e + vn * cn_n
        ue2 = ug * he_e + vg * hn_e         # grid -> earth at the same V point (io-wind-rotate)
        vn2 = ug * he_n + vg * hn_n
        rt[name] = float(max(np.max(np.abs(ue2 - ue)), np.max(np.abs(vn2 - vn))))
    jc, ic = g.jm // 2, g.im // 2
    out["wind_rotate"] = {"roundtrip_max_err_ms": rt, "centre_grid_i_east": [float(ce_e[jc, ic]), float(ce_n[jc, ic])],
                          "centre_grid_j_north": [float(cn_e[jc, ic]), float(cn_n[jc, ic])],
                          "nonfinite": int(sum((~np.isfinite(a)).sum() for a in (ce_e, ce_n, cn_e, cn_n, he_e, he_n, hn_e, hn_n)))}
    la = cp.concatenate([cp.degrees(st["GLAT"].ravel()), cp.degrees(st["VLAT"].ravel())])
    lo = cp.concatenate([cp.degrees(st["GLON"].ravel()), cp.degrees(st["VLON"].ravel())])
    j0, j1, cols, periodic = gfs_start.source_window(la, lo, 1440, 721, 0.25, 0.25)
    out["window"] = {"rows": [j0, j1], "lat": [-90 + 0.25 * j0, -90 + 0.25 * (j1 - 1)], "periodic": periodic,
                     "ncol": 1440 if cols is None else len(cols),
                     "lon": None if cols is None else [float(cols[0]) * 0.25, float(cols[-1]) * 0.25]}
    # synthetic smooth field through the same window: f = 100 sin(lon) + 50 cos(lat) + 20 cos(lon) sin(lat)
    glat = np.radians(-90.0 + 0.25 * np.arange(721))[:, None]
    glon = np.radians(0.25 * np.arange(1440))[None, :]
    full = (100 * np.sin(glon) + 50 * np.cos(glat) + 20 * np.cos(glon) * np.sin(glat)).astype(np.float32)
    a = full[j0:j1] if cols is None else full[j0:j1][:, cols]
    grid = (hin.Grid(lat0=-90 + 0.25 * j0, lon0=0.0, dlat=0.25, dlon=0.25, lon_periodic=True) if periodic else
            hin.Grid(lat0=-90 + 0.25 * j0, lon0=float(cols[0]) * 0.25, dlat=0.25, dlon=0.25, lon_periodic=False,
                     lon_window=True))
    src = cp.asarray(np.ascontiguousarray(a))[None]
    res, nmiss = hin.interp_full(src, cp.ones(src.shape, cp.uint8), st["GLAT"], st["GLON"], grid=grid)
    tl, tn = cp.asnumpy(st["GLAT"]).astype(np.float64), cp.asnumpy(st["GLON"]).astype(np.float64)
    exact = 100 * np.sin(tn) + 50 * np.cos(tl) + 20 * np.cos(tn) * np.sin(tl)
    err = np.abs(cp.asnumpy(res[0]) - exact)
    out["hinterp_synthetic"] = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                                "nmiss": int(nmiss.get()[0]),
                                "lon_range_deg": [float(np.degrees(tn).min()), float(np.degrees(tn).max())]}
    return out


if __name__ == "__main__":
    import json
    for c in sys.argv[1:]:
        print(json.dumps(one(c)), flush=True)
