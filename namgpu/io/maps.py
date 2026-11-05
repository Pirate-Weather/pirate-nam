"""Weather maps from WOOF-NAM history files with the Rust renderer rw_wrfbatch (render law: no
matplotlib for weather fields), drawn exactly like the case referee maps (same CF table, same colour
scales), so a WOOF-NAM panel sits beside the referee panels (MRMS, IMERG, the source's analysis, the
archived NAM) on the same frame.

Frame: taken from the history's own XLAT/XLONG (H points), dateline-aware (frame_bounds): a domain
across 180 gets W < 180 < E with E past 180, which rw_wrfbatch draws on one Lambert frame centred on
the domain. That needs a renderer from gpuwm 2.9.12 or later; an older one refuses the bounds by name,
and the failure says which renderer to use (NAMGPU_RENDERER). No fixed CONUS frame.
Titles: the renderer writes "<source> | <row title>"; every panel reads
"WOOF-NAM <res> km | <field>  init YYYY-MM-DD HHZ  valid YYYY-MM-DD HHZ (+FFh)", res from the
history's DY (NMMB dphd = 0.009 deg x res).

Usage:  python -m namgpu.io.maps <wrfout ...> --out <dir> [--products composite_reflectivity,t2m,...]

Route: rw_wrfbatch --cf on the history's own curvilinear XLAT/XLONG (H points). The renderer's wrfout
route does not place MAP_PROJ 203 (the rotated lat-lon B grid): it rendered the 12 km parent as a band
across the top of the frame (2026-11-04, runs/init_native), so it is not used. The CF route is the one
the referee maps use for the NAM native B grid (cases/<case>/referee/make_referee.py).

Products (CF variable, from the history fields):
  composite_reflectivity  refc   COMPOSITE (dBZ); MRMS overlay where the case has the hour
  t2m                     t2m    T2 (K -> degF)
  d2m                     d2m    dewpoint from Q2 (2 m q) and PSFC (Bolton 1980 inverse), degF
  wind10                  ws10m  |U10, V10| (m/s, drawn in kt)
  wind850                 ws850  |U, V| at 850 hPa, log-p linear between the bracketing layers of PRES;
                                 blank where 850 hPa is below ground
  mslp                    mslp   MSLP (NMMB QUASIPOST membrane MSLP, Pa -> hPa)
  qpf_1h                  qpf1h  RAINNC + RAINC difference from the history one hour earlier (mm)
  qpf_6h                  qpf6h  RAINNC + RAINC difference from the history six hours earlier (mm)
A product whose input is not in a frame is SKIPPED with the reason (composite reflectivity before the
microphysics has run, qpf_1h on the first frame); the caller decides what a skip means.
Host work, all cores: each drawn frame is read (only the variables its products need), derived and
written as CF in its own worker process; renderer subprocesses then run in parallel threads, one per
product and frame.
"""
from __future__ import annotations

import re
from namgpu import cores as _cores

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DEFAULT_PRODUCTS = "composite_reflectivity,t2m,d2m,wind10,wind850,mslp,qpf_1h,qpf_6h"
RECEIPT_ENV = "NAMGPU_MAPS_RECEIPT"
BOUNDS = "-126,-66,22,52"            # the CONUS frame of the side-by-side panels (io_referee_panels); maps.py
                                     # frames come from each history's XLAT/XLONG (frame_bounds)
RENDERER_ENV = "NAMGPU_RENDERER"     # full path of an rw_wrfbatch that draws dateline frames (gpuwm >= 2.9.12)
# The pinned renderer lives in a checkout (tools/renderer/bin, not committed: tools/renderer/install_renderer.sh
# builds it from a gpuwm cut with the patches in tools/renderer, or copies a built one, and puts the Natural Earth
# basemap beside it). An installed wheel uses NAMGPU_RENDERER or the installed gpuwm renderer.
from namgpu import paths as _paths  # noqa: E402
PINNED_RENDERER = (_paths.checkout_root() or _paths.work_root()) / "tools" / "renderer" / "bin" / "rw_wrfbatch"
FRAME_PAD = 0.03                     # fraction of the domain span added on each side of the frame

# The 1 h precipitation ladder (in), as the CF table's "scale" (the renderer takes exactly ten edges, nine colour
# bands). Amounts under the first edge are left blank. The edges are (near) evenly spaced because rw_wrfbatch draws
# the nine bands at equal heights on a linear-labelled bar: with the old 0.01/0.05/0.1/0.2/0.3/0.5/0.75/1/1.5/2 edges
# the 0.2-0.3 in band sat beside the bar's "0.8-0.9" labels (f12, 2026-11-04), so every reading of the bar was wrong.
QPF1H_LEVELS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QPF1H_COLORS = [[139, 205, 132], [66, 170, 72], [21, 120, 36], [255, 250, 140], [255, 200, 0],
                [255, 130, 0], [230, 30, 0], [150, 0, 30], [200, 0, 200]]
# 6 h precipitation (in): the same nine colours on evenly spaced edges, 0.05 to 2.0 in.
QPF6H_LEVELS = [0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

# product -> (CF variable, CF table row); rows match cases/<case>/referee/cf-table.json
TABLE = {
    "composite_reflectivity": ("refc", {"names": ["refc"], "standard_names": [], "product": "composite_reflectivity",
                                        "title": "Composite reflectivity", "recipe": "composite_reflectivity",
                                        "default_units": "dBZ", "display_units": "dBZ", "unit_factors": {"dBZ": 1.0},
                                        "observation_product": "refc"}),
    "t2m": ("t2m", {"names": ["t2m"], "standard_names": [], "product": "t2m", "title": "2 m temperature",
                    "recipe": "2m_temperature", "default_units": "degF", "display_units": "degF",
                    "unit_factors": {"degF": 1.0}}),
    "d2m": ("d2m", {"names": ["d2m"], "standard_names": [], "product": "d2m", "title": "2 m dewpoint",
                    "recipe": "2m_dewpoint", "default_units": "degF", "display_units": "degF",
                    "unit_factors": {"degF": 1.0}}),
    "wind10": ("ws10m", {"names": ["ws10m"], "standard_names": [], "product": "wind10", "title": "10 m wind speed",
                         "recipe": "10m_wind_speed_and_direction", "default_units": "m s-1", "display_units": "kt",
                         "unit_factors": {"m s-1": 1.9438445}}),
    "wind850": ("ws850", {"names": ["ws850"], "standard_names": [], "product": "wind850",
                          "title": "850 hPa wind speed", "recipe": "10m_wind_speed_and_direction",
                          "default_units": "m s-1", "display_units": "kt", "unit_factors": {"m s-1": 1.9438445}}),
    "mslp": ("mslp", {"names": ["mslp"], "standard_names": [], "product": "mslp", "title": "Mean sea level pressure",
                      "recipe": "href_mean_mslp", "default_units": "hPa", "display_units": "hPa",
                      "unit_factors": {"hPa": 1.0}}),
    "qpf_1h": ("qpf1h", {"names": ["qpf1h"], "standard_names": [], "product": "qpf_1h", "title": "1 h precipitation",
                         "recipe": "total_qpf", "default_units": "kg m-2", "display_units": "in",
                         "unit_factors": {"kg m-2": 0.0393700787},
                         # the standard 1 h precipitation ladder (in), not total_qpf's run-total 0-15 in
                         # scale, which painted every real hourly amount in its first colour (judge 11:30Z)
                         # mask_below: under the first edge is blank, not the first colour (renderer patch
                         # tools/renderer/02-cf_scale-mask-below.patch; a dry hour painted a domain-wide green band)
                         "scale": {"levels": QPF1H_LEVELS, "colors": QPF1H_COLORS, "mask_below": QPF1H_LEVELS[0]}}),
    "qpf_6h": ("qpf6h", {"names": ["qpf6h"], "standard_names": [], "product": "qpf_6h", "title": "6 h precipitation",
                         "recipe": "total_qpf", "default_units": "kg m-2", "display_units": "in",
                         "unit_factors": {"kg m-2": 0.0393700787},
                         "scale": {"levels": QPF6H_LEVELS, "colors": QPF1H_COLORS, "mask_below": QPF6H_LEVELS[0]}}),
}

# history variables each product reads (the read takes only these, not every 3-D field of the file)
NEEDS = {"composite_reflectivity": ("COMPOSITE",), "t2m": ("T2",), "d2m": ("Q2", "PSFC"),
         "wind10": ("U10", "V10"), "wind850": ("U", "V", "PRES"), "mslp": ("MSLP",),
         "qpf_1h": ("RAINNC", "RAINC"), "qpf_6h": ("RAINNC", "RAINC")}
WINDOW_H = {"qpf_1h": 1, "qpf_6h": 6}   # accumulation products: hours back to the frame they difference


def _parse(argv):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Draw weather maps from WOOF-NAM history files (wrfout_d01_*) with gpuwm's Rust "
                    "renderer: composite reflectivity, 2 m temperature and dewpoint, 10 m and 850 hPa "
                    "wind, MSLP and 1 h precipitation. A forecast draws its own maps when it ends; use "
                    "this to redraw them. Host only, no GPU.",
        epilog="example: woof-nam maps runs/2025-03-14_12z/wrfout_d01_* --out runs/2025-03-14_12z/maps",
        formatter_class=HelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="history files (wrfout_d01_<valid time>)")
    ap.add_argument("--out", type=Path, required=True, help="output folder; maps go to <out>/<product>/")
    ap.add_argument("--products", default=DEFAULT_PRODUCTS, help=f"comma list (default {DEFAULT_PRODUCTS})")
    ap.add_argument("--frames", default="all", help="which history frames to draw (default all)")
    ap.add_argument("--width", type=int, default=1200, help="image width in pixels (default 1200)")
    ap.add_argument("--height", type=int, default=900, help="image height in pixels (default 900)")
    ap.add_argument("--mrms", type=Path, default=None, help="folder of mrms_refc_YYYYmmddHH.grib2 (overlay)")
    ap.add_argument("--label", default=None,
                    help="panel source label (default: 'WOOF-NAM <res> km', res from the history's DY)")
    ap.add_argument("--bounds", default="auto",
                    help="map frame W,E,S,N (E may pass 180 for a dateline frame); auto: from XLAT/XLONG")
    return ap.parse_args(argv)


def basemap_dir(rw: Path, env: dict | None = None) -> Path | None:
    """The Natural Earth basemap rw_wrfbatch will find (rustwx-render features.rs search order:
    RUSTWX_BASEMAP_DIR, RUSTWX_ASSETS_DIR/basemap, <exe dir and 7 ancestors>/assets/basemap), or None."""
    env = os.environ if env is None else env
    roots = []
    if env.get("RUSTWX_BASEMAP_DIR"):
        roots.append(Path(env["RUSTWX_BASEMAP_DIR"]))
    if env.get("RUSTWX_ASSETS_DIR"):
        roots.append(Path(env["RUSTWX_ASSETS_DIR"]) / "basemap")
    exe_dir = Path(rw).resolve().parent
    roots += [a / "assets" / "basemap" for a in [exe_dir, *exe_dir.parents][:8]]
    return next((r for r in roots if (r / "natural_earth_10m").is_dir() or (r / "natural_earth_110m").is_dir()),
                None)


# The basemap shipped in the wheel (namgpu/data/basemap: Natural Earth and US Census, public domain). The released
# gpuwm wheel carries rw_wrfbatch with no assets/basemap beside it, so on a bare `pip install woof-nam` every map was drawn
# with no coastlines (2026-11-04), and then the basemap check refused the renderer and the run ended rc 1 at maps
# after a finished forecast (2026-11-05, ERA5 lane).
PACKAGED_BASEMAP = Path(__file__).resolve().parent.parent / "data" / "basemap"
_BASEMAP_WARNED = False


def _has_basemap(root: Path) -> bool:
    return (root / "natural_earth_10m").is_dir() or (root / "natural_earth_110m").is_dir()


def ensure_basemap(rw: Path) -> Path | None:
    """Make sure rw_wrfbatch (and every renderer call gpuwm.rustwx makes from this process) finds a basemap.

    Keeps what the renderer already finds (RUSTWX_BASEMAP_DIR, RUSTWX_ASSETS_DIR/basemap, assets/basemap beside the
    binary); otherwise sets RUSTWX_BASEMAP_DIR in this process's environment to gpuwm's vendored basemap or, failing
    that, the copy shipped in this package. Returns the basemap folder, or None after one warning: the maps are then
    drawn without coastlines, borders or state lines, and the run goes on."""
    global _BASEMAP_WARNED
    found = basemap_dir(rw)
    if found is not None:
        return found
    candidates = []
    try:
        from gpuwm import rustwx  # noqa: PLC0415
        candidates.append(Path(rustwx.basemap_dir()))
    except Exception:  # noqa: BLE001  (no gpuwm, or an older one without basemap_dir)
        pass
    candidates.append(PACKAGED_BASEMAP)
    for root in candidates:
        if _has_basemap(root):
            os.environ["RUSTWX_BASEMAP_DIR"] = str(root)
            return root
    if not _BASEMAP_WARNED:
        _BASEMAP_WARNED = True
        print(f"woof-nam maps: WARNING: {rw} finds no Natural Earth basemap (none beside it, RUSTWX_BASEMAP_DIR "
              f"unset, {PACKAGED_BASEMAP} missing): maps are drawn without coastlines, borders or state lines. "
              f"Set RUSTWX_BASEMAP_DIR to a folder holding natural_earth_10m to draw them.", file=sys.stderr,
              flush=True)
    return None


def renderer_env(rw: Path) -> dict:
    """The environment to run rw_wrfbatch with: this process's, with the basemap pinned (ensure_basemap)."""
    ensure_basemap(rw)
    return dict(os.environ)


def renderer() -> Path:
    """The rw_wrfbatch to draw with (see _renderer), with its basemap pinned (ensure_basemap). A missing basemap is
    a warning, never a refusal: the forecast is done by the time maps are drawn."""
    rw = _renderer()
    ensure_basemap(rw)
    return rw


def _gpuwm_renderer() -> Path | None:
    """The released gpuwm renderer: GPUWM_BRIDGES/rw_wrfbatch, else gpuwm.rustwx.find_renderer(), or None."""
    env = os.environ.get("GPUWM_BRIDGES")
    if env and (Path(env) / "rw_wrfbatch").is_file():
        return Path(env) / "rw_wrfbatch"
    try:
        from gpuwm import rustwx  # noqa: PLC0415
    except ImportError:
        return None
    r = rustwx.find_renderer()
    return Path(r) if r else None


def _renderer() -> Path:
    """The rw_wrfbatch to draw with: NAMGPU_RENDERER, else the pinned build (tools/renderer/bin in a checkout),
    else the installed gpuwm's. Only gpuwm 2.9.12 and later draw a frame across the dateline; an older one
    refuses such a frame by name (maps_main says which renderer to set)."""
    env = os.environ.get(RENDERER_ENV)
    if env:
        if not Path(env).is_file():
            raise SystemExit(f"woof-nam maps: {RENDERER_ENV}={env} is not a file")
        return Path(env)
    if PINNED_RENDERER.is_file():
        return PINNED_RENDERER
    rw = _gpuwm_renderer()
    if rw is not None:
        return rw
    raise SystemExit(f"woof-nam maps: no renderer: set {RENDERER_ENV}, install gpuwm (its rw_wrfbatch), or in a "
                     f"checkout run tools/renderer/install_renderer.sh (builds rw_wrfbatch from a gpuwm cut with "
                     f"the tree's renderer patches into {PINNED_RENDERER}, with the basemap beside it).")


def _bridges() -> Path:
    return renderer().parent


def frame_bounds(lat, lon, pad: float = FRAME_PAD):
    """(W, E, S, N) of a curvilinear domain, dateline-aware: longitudes are unwrapped around the domain's
    centre point, so a domain across 180 gets W < 180 < E (E up to W + 360), never a -180..180 frame.
    A domain that holds a pole gets the whole longitude circle from its centre."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    j, i = lat.shape[0] // 2, lat.shape[1] // 2
    c = float(lon[j, i])
    rel = (lon - c + 180.0) % 360.0 - 180.0
    s, n = float(lat.min()), float(lat.max())
    w, e = c + float(rel.min()), c + float(rel.max())
    dx, dy = (e - w) * pad, (n - s) * pad
    w, e = w - dx, e + dx
    s, n = max(s - dy, -89.9), min(n + dy, 89.9)
    if e - w >= 359.0 or n >= 89.9 or s <= -89.9:        # pole inside: the whole circle
        w, e = c - 180.0, c + 180.0
    shift = 360.0 * np.floor((w + 180.0) / 360.0)          # W into [-180, 180); E keeps the span
    return (round(w - shift, 3), round(e - shift, 3), round(s, 3), round(n, 3))


def legacy_frame(frame) -> tuple | None:
    """A frame a renderer older than gpuwm 2.9.12 can draw (-180 <= W < E <= 180), for a domain whose own frame
    crosses 180 or holds the pole: the CONUS frame (BOUNDS, the frame woof-nam 0.1.0 used) when the domain covers
    it, else None. The released gpuwm 2.9.11 renderer refuses the 12 km parent's own frame (it holds the pole), so
    without this a pip install made no map at all (2026-11-05)."""
    w, e, s, n = frame
    cw, ce, cs, cn = (float(v) for v in BOUNDS.split(","))
    if s > cs or n < cn:
        return None
    if any(w <= cw + k and ce + k <= e for k in (-360.0, 0.0, 360.0)):
        return (cw, ce, cs, cn)
    return None


def bounds_arg(b) -> str:
    return ",".join(f"{v:g}" for v in b)


def resolution_km(path: Path) -> float | None:
    """Grid length in km from the history's DY (NMMB: dphd = 0.009 deg x res, 1.0008 km x res)."""
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        dy = getattr(nc, "DY", None)
    if dy is None:
        return None
    return round(float(dy) / 1000.8 * 2.0) / 2.0


def source_label(res_km: float | None) -> str:
    return f"WOOF-NAM {res_km:g} km" if res_km else "WOOF-NAM"


CREDIT_ENV = "NAMGPU_DATA_CREDIT"    # short data credit (start/boundary source, static datasets) the door sets


def credit_note() -> str | None:
    """The run's data credit for the panel subtitle (CC BY sources need it on every map), or None. In the subtitle
    (CF row "note", renderer patch tools/renderer/03-cf_row-note-subtitle.patch), not the title: a longer title
    widened the panel and cut the valid time, and panels of two widths cannot be composed into a sheet."""
    c = os.environ.get(CREDIT_ENV, "").strip()
    return f"data: {c}" if c else None


def panel_title(field: str, start: datetime, valid: datetime) -> str:
    fh = int(round((valid - start).total_seconds() / 3600))
    return f"{field}  init {start:%Y-%m-%d %H}Z  valid {valid:%Y-%m-%d %H}Z (+{fh}h)"


def cf_row(product: str, title: str, note: str | None = None) -> dict:
    """The product's CF table row with its panel title (and an optional subtitle note, e.g. the data credit)."""
    row = json.loads(json.dumps(TABLE[product][1]))
    row["title"] = title
    if note:
        row["note"] = note
    return row


def _k2f(k):
    return (k - 273.15) * 1.8 + 32.0


def _dewpoint_k(q, p):
    e = np.maximum(q * p / (0.622 + 0.378 * q), 1e-3) / 100.0          # hPa
    ln = np.log(e / 6.112)
    return 273.15 + 243.5 * ln / (17.67 - ln)


def _at_pressure(a, pres, target):
    """Log-p linear value of a [L, J, I] at pressure `target` (Pa); NaN below ground or above the top."""
    lp = np.log(pres)
    lt = np.log(target)
    above = pres <= target                                             # file is bottom-up
    k = np.argmax(above, axis=0)                                       # first level at or above target
    ok = above.any(axis=0) & (k > 0)
    k = np.clip(k, 1, a.shape[0] - 1)
    jj, ii = np.indices(k.shape)
    p0, p1 = lp[k - 1, jj, ii], lp[k, jj, ii]
    w = (lt - p0) / np.where(p1 != p0, p1 - p0, 1.0)
    out = a[k - 1, jj, ii] + w * (a[k, jj, ii] - a[k - 1, jj, ii])
    return np.where(ok, out, np.nan)


def _times(path: Path):
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        t = "".join(c.decode() for c in nc.variables["Times"][0])
        return datetime.strptime(t, "%Y-%m-%d_%H:%M:%S"), datetime.strptime(nc.START_DATE, "%Y-%m-%d_%H:%M:%S")


def _read(path: Path, names=None):
    """(valid, start, {name: array}) of one history frame; with `names`, only those variables (a frame is
    ~1 GB of 3-D fields and the maps need a handful of 2-D ones)."""
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        v = nc.variables
        t = "".join(c.decode() for c in nc.variables["Times"][0])
        valid = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S")
        start = datetime.strptime(nc.START_DATE, "%Y-%m-%d_%H:%M:%S")
        get = {n: np.asarray(v[n][0], dtype=np.float64) for n in v
               if n not in ("Times", "XTIME") and v[n].dimensions[0] == "Time" and (names is None or n in names)}
    return valid, start, get


def fields_for(prevs, cur, products):
    """CF fields of one frame: {cf_var: (array, units)}; skipped: [(product, reason)].
    prevs: {hours back: (valid, start, fields, path)} of the frames the accumulation products difference."""
    if not isinstance(prevs, dict):                      # older callers pass the frame one hour earlier
        prevs = {1: prevs} if prevs is not None else {}
    valid, start, f, _ = cur
    out, skipped = {}, []
    for prod in products:
        var = TABLE[prod][0]
        if prod == "composite_reflectivity":
            if "COMPOSITE" not in f:
                skipped.append((prod, f"{cur[3].name}: no COMPOSITE in the history (the microphysics has not written REFL_10CM yet)"))
                continue
            out[var] = (np.maximum(f["COMPOSITE"], -30.0), "dBZ")
        elif prod == "t2m":
            out[var] = (_k2f(f["T2"]), "degF")
        elif prod == "d2m":
            out[var] = (_k2f(_dewpoint_k(np.maximum(f["Q2"], 0.0), f["PSFC"])), "degF")
        elif prod == "wind10":
            out[var] = (np.hypot(f["U10"], f["V10"]), "m s-1")
        elif prod == "wind850":
            out[var] = (np.hypot(_at_pressure(f["U"], f["PRES"], 85000.0),
                                 _at_pressure(f["V"], f["PRES"], 85000.0)), "m s-1")
        elif prod == "mslp":
            if "MSLP" not in f:
                skipped.append((prod, f"{cur[3].name}: no MSLP in the history (io-quasipost-driver did not run for it)"))
                continue
            out[var] = (f["MSLP"] / 100.0, "hPa")
        elif prod in WINDOW_H:
            n = WINDOW_H[prod]
            prev = prevs.get(n)
            if prev is None or (valid - prev[0]) != timedelta(hours=n):
                skipped.append((prod, f"{cur[3].name}: no history frame {n} h earlier to difference"))
                continue
            tot = f["RAINNC"] + f["RAINC"]
            q = np.maximum(tot - (prev[2]["RAINNC"] + prev[2]["RAINC"]), 0.0)
            first = TABLE[prod][1]["scale"]["levels"][0] * 25.4
            q = np.where(q >= first, q, np.nan)                          # below the first edge: blank
            if not np.isfinite(q).any():
                # a dry window (no point at the first edge): zeros, which the scale's mask_below leaves blank, not an
                # all-NaN field the renderer refuses ("selected CF field has no finite values"), which failed the
                # run and skipped every sheet
                q = np.zeros_like(q)
            out[var] = (q, "kg m-2")
    return out, skipped


def _write_cf(path, lat, lon, valid, fields, label):
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path, "w") as nc:
        ny, nx = lat.shape
        nc.createDimension("time", 1)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.source = label
        t = nc.createVariable("time", "f8", ("time",))
        t.units = f"hours since {valid:%Y-%m-%d %H:%M:%S}"
        t.standard_name = "time"
        t[:] = [0.0]
        a = nc.createVariable("lat", "f4", ("y", "x"))
        a.standard_name, a.units = "latitude", "degrees_north"
        a[:] = lat
        b = nc.createVariable("lon", "f4", ("y", "x"))
        b.standard_name, b.units = "longitude", "degrees_east"
        b[:] = lon
        for name, (arr, units) in fields.items():
            v = nc.createVariable(name, "f4", ("time", "y", "x"), fill_value=np.float32(np.nan))
            v.units = units
            v[0] = arr.astype(np.float32)


def _prep_frame(job):
    """One drawn frame: read the variables its products need (and the earlier frames' rain for the
    accumulation products), derive the CF fields and write the frame's CF file and its CF table (the table
    carries the panel titles, which name init, valid and lead). Runs in a worker process.
    Returns (cf path or None, valid, start, CF variables written, skipped, table path, frame bounds)."""
    path, prev_paths, need, products, work, label = job
    path = Path(path)
    valid, start, f = _read(path, set(need))
    prevs = {}
    for n, pp in (prev_paths or {}).items():
        pv, ps, pf = _read(Path(pp), {"RAINNC", "RAINC"})
        prevs[int(n)] = (pv, ps, pf, Path(pp))
    flds, sk = fields_for(prevs, (valid, start, f, path), products)
    lat, lon = f.get("XLAT"), f.get("XLONG")
    if lat is None or lon is None:
        raise RuntimeError(f"woof-nam maps: {path.name} has no XLAT/XLONG")
    frame = frame_bounds(lat, lon)
    if not flds:
        return None, valid, start, [], sk, None, frame
    lon = (lon + 180.0) % 360.0 - 180.0
    cf = Path(work) / f"namgpu_{valid:%Y%m%d%H%M}.nc"
    _write_cf(cf, lat, lon, valid, flds, label)
    table = Path(work) / f"cf-table_{valid:%Y%m%d%H%M}.json"
    note = credit_note()
    table.write_text(json.dumps([cf_row(p, panel_title(TABLE[p][1]["title"], start, valid), note) for p in products]))
    return str(cf), valid, start, sorted(flds), sk, str(table), frame


def render(argv=None) -> dict:
    """Render history files. Returns {"written": [...], "failures": [...], "skipped": [[product, reason]]}.
    Does not raise on a renderer failure or a skip; the caller decides what they mean."""
    a = _parse(list(sys.argv[1:] if argv is None else argv))
    products = [p.strip() for p in a.products.split(",") if p.strip()]
    unknown = [p for p in products if p not in TABLE]
    if unknown:
        raise SystemExit(f"woof-nam maps: unknown products {unknown} (known: {list(TABLE)})")
    rw = renderer()
    renderer_env = globals()["renderer_env"](rw)
    a.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cf-", dir=a.out))

    files = sorted(a.files)
    # frame times first (header reads only), then each drawn frame is read, derived and written as CF in its
    # own process across the cores (the old loop read every variable of every frame on one core: ~1 min per
    # 914 MB frame, 49 frames ~50 min after a 48 h run, run lead 11:05Z)
    times = [_times(f) for f in files]           # header reads; netCDF4/HDF5 is not thread-safe, keep these serial
    order = sorted(range(len(files)), key=lambda k: times[k][0])
    files = [files[k] for k in order]
    first = 0 if a.frames == "all" else max(0, len(files) - int(a.frames))
    need = {"XLAT", "XLONG"}
    for p in products:
        need.update(NEEDS[p])
    label = a.label or source_label(resolution_km(files[-1]))
    valids = [times[k][0] for k in order]
    by_valid = dict(zip(valids, files))
    windows = sorted({WINDOW_H[p] for p in products if p in WINDOW_H})
    preps = []
    for k in range(first, len(files)):
        back = {n: str(by_valid[valids[k] - timedelta(hours=n)]) for n in windows
                if (valids[k] - timedelta(hours=n)) in by_valid}
        preps.append((str(files[k]), back, sorted(need), products, str(work), label))
    written, failures, skipped = [], [], []
    jobs = []
    workers = max(1, min(len(preps), _cores.usable() // 2, 48))
    if workers == 1:
        results = [_prep_frame(p) for p in preps]
    else:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        # spawn: the caller may hold a CUDA context, which a forked child must not inherit
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_prep_frame, preps))
    frame = None
    if a.bounds != "auto":
        frame = tuple(float(v) for v in a.bounds.split(","))
    elif results:
        frame = results[-1][6]                 # the domain does not move between frames
    from namgpu.io.referee_rows import ROWS, _overlaps  # noqa: PLC0415
    mrms_region = next(r["region"] for r in ROWS if r["name"] == "mrms")
    for cf, valid, start, have, sk, table, _frame in results:
        skipped += [[p, r] for p, r in sk]
        if cf is None:
            continue
        cf = Path(cf)
        fhr = int(round((valid - start).total_seconds() / 3600))
        for prod in products:
            var = TABLE[prod][0]
            if var not in have:
                continue
            obs = None
            if prod == "composite_reflectivity" and a.mrms is not None and _overlaps(frame, mrms_region):
                src = a.mrms / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
                if src.is_file():
                    obs = work / f"MRMS_MergedReflectivityQCComposite_00.50_{valid:%Y%m%d-%H}0000.grib2"
                    if not obs.exists():
                        obs.symlink_to(src.resolve())
            dst = a.out / prod / f"namgpu_{start:%Y%m%d_%H}z_f{fhr:03d}_{prod}.png"
            jobs.append((cf, var, valid, obs, dst, prod, table))

    legacy = legacy_frame(frame) if frame else None
    used_legacy = []

    def one(job):
        cf, var, valid, obs, dst, prod, table = job
        od = work / f"r_{var}_{valid:%Y%m%d%H%M}"
        cmd = [rw, "--cf", cf, "--variable", var, "--out-dir", od, "--valid",
               valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", bounds_arg(frame), "--cf-table", table,
               "--width", str(a.width), "--height", str(a.height)]
        if obs is not None:
            cmd += ["--observation", obs]
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=renderer_env)
        pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
        if (p.returncode or not pngs) and "bounds require" in (p.stderr or p.stdout) and legacy is not None:
            # a renderer older than 2.9.12 (the released gpuwm 2.9.11 wheel's) cannot draw this frame: draw the
            # CONUS frame instead of no map
            cmd[cmd.index("--bounds") + 1] = bounds_arg(legacy)
            p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=renderer_env)
            pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
            if not (p.returncode or not pngs):
                used_legacy.append(True)
        if p.returncode or not pngs:
            why = (p.stderr or p.stdout).strip()[-400:]
            if "bounds require" in why:
                why += (f" [{rw} predates dateline frames (gpuwm 42631c9f83e, 2.9.12): set {RENDERER_ENV} "
                        f"to a 2.9.12 rw_wrfbatch, e.g. {PINNED_RENDERER}]")
            return None, f"{prod} {valid:%Y%m%d%H}: exit {p.returncode}: {why}"
        # with an observation the renderer also writes the side-by-side; keep the model panel as the product
        model = [q for q in pngs if "comparison" not in q.name and "observed" not in q.name] or pngs
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(model[0]), dst)
        extra = []
        for q in pngs:
            if q is model[0] or not q.exists():
                continue
            tag = "comparison" if "comparison" in q.name else ("observed" if "observed" in q.name else "extra")
            e = dst.with_name(dst.stem + f"_{tag}.png")
            shutil.move(str(q), e)
            extra.append(str(e))
        return [str(dst), *extra], None

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), _cores.usable(), 64))) as ex:
        for w, err in ex.map(one, jobs):
            if err:
                failures.append(err)
            else:
                written += w
    shutil.rmtree(work, ignore_errors=True)
    if used_legacy:
        print(f"woof-nam maps: {rw} predates dateline frames (gpuwm 2.9.12): maps use the CONUS frame "
              f"{bounds_arg(legacy)} instead of the domain's own {bounds_arg(frame)}", file=sys.stderr, flush=True)
        frame = legacy
    result = {"written": written, "failures": failures, "skipped": skipped,
              "frame": list(frame) if frame else None, "renderer": str(rw), "label": label}
    receipt = os.environ.get(RECEIPT_ENV)
    if receipt:
        Path(receipt).write_text(json.dumps(result))
    return result


def main(argv=None):
    result = render(argv)
    for path in result["written"]:
        print(f"RENDERED {path}")
    for slug, reason in result["skipped"]:
        print(f"SKIPPED {slug} {reason}")
    for line in result["failures"]:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
