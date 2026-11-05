"""Maps of a built static file with the Rust renderer rw_wrfbatch (CF route), titled WOOF-NAM.

    python tools/plot_static.py static/nam4km_static.npz OUTDIR "WOOF-NAM 4 km static" [--fields hgt,land,vgtyp,sotyp,tg]
        [--bounds W,E,S,N]

Route: the static's own GLAT/GLON (H points, the rotated B grid) are written as a curvilinear CF file, one
variable per field, and rw_wrfbatch --cf draws each with a CF table row carrying its own class or physical
ladder (land use: the 20 IGBP classes + water 0; soil: the 16 STATSGO/SoilGrids texture classes + water 0).
Bounds follow the domain, unwrapped about its centre so a domain across the dateline is drawn whole
(the renderer recentres when east > 180; rw_wrfbatch from gpuwm 2.9.12 source 42631c9f83e on, older bridges
refuse east > 180). One renderer process per field, in parallel.
Class fields (land, land use, soil): the CF route draws with a smooth legend and interpolated sampling, which
paints codes no cell holds at class edges (crop 12 | mosaic 14 painted as urban 13). So those are written on a
grid 4x finer, each cell's code repeated and the coordinates computed exactly from the rotated grid (META),
which keeps any in-between colour to a one-pixel seam.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

WATER = [70, 130, 180]
IGBP = [WATER, [0, 100, 0], [0, 140, 0], [80, 160, 60], [120, 190, 80], [60, 140, 90], [190, 140, 80],
        [220, 190, 120], [180, 200, 80], [230, 220, 100], [170, 220, 120], [80, 180, 200], [250, 230, 110],
        [220, 0, 0], [200, 210, 60], [250, 250, 250], [190, 190, 190], WATER, [130, 110, 160], [170, 150, 190],
        [210, 200, 220]]
SOIL = [WATER, [255, 240, 150], [240, 215, 120], [220, 180, 90], [190, 160, 120], [170, 140, 110],
        [150, 110, 70], [200, 120, 60], [160, 90, 90], [140, 70, 50], [190, 90, 40], [120, 60, 80],
        [100, 40, 30], [40, 40, 40], WATER, [130, 130, 130], [235, 245, 255]]
# land only (water cells are left blank): a water band at the bottom of a smooth ladder tinted low coastal
# plains as sea (east Sumatra at 0-50 m painted blue, 2026-11-04)
TERRAIN_LV = [-50.0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4500]
TERRAIN_C = [[40, 120, 60], [90, 150, 70], [150, 180, 90], [200, 200, 110], [210, 180, 100],
             [190, 140, 80], [160, 110, 70], [140, 100, 90], [170, 160, 160], [215, 215, 215], [250, 250, 250]]
TG_LV = [float(x) for x in range(240, 315, 5)]
TG_C = [[49, 54, 149], [59, 85, 164], [69, 117, 180], [93, 145, 195], [116, 173, 209], [144, 195, 221],
        [171, 217, 233], [224, 243, 248], [254, 224, 144], [253, 174, 97], [244, 109, 67], [215, 48, 39],
        [190, 20, 38], [165, 0, 38]]


CLASS_FIELDS = ("land", "vgtyp", "sotyp")
UPSAMPLE = 4


def rotated_latlon(meta, ii, jj):
    """Geographic lat/lon (deg) at fractional H indices of the (cropped) grid: NMMB rotated lat-lon."""
    d = np.pi / 180.0
    tph0, tlm0 = meta["tph0d"] * d, meta["tlm0d"] * d
    dlm, dph = -2.0 * meta["wbd"] / (meta["im"] - 1) * d, -2.0 * meta["sbd"] / (meta["jm"] - 1) * d
    tlm, tph = meta["wbd"] * d + ii * dlm, meta["sbd"] * d + jj * dph
    st0, ct0 = np.sin(tph0), np.cos(tph0)
    lat = np.arcsin(np.clip(st0 * np.cos(tph) * np.cos(tlm) + ct0 * np.sin(tph), -1.0, 1.0))
    lon = tlm0 + np.arctan2(np.cos(tph) * np.sin(tlm), ct0 * np.cos(tph) * np.cos(tlm) - st0 * np.sin(tph))
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    return lat / d, lon / d


def _row(var, title, units, levels, colors):
    return {"names": [var], "standard_names": [], "product": f"static_{var}", "title": title,
            "default_units": units, "display_units": units, "unit_factors": {units: 1.0},
            "scale": {"levels": [float(v) for v in levels], "colors": colors}}


FIELDS = {
    "hgt": ("HGT_SFC", _row("hgt", "Terrain height over land (GMTED2010)", "m", TERRAIN_LV, TERRAIN_C)),
    "land": ("LAND", _row("land", "Land mask (1 land, 0 water; MODIS IGBP with lakes)", "1", [-0.5, 0.5, 1.5],
                          [WATER, [200, 170, 110]])),
    "vgtyp": ("VGTYP", _row("vgtyp", "Land use (IGBP 1-20, water 0)", "category",
                            [c - 0.5 for c in range(0, 22)], IGBP)),
    "sotyp": ("SOTYP", _row("sotyp", "Soil type (SoilGrids250m texture 1-16, water 0)", "category",
                            [c - 0.5 for c in range(0, 18)], SOIL)),
    "tg": ("TG", _row("tg", "Deep soil temperature (ERA5 sea-level climatology)", "K", TG_LV, TG_C)),
}


def _bridges() -> Path:
    env = os.environ.get("GPUWM_BRIDGES")
    if env and (Path(env) / "rw_wrfbatch").is_file():
        return Path(env)
    from gpuwm import rustwx  # noqa: PLC0415
    r = rustwx.find_renderer()
    if r is None:
        raise SystemExit(rustwx.renderer_remedy())
    return Path(r).parent


def bounds(glat, glon, pad=0.03):
    jc, ic = glat.shape[0] // 2, glat.shape[1] // 2
    c = float(glon[jc, ic])
    lon = (glon - c + 180.0) % 360.0 - 180.0 + c
    w, e, s, n = float(lon.min()), float(lon.max()), float(glat.min()), float(glat.max())
    dx, dy = (e - w) * pad, (n - s) * pad
    if e - w + 2 * dx >= 360.0 or n > 89.0 or s < -89.0:
        # the domain holds a pole (NAM's own 12 km does): the CF route's geographic frame cannot draw it whole
        # (strips across the pole row), so pass --bounds for a regional frame instead
        print("plot_static: domain holds a pole; drawing the full-longitude frame, use --bounds W,E,S,N", flush=True)
        w, e, dx = c - 180.0, c + 180.0, 0.0
    return f"{w - dx:.3f},{e + dx:.3f},{max(-90.0, s - dy):.3f},{min(90.0, n + dy):.3f}"


def main(path, out, title, fields="hgt,land,vgtyp,sotyp,tg", width=1100, height=1000, frame=None):
    path, out = Path(path), Path(out)
    z = np.load(path)
    meta = json.loads(str(z["META"])) if "META" in z.files else {}
    valid = datetime.fromisoformat(meta.get("valid", "2026-01-01T00:00:00"))
    names = [f.strip() for f in fields.split(",") if f.strip()]
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cf-static-", dir=out))
    native = [f for f in names if f not in CLASS_FIELDS]
    classes = [f for f in names if f in CLASS_FIELDS]
    files = {}
    if native:
        land = z["LAND"] > 0.5
        _write(work / "static.nc", z["GLAT"].astype(np.float32), z["GLON"].astype(np.float32),
               {f: np.where(land, z[FIELDS[f][0]], np.nan) if f == "hgt" else z[FIELDS[f][0]] for f in native},
               title, valid)
        files.update({f: work / "static.nc" for f in native})
    if classes:
        k = UPSAMPLE
        jm, im = z["GLAT"].shape
        sub = (np.arange(k) + 0.5) / k - 0.5
        ii = (np.arange(im)[:, None] + sub[None, :]).ravel()
        jj = (np.arange(jm)[:, None] + sub[None, :]).ravel()
        J, I = np.meshgrid(jj, ii, indexing="ij")
        la, lo = rotated_latlon(meta, I, J)
        up = {f: np.repeat(np.repeat(z[FIELDS[f][0]], k, 0), k, 1) for f in classes}
        _write(work / "static_classes.nc", la.astype(np.float32), lo.astype(np.float32), up, title, valid)
        files.update({f: work / "static_classes.nc" for f in classes})
    table = work / "cf-table.json"
    table.write_text(json.dumps([FIELDS[f][1] for f in names]))
    renderer = _bridges() / "rw_wrfbatch"
    bb = frame or bounds(z["GLAT"].astype(np.float64), z["GLON"].astype(np.float64))
    stem = path.name.replace(".npz", "")

    def one(f):
        od = work / f"r_{f}"
        cmd = [str(renderer), "--cf", str(files[f]), "--variable", f, "--out-dir", str(od), "--valid",
               valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", bb, "--cf-table", str(table),
               "--width", str(width), "--height", str(height)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
        if p.returncode or not pngs:
            return f, None, (p.stderr or p.stdout).strip()[-600:]
        dst = out / f"{stem}_{f}.png"
        shutil.move(str(pngs[0]), dst)
        return f, str(dst), None

    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        res = list(ex.map(one, names))
    shutil.rmtree(work, ignore_errors=True)
    bad = [(f, e) for f, d, e in res if d is None]
    for f, d, e in res:
        print(f"plot_static: {f} -> {d}" if d else f"plot_static: {f} FAILED: {e}")
    return 1 if bad else 0


def _write(cf, lat, lon, fields, title, valid):
    import netCDF4  # noqa: PLC0415
    ny, nx = lat.shape
    with netCDF4.Dataset(cf, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.source = f"{title}, static (no init), built for {valid:%Y-%m-%d %HZ}"
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
        for f, arr in fields.items():
            v = nc.createVariable(f, "f4", ("time", "y", "x"), fill_value=np.float32(np.nan))
            v.units = FIELDS[f][1]["default_units"]
            v[0] = arr.astype(np.float32)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    for flag, key in (("--fields", "fields"), ("--bounds", "frame")):
        if flag in a:
            k = a.index(flag)
            kw[key] = a[k + 1]
            del a[k:k + 2]
    raise SystemExit(main(*a[:3], **kw))
