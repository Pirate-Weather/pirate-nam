"""Referee sheets for case 2025-03-14_12z: archived NAM forecast | MRMS, plus the native 12Z analysis.

Decode: grib2_inventory / grib2_dump (Rust). Draw: rw_wrfbatch --cf (Rust, production panels and
ladders). Python only selects records, writes the CF container and tiles finished PNGs.
Small numpy steps, stated: Lambert lat/lon of grid 218 from its GRIB header, K->degF, 1 h precip
as the difference of two NAM 3 h-bucket accumulations, 10 m speed from U/V.
"""
import json, os, subprocess, shutil
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, netCDF4

B = Path.home() / "venv/lib/python3.12/site-packages/gpuwm/libexec/bridges"
CASE = Path.home() / "nam-gpu/cases/2025-03-14_12z"
RAW = CASE / "raw"; OUT = CASE / "referee"; TMP = OUT / "tmp"
INIT = datetime(2025, 3, 14, 12)
TMP.mkdir(parents=True, exist_ok=True)


def inventory(path):
    lines = subprocess.run([B / "grib2_inventory", path], capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [l for l in lines if not l.startswith("#")]
    head = rows[0].split("\t")
    return [dict(zip(head, r.split("\t"))) for r in rows[1:]]


def pick(inv, d, c, p, lt, lv=None, pdt=None):
    for r in inv:
        if (int(r["discipline"]), int(r["category"]), int(r["parameter"]), int(r["level_type"])) == (d, c, p, lt) \
           and (lv is None or float(r["level_value"]) == lv) and (pdt is None or int(r["pdt"]) == pdt):
            return int(r["index"])
    raise KeyError((d, c, p, lt, lv, pdt))


def dump(path, idxs, tag):
    dd = TMP / tag
    if dd.exists():
        shutil.rmtree(dd)
    subprocess.run([B / "grib2_dump", path, dd, *map(str, idxs)], check=True, capture_output=True)
    meta = [l.split("\t") for l in (dd / "metadata.tsv").read_text().splitlines()]
    h = meta[0]; out = {}
    for m in meta[1:]:
        r = dict(zip(h, m))
        out[int(r["index"])] = np.fromfile(dd / r["filename"], "<f8").reshape(int(r["ny"]), int(r["nx"]))
    shutil.rmtree(dd)
    return out


def lcc218(nx, ny):
    # GDT 3.30, earth shape 6 (sphere R=6371229 m), latin1=latin2=25N, LoV=265E, first point = SW corner, dx=dy=12191 m
    R = 6371229.0; phi0 = np.radians(25.0); lov = np.radians(265.0 - 360.0)
    n = np.sin(phi0); F = np.cos(phi0) * np.tan(np.pi / 4 + phi0 / 2) ** n / n
    rho = lambda phi: R * F / np.tan(np.pi / 4 + phi / 2) ** n
    rho0 = rho(phi0)
    la1, lo1 = np.radians(12.19), np.radians(226.541 - 360.0)
    r1 = rho(la1)
    x0 = r1 * np.sin(n * (lo1 - lov)); y0 = rho0 - r1 * np.cos(n * (lo1 - lov))
    x = x0 + 12191.0 * np.arange(nx)[None, :]; y = y0 + 12191.0 * np.arange(ny)[:, None]
    yy = rho0 - y
    rr = np.hypot(x, yy)
    th = np.arctan2(x, yy)
    lat = 2 * np.arctan((R * F / rr) ** (1 / n)) - np.pi / 2
    lon = lov + th / n
    return np.degrees(lat), (np.degrees(lon) + 180) % 360 - 180


def write_cf(path, lat, lon, valid, fields, source):
    with netCDF4.Dataset(path, "w") as nc:
        ny, nx = lat.shape
        nc.createDimension("time", 1); nc.createDimension("y", ny); nc.createDimension("x", nx)
        nc.source = source
        t = nc.createVariable("time", "f8", ("time",)); t.units = "hours since 2025-03-14 00:00:00"; t.standard_name = "time"
        t[:] = [(valid - datetime(2025, 3, 14)).total_seconds() / 3600]
        a = nc.createVariable("lat", "f4", ("y", "x")); a.standard_name = "latitude"; a.units = "degrees_north"; a[:] = lat
        b = nc.createVariable("lon", "f4", ("y", "x")); b.standard_name = "longitude"; b.units = "degrees_east"; b[:] = lon
        for name, (arr, units) in fields.items():
            v = nc.createVariable(name, "f4", ("time", "y", "x")); v.units = units
            v[0] = arr.astype(np.float32)


TABLE = [
    {"names": ["refc"], "standard_names": [], "product": "composite_reflectivity", "title": "Composite reflectivity", "recipe": "composite_reflectivity", "default_units": "dBZ", "display_units": "dBZ", "unit_factors": {"dBZ": 1.0}, "observation_product": "refc"},
    {"names": ["t2m"], "standard_names": [], "product": "t2m", "title": "2 m temperature", "recipe": "2m_temperature", "default_units": "degF", "display_units": "degF", "unit_factors": {"degF": 1.0}},
    {"names": ["d2m"], "standard_names": [], "product": "d2m", "title": "2 m dewpoint", "recipe": "2m_dewpoint", "default_units": "degF", "display_units": "degF", "unit_factors": {"degF": 1.0}},
    {"names": ["ws10m"], "standard_names": [], "product": "wind10", "title": "10 m wind speed", "recipe": "10m_wind_speed_and_direction", "default_units": "m s-1", "display_units": "kt", "unit_factors": {"m s-1": 1.9438445}},
    {"names": ["ws850"], "standard_names": [], "product": "wind850", "title": "850 hPa wind speed", "recipe": "10m_wind_speed_and_direction", "default_units": "m s-1", "display_units": "kt", "unit_factors": {"m s-1": 1.9438445}},
    {"names": ["qpf1h"], "standard_names": [], "product": "qpf_1h", "title": "1 h precipitation", "recipe": "total_qpf", "default_units": "kg m-2", "display_units": "in", "unit_factors": {"kg m-2": 0.0393700787}},
    {"names": ["mslp"], "standard_names": [], "product": "mslp", "title": "Mean sea level pressure", "recipe": "href_mean_mslp", "default_units": "hPa", "display_units": "hPa", "unit_factors": {"hPa": 1.0}},
]
(OUT / "cf-table.json").write_text(json.dumps(TABLE))
K2F = lambda k: (k - 273.15) * 1.8 + 32
BOUNDS = os.environ.get("BOUNDS") or "-126,-66,22,52"
PNG = OUT / "png"; PNG.mkdir(exist_ok=True)
MDIR = TMP / "mrms"; MDIR.mkdir(exist_ok=True)


def render(cf, var, valid, obs=None):
    cmd = [B / "rw_wrfbatch", "--cf", cf, "--variable", var, "--out-dir", OUT / "render", "--valid",
           valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", BOUNDS, "--cf-table", OUT / "cf-table.json",
           "--width", "1200", "--height", "900"]
    if obs:
        cmd += ["--observation", obs]
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    if p.returncode:
        print("FAIL", var, valid, (p.stderr or p.stdout).strip()[-600:])
        return []
    return [l.split()[-1] for l in p.stdout.splitlines() if l.startswith("RENDERED")]


def mrms_link(valid):
    link = MDIR / f"MRMS_MergedReflectivityQCComposite_00.50_{valid:%Y%m%d-%H}0000.grib2"
    if not link.exists():
        link.symlink_to(RAW / f"mrms/mrms_refc_{valid:%Y%m%d%H}.grib2")
    return link


def nam_file(h):
    f = RAW / f"nam/awphys{h:02d}.grib2"
    return f if f.exists() else RAW / f"nam/awphys{h:02d}.sub.grib2"


inv3 = inventory(RAW / "nam/bgrd3d00.grib2")
lat218 = lon218 = None
for h in map(int, os.environ.get("HOURS", "1,6,12,24").split(",")):
    if h <= 0:
        continue
    valid = INIT + timedelta(hours=h)
    f = nam_file(h)
    inv = inventory(f)
    ix = dict(refc=pick(inv, 0, 16, 196, 200), t=pick(inv, 0, 0, 0, 103, 2, 0), td=pick(inv, 0, 0, 6, 103, 2, 0),
              u=pick(inv, 0, 2, 2, 103, 10, 0), v=pick(inv, 0, 2, 3, 103, 10, 0), ap=pick(inv, 0, 1, 8, 1, 0, 8))
    d = dump(f, ix.values(), f"f{h}")
    ny, nx = d[ix["t"]].shape
    if lat218 is None:
        lat218, lon218 = lcc218(nx, ny)
        print("grid218 lat", lat218.min(), lat218.max(), "lon", lon218.min(), lon218.max())
    ap = d[ix["ap"]]
    if h % 3 != 1:  # NAM buckets precip every 3 h: 1 h = this hour's accumulation minus the previous hour's
        fp = nam_file(h - 1)
        ipa = pick(inventory(fp), 0, 1, 8, 1, 0, 8)
        prev = dump(fp, [ipa], f"p{h}")[ipa]
        print(f"+{h}h precip bucket check: min(cur-prev)={np.nanmin(ap - prev):.3f} max cur={np.nanmax(ap):.1f}")
        ap = np.maximum(ap - prev, 0)
    fields = {"refc": (d[ix["refc"]], "dBZ"), "t2m": (K2F(d[ix["t"]]), "degF"), "d2m": (K2F(d[ix["td"]]), "degF"),
              "ws10m": (np.hypot(d[ix["u"]], d[ix["v"]]), "m s-1"), "qpf1h": (ap, "kg m-2")}
    cf = TMP / f"nam218_f{h:02d}.nc"
    write_cf(cf, lat218, lon218, valid, fields, f"Archived NAM 12 km forecast (grid 218), 2025-03-14 12Z +{h} h")
    for var in fields:
        for p in render(cf, var, valid, mrms_link(valid) if var == "refc" else None):
            q = PNG / f"f{h:02d}_{Path(p).name}"; shutil.copy(p, q); print("PNG", q)

if os.environ.get("ANL", "1") == "1":
    ix = dict(mslp=pick(inv3, 0, 3, 1, 101), refc=pick(inv3, 0, 16, 196, 200), t=pick(inv3, 0, 0, 0, 103, 2), td=pick(inv3, 0, 0, 6, 103, 2),
              u10=pick(inv3, 0, 2, 2, 103, 10), v10=pick(inv3, 0, 2, 3, 103, 10), u85=pick(inv3, 0, 2, 2, 100, 85000),
              v85=pick(inv3, 0, 2, 3, 100, 85000), nlat=1481, elon=1482)
    d = dump(RAW / "nam/bgrd3d00.grib2", ix.values(), "anl")
    nlat, nlon = d[1481], (d[1482] + 180) % 360 - 180
    print("native parent lat", nlat.min(), nlat.max(), "lon", nlon.min(), nlon.max())
    fields = {"mslp": (d[ix["mslp"]] / 100, "hPa"), "refc": (d[ix["refc"]], "dBZ"), "t2m": (K2F(d[ix["t"]]), "degF"),
              "d2m": (K2F(d[ix["td"]]), "degF"), "ws10m": (np.hypot(d[ix["u10"]], d[ix["v10"]]), "m s-1"),
              "ws850": (np.hypot(d[ix["u85"]], d[ix["v85"]]), "m s-1")}
    cf = TMP / "native_anl_12z.nc"
    write_cf(cf, nlat, nlon, INIT, fields, "NAM native B-grid analysis (bgrd3d00), 2025-03-14 12Z")
    for var in fields:
        for p in render(cf, var, INIT, mrms_link(INIT) if var == "refc" else None):
            q = PNG / f"anl12z_{Path(p).name}"; shutil.copy(p, q); print("PNG", q)
