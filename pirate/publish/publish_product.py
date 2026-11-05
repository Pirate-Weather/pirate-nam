"""Turn a run's product_fields npz files into what gets published: hourly GRIB2, a browser-readable Zarr, and a manifest.
Runs on the host that holds storage credentials, so the GPU machine never needs them. Nothing is uploaded with --no-upload.

GRIB2: rotated lat-lon (template 3.1) on the NAM H grid, earth-relative winds, hourly and running-total precipitation.
Only WMO-standard parameter numbers, and the originating centre stays unset: this is not an NCEP product and must not say so.
Zarr: the viewer's layout, one gzip chunk per forecast hour; xarray opens it with open_zarr(consolidated=True).
Every product is validated first: missing hours or broken ranges stop publication.
Usage: publish_product.py NPZ_DIR OUT_DIR CYCLE [--no-upload]
"""
import gzip, json, os, sys, glob
import numpy as np, eccodes
NPZ, OUT, CYCLE = sys.argv[1:4]; UPLOAD = "--no-upload" not in sys.argv
NX, NY, WBD, SBD, DLON, DLAT, TPH0, TLM0 = 954, 835, -60.039, -45.036, 0.126, 0.108, 54.0, -106.0
FILL = -32768
LEVELS = (1000, 925, 850, 700, 500, 300, 250)
SOURCE = "Pirate NAM (WOOF-NAM GPU NMMB), experimental, not a NOAA or NCEP product"
# Sane ranges by field name prefix, in file units. Outside these a run isn't published.
RANGES = {"t2m": (180, 340), "d2m": (150, 330), "mslp": (85000, 110000), "psfc": (40000, 110000), "t": (150, 340), "z": (-500, 14000), "u": (-150, 150), "v": (-150, 150), "rh": (0, 100.01), "pwat": (0, 120), "refc": (-40, 90), "apcp_total": (0, 3000)}
# Viewer variables: key, title, units, function of the fields dict, scale factor (matches the page's colour scales).
VIEW = [("t2m", "2 m temperature", "°C", lambda f: f["t2m"] - 273.15, 0.01), ("d2m", "2 m dewpoint", "°C", lambda f: f["d2m"] - 273.15, 0.01),
        ("mslp", "Sea-level pressure", "hPa", lambda f: f["mslp"] * 0.01, 0.1), ("wind10", "10 m wind speed", "m/s", lambda f: np.hypot(f["u10"], f["v10"]), 0.02),
        ("apcp", "Total precipitation since start", "mm", lambda f: f["apcp"], 0.05), ("refc", "Composite reflectivity", "dBZ", lambda f: f["refc"], 0.05),
        ("pwat", "Precipitable water", "mm", lambda f: f["pwat"], 0.05), ("z500", "500 mb height", "dam", lambda f: f["z500"] * 0.1, 0.1),
        ("t850", "850 mb temperature", "°C", lambda f: f["t850"] - 273.15, 0.01), ("wind250", "250 mb wind speed", "m/s", lambda f: np.hypot(f["u250"], f["v250"]), 0.05)]

files = sorted(glob.glob(f"{NPZ}/f*.npz")); hours = [int(os.path.basename(p)[1:-4]) for p in files]
if not hours or hours != list(range(hours[-1] + 1)): raise SystemExit(f"hours are not continuous: {hours}")

def check(name, a, fh):
    key = next((k for k in sorted(RANGES, key=len, reverse=True) if name == k or (k in ("t", "z", "u", "v", "rh") and name.startswith(k) and name[len(k):].isdigit())), None)
    if key is None: return
    good = a[np.isfinite(a)]
    below_ground_ok = key in ("t", "z", "u", "v", "rh")
    if good.size == 0 or (not below_ground_ok and good.size != a.size): raise SystemExit(f"{name} f{fh:02d}: has missing values")
    if good.min() < RANGES[key][0] or good.max() > RANGES[key][1]: raise SystemExit(f"{name} f{fh:02d}: {good.min()}..{good.max()} outside {RANGES[key]}")

def message(fid, base, fh, cat, num, ltype, lev, values, disc=0, accum_from=None):
    h = eccodes.codes_clone(base)
    eccodes.codes_set(h, "discipline", disc); eccodes.codes_set(h, "parameterCategory", cat); eccodes.codes_set(h, "parameterNumber", num)
    if accum_from is not None:
        # Accumulation: statistically processed product over the window ending at this hour.
        eccodes.codes_set(h, "productDefinitionTemplateNumber", 8); eccodes.codes_set(h, "typeOfStatisticalProcessing", 1)
        eccodes.codes_set(h, "stepType", "accum"); eccodes.codes_set(h, "startStep", accum_from); eccodes.codes_set(h, "endStep", fh)
    else:
        eccodes.codes_set(h, "stepRange", str(fh))
    eccodes.codes_set(h, "typeOfFirstFixedSurface", ltype)
    eccodes.codes_set(h, "scaleFactorOfFirstFixedSurface", 0); eccodes.codes_set(h, "scaledValueOfFirstFixedSurface", lev)
    miss = ~np.isfinite(values)
    eccodes.codes_set(h, "bitmapPresent", 1 if miss.any() else 0)
    eccodes.codes_set(h, "missingValue", 9999.0)
    eccodes.codes_set_values(h, np.where(miss, 9999.0, values).astype("f8").ravel())
    eccodes.codes_write(h, fid); eccodes.codes_release(h)

def grid_lat_lon():
    """Geographic position of every H point, from the NMMB rotated grid."""
    d = np.deg2rad; x, y = np.meshgrid(WBD + np.arange(NX) * DLON, SBD + np.arange(NY) * DLAT)
    tp, lam, phi = d(TPH0), d(x), d(y)
    lat = np.arcsin(np.cos(tp) * np.sin(phi) + np.sin(tp) * np.cos(phi) * np.cos(lam))
    lon = TLM0 + np.rad2deg(np.arctan2(np.cos(phi) * np.sin(lam), np.cos(tp) * np.cos(phi) * np.cos(lam) - np.sin(tp) * np.sin(phi)))
    return np.rad2deg(lat), (lon + 180.0) % 360.0 - 180.0

def write_coord(zdir, meta, name, values, dims, units, long_name):
    """Coordinates, so xr.open_zarr shows real positions and forecast hours instead of bare indices."""
    os.makedirs(f"{zdir}/{name}", exist_ok=True); a = np.asarray(values, "<f4")
    za = {"zarr_format": 2, "shape": list(a.shape), "chunks": list(a.shape), "dtype": "<f4", "compressor": {"id": "gzip", "level": 6}, "fill_value": None, "filters": None, "order": "C", "dimension_separator": "."}
    at = {"_ARRAY_DIMENSIONS": dims, "units": units, "long_name": long_name}
    with gzip.open(f"{zdir}/{name}/" + ".".join("0" * a.ndim), "wb", compresslevel=6) as g: g.write(a.tobytes())
    json.dump(za, open(f"{zdir}/{name}/.zarray", "w")); json.dump(at, open(f"{zdir}/{name}/.zattrs", "w")); meta[f"{name}/.zarray"] = za; meta[f"{name}/.zattrs"] = at

def template():
    # The eccodes sample says centre 98, ECMWF. This product is nobody's but ours, so the centre has to be set to missing.
    h = eccodes.codes_grib_new_from_samples("rotated_ll_sfc_grib2")
    for k, v in (("centre", 255), ("subCentre", 0), ("generatingProcessIdentifier", 255), ("tablesVersion", 28), ("localTablesVersion", 0), ("jScansPositively", 1), ("Ni", NX), ("Nj", NY), ("latitudeOfFirstGridPointInDegrees", SBD), ("longitudeOfFirstGridPointInDegrees", WBD),
                 ("latitudeOfLastGridPointInDegrees", SBD + (NY - 1) * DLAT), ("longitudeOfLastGridPointInDegrees", WBD + (NX - 1) * DLON),
                 ("iDirectionIncrementInDegrees", DLON), ("jDirectionIncrementInDegrees", DLAT), ("latitudeOfSouthernPoleInDegrees", -(90.0 - TPH0)),
                 ("longitudeOfSouthernPoleInDegrees", TLM0 + 360.0), ("dataDate", int(CYCLE[:8])), ("dataTime", int(CYCLE[8:]) * 100), ("packingType", "grid_ccsds"),
                 ("bitsPerValue", 16)):
        try: eccodes.codes_set(h, k, v)
        except eccodes.GribInternalError:
            if k != "packingType": raise
            eccodes.codes_set(h, "packingType", "grid_simple")
    return h

os.makedirs(f"{OUT}/grib", exist_ok=True); zdir = f"{OUT}/zarr"
base = template(); rng = {}
apcp0 = None; prev_total = None
for fh, path in zip(hours, files):
    f = dict(np.load(path))
    for k, a in f.items(): check(k, a, fh)
    total = f["apcp_total"]
    if apcp0 is None: apcp0 = total.copy(); prev_total = total.copy()
    f["apcp"] = np.maximum(total - apcp0, 0.0)
    gp = f"{OUT}/grib/woofnam.t{CYCLE[8:]}z.f{fh:03d}.grib2"
    with open(gp + ".part", "wb") as g:
        message(g, base, fh, 0, 0, 103, 2, f["t2m"]); message(g, base, fh, 0, 6, 103, 2, f["d2m"])
        message(g, base, fh, 2, 2, 103, 10, f["u10"]); message(g, base, fh, 2, 3, 103, 10, f["v10"])
        message(g, base, fh, 3, 1, 101, 0, f["mslp"]); message(g, base, fh, 3, 0, 1, 0, f["psfc"])
        message(g, base, fh, 1, 3, 10, 0, f["pwat"]); message(g, base, fh, 16, 5, 10, 0, f["refc"])
        if fh > 0:
            message(g, base, fh, 1, 8, 1, 0, f["apcp"], accum_from=0)
            message(g, base, fh, 1, 8, 1, 0, np.maximum(total - prev_total, 0.0), accum_from=fh - 1)
        for lev in LEVELS:
            pa = lev * 100
            message(g, base, fh, 0, 0, 100, pa, f[f"t{lev}"]); message(g, base, fh, 3, 5, 100, pa, f[f"z{lev}"]); message(g, base, fh, 1, 1, 100, pa, f[f"rh{lev}"])
            message(g, base, fh, 2, 2, 100, pa, f[f"u{lev}"]); message(g, base, fh, 2, 3, 100, pa, f[f"v{lev}"])
    os.replace(gp + ".part", gp)
    prev_total = total
    for key, _, _, fn, sf in VIEW:
        a = np.asarray(fn(f), "f8").ravel(); os.makedirs(f"{zdir}/{key}", exist_ok=True)
        ok = np.isfinite(a)
        # Fail rather than clip: a saturated field looks like plausible data on the map. z500 did this once.
        if ok.any() and np.abs(a[ok]).max() / sf > 32767: raise SystemExit(f"{key} f{fh} overflows int16 at scale {sf}: max {np.abs(a[ok]).max()}")
        i = np.full(a.size, FILL, "<i2"); i[ok] = np.clip(np.rint(a[ok] / sf), -32767, 32767).astype("<i2")
        with gzip.open(f"{zdir}/{key}/{fh}.0.0", "wb", compresslevel=6) as g: g.write(i.tobytes())
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a)); old = rng.get(key, (lo, hi)); rng[key] = (min(lo, old[0]), max(hi, old[1]))
    print("hour", fh, flush=True)
meta = {".zgroup": {"zarr_format": 2}}
for key, title, units, _, sf in VIEW:
    za = {"zarr_format": 2, "shape": [len(hours), NY, NX], "chunks": [1, NY, NX], "dtype": "<i2", "compressor": {"id": "gzip", "level": 6}, "fill_value": FILL, "filters": None, "order": "C", "dimension_separator": "."}
    at = {"_ARRAY_DIMENSIONS": ["fhr", "y", "x"], "scale_factor": sf, "units": units, "long_name": title, "stagger_v": 0, "min": rng[key][0], "max": rng[key][1], "coordinates": "latitude longitude"}
    json.dump(za, open(f"{zdir}/{key}/.zarray", "w")); json.dump(at, open(f"{zdir}/{key}/.zattrs", "w")); meta[f"{key}/.zarray"] = za; meta[f"{key}/.zattrs"] = at
lat2d, lon2d = grid_lat_lon()
write_coord(zdir, meta, "latitude", lat2d, ["y", "x"], "degrees_north", "latitude"); write_coord(zdir, meta, "longitude", lon2d, ["y", "x"], "degrees_east", "longitude")
write_coord(zdir, meta, "fhr", hours, ["fhr"], "hours", "forecast hour")
meta[".zattrs"] = {"cycle": CYCLE, "domain": "woof", "fhrs": hours, "nx": NX, "ny": NY, "wbd": WBD, "sbd": SBD, "dlon": DLON, "dlat": DLAT, "tph0": TPH0, "tlm0": TLM0, "source": SOURCE}
json.dump({"zarr_consolidated_format": 1, "metadata": meta}, open(f"{zdir}/.zmetadata", "w")); json.dump({"zarr_format": 2}, open(f"{zdir}/.zgroup", "w")); json.dump(meta[".zattrs"], open(f"{zdir}/.zattrs", "w"))
print("built", len(hours), "hours", OUT, flush=True)
