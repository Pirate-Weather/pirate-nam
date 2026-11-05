"""Turn the newest published run into a local regular lat-lon Zarr for the WMS server, and validate it before anything serves it.
Runs on the WMS server, reading only the public bucket. Prints the cycle's store path, or nothing if the newest run is already built.
The native grid contains the North Pole, which xpublish-wms draws wrongly, so the WMS serves this regridded copy instead.
Usage: build_store.py DATA_DIR [CYCLE]"""
import json, logging, os, shutil, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import dask.array as da, numpy as np, pandas as pd, xarray as xr, zarr
from scipy.spatial import cKDTree

BASE = "https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net"
DATA = sys.argv[1]
GRID_STEP = 0.1                # degrees, about 11 km: close to the native 12 km.
LAT_MIN, LAT_MAX = -8.0, 90.0  # The domain's southern edge is near 8 S and it reaches past the pole.
MAX_DISTANCE_KM = 12.0         # Further than a grid cell from any model point means off the domain.
EARTH_RADIUS_KM = 6371.0
CHUNK_LAT, CHUNK_LON = 256, 512  # A map tile touches a few chunks, not a whole frame of 3.5 million points.
WORKERS = 4
TITLE = "Pirate NAM, experimental, not a NOAA or NCEP product"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s"); log = logging.getLogger("build_store")

def newest_cycle():
    with urllib.request.urlopen(f"{BASE}/runs.json", timeout=60) as r:
        return max(run["cycle"] for run in json.load(r)["runs"] if "woof" in run.get("domains", []))

def unit_vectors(lat, lon):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)

def load_mapping(ds, lat, lon):
    """Nearest native point for each output point. The grid never changes, so build it once and keep it."""
    path = f"{DATA}/mapping.npz"
    if os.path.exists(path):
        m = np.load(path); return m["jj"], m["ii"], m["inside"]
    t0 = time.time(); nlat = np.radians(ds.latitude.values.astype("f8")); nlon = np.radians(ds.longitude.values.astype("f8"))
    tree = cKDTree(unit_vectors(nlat, nlon).reshape(-1, 3))
    glon, glat = np.meshgrid(np.radians(lon), np.radians(lat))
    dist, idx = tree.query(unit_vectors(glat, glon).reshape(-1, 3), workers=WORKERS)
    inside = (dist * EARTH_RADIUS_KM <= MAX_DISTANCE_KM).reshape(glat.shape)
    jj, ii = np.unravel_index(idx.reshape(glat.shape), nlat.shape)
    np.savez(path + ".tmp.npz", jj=jj.astype("i4"), ii=ii.astype("i4"), inside=inside); os.replace(path + ".tmp.npz", path)
    log.info("mapping built in %.0f s, %.1f%% of the grid is on the domain", time.time() - t0, 100 * inside.mean())
    return jj, ii, inside

CYCLE = sys.argv[2] if len(sys.argv) > 2 else newest_cycle()
FINAL = f"{DATA}/stores/{CYCLE}.zarr"
if os.path.exists(FINAL): sys.exit(0)
t_start = time.time()
ds = xr.open_zarr(f"{BASE}/{CYCLE}/woof/zarr", consolidated=True, zarr_format=3)
if ds.attrs.get("cycle") != CYCLE: raise SystemExit(f"store says cycle {ds.attrs.get('cycle')}, expected {CYCLE}")
lat = np.round(np.arange(LAT_MIN, LAT_MAX + GRID_STEP / 2, GRID_STEP), 1); lon = np.round(np.arange(-180.0, 180.0, GRID_STEP), 1)
jj, ii, inside = load_mapping(ds, lat, lon)
fhr = ds.fhr.values.astype(int); times = pd.Timestamp(f"{CYCLE[:8]}T{CYCLE[8:]}") + pd.to_timedelta(fhr, "h")
names = list(ds.data_vars)

template = xr.Dataset(attrs={"title": TITLE, "cycle": CYCLE, "source": ds.attrs.get("source", ""), "institution": "Pirate Weather"},
                      coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)})
template.lat.attrs.update(standard_name="latitude", units="degrees_north", axis="Y", long_name="latitude")
template.lon.attrs.update(standard_name="longitude", units="degrees_east", axis="X", long_name="longitude")
template.time.attrs.update(standard_name="time", axis="T", long_name="time")
for name in names:
    attrs = {k: v for k, v in ds[name].attrs.items() if k in ("units", "long_name", "standard_name")}
    template[name] = (("time", "lat", "lon"), da.full((len(times), len(lat), len(lon)), np.nan, dtype="f4", chunks=(1, CHUNK_LAT, CHUNK_LON)), attrs)
encoding = {n: {"chunks": (1, CHUNK_LAT, CHUNK_LON), "compressors": [zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")], "fill_value": np.nan} for n in names}
encoding["time"] = {"units": f"hours since {CYCLE[:4]}-{CYCLE[4:6]}-{CYCLE[6:8]} {CYCLE[8:]}:00:00", "dtype": "f8"}
work = FINAL + ".part"; shutil.rmtree(work, ignore_errors=True); os.makedirs(os.path.dirname(FINAL), exist_ok=True)
template.to_zarr(work, compute=False, zarr_format=3, consolidated=True, encoding=encoding, mode="w")

def write_hour(n):
    arrays = {}
    for name in names:
        native = ds[name].isel(fhr=n).values
        arrays[name] = (("time", "lat", "lon"), np.where(inside, native[jj, ii], np.nan)[None].astype("f4"))
    xr.Dataset(arrays).to_zarr(work, region={"time": slice(n, n + 1)}, zarr_format=3, consolidated=False)
with ThreadPoolExecutor(WORKERS) as pool: list(pool.map(write_hour, range(len(times))))
zarr.consolidate_metadata(work, zarr_format=3)
log.info("wrote %d hours of %d fields in %.0f s", len(times), len(names), time.time() - t_start)

# Validate before serving: a bad store must never replace a good one.
out = xr.open_zarr(work, consolidated=True, zarr_format=3)
if set(out.data_vars) != set(names) or out.sizes["time"] != ds.sizes["fhr"]: raise SystemExit("regridded store is incomplete")
for name in names:
    for n in (0, len(times) // 2, len(times) - 1):
        native, regrid = ds[name].isel(fhr=n).values.astype("f4"), out[name].isel(time=n).values  # Compare in the stored precision.
        if not np.isfinite(regrid).any(): raise SystemExit(f"{name} hour {n} is empty")
        lo, hi = np.nanmin(native), np.nanmax(native)
        if np.nanmin(regrid) < lo or np.nanmax(regrid) > hi: raise SystemExit(f"{name} hour {n} is outside the native range")
os.rename(work, FINAL)
print(FINAL)
