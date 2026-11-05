"""Compare a published run with the GFS cycle of the same time, at the same valid times. A live run has no NAM to score against,
so this is a sanity check on drift, not a skill score: GFS isn't the truth. Usage: check_against_gfs.py CYCLE OUT.csv"""
import csv, sys
import eccodes, numpy as np, requests, xarray as xr
from scipy.interpolate import RegularGridInterpolator

CYCLE, OUT = sys.argv[1:3]
STORE = f"https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net/{CYCLE}/woof/zarr"
GFS = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{CYCLE[:8]}/{CYCLE[8:]}/atmos/gfs.t{CYCLE[8:]}z.pgrb2.0p25.f%03d"
# Store variable -> GFS inventory match, GFS-to-store scale and offset, then store-to-table scale.
FIELDS = {"z500": ("HGT:500 mb", .1, 0, 10), "mslp": ("PRMSL:mean sea level", .01, 0, 1), "t850": ("TMP:850 mb", 1, -273.15, 1), "t2m": ("TMP:2 m above ground", 1, -273.15, 1)}
HOURS = (0, 12, 24, 48, 84)
GFS_NY, GFS_NX = 721, 1440

def read_gfs(hour, match):
    """Fetch one message by byte range, using the .idx inventory."""
    url = GFS % hour; idx = requests.get(url + ".idx", timeout=60); idx.raise_for_status(); lines = idx.text.splitlines()
    for n, line in enumerate(lines):
        if match in line:
            first, last = int(line.split(":")[1]), int(lines[n + 1].split(":")[1]) - 1
            r = requests.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=120); r.raise_for_status()
            g = eccodes.codes_new_from_message(r.content); v = eccodes.codes_get_values(g).reshape(GFS_NY, GFS_NX); eccodes.codes_release(g); return v
    raise SystemExit(f"{match} not in {url}.idx")

ds = xr.open_zarr(STORE, consolidated=True, zarr_format=3); lat, lon = ds.latitude.values, ds.longitude.values
glat = (90 - .25 * np.arange(GFS_NY))[::-1]; glon = np.r_[.25 * np.arange(GFS_NX), 360]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["field", "fhr", "bias", "rmse"])
    for name, (match, scale, offset, table_scale) in FIELDS.items():
        for h in HOURS:
            g = read_gfs(h, match) * scale + offset; wrapped = np.c_[g, g[:, 0]][::-1]
            on_grid = RegularGridInterpolator((glat, glon), wrapped)(np.c_[lat.ravel(), lon.ravel() % 360]).reshape(lat.shape)
            e = (ds[name].values[h] - on_grid) * table_scale
            w.writerow([name, h, f"{np.nanmean(e):.2f}", f"{np.sqrt(np.nanmean(e ** 2)):.2f}"])
