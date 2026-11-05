"""Build production's refd3d.tHHz.grb2f00 (blocked on NOMADS) from public MRMS merged reflectivity.

nam_ref2nemsio reads 31 REFD levels from 500 m to 18 km on a 0.05 degree grid (-130..-60, 20..55). MRMS
publishes those same heights on a 0.01 degree grid; 17 km and 19 km exist in MRMS but aren't in the file.
Usage: make_refd3d.py YYYYMMDDHH
"""
import gzip, os, re, shutil, subprocess, sys, tempfile, urllib.request
MRMS = "https://noaa-mrms-pds.s3.amazonaws.com"; OUT = "/nam/com/radarl2/v1.2"
LEVELS_KM = [0.5 + 0.25 * i for i in range(11)] + [3.5 + 0.5 * i for i in range(11)] + [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0]
MAX_OFFSET_S = 300   # Refuse a volume more than 5 minutes from the analysis time.
WGRIB2 = "/nam/util_exec/wgrib2"

def nearest_key(km, c):
    prod = f"MergedReflectivityQC_{km:05.2f}"; want = int(c[8:]) * 3600; best = None
    # Listing the day and the previous day would cost far more; a volume near 00Z is within this day's first files.
    url = f"{MRMS}/?list-type=2&prefix=CONUS/{prod}/{c[:8]}/MRMS_{prod}_{c[:8]}-{c[8:]}"
    s = urllib.request.urlopen(url, timeout=60).read().decode()
    for k in re.findall(r"<Key>([^<]+)</Key>", s):
        m = re.search(r"-(\d\d)(\d\d)(\d\d)\.grib2", k); off = abs(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) - want)
        if off <= MAX_OFFSET_S and (best is None or off < best[0]): best = (off, k)
    if best is None: raise SystemExit(f"no MRMS volume near {c} for {prod}")
    return best[1]

# Written to .off so runs in progress cannot find it; variant.sh renames it for tests that want radar.
c = sys.argv[1]; d = f"{OUT}/radar.{c[:8]}.off"; os.makedirs(d, exist_ok=True); out = f"{d}/refd3d.t{c[8:]}z.grb2f00"
tmp = tempfile.mkdtemp(dir="/nam/ens_tmp"); part = out + ".part"; open(part, "wb").close()
for km in LEVELS_KM:
    k = nearest_key(km, c); gz = f"{tmp}/a.gz"; raw = f"{tmp}/a.grib2"; g = f"{tmp}/b.grib2"
    urllib.request.urlretrieve(f"{MRMS}/{k}", gz)
    with gzip.open(gz) as i, open(raw, "wb") as o: shutil.copyfileobj(i, o)
    subprocess.run([WGRIB2, raw, "-set_grib_type", "c3", "-new_grid_interpolation", "neighbor", "-new_grid_winds", "earth",
                    "-new_grid", "latlon", "230:1401:0.05", "20:701:0.05", g], check=True, capture_output=True)
    subprocess.run([WGRIB2, g, "-set_var", "REFD", "-set_lev", f"{int(km*1000)} m above mean sea level", "-append", "-grib", part], check=True, capture_output=True)
shutil.rmtree(tmp); os.rename(part, out); print("REFD_DONE", out, len(LEVELS_KM))
