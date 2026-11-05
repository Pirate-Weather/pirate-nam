"""Score WOOF-NAM fields (woof_fields.py npz) against archived NAM awphys, same statistics as compare_series.py.
APCP takes the same accumulation window as the first APCP record in each awphys file, differencing WOOF-NAM's running total.
Usage: woof_compare.py NPZ_DIR ARC_DIR OUT.csv [hours...]
"""
import csv, importlib.util, re, subprocess, sys
import numpy as np, xarray as xr
spec = importlib.util.spec_from_file_location("ca", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_awphys_lib.py")); ca = importlib.util.module_from_spec(spec); spec.loader.exec_module(ca)
npz_d, arc_d, out = sys.argv[1:4]; hours = [int(h) for h in sys.argv[4:]] or [0, 3, 6, 9, 12]
ds = xr.open_dataset(f"{arc_d}/awphys00.grib2", engine="cfgrib", backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}, "indexpath": ""})
lat, lon = ds.latitude.values, ds.longitude.values; ny, nx = lat.shape; rx, ry = ca.rotated(lat, lon)

def stats(arc, mine):
    ok = np.isfinite(arc) & np.isfinite(mine) & (np.abs(arc) < 9e15) & (np.abs(mine) < 9e15); a, m = arc[ok], mine[ok]; d = m - a
    return d.mean(), np.sqrt((d**2).mean()), (np.corrcoef(a, m)[0, 1] if a.std() > 0 and m.std() > 0 else np.nan)

def acc_window(path):
    inv = subprocess.run(["wgrib2", path, "-match", ":APCP:surface:"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"(\d+)-(\d+) hour acc", inv)
    return (int(m[1]), int(m[2])) if m else None

SCAL = [("PRMSL", ":PRMSL:mean sea level:", "PRMSL"), ("MSLET", ":MSLET:mean sea level:", "PRMSL"), ("TMP2m", ":TMP:2 m above ground:", "TMP2m"), ("DPT2m", ":DPT:2 m above ground:", "DPT2m"),
        ("TMP850", ":TMP:850 mb:", "TMP850"), ("TMP500", ":TMP:500 mb:", "TMP500"), ("HGT850", ":HGT:850 mb:", "HGT850"), ("HGT500", ":HGT:500 mb:", "HGT500"), ("HGT250", ":HGT:250 mb:", "HGT250"),
        ("RH700", ":RH:700 mb:", "RH700"), ("REFC", ":REFC:", "REFC")]
# WOOF-NAM history winds are rotated to earth-relative and averaged to H points (io-wind-rotate), so no V-point offset here.
WIND = [("WIND10m", "10 m above ground"), ("WIND850", "850 mb"), ("WIND250", "250 mb")]
w = csv.writer(open(out, "w")); w.writerow("fhr field source bias rmse corr arc_std".split())
for h in hours:
    f = np.load(f"{npz_d}/f{h:02d}.npz"); a = f"{arc_d}/awphys{h:02d}.grib2"
    for name, pat, key in SCAL:
        A = ca.read(a, pat)
        if A is None: continue
        A2 = A[:nx*ny].reshape(ny, nx); w.writerow([h, name, "woof", *stats(A2, ca.to_arc(f[key].ravel(), rx, ry)), A2.std()])
    for name, lev in WIND:
        ua, va = ca.read(a, f":UGRD:{lev}:"), ca.read(a, f":VGRD:{lev}:")
        A2 = np.hypot(ua[:nx*ny], va[:nx*ny]).reshape(ny, nx); w.writerow([h, name, "woof", *stats(A2, ca.to_arc(f[name].ravel(), rx, ry)), A2.std()])
    win = acc_window(a) if h else None
    if win:
        A2 = ca.read(a, ":APCP:surface:")[:nx*ny].reshape(ny, nx)
        # A cycled run has rained since tm06, so even a window from hour 0 has to difference against that hour.
        total = f["APCP_total"] - np.load(f"{npz_d}/f{win[0]:02d}.npz")["APCP_total"]
        w.writerow([h, "APCP", "woof", *stats(A2, ca.to_arc(total.ravel(), rx, ry)), A2.std()])
    print(h, flush=True)
