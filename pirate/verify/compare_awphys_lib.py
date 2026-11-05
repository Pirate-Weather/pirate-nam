"""Compare NAM-revival native-grid BGDAWP output with an archived NCEP awphys (grid 218) product.

Native output is on the rotated lat/lon B-grid from parm/nam_configfile: tph0d 54, tlm0d -106,
wbd -60.039, sbd -45.036, dlon 0.126, dlat 0.108. Mass fields sit on H points; winds on V points,
offset half a cell in both directions. Winds are compared as speed because native winds are
rotated-grid-relative while grid 218 winds are Lambert-grid-relative.

Usage: compare_awphys.py MINE.grib2 ARC.grib2 [--gfs GFS.grib2]
"""
import subprocess, sys, tempfile, os, numpy as np, xarray as xr
from scipy.ndimage import map_coordinates

NX, NY, TPH0, TLM0, WBD, SBD, DLON, DLAT = 954, 835, 54.0, -106.0, -60.039, -45.036, 0.126, 0.108

def read(path, match, nx=None, ny=None):
    with tempfile.NamedTemporaryFile(delete=False) as t: tn = t.name
    r = subprocess.run(["wgrib2", path, "-match", match, "-order", "we:sn", "-no_header", "-ieee", tn, "-big_endian", "-inv", "/dev/null"], capture_output=True, text=True)
    data = np.fromfile(tn, dtype=">f4"); os.unlink(tn)
    return None if data.size == 0 else data

def rotated(lat, lon):
    """Geographic -> rotated coordinates of the NMM-B grid (degrees)."""
    d = np.deg2rad; tp, tl = d(TPH0), d(TLM0); la, lo = d(lat), d(lon)
    x = np.cos(tp)*np.cos(la)*np.cos(lo-tl) + np.sin(tp)*np.sin(la)
    y = np.cos(la)*np.sin(lo-tl)
    z = np.cos(tp)*np.sin(la) - np.sin(tp)*np.cos(la)*np.cos(lo-tl)
    return np.rad2deg(np.arctan2(y, x)), np.rad2deg(np.arcsin(z))

def to_arc(native, rx, ry, vpt=False):
    off = 0.5 if vpt else 0.0
    i = (rx-WBD)/DLON - off; j = (ry-SBD)/DLAT - off
    return map_coordinates(native.reshape(NY, NX), [j, i], order=1, mode="nearest")

FIELDS = [("PRMSL Pa", r":PRMSL:mean sea level:", False), ("MSLET Pa", r":MSLET:mean sea level:", False),
  ("TMP 2 m K", r":TMP:2 m above ground:", False), ("DPT 2 m K", r":DPT:2 m above ground:", False),
  ("TMP 850 K", r":TMP:850 mb:", False), ("TMP 500 K", r":TMP:500 mb:", False), ("TMP 250 K", r":TMP:250 mb:", False),
  ("HGT 850 m", r":HGT:850 mb:", False), ("HGT 500 m", r":HGT:500 mb:", False), ("HGT 250 m", r":HGT:250 mb:", False),
  ("RH 700 %", r":RH:700 mb:", False), ("REFC dBZ", r":REFC:", False), ("APCP kg/m2", r":APCP:surface:", False),
  ("CAPE sfc J/kg", r":CAPE:surface:", False), ("TSOIL 0-0.1 K", r":TSOIL:0-0.1 m below", False), ("SOILW 0-0.1", r":SOILW:0-0.1 m below", False),
  ("WEASD kg/m2", r":WEASD:surface:", False), ("TMP sfc K", r":TMP:surface:", False)]
WINDS = [("WIND 10 m m/s", "10 m above ground"), ("WIND 850 m/s", "850 mb"), ("WIND 500 m/s", "500 mb"), ("WIND 250 m/s", "250 mb")]

def main():
    mine, arc = sys.argv[1], sys.argv[2]
    ds = xr.open_dataset(arc, engine="cfgrib", backend_kwargs={"filter_by_keys":{"typeOfLevel":"meanSea"},"indexpath":""})
    lat, lon = ds.latitude.values, ds.longitude.values; ny, nx = lat.shape
    rx, ry = rotated(lat, lon)
    print(f"{'field':16s} {'bias':>9s} {'RMSE':>9s} {'corr':>7s} {'arc std':>9s} {'RMSE/std':>9s}")
    def stats(name, a, m):
        ok = np.isfinite(a) & np.isfinite(m) & (np.abs(a) < 9e15) & (np.abs(m) < 9e15)
        a, m = a[ok], m[ok]; d = m-a
        c = np.corrcoef(a, m)[0,1] if a.std() > 0 and m.std() > 0 else np.nan
        print(f"{name:16s} {d.mean():9.3f} {np.sqrt((d**2).mean()):9.3f} {c:7.4f} {a.std():9.3f} {np.sqrt((d**2).mean())/max(a.std(),1e-9):9.3f}")
    for name, pat, v in FIELDS:
        n, a = read(mine, pat), read(arc, pat)
        if n is None or a is None or n.size % (NX*NY) or a.size % (nx*ny): print(f"{name:16s} missing ({None if n is None else n.size}, {None if a is None else a.size})"); continue
        stats(name, a[:nx*ny].reshape(ny, nx), to_arc(n[:NX*NY], rx, ry, v))
    for name, lev in WINDS:
        un, vn, ua, va = read(mine, f":UGRD:{lev}:"), read(mine, f":VGRD:{lev}:"), read(arc, f":UGRD:{lev}:"), read(arc, f":VGRD:{lev}:")
        if un is None or ua is None: print(name, "missing"); continue
        sn = np.hypot(un[:NX*NY], vn[:NX*NY]); sa = np.hypot(ua[:nx*ny], va[:nx*ny]).reshape(ny, nx)
        stats(name, sa, to_arc(sn, rx, ry, True))
if __name__ == "__main__":
    main()
