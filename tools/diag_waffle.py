"""2-dx 'waffle' census and East-box 2 m T bias vs the archived NAM (grid 218) on real history files.
python tools/diag_waffle.py --hist wrfout_... [--nam awphys24.grib2]
Waffle = RMS of the (i+j) checkerboard component and of the i-only / j-only 2-dx components in 9x9 windows.
"""
import argparse, sys
from pathlib import Path
import numpy as np, netCDF4

sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser()
ap.add_argument("--hist", nargs="+"); ap.add_argument("--nam", default="")
ap.add_argument("--box", default="30,42,-90,-75")
a = ap.parse_args()


def _box(g, n):
    k = np.zeros_like(g)
    for dj in range(n):
        for di in range(n):
            k += np.roll(np.roll(g, -dj, 0), -di, 1)
    return k / (n * n)


def comp2dx(f):
    """Local amplitude of 2-dx components after a 3x3 high-pass: checkerboard (-1)^(i+j), i-stripes, j-stripes."""
    f = f.astype(np.float64)
    r = f - np.roll(np.roll(_box(f, 3), 1, 0), 1, 1)
    jj, ii = np.indices(f.shape)
    return {name: _box(r * s, 4) for name, s in (("ij", (-1.0) ** (ii + jj)), ("i", (-1.0) ** ii), ("j", (-1.0) ** jj))}


la0, la1, lo0, lo1 = map(float, a.box.split(","))
for h in a.hist:
    d = netCDF4.Dataset(h)
    lat = np.asarray(d["XLAT"][0]); lon = np.asarray(d["XLONG"][0])
    m = (lat > la0) & (lat < la1) & (lon > lo0) & (lon < lo1)
    P = np.asarray(d["PRES"][0]); kb = 0 if P[0].mean() > P[-1].mean() else P.shape[0] - 1
    fields = {"T2": np.asarray(d["T2"][0]), "Q2": np.asarray(d["Q2"][0]) * 1e3, "PSFC": np.asarray(d["PSFC"][0]) / 100,
              "TH_k0": np.asarray(d["T"][0, kb]), "Q_k0": np.asarray(d["QVAPOR"][0, kb]) * 1e3,
              "U10": np.asarray(d["U10"][0]), "V10": np.asarray(d["V10"][0])}
    print(f"== {Path(h).name}  box {a.box}: T2 mean {fields['T2'][m].mean()-273.15:.2f} C")
    for n, f in fields.items():
        c = comp2dx(f)
        print(f"   {n:6s} 2dx-amp(box rms): ij {np.sqrt((c['ij'][m]**2).mean()):.3f} i {np.sqrt((c['i'][m]**2).mean()):.3f} "
              f"j {np.sqrt((c['j'][m]**2).mean()):.3f}   field std {f[m].std():.3f}")

if a.nam:
    from diag_cape import inventory, dump, lcc218  # noqa: E402
    inv = inventory(a.nam)
    idx = [int(r["index"]) for r in inv if (int(r["discipline"]), int(r["category"]), int(r["parameter"]), int(r["level_type"])) == (0, 0, 0, 103)
           and float(r["level_value"]) == 2][:1]
    f = dump(a.nam, idx, "namt2")[idx[0]]
    ny, nx = f.shape
    nlat, nlon = lcc218(nx, ny); nlon = np.where(nlon > 180, nlon - 360, nlon)
    nb = (nlat > la0) & (nlat < la1) & (nlon > lo0) & (nlon < lo1)
    print(f"NAM {Path(a.nam).name} T2 box mean {f[nb].mean()-273.15:.2f} C")
    from scipy.spatial import cKDTree
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    tree = cKDTree(xyz(nlat.ravel(), nlon.ravel()))
    _, ix = tree.query(xyz(lat[m], lon[m]), workers=-1)
    namv = f.ravel()[ix]
    ours = fields["T2"][m]
    diff = (ours - namv) * 1.8
    print(f"last hist minus NAM T2 (F) in box: mean {diff.mean():.1f} p10 {np.percentile(diff,10):.1f} p50 {np.percentile(diff,50):.1f} p90 {np.percentile(diff,90):.1f}; frac < -10F {np.mean(diff<-10):.2f}")
    for la_, lo_ in ((35, -85), (38, -82), (33, -84), (40, -80), (36, -79), (32, -88)):
        k = np.argmin((lat[m]-la_)**2 + (lon[m]-lo_)**2)
        print(f"   pt {la_},{lo_}: ours {(ours[k]-273.15)*1.8+32:.0f}F NAM {(namv[k]-273.15)*1.8+32:.0f}F")
