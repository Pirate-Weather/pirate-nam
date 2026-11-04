"""Is horizontal advection acting on Q and theta? Regress the 1 h change at one level against -V.grad(X)*dt
from two consecutive history files (real run output, whole interior domain). Ratio ~1 = advected.
python tools/diag_adv.py --a wrfout_..._12 --b wrfout_..._13 --k 5
"""
import argparse
import numpy as np, netCDF4

ap = argparse.ArgumentParser()
ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--k", default="3,8,14,22")
ap.add_argument("--box", default="25,45,-105,-80")
a = ap.parse_args()
da = netCDF4.Dataset(a.a); db = netCDF4.Dataset(a.b)
lat = np.asarray(da["XLAT"][0], np.float64); lon = np.asarray(da["XLONG"][0], np.float64)
R = 6371e3
la = np.radians(lat); lo = np.radians(lon)
# metric terms by centred differences
def d_i(f): g = np.full_like(f, np.nan); g[:, 1:-1] = 0.5 * (f[:, 2:] - f[:, :-2]); return g
def d_j(f): g = np.full_like(f, np.nan); g[1:-1, :] = 0.5 * (f[2:, :] - f[:-2, :]); return g
xi = R * np.cos(la) * d_i(lo); yi = R * d_i(la)
xj = R * np.cos(la) * d_j(lo); yj = R * d_j(la)
det = xi * yj - xj * yi
la0, la1, lo0, lo1 = map(float, a.box.split(","))
m = (lat > la0) & (lat < la1) & (lon > lo0) & (lon < lo1)
P = np.asarray(da["PRES"][0])
bot_first = P[0].mean() > P[-1].mean()
for k in [int(x) for x in a.k.split(",")]:
    kk = k if bot_first else P.shape[0] - 1 - k
    u = 0.5 * (np.asarray(da["U"][0, kk], np.float64) + np.asarray(db["U"][0, kk], np.float64))
    v = 0.5 * (np.asarray(da["V"][0, kk], np.float64) + np.asarray(db["V"][0, kk], np.float64))
    for name in ("QVAPOR", "T"):
        fa = np.asarray(da[name][0, kk], np.float64); fb = np.asarray(db[name][0, kk], np.float64)
        fi = d_i(fa); fj = d_j(fa)
        # grad in east/north: [fx, fy] solves [xi yi; xj yj][fx fy]^T = [fi fj]
        fx = (fi * yj - fj * yi) / det
        fy = (xi * fj - xj * fi) / det
        adv = -(u * fx + v * fy) * 3600.0
        dch = fb - fa
        ok = m & np.isfinite(adv) & np.isfinite(dch)
        x = adv[ok]; y = dch[ok]
        slope = float((x * y).sum() / (x * x).sum())
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"k{k:2d} p~{P[kk][m].mean()/100:6.1f} {name:7s} slope(dX vs adv) {slope:6.3f} corr {r:5.2f} "
              f"|adv| p90 {np.percentile(np.abs(x), 90):.3g} |dX| p90 {np.percentile(np.abs(y), 90):.3g}")
