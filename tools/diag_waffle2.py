import os, sys, numpy as np, netCDF4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
h = sys.argv[1]; la0, la1, lo0, lo1 = map(float, sys.argv[2].split(","))
d = netCDF4.Dataset(h)
lat = np.asarray(d["XLAT"][0]); lon = np.asarray(d["XLONG"][0])
m = (lat > la0) & (lat < la1) & (lon > lo0) & (lon < lo1)
P = np.asarray(d["PRES"][0]); bf = P[0].mean() > P[-1].mean()
def lev(name, k):
    a = d[name][0]; return np.asarray(a[k] if bf else a[a.shape[0] - 1 - k])
def amp(f):
    f = f.astype(np.float64)
    # pure 2-dx in i: half the |second difference| / 2 ; same in j
    di = np.abs(f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / 4.0
    dj = np.abs(f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / 4.0
    mi = m[:, 1:-1]; mj = m[1:-1, :]
    return np.percentile(di[mi], 50), np.percentile(di[mi], 95), np.percentile(dj[mj], 50), np.percentile(dj[mj], 95)
rows = [("T2", np.asarray(d["T2"][0])), ("Q2g", np.asarray(d["Q2"][0]) * 1e3), ("U10", np.asarray(d["U10"][0])), ("V10", np.asarray(d["V10"][0])), ("PSFC", np.asarray(d["PSFC"][0]) / 100)]
for k in (0, 3, 8, 15, 25):
    rows += [(f"TH_k{k}", lev("T", k)), (f"Q_k{k}", lev("QVAPOR", k) * 1e3), (f"U_k{k}", lev("U", k)), (f"V_k{k}", lev("V", k)), (f"W_k{k}", lev("W", k))]
print(f"{h[-19:]} box {sys.argv[2]}: half 2nd-difference (2-dx amplitude) p50/p95 along i | along j")
for n, f in rows:
    a = amp(f); print(f"  {n:7s} i {a[0]:.3f}/{a[1]:.3f}   j {a[2]:.3f}/{a[3]:.3f}")
