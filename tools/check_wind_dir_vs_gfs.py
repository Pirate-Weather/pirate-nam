"""10 m wind direction of a nest f00 history against the source GFS f000 10 m wind at the nearest GFS point
(lane init-gfs-anywhere): mean U/V, direction difference, correlation.
    python tools/check_wind_dir_vs_gfs.py <wrfout f00> <gfs f000 subset> <tmpdir>
"""
import sys, numpy as np, cupy as cp, netCDF4
from types import SimpleNamespace
from namgpu.init import init_grib2_decode as g2
hist, gfs, tmp = sys.argv[1:4]
with netCDF4.Dataset(hist) as nc:
    u, v = np.asarray(nc["U10"][0], float), np.asarray(nc["V10"][0], float)
    la, lo = np.asarray(nc["XLAT"][0], float), np.asarray(nc["XLONG"][0], float)
inv = g2.inventory(gfs); nx, ny = int(inv[0]["nx"]), int(inv[0]["ny"])
s = lambda dcp: (lambda r: r["dcp"] == dcp and str(r["level_type"]) == "103" and abs(float(r["level_value"]) - 10) < .01)
got = g2.run(SimpleNamespace(stream=None, tables={}), gfs, {"U": s((0, 2, 2)), "V": s((0, 2, 3))}, missing_value=-9999.0, tmpdir=tmp, expect_shape=(nx, ny))
GU, GV = cp.asnumpy(got["U"]), cp.asnumpy(got["V"])
j = np.clip(np.rint((la + 90) / 0.25).astype(int), 0, ny - 1); i = np.rint((lo % 360) / 0.25).astype(int) % nx
gu, gv = GU[j, i], GV[j, i]
m = np.hypot(gu, gv) > 3
ang = np.degrees(np.angle((u + 1j * v)[m] * np.conj((gu + 1j * gv)[m])))
print(hist.split("/")[-2], "points", int(m.sum()), "mean U nest/gfs %.2f/%.2f V %.2f/%.2f" % (u[m].mean(), gu[m].mean(), v[m].mean(), gv[m].mean()),
      "direction diff deg median %.1f p90|.| %.1f" % (np.median(ang), np.percentile(abs(ang), 90)), "corr u %.3f v %.3f" % (np.corrcoef(u[m], gu[m])[0, 1], np.corrcoef(v[m], gv[m])[0, 1]))
