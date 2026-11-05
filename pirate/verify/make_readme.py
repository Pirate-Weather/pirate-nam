"""Build the score figures and the results section of the repo README from the scored test case, so numbers are read, not typed.
Usage: PIRATE_DATA=... PIRATE_VIEWER=... make_readme.py REPO_DIR. Runs: hyb6 = GSI cycle, gfs36/gfs = plain GFS start, hyb8 and gfs36 for long leads."""
import importlib.util, json, os, re, sys
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = sys.argv[1]; DATA = os.environ["PIRATE_DATA"]; ASSETS = f"{REPO}/pirate/assets/"
W = f"{DATA}/woof/scored/"; ARC = f"{DATA}/compare/f36/arc"; FORT = f"{DATA}/compare/var/ens_radar_rap/series.csv"
PIRATE, PLAIN = "hyb6", "gfs36" if os.path.exists(W + "gfs36/series.csv") else "gfs"
def rmse(path, source="woof"):
    d = pd.read_csv(path); d = d[d.source == source]; return d.set_index(["fhr", "field"]).rmse
pirate, plain, fortran, native = rmse(W + PIRATE + "/series.csv"), rmse(W + PLAIN + "/series.csv"), rmse(FORT, "mine"), rmse(W + "native_nodfi/series.csv")
def drop_prmsl(s): return s[s.index.get_level_values(1) != "PRMSL"]
ratio = drop_prmsl(pirate / plain).dropna(); ratio_f = drop_prmsl(pirate / fortran).dropna()

PANELS = [("HGT500", "500 mb height", "m"), ("HGT250", "250 mb height", "m"), ("RH700", "700 mb humidity", "%"), ("TMP2m", "2 m temperature", "K"), ("DPT2m", "2 m dew point", "K"), ("WIND850", "850 mb wind", "m/s")]
HOURS = [0, 3, 6, 9, 12]; S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), facecolor="#fcfcfb")
for ax, (f, t, u) in zip(axs.flat, PANELS):
    for s, c, lab, mk in ((pirate, S1, "WOOF-NAM + GSI", "o"), (plain, S2, "Plain WOOF-NAM", "s"), (fortran, S3, "Fortran rerun", "^")):
        v = [s.get((h, f), np.nan) for h in HOURS]; ax.plot(HOURS, v, color=c, lw=2, marker=mk, ms=6, mec="#fcfcfb", label=lab)
    ax.set_title(f"{t} ({u})", loc="left", fontsize=11); ax.set_xticks(HOURS); ax.set_xticklabels([f"{h} h" for h in HOURS], fontsize=9); ax.tick_params(axis="y", labelsize=9, length=0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e4e3dd", lw=0.8); ax.set_facecolor("#fcfcfb"); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
h, l = axs.flat[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(ASSETS + "scores.png", dpi=110); plt.close(fig)

spec = importlib.util.spec_from_file_location("ca", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_awphys_lib.py")); ca = importlib.util.module_from_spec(spec); spec.loader.exec_module(ca)
ds = xr.open_dataset(f"{ARC}/awphys00.grib2", engine="cfgrib", backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}, "indexpath": ""})
lat, lon = ds.latitude.values, ds.longitude.values; ny, nx = lat.shape; rx, ry = ca.rotated(lat, lon); lon = (lon + 180) % 360 - 180
a = f"{ARC}/awphys12.grib2"; arc = lambda m: ca.read(a, m)[:nx * ny].reshape(ny, nx)
pn, pl = np.load(W + PIRATE + "/f12.npz"), np.load(W + PLAIN + "/f12.npz"); to = lambda z, k: ca.to_arc(z[k].ravel(), rx, ry)
geo = json.load(open(os.path.join(os.environ["PIRATE_VIEWER"], "geo/lines.json")))
def coast(ax):
    for key in ("coast", "states", "borders"):
        for ln in geo.get(key, []): ax.plot(ln[0::2], ln[1::2], color="#52514e", lw=0.4)
    ax.set_xlim(-127, -66); ax.set_ylim(23, 51); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(1.3)
    for s in ax.spines.values(): s.set_visible(False)
seq = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#bcd6f4", "#2a78d6", "#0d366b"]); div = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f0efec", "#e34948"])
fig, ax = plt.subplots(3, 3, figsize=(14, 8.2), facecolor="#fcfcfb")
def panel(x, f, title, **k):
    m = x.pcolormesh(lon, lat, f, shading="auto", rasterized=True, **k); coast(x); x.set_title(title, fontsize=11, loc="left"); return m
for x, f, t in ((ax[0, 0], arc(":REFC:"), "NOAA NAM"), (ax[0, 1], to(pl, "REFC"), "Plain WOOF-NAM"), (ax[0, 2], to(pn, "REFC"), "WOOF-NAM + GSI")):
    m = panel(x, np.ma.masked_less(f, 5), "Reflectivity, " + t, cmap=seq, vmin=5, vmax=55)
fig.colorbar(m, ax=ax[0], fraction=0.015, pad=0.01, label="dBZ")
P = arc(":PRMSL:mean sea level:") / 100; H = arc(":HGT:500 mb:")
fig.colorbar(panel(ax[1, 0], P, "Sea level pressure, NOAA NAM", cmap=seq, vmin=990, vmax=1035), ax=ax[1, 0], fraction=0.03, pad=0.01, label="mb")
n = TwoSlopeNorm(0, -3, 3); m = panel(ax[1, 1], to(pl, "PRMSL") / 100 - P, "Plain WOOF-NAM minus NAM", cmap=div, norm=n); panel(ax[1, 2], to(pn, "PRMSL") / 100 - P, "WOOF-NAM + GSI minus NAM", cmap=div, norm=n); fig.colorbar(m, ax=ax[1, 1:], fraction=0.015, pad=0.01, label="mb")
fig.colorbar(panel(ax[2, 0], H / 10, "500 mb height, NOAA NAM", cmap=seq, vmin=520, vmax=590), ax=ax[2, 0], fraction=0.03, pad=0.01, label="dam")
n = TwoSlopeNorm(0, -25, 25); m = panel(ax[2, 1], to(pl, "HGT500") - H, "Plain WOOF-NAM minus NAM", cmap=div, norm=n); panel(ax[2, 2], to(pn, "HGT500") - H, "WOOF-NAM + GSI minus NAM", cmap=div, norm=n); fig.colorbar(m, ax=ax[2, 1:], fraction=0.015, pad=0.01, label="m")
fig.savefig(ASSETS + "maps_f12.png", dpi=95, bbox_inches="tight"); plt.close(fig)

TABLE = PANELS + [("WIND10m", "10 m wind", "m/s"), ("WIND250", "250 mb wind", "m/s"), ("REFC", "Composite reflectivity", "dBZ"), ("APCP", "Precipitation", "mm"), ("MSLET", "Sea level pressure", "Pa")]
def cell(s, h, f):
    v = s.get((h, f), np.nan); return "n/a" if not np.isfinite(v) else (f"{v:.1f}" if abs(v) >= 20 else f"{v:.2f}")
rows = ["| Field | GSI cycle | Plain WOOF | Fortran rerun | WOOF from NAM analysis |", "|---|--:|--:|--:|--:|"]
for f, t, u in TABLE:
    # The native run's humidity was scored before the mixing ratio fix, so it isn't comparable.
    rows.append(f"| {t} ({u}) | **{cell(pirate, 12, f)}** | {cell(plain, 12, f)} | {cell(fortran, 12, f)} | {'n/a' if f == 'RH700' else cell(native, 12, f)} |")
long_md = ""
if os.path.exists(W + "hyb8_long/series.csv") and os.path.exists(W + "gfs36_long/series.csv"):
    c, g = rmse(W + "hyb8_long/series.csv"), rmse(W + "gfs36_long/series.csv"); r = drop_prmsl(c / g).dropna().reset_index(name="r"); by = r.groupby("fhr").r.median(); ld = [0, 6, 12, 18, 24, 30, 36]
    head = "| Field | " + " | ".join(f"{h} h" for h in ld) + " |\n|---|" + "--:|" * len(ld) + "\n| **Median ratio, all fields** | " + " | ".join(f"**{by.get(h, np.nan):.2f}**" for h in ld) + " |\n"
    body = "".join(f"| {t} ({u}) | " + " | ".join(f"{c.get((h, f), np.nan):.2f} / {g.get((h, f), np.nan):.2f}" for h in ld) + " |\n" for f, t, u in PANELS)
    long_md = f"\n### How long the observations help\n\nThe median ratio is the GSI cycle's error relative to a plain GFS start: below 1 means closer to the NAM. Below that, each cell is RMSE for the GSI cycle / plain start.\n\n{head}{body}"
out = open(ASSETS + "results.md", "w")
out.write(f"""- The GSI cycle's median error is **{ratio.median():.2f}×** plain WOOF-NAM from GFS, and it's closer to the NAM on **{int((ratio < 1).sum())} of {len(ratio)}** scores.
- Its median error is **{ratio_f.median():.2f}×** our Fortran rerun of the NAM.

![RMSE against the NAM by forecast hour, for six fields](pirate/assets/scores.png)

### Scores at 12 hours

{chr(10).join(rows)}

GSI cycle: WOOF-NAM from GFS with three GSI analyses. Plain WOOF: WOOF-NAM from GFS. Fortran rerun: the original Fortran NAM and GSI rebuilt with gfortran on the same substitute inputs, with land seeded from the NAM's archive. WOOF from NAM analysis: WOOF-NAM started from the NAM's own archived analysis, which tests the port itself and isn't something a live system could do.

### Maps at 12 hours

![Maps of reflectivity, sea level pressure and 500 mb height at 12 hours, with differences from the NAM](pirate/assets/maps_f12.png)

Valid 12Z 3 November 2026. Red is higher than the NAM and blue is lower. The noisy pressure differences over the western mountains appear in both WOOF-NAM runs, and sea level pressure is an extrapolated quantity there.
{long_md}""")
print("plain run", PLAIN, "ratio", round(ratio.median(), 3), int((ratio < 1).sum()), len(ratio), "vs fortran", round(ratio_f.median(), 3))
