"""Build BENCHMARKS.md and its figures from the scored test case, so score numbers are read, not typed. Timings of the live run are typed from its log.
Usage: PIRATE_DATA=... PIRATE_VIEWER=... make_benchmarks.py REPO_DIR. Runs: hyb11 = Pirate NAM's live setup, hyb6 = the same with NAM's own observation files,
gfs36/gfs = plain GFS start, hyb8 and gfs36 for long leads. Other hyb runs are the one-change tests in the decisions table."""
import importlib.util, json, os, re, sys
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = sys.argv[1]; DATA = os.environ["PIRATE_DATA"]; ASSETS = f"{REPO}/pirate/assets/"
W = f"{DATA}/woof/scored/"; ARC = f"{DATA}/compare/f36/arc"; FORT = f"{DATA}/compare/var/ens_radar_rap/series.csv"
PIRATE, NAMFILES, PLAIN = "hyb11", "hyb6", "gfs36" if os.path.exists(W + "gfs36/series.csv") else "gfs"
def rmse(path, source="woof"):
    d = pd.read_csv(path); d = d[d.source == source]; return d.set_index(["fhr", "field"]).rmse
pirate, namfiles, plain, fortran, native = rmse(W + PIRATE + "/series.csv"), rmse(W + NAMFILES + "/series.csv"), rmse(W + PLAIN + "/series.csv"), rmse(FORT, "mine"), rmse(W + "native_nodfi/series.csv")
def drop_prmsl(s): return s[s.index.get_level_values(1) != "PRMSL"]
ratio = drop_prmsl(pirate / plain).dropna(); ratio_f = drop_prmsl(pirate / fortran).dropna(); ratio_n = drop_prmsl(pirate / namfiles).dropna()

PANELS = [("HGT500", "500 mb height", "m"), ("HGT250", "250 mb height", "m"), ("RH700", "700 mb humidity", "%"), ("TMP2m", "2 m temperature", "K"), ("DPT2m", "2 m dew point", "K"), ("WIND850", "850 mb wind", "m/s")]
HOURS = [0, 3, 6, 9, 12]; S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), facecolor="#fcfcfb")
for ax, (f, t, u) in zip(axs.flat, PANELS):
    for s, c, lab, mk in ((pirate, S1, "Pirate NAM", "o"), (plain, S2, "Plain WOOF-NAM", "s"), (fortran, S3, "Fortran rerun", "^")):
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
for x, f, t in ((ax[0, 0], arc(":REFC:"), "NOAA NAM"), (ax[0, 1], to(pl, "REFC"), "Plain WOOF-NAM"), (ax[0, 2], to(pn, "REFC"), "Pirate NAM")):
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
rows = ["| Field | Pirate NAM | With NAM's files | Plain GFS start | Fortran rerun | WOOF from NAM analysis |", "|---|--:|--:|--:|--:|--:|"]
for f, t, u in TABLE:
    # The native run's humidity was scored before the mixing ratio fix, so it isn't comparable.
    rows.append(f"| {t} ({u}) | **{cell(pirate, 12, f)}** | {cell(namfiles, 12, f)} | {cell(plain, 12, f)} | {cell(fortran, 12, f)} | {'n/a' if f == 'RH700' else cell(native, 12, f)} |")
long_md = ""
if os.path.exists(W + "hyb8_long/series.csv") and os.path.exists(W + "gfs36_long/series.csv"):
    c, g = rmse(W + "hyb8_long/series.csv"), rmse(W + "gfs36_long/series.csv"); r = drop_prmsl(c / g).dropna().reset_index(name="r"); by = r.groupby("fhr").r.median(); ld = [0, 6, 12, 18, 24, 30, 36]
    head = "| Field | " + " | ".join(f"{h} h" for h in ld) + " |\n|---|" + "--:|" * len(ld) + "\n| **Median ratio, all fields** | " + " | ".join(f"**{by.get(h, np.nan):.2f}**" for h in ld) + " |\n"
    body = "".join(f"| {t} ({u}) | " + " | ".join(f"{c.get((h, f), np.nan):.2f} / {g.get((h, f), np.nan):.2f}" for h in ld) + " |\n" for f, t, u in PANELS)
    long_md = f"\n## How long the observations help\n\nThis test used the NAM's own observation files and was scored to 36 hours, so hours 36 to 84 have no score against the NAM. The median ratio is the GSI cycle's error relative to a plain GFS start: below 1 means closer to the NAM. Below that, each cell is RMSE for the GSI cycle / plain start.\n\n{head}{body}"
# One-change tests, each compared with the cycle on NAM's own files, since that's what they were run against.
def versus(run, base=None, kind="woof"):
    a = rmse(run, kind) if base else rmse(W + run + "/series.csv"); b = rmse(base, kind) if base else namfiles
    r = drop_prmsl(a / b).dropna(); return f"{r.median():.3f} | {int((r < 1).sum())} of {len(r)}"
VAR = f"{DATA}/compare/var/"
tests = [("Live inputs: RAP's files, no GPS radio occultation", "hyb11", "**Used.** It's the only option live"),
         ("Land seeded from the NAM archive", "hyb9", "Not used. A small gain, and a live run has no archive to seed from"),
         ("Start from a Fortran NAM cycle's analysis", "hyb5", "Not used. Needs a Fortran model run as well"),
         ("Digital filter after each analysis", "hyb7", "Not used. No gain, and hour 0 2 m temperature got worse"),
         ("Start from RRFS instead of GFS", "hyb10", "Not used. Much further from the NAM")]
test_rows = "".join(f"| {t} | {versus(r)} | {d} |\n" for t, r, d in tests)
radiance, lightning = versus(VAR + "ens_radar_rap_rad/series.csv", VAR + "ens_radar_rap/series.csv", "mine"), versus(VAR + "all/series.csv", VAR + "ens_radar_rap_rad/series.csv", "mine")
seven, four = (drop_prmsl(rmse(W + r + "/series.csv") / plain).dropna().median() for r in ("hyb1", "hyb2"))
by_hour = ratio.groupby(level=0).median()
gc = pd.read_csv(ASSETS + "gfs_check_2026110512.csv"); GH = sorted(gc.fhr.unique()); GN = {"z500": "500 mb height (m)", "mslp": "Sea level pressure (mb)", "t850": "850 mb temperature (K)", "t2m": "2 m temperature (K)"}
gfs_rows = "".join(f"| {t} | " + " | ".join(f"{gc[(gc.field == k) & (gc.fhr == h)].rmse.iloc[0]:.2f} ({gc[(gc.field == k) & (gc.fhr == h)].bias.iloc[0]:+.2f})" for h in GH) + " |\n" for k, t in GN.items())
out = open(f"{REPO}/BENCHMARKS.md", "w")
out.write(f"""# Pirate NAM benchmarks

Detail behind the [README](README.md): how close Pirate NAM gets to the archived NAM, what each test decided, and what a live run costs.

## Closeness to the NAM

All scores are RMSE against the archived NAM forecast for the 00Z 3 November 2026 cycle, on the NAM's own output grid, to 12 hours. Lower means closer to the NAM.

- Pirate NAM's median error is **{ratio.median():.2f}×** plain WOOF-NAM from GFS, and it's closer to the NAM on **{int((ratio < 1).sum())} of {len(ratio)}** scores. That ratio runs from {by_hour.iloc[0]:.2f} at hour 0 to {by_hour.iloc[-1]:.2f} at hour 12.
- Its median error is **{ratio_f.median():.2f}×** our Fortran rerun of the NAM.
- Live inputs cost **{(ratio_n.median() - 1) * 100:.1f}%** in median error against the same cycle on the NAM's own observation files. Sea level pressure, low-level fields and humidity lose the most, while upper-level heights are slightly closer.

![RMSE against the NAM by forecast hour, for six fields](pirate/assets/scores.png)

### Scores at 12 hours

{chr(10).join(rows)}

- **Pirate NAM:** WOOF-NAM from GFS with three GSI analyses, using only inputs a live run can get.
- **With NAM's files:** the same cycle using the NAM's final observation files, which ended with the NAM.
- **Plain GFS start:** WOOF-NAM from GFS, with no analyses.
- **Fortran rerun:** the original Fortran NAM and GSI rebuilt with gfortran on the same substitute inputs, with land seeded from the NAM's archive.
- **WOOF from NAM analysis:** WOOF-NAM started from the NAM's own archived analysis. This tests the port itself and isn't something a live system could do.

### How to read this

- **This is one case.** Other weather could score differently.
- **Close to the NAM isn't the same as accurate.** These scores don't compare any of the runs with observations. The NAM's own errors count as the truth here, and the NAM also used GSI, which favours the GSI cycle on this measure.
- **The Fortran rerun has an advantage near the surface:** its land was seeded from the same archive it's scored against. A live run couldn't do that.

### Maps at 12 hours

![Maps of reflectivity, sea level pressure and 500 mb height at 12 hours, with differences from the NAM](pirate/assets/maps_f12.png)

Valid 12Z 3 November 2026. Red is higher than the NAM and blue is lower. The noisy pressure differences over the western mountains appear in both WOOF-NAM runs, and sea level pressure is an extrapolated quantity there.
{long_md}
## What each test decided

Each row changes one thing from the GSI cycle on the NAM's own files, on the same case. A ratio below 1 means closer to the NAM.

| Change | Median error ratio | Closer on | Decision |
|---|--:|--:|---|
{test_rows}
Three other choices come from separate comparisons:

- **Radiances:** {radiance.replace(" | ", " median ratio, closer on ")} scores, tested in the Fortran rerun. Left out.
- **Lightning:** {lightning.replace(" | ", " median ratio, closer on ")} scores, tested in the Fortran rerun on top of radiances. Left out.
- **Number of analyses:** seven analyses scored {seven:.3f}× a plain GFS start and four scored {four:.3f}×, so seven were no closer. Both tests started from a Fortran analysis and predate the fix that reloads lateral boundaries after each analysis, and neither was repeated afterwards.

## The first live run

The 12Z 5 November 2026 cycle, on a two-card t2-90 machine already set up. The first attempt at this cycle failed at the handoff to the forecast and was fixed; these times are from the run that completed.

| Step | Time | Notes |
|---|--:|---|
| Inputs | 27 min | RAP observations, MRMS radar, GEFS ensemble, boundaries. GFS was reused from the failed attempt, where it took another 4.5 min |
| Catch-up with 3 GSI analyses | 45 min | GSI took 661, 664 and 691 s on 28 CPU cores, with a peak of 39 GB |
| 84-hour forecast on two cards | 74 min | 53 s per forecast hour, including writing output |
| Build and upload | 17 min | |
| **Total** | **2 h 43 min** | **About CAD 6.30** at CAD 2.32 an hour, or about CAD 6.50 with the GFS download. Machine setup isn't counted |

Output was 85 GRIB2 files of about 43 MB each and a 679 MB Zarr store.

### Compared with GFS

There's no NAM left to score a live run against, so this compares it with the 12Z GFS at the same valid times. GFS isn't the truth, and the two models should drift apart, so this is a check for problems and not a skill score. Each cell is RMSE, with the mean difference in brackets.

| Field | {" | ".join(f"{h} h" for h in GH)} |
|---|{"--:|" * len(GH)}
{gfs_rows}
**Pirate NAM gets steadily colder than GFS,** by 1.5 K at 2 m and 0.8 K at 850 mb by hour 84. Scoring against the NAM stops at 36 hours, so it's not yet known whether the NAM did the same or whether this is specific to this setup.

### Estimated costs for other setups

| 84-hour forecast | Time | Cost (OVHcloud) | Status |
|---|---|---|---|
| GSI cycle, one card (t2-45) | About 3.2 h | About CAD 3.70 | Estimated from measured parts |
| WOOF-NAM from GFS, one card | About 2.2 h | About CAD 2.55 | Scaled from a measured 48-hour run |

These cover the model and GSI only, not fetching inputs, building products or uploading.

{open(f"{REPO}/pirate/docs/benchmarks_notes.md").read()}""")
out.close()
print("plain run", PLAIN, "ratio", round(ratio.median(), 3), int((ratio < 1).sum()), len(ratio), "vs fortran", round(ratio_f.median(), 3), "vs nam files", round(ratio_n.median(), 3))
