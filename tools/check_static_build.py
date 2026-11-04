"""Proof of init-static-build on the NAM 12 km grid: the built static vs NAM's own (bgrd3d00 -> nam12_static.npz).

    python tools/check_static_build.py --built scratch/e3/nam12_built.npz --ref static/nam12_static.npz \
        --out scratch/e3/check12 [--maps]

Per field: continuous -> mean diff, mean |diff|, RMS, p99 |diff| over the cells where both are defined
(land only for the land-surface fields); categorical -> agreement fraction on cells land in both. LAND ->
agreement over all cells. Writes check.json and, with --maps, one PNG per field (NAM | built | built - NAM,
analysis charts of static fields drawn with matplotlib; no weather fields).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONT = ("HGT_SFC", "SFCR", "VEG", "ALBDO")
CAT = ("VGTYP", "SOTYP")


def compare(b, r):
    out = {}
    lb, lr = b["LAND"] > 0.5, r["LAND"] > 0.5
    both = lb & lr
    out["LAND"] = {"agree": round(float((lb == lr).mean()), 5), "land_built": int(lb.sum()), "land_nam": int(lr.sum()),
                   "built_land_nam_water": int((lb & ~lr).sum()), "built_water_nam_land": int((~lb & lr).sum())}
    for k in CONT:
        m = both if k != "HGT_SFC" else np.ones_like(both)
        d = (b[k] - r[k])[m].astype(np.float64)
        out[k] = {"cells": int(m.sum()), "mean_nam": round(float(r[k][m].mean()), 4),
                  "mean_built": round(float(b[k][m].mean()), 4), "mean_diff": round(float(d.mean()), 4),
                  "mean_abs": round(float(np.abs(d).mean()), 4), "rms": round(float(np.sqrt((d * d).mean())), 4),
                  "p99_abs": round(float(np.percentile(np.abs(d), 99)), 4),
                  "corr": round(float(np.corrcoef(b[k][m].ravel(), r[k][m].ravel())[0, 1]), 4)}
        if k == "HGT_SFC":
            dl = (b[k] - r[k])[both].astype(np.float64)
            out[k]["land_mean_abs"] = round(float(np.abs(dl).mean()), 3)
            out[k]["land_rms"] = round(float(np.sqrt((dl * dl).mean())), 3)
    for k in CAT:
        vb, vr = np.rint(b[k]).astype(int), np.rint(r[k]).astype(int)
        out[k] = {"cells": int(both.sum()), "agree": round(float((vb == vr)[both].mean()), 4)}
    return out


def maps(b, r, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = {"HGT_SFC": ("terrain", 0, 3500, 300), "LAND": ("gray", 0, 1, 1), "VGTYP": ("tab20", 0, 20, 5),
            "SOTYP": ("tab20", 0, 16, 5), "SFCR": ("viridis", 0, 3, 0.5), "VEG": ("YlGn", 0, 100, 20),
            "ALBDO": ("magma", 0, 80, 20)}
    for k, (cm, lo, hi, dd) in spec.items():
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
        for a, arr, t in ((ax[0], r[k], f"NAM 12 km static (bgrd3d00) {k}"), (ax[1], b[k], f"built (init-static-build) {k}")):
            im = a.imshow(arr, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            a.set_title(t)
            fig.colorbar(im, ax=a, shrink=0.8)
        d = b[k] - r[k]
        if k in CAT:
            d = (np.rint(b[k]) != np.rint(r[k])).astype(float)
            im = ax[2].imshow(d, origin="lower", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
            ax[2].set_title(f"{k} differs (1 = different class)")
        else:
            im = ax[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-dd, vmax=dd, interpolation="nearest")
            ax[2].set_title(f"built - NAM {k} (scale +-{dd})")
        fig.colorbar(im, ax=ax[2], shrink=0.8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.savefig(outdir / f"static12_{k}.png", dpi=70)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--maps", action="store_true")
    a = ap.parse_args(argv)
    b, r = dict(np.load(a.built)), dict(np.load(a.ref))
    a.out.mkdir(parents=True, exist_ok=True)
    res = compare(b, r)
    (a.out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if a.maps:
        maps(b, r, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
