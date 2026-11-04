"""One sheet of a built static file (analysis chart of static fields, matplotlib; no weather fields).

    python tools/plot_static.py static/nam4km_static.npz scratch/e3/static_4km.png "NAM 4 km"
"""
import json
import sys

import numpy as np


def main(path, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(path)
    meta = json.loads(str(z["META"])) if "META" in z.files else {}
    panels = (("HGT_SFC", "terrain", 0, 3500, "m"), ("LAND", "gray", 0, 1, ""), ("VGTYP", "tab20", 0, 20, "IGBP class"),
              ("SOTYP", "tab20", 0, 16, "STATSGO class"), ("VEG", "YlGn", 0, 100, "%"), ("ALBDO", "magma", 0, 40, "%"),
              ("SFCR", "viridis", 0, 3, "m"), ("TG", "coolwarm", 260, 305, "K"), ("MXSNAL", "Blues", 0, 0.9, ""))
    fig, ax = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    for a, (k, cm, lo, hi, u) in zip(ax.ravel(), panels):
        im = a.imshow(z[k], origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        a.contour(z["GLAT"], levels=range(10, 70, 10), colors="k", linewidths=0.4)
        a.contour(z["GLON"], levels=range(-160, -40, 10), colors="k", linewidths=0.4)
        a.set_title(f"{k} {u}")
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f"{title}: {z['HGT_SFC'].shape[1]} x {z['HGT_SFC'].shape[0]} static from WPS_GEOG (init-static-build), "
                 f"valid {meta.get('valid', '?')[:10]}, land {meta.get('land_frac', '?')}; 10-degree lat/lon lines")
    fig.savefig(out, dpi=60)


if __name__ == "__main__":
    main(*sys.argv[1:4])
