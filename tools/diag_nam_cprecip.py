import sys, numpy as np
sys.path.insert(0, "/work/nam-gpu/tools")
from diag_cape import inventory, dump, lcc218
f = sys.argv[1]; boxes = {"MO-AR 33-39N 94-89W": (33, 39, -94, -89), "IA-NE-KS 38-44N 100-91W": (38, 44, -100, -91), "warm 30-40N 98-86W": (30, 40, -98, -86)}
inv = inventory(f)
sel = {}
for r in inv:
    k = (int(r["discipline"]), int(r["category"]), int(r["parameter"]))
    if k in ((0, 1, 10), (0, 1, 8), (0, 1, 9)) and int(r["level_type"]) == 1:
        sel.setdefault(k, int(r["index"]))
fl = dump(f, list(sel.values()), "namcp")
ny, nx = next(iter(fl.values())).shape
la, lo = lcc218(nx, ny); lo = np.where(lo > 180, lo - 360, lo)
for nm, b in boxes.items():
    m = (la > b[0]) & (la < b[1]) & (lo > b[2]) & (lo < b[3])
    print(nm, {str(k): (round(float(fl[i][m].max()), 2), round(float(fl[i][m].mean()), 3), round(float((fl[i][m] > 1).mean()), 3)) for k, i in sel.items()})
