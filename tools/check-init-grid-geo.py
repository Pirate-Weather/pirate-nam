"""check-init-grid-geo: init-grid-geo at full size (954 x 835 x 60) against the 2025-03-14 12Z analysis.

Runs the unit's run() on a full-size context, prints the measured analysis errors (H and V), the
NaN/Inf count of every output, and the decision. Prints the report even when the unit refuses.
Run on a card lease: GPU_MUTEX_POOL=0 pool-v3/run.sh --cards 1 --vram-mib 4000 geo -- python this.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu import config
from namgpu.init import init_grid_geo as unit

CASE = Path(_ROOT + "/cases/2025-03-14_12z")
cfg = config.load(Path(_ROOT + "/parm/nam_configfile"), None)
im, jm = int(cfg.im), int(cfg.jm)
st = {n: cp.zeros((jm, im), dtype=cp.float32) for n in ("GLAT", "GLON", "VLAT", "VLON")}
ctx = SimpleNamespace(cfg=cfg, grid=SimpleNamespace(im=im, jm=jm), st=st, tables={},
                      stream=None, case_dir=CASE)
try:
    unit.run(ctx)
    decision = "PASS"
except RuntimeError as e:
    decision = f"REFUSED: {e}"
rep = ctx.tables.get("geo_check", {})
print("report", rep)
nonfinite = {n: int((~cp.isfinite(a)).sum()) for n, a in st.items()}
nonfinite["grid_sw"] = int((~cp.isfinite(ctx.tables["grid_sw"])).sum()) if "grid_sw" in ctx.tables else -1
print("nonfinite", nonfinite)
print("GLAT deg range", float(cp.degrees(st["GLAT"]).min()), float(cp.degrees(st["GLAT"]).max()))
print("grid_sw deg", [float(np.degrees(x)) for x in cp.asnumpy(ctx.tables.get("grid_sw", cp.zeros(2)))])
print("decision", decision)
