"""Run one init source's stage (and optionally its boundary build) on the configured domain; print and save
the SHA-256, finite count and range of every array it writes, the wall time and the device peak.

    NAMGPU_CONFIGFILE=parm/nam_configfile_eu3km python tools/source_stage_hash.py --case 2026-11-03_12z \
        --source gfs --bocos 6 --out runs/hash-gfs.json

Used to show that a change to the source tables leaves the GFS start and boundaries byte-identical (compare two
JSON files: python tools/source_stage_hash.py --compare a.json b.json), and that a new source is finite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE_NAMES = ("T", "Q", "CW", "U", "V", "PD", "PINT", "PSFC", "FIS", "HGT_M", "SOILHGT", "P3D_OUT",
               "PSHLTR", "TSHLTR", "QSHLTR", "U10", "V10", "TH10", "Q10")


def _describe(a):
    import cupy as cp
    import numpy as np
    h = cp.asnumpy(a)
    fin = np.isfinite(h) if h.dtype.kind == "f" else np.ones(h.shape, bool)
    return {"sha256": hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest(), "shape": list(h.shape),
            "dtype": str(h.dtype), "nonfinite": int((~fin).sum()),
            "min": float(h[fin].min()) if fin.any() else None, "max": float(h[fin].max()) if fin.any() else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--source", default="gfs")
    ap.add_argument("--bocos", type=int, default=0, help="also build the boundary files for this many hours")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--save", type=Path, help="also write the state arrays to this .npz (host copies)")
    a = ap.parse_args(argv)
    if a.compare:
        x, y = (json.loads(p.read_text()) for p in a.compare)
        keys = sorted(set(x["arrays"]) | set(y["arrays"]))
        diff = [k for k in keys if x["arrays"].get(k, {}).get("sha256") != y["arrays"].get(k, {}).get("sha256")]
        print(f"{len(keys)} arrays/files compared, {len(diff)} differ" + (f": {diff}" if diff else ""))
        return 1 if diff else 0

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.init import init_grib2_decode

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.first_step = True
    pool = cp.get_default_memory_pool()
    t0 = time.perf_counter()
    with ctx.stream:
        init_grib2_decode.run(ctx)
    ctx.stream.synchronize()
    t_stage = time.perf_counter() - t0
    arrays = {f"st.{n}": _describe(st[n]) for n in STATE_NAMES if n in st.fields}
    for k, v in ctx.tables.get("raw_sfc", {}).items():
        arrays[f"raw_sfc.{k}"] = _describe(v)
    if a.save:
        import numpy as np
        a.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, **{n: cp.asnumpy(st[n]) for n in STATE_NAMES if n in st.fields},
                 **{f"raw_{k}": cp.asnumpy(v) for k, v in ctx.tables.get("raw_sfc", {}).items()},
                 GLAT=cp.asnumpy(st["GLAT"]), GLON=cp.asnumpy(st["GLON"]))
    rec = {"case": a.case, "source": a.source, "grid": [g.im, g.jm, g.lm], "stage_s": round(t_stage, 2),
           "grid_source": ctx.tables.get("grid_source"), "pool_peak_bytes_after_stage": int(pool.total_bytes())}
    if a.bocos:
        from namgpu.init import gfs_bocos
        t1 = time.perf_counter()
        with ctx.stream:
            out = gfs_bocos.run(ctx, a.bocos)
        ctx.stream.synchronize()
        rec["bocos_s"] = round(time.perf_counter() - t1, 2)
        rec["boco_dir"] = str(out)
        for p in sorted(Path(out).glob("boco.*")):
            if int(p.name.split(".")[1]) < a.bocos:
                arrays[f"boco.{p.name}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                            "bytes": p.stat().st_size}
    rec["arrays"] = arrays
    bad = {k: v["nonfinite"] for k, v in arrays.items() if v.get("nonfinite")}
    rec["nonfinite_arrays"] = bad
    print(json.dumps({k: v for k, v in rec.items() if k != "arrays"}, default=str, indent=1))
    for k, v in arrays.items():
        print(f"  {k:18s} {v['sha256'][:16]} " + (f"[{v['min']:.4g}, {v['max']:.4g}] nonfinite {v['nonfinite']}"
                                                  if "min" in v and v["min"] is not None else f"{v.get('bytes', '')}"))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, default=str, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
