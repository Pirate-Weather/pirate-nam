"""Banded vs whole-domain column physics, every call of a real forecast, every field, bit for bit.

    python tools/check_phys_bands.py --rows 16 [run_forecast.py arguments: --case ... --source ... --hours 1 --out ...]

Why a per-call A/B and not only two whole runs: on configurations whose stepping has float atomics (the 12 km
parent: two runs of one tree already differ at f01, PORT/NESTS-ANYWHERE.md 20:50Z), two whole runs cannot show
that bands change nothing. Here each call of phys-turbl-driver, phys-bmj-driver and phys-fer-driver runs twice
on the same state in the same process:
  A: the whole domain (one band, exactly the code without bands),
  B: row bands of --rows rows,
and every array in ctx.st.fields after A and after B is compared bit for bit (a name B kept band-local, i.e.
in the unit's proven write-before-read scratch, is absent after B by design and is not compared). B's result is kept and the run goes
on, so the forecast is the banded forecast. Before B the state is put back to its value before A: every array is
restored from a copy, names A added are removed (B creates them itself) and table keys A added are removed.
NAMGPU_SCRATCH_EMPTY=0 is set so a newly created array is zero-filled in both arms (no uninitialised bytes).
Prints one line per unit at the end (calls, calls with any difference, the differing fields) and writes
<out>/phys-bands-ab.json. Exit 1 when any call differs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["NAMGPU_SCRATCH_EMPTY"] = "0"


def main():
    argv = sys.argv[1:]
    rows = 16
    if "--rows" in argv:
        i = argv.index("--rows")
        rows = int(argv[i + 1])
        del argv[i:i + 2]
    out = Path(argv[argv.index("--out") + 1])
    import cupy as cp

    from namgpu.physics import band

    orig = band.run
    stats = {}

    def ab(ctx, uid, body, **kw):
        st = ctx.st
        names0 = dict(st.fields)
        uniq = {}
        for n, a in names0.items():
            if isinstance(a, cp.ndarray):
                uniq.setdefault((a.data.ptr, a.shape, a.strides, str(a.dtype)), (a, a.copy()))
        tkeys0 = set(ctx.tables)
        old = os.environ.get(band.ENV_ROWS)
        os.environ[band.ENV_ROWS] = "0"
        try:
            orig(ctx, uid, body, **kw)
        finally:
            os.environ[band.ENV_ROWS] = str(rows)
        res_a = {n: a.copy() for n, a in st.fields.items() if isinstance(a, cp.ndarray)}
        # back to the state before A
        for n in list(st.fields):
            if n not in names0:
                del st.fields[n]
        for n, a in names0.items():
            st.fields[n] = a
        for a, c in uniq.values():
            a[...] = c
        for k in list(ctx.tables):
            if k not in tkeys0 and not str(k).startswith("band:"):
                del ctx.tables[k]
        try:
            r = orig(ctx, uid, body, **kw)
        finally:
            if old is None:
                os.environ.pop(band.ENV_ROWS, None)
            else:
                os.environ[band.ENV_ROWS] = old
        diff = []
        band_local = frozenset(kw.get("scratch", ())) - frozenset(kw.get("keep_whole", ()))
        for n in sorted(set(res_a) | {n for n, a in st.fields.items() if isinstance(a, cp.ndarray)}):
            a, b = res_a.get(n), st.fields.get(n)
            if a is not None and b is None and n in band_local:
                continue                    # B kept it band-local: proven write-before-read, gone after the unit
            if a is None or b is None or not isinstance(b, cp.ndarray):
                diff.append(f"{n}:missing-{'A' if a is None else 'B'}")
                continue
            if a.shape != b.shape or a.dtype != b.dtype:
                diff.append(f"{n}:shape")
                continue
            va = a.view(cp.uint8).ravel() if a.dtype.itemsize == 1 else a.view(f"u{a.dtype.itemsize}")
            vb = b.view(cp.uint8).ravel() if b.dtype.itemsize == 1 else b.view(f"u{b.dtype.itemsize}")
            nd = int(cp.count_nonzero(va != vb))
            if nd:
                diff.append(f"{n}:{nd}")
        s = stats.setdefault(uid, {"calls": 0, "calls_differ": 0, "fields": {}, "rows": rows})
        s["calls"] += 1
        if diff:
            s["calls_differ"] += 1
            for d in diff:
                f = d.split(":")[0]
                s["fields"][f] = s["fields"].get(f, 0) + 1
            if s["calls_differ"] <= 3:
                print(f"BANDS A/B DIFFER {uid} call {s['calls']} step {ctx.ntimestep}: {diff[:12]}", flush=True)
        return r

    band.run = ab
    sys.argv = [str(ROOT / "tools" / "run_forecast.py")] + argv
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_forecast", ROOT / "tools" / "run_forecast.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    rc = rf.main(argv)
    bad = 0
    for uid, s in sorted(stats.items()):
        print(f"bands A/B {uid}: {s['calls']} calls at {rows} rows, {s['calls_differ']} differ"
              + (f" ({s['fields']})" if s["fields"] else ""), flush=True)
        bad += s["calls_differ"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "phys-bands-ab.json").write_text(json.dumps({"rows": rows, "units": stats, "run_rc": rc}, indent=1))
    print("BANDS A/B: " + ("ALL CALLS BITWISE IDENTICAL" if bad == 0 and stats else "DIFFERENCES" if bad else
                           "NO CALLS SEEN"), flush=True)
    return 1 if (bad or rc or not stats) else 0


if __name__ == "__main__":
    sys.exit(main())
