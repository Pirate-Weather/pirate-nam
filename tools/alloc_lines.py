"""Which code lines allocate device memory in a real forecast, with the dtype of what they made.

    python tools/alloc_lines.py --case 2025-03-14_12z --source gfs --hours 1 --out <dir> [--init-from <saved init>]

Runs tools/run_forecast.py in this process with a CuPy memory hook. Every allocation of at least --min-mib is
charged to the innermost namgpu source line on the stack (file:line), with count, total and largest bytes.
After the run the lines are listed largest-first. A float64 temporary shows as a size that is 2x the float32
field it was made from; the dtype column is filled by re-reading the line for 'float64'/'np.float64'.
Diagnostics only: the run itself is unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--min-mib", type=float, default=1.0)
    a = ap.parse_args()
    import cupy as cp
    from cupy.cuda import memory_hook

    floor = int(a.min_mib * 2**20)
    stats = collections.defaultdict(lambda: [0, 0, 0])   # line -> [count, total, max]

    class Hook(memory_hook.MemoryHook):
        name = "alloc_lines"

        def malloc_preprocess(self, device_id, size, mem_size):
            if mem_size < floor:
                return
            f = sys._getframe(1)
            where = "?"
            while f is not None:
                fn = f.f_code.co_filename
                if "/namgpu/" in fn:
                    where = f"{fn.split('/namgpu/', 1)[1]}:{f.f_lineno}"
                    break
                f = f.f_back
            s = stats[where]
            s[0] += 1
            s[1] += mem_size
            s[2] = max(s[2], mem_size)

    import importlib.util
    spec = importlib.util.spec_from_file_location("run_forecast", ROOT / "tools" / "run_forecast.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    argv = ["--case", a.case, "--source", a.source, "--hours", str(a.hours), "--out", str(a.out / "run"), "--no-maps"]
    if a.init_from:
        argv += ["--init-from", a.init_from]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with Hook():
        rc = rf.main(argv)
    rows = []
    for where, (n, tot, mx) in stats.items():
        src = ""
        if ":" in where and where != "?":
            fn, ln = where.rsplit(":", 1)
            try:
                src = (ROOT / "namgpu" / fn).read_text().splitlines()[int(ln) - 1].strip()
            except Exception:  # noqa: BLE001
                pass
        rows.append({"line": where, "count": n, "total_mib": round(tot / 2**20, 1), "max_mib": round(mx / 2**20, 2),
                     "f64": ("float64" in src or "np.double" in src), "src": src[:160]})
    rows.sort(key=lambda r: -r["max_mib"])
    (a.out / "alloc-lines.json").write_text(json.dumps({"rc": rc, "wall_s": round(time.perf_counter() - t0, 1),
                                                        "lines": rows}, indent=1))
    for r in rows[:80]:
        print(f"{r['max_mib']:9.2f} MiB max {r['count']:7d}x {r['total_mib']:10.1f} MiB  {'F64 ' if r['f64'] else '    '}"
              f"{r['line']:55s} {r['src'][:90]}")
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(int(rc or 0))
