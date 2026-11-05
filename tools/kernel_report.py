"""Device memory that is not CuPy pool: the CUDA context, the loaded kernel modules, and the local-memory
(stack) reservation each kernel would make. Prints a table and writes <out>/kernel-report.json.

    python tools/kernel_report.py --out scratch/kernel-report [--units phys_myj_tables,phys_rrtm_lw_rtrnmr]

Measured with nvidia-smi for this process (the card is shared, so card-wide memGetInfo can include other jobs):
  context   : process memory right after the CUDA context is created (cupy.cuda.runtime.free(0) style touch)
  modules   : process memory after every unit module (kernels/*.cu) is compiled and loaded, minus the context
  per kernel: local_size_bytes (per-thread frame), num_regs, const/shared bytes; the local-memory reservation the
              driver makes when it is the largest frame launched = frame x max resident threads (SMs x threads/SM)
Nothing is launched; nothing in a forecast changes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def proc_mib() -> int:
    me = str(os.getpid())
    for _ in range(5):
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        for line in q.stdout.splitlines():
            pid, mib = [x.strip() for x in line.split(",")[:2]]
            if pid == me:
                return int(mib)
        time.sleep(0.5)
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--units", default="", help="comma list of kernel units (default: every kernels/*.cu)")
    a = ap.parse_args()
    import cupy as cp
    from namgpu import kernels
    rt = cp.cuda.runtime
    rt.memGetInfo()                                   # creates the context
    cp.cuda.Device().synchronize()
    dev = cp.cuda.Device()
    props = rt.getDeviceProperties(dev.id)
    sms, thr = int(props["multiProcessorCount"]), int(props["maxThreadsPerMultiProcessor"])
    ctx_mib = proc_mib()
    stack0 = rt.deviceGetLimit(rt.cudaLimitStackSize)
    heap = rt.deviceGetLimit(rt.cudaLimitMallocHeapSize)
    fifo = rt.deviceGetLimit(rt.cudaLimitPrintfFifoSize)
    units = [u for u in a.units.split(",") if u] or sorted(p.stem for p in kernels.HERE.glob("*.cu"))
    rows, failed = [], {}
    for u in units:
        src = (kernels.HERE / f"{u}.cu").read_text()
        names = re.findall(r"extern\s+\"C\"\s+__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)", src)
        try:
            m = kernels.module(u)
            for n in names:
                at = m.get_function(n).attributes
                rows.append({"unit": u, "kernel": n, "local_B": int(at["local_size_bytes"]),
                             "regs": int(at["num_regs"]), "const_B": int(at["const_size_bytes"]),
                             "shared_B": int(at["shared_size_bytes"]),
                             "lmem_reserve_gib": round(int(at["local_size_bytes"]) * sms * thr / 2**30, 3)})
        except Exception as e:  # noqa: BLE001
            failed[u] = f"{type(e).__name__}: {str(e)[:200]}"
    cp.cuda.Device().synchronize()
    mods_mib = proc_mib()
    rows.sort(key=lambda r: -r["local_B"])
    rec = {"device": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
           "sms": sms, "threads_per_sm": thr, "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "(default)"),
           "context_mib": ctx_mib, "after_modules_mib": mods_mib, "modules_mib": mods_mib - ctx_mib,
           "units_loaded": len(units) - len(failed), "kernels": len(rows), "failed": failed,
           "limits": {"stack_B": stack0, "malloc_heap_B": heap, "printf_fifo_B": fifo},
           "kernels_by_frame": rows}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "kernel-report.json").write_text(json.dumps(rec, indent=1))
    print(f"{rec['device']}: {sms} SMs x {thr} threads; CUDA_MODULE_LOADING={rec['cuda_module_loading']}")
    print(f"context {ctx_mib} MiB; {rec['units_loaded']} unit modules, {len(rows)} kernels loaded: +{rec['modules_mib']} MiB")
    print(f"limits: stack {stack0} B, malloc heap {heap} B, printf fifo {fifo} B; failed: {len(failed)}")
    print("largest per-thread frames (local memory reservation if it is the largest launched):")
    for r in rows[:15]:
        print(f"  {r['local_B']:>7} B  {r['lmem_reserve_gib']:>6.2f} GiB  regs {r['regs']:>3}  {r['unit']}:{r['kernel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
