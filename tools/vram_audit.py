"""VRAM audit of one real WOOF-NAM forecast: who holds the device memory at the high-water mark.

    source env.sh
    python tools/vram_audit.py --case 2025-03-14_12z --hours 1 --out scratch/vram-audit/before

Runs the normal forecast entry (namgpu.run_forecast: init, TDFI, forecast, hourly history; no maps) with
hooks on the CuPy allocator, State.fields, ctx.tables and the timing stamps:
  - every array put in st.fields / ctx.tables is tagged with the code that created it (module file);
  - at each new pool high-water mark (32 MiB steps) the live named arrays are listed (holder, name, bytes,
    dtype, shape, owner group); pool bytes not reachable from a name are the running unit's temporaries;
  - per plan unit: live bytes at its start, its peak inside (transient working set);
  - card used (memGetInfo) at every stamp: pool reserved + everything outside the pool (CUDA context,
    kernel modules, the device local-memory reservation).
Writes <out>/vram-audit.json and prints the per-group table (GiB and bytes per grid column). Audit only:
no change to the run (allocations go through the same pool).
"""
from __future__ import annotations

import argparse
import array
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = (  # (path fragment of the creating module, group); first match wins
    ("dynamics/dyn_adv", "advection"), ("dynamics/dyn_mono", "advection"), ("dynamics/dyn_updates", "advection"),
    ("dynamics/", "dynamics"), ("boundary/", "halo/boundary"), ("steps/drv_halo", "halo/boundary"),
    ("physics/bmj", "BMJ"), ("phys_bmj", "BMJ"), ("physics/fer", "Ferrier"), ("phys_fer", "Ferrier"),
    ("physics/myj", "MYJ PBL"), ("physics/jsfc", "MYJ PBL"), ("phys_turbl", "MYJ PBL"), ("arwen/core/myj", "MYJ PBL"),
    ("physics/noah", "Noah"), ("physics/gwd", "GWD"), ("init/init_gwd", "GWD"),
    ("physics/rrtm", "RRTM LW/SW"), ("physics/rad", "RRTM LW/SW"), ("phys_rad", "RRTM LW/SW"),
    ("steps/drv_dfi", "TDFI"), ("io/", "history writer"), ("init/", "init leftovers"),
    ("steps/", "driver steps"), ("physics/", "physics other"),
)


def group_of(path: str) -> str:
    for frag, g in GROUPS:
        if frag in path:
            return g
    return "other"


def _creator() -> str:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/namgpu/" in fn and not fn.endswith(("/state.py", "/driver.py")):
            return fn.split("/namgpu/", 1)[1]
        f = f.f_back
    return "?"


class TrackDict(dict):
    """dict that remembers which module put each key in (first insert)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.origin = {key: "pre" for key in self}

    def __setitem__(self, k, v):
        if k not in self.origin:
            self.origin[k] = _creator()
        super().__setitem__(k, v)

    def setdefault(self, k, v=None):
        if k not in self:
            self.__setitem__(k, v)
        return self[k]

    def update(self, *a, **k):
        for kk, vv in dict(*a, **k).items():
            self.__setitem__(kk, vv)


def _arrays(v, seen, path, out, depth=0):
    """(path, ptr, bytes, dtype, shape) of every device allocation reachable from v, each allocation once."""
    if depth > 6:
        return
    if hasattr(v, "data") and hasattr(v, "dtype") and hasattr(getattr(v, "data", None), "mem"):
        mem = v.data.mem
        if mem.ptr in seen or not mem.size:
            return
        seen.add(mem.ptr)
        out.append((path, mem.ptr, int(mem.size), str(v.dtype), list(v.shape)))
        return
    if isinstance(v, (str, bytes, int, float, type(None))) or hasattr(v, "__array_interface__"):
        return
    if id(v) in seen:
        return
    seen.add(id(v))
    if isinstance(v, dict):
        for k, x in list(dict.items(v)):
            _arrays(x, seen, f"{path}[{k}]", out, depth + 1)
    elif isinstance(v, (list, tuple, set)):
        for i, x in enumerate(list(v)):
            _arrays(x, seen, f"{path}[{i}]", out, depth + 1)
    else:
        d = getattr(v, "__dict__", None)
        if isinstance(d, dict) and not isinstance(v, type) and (
                type(v).__module__.startswith("namgpu") or type(v).__name__ in ("SimpleNamespace",)):
            for k, x in list(d.items()):
                _arrays(x, seen, f"{path}.{k}", out, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step-mib", type=int, default=32, help="snapshot every N MiB of new high-water")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a saved init (python -m namgpu.init.slab_init save): both arms of a bitwise check start from "
                         "the same bits (the init chain sums with float atomics)")
    a = ap.parse_args()
    import cupy as cp
    from namgpu import state as statemod
    from namgpu import driver as drvmod
    from namgpu.steps import drv_timing_receipt as timing

    try:
        from namgpu import devmem      # the run's allocator (NAMGPU_POOL / NAMGPU_POOL_CAP)
        devmem.setup()
        pool = devmem.pool()
    except ImportError:
        pool = cp.get_default_memory_pool()
    spec_names = set(statemod.all_specs())     # the NAM tables: every field of these is "NAM state"
    cur = {"label": "setup", "phase": "setup"}
    holder = {"ctx": None, "st": None}
    unit_start = {}
    unit_peak_tr = collections.defaultdict(int)        # label -> max(transient) bytes
    unit_start_live = collections.defaultdict(int)     # label -> max live at start
    card = {"peak": 0, "label": None, "pool_total": 0, "pool_used": 0, "phase": None}
    hw = {"used": 0, "label": None, "snap_used": 0, "snap": None, "phase": None}
    step_mib = a.step_mib << 20

    def snapshot():
        ctx, st = holder["ctx"], holder["st"]
        if ctx is not None and getattr(ctx, "st", None) is not None:
            st = ctx.st                      # the live state (the GFS start builds helper States too)
        seen = set()
        items = []
        if st is not None:
            for k, v in list(st.fields.items()):
                out = []
                _arrays(v, seen, k, out)
                # a NAM table field is model state whoever filled it: the GFS start builds it in State(),
                # a saved init (--init-from) puts it in from slab_init.load
                org = "pre" if k in spec_names else getattr(st.fields, "origin", {}).get(k, "?")
                items += [("st", p, b, dt, sh, org) for p, _, b, dt, sh in out]
        if ctx is not None:
            for k, v in list(ctx.tables.items()):
                out = []
                _arrays(v, seen, k, out)
                org = getattr(ctx.tables, "origin", {}).get(k, "?")
                items += [("tables", p, b, dt, sh, org) for p, _, b, dt, sh in out]
            for attr in ("bc", "filt", "history_fields", "slab", "comm"):
                out = []
                _arrays(getattr(ctx, attr, None), seen, attr, out)
                items += [(attr, p, b, dt, sh, attr) for p, _, b, dt, sh in out]
        # module-level device caches in namgpu modules
        for mname, mod in list(sys.modules.items()):
            if not mname.startswith("namgpu") or mod is None:
                continue
            for k, v in list(vars(mod).items()):
                if k.startswith("__") or isinstance(v, type) or (callable(v) and not isinstance(v, dict)):
                    continue
                out = []
                _arrays(v, seen, f"{mname}.{k}", out)
                items += [("module", p, b, dt, sh, mname.replace(".", "/") + ".py") for p, _, b, dt, sh in out]
        return items

    phase_max = collections.defaultdict(lambda: [0, 0, 0])   # phase -> [card used, pool reserved, pool live]
    orig_malloc = cp.cuda.get_allocator()      # the installed allocator (devmem's cap, or pool.malloc)

    def alloc(n):
        p = orig_malloc(n)
        u = pool.used_bytes()
        pm = phase_max[cur["phase"]]
        t = pool.total_bytes()
        if t > pm[1]:
            pm[1] = t
        if u > pm[2]:
            pm[2] = u
        lab = cur["label"]
        tr = u - unit_start.get(lab, u)
        if tr > unit_peak_tr[lab]:
            unit_peak_tr[lab] = tr
        if u > hw["used"]:
            hw["used"], hw["label"], hw["phase"] = u, lab, cur["phase"]
            if u >= hw["snap_used"] + step_mib:
                hw["snap_used"] = u
                hw["snap"] = (u, lab, cur["phase"], snapshot(), pool.total_bytes())
        return p

    cp.cuda.set_allocator(alloc)

    def card_sample(lab):
        fr, tt = cp.cuda.runtime.memGetInfo()
        used = tt - fr
        pm = phase_max[cur["phase"]]
        pm[0] = max(pm[0], used)
        pm[1] = max(pm[1], pool.total_bytes())
        pm[2] = max(pm[2], pool.used_bytes())
        if used > card["peak"]:
            card.update(peak=used, label=lab, phase=cur["phase"], pool_total=pool.total_bytes(),
                        pool_used=pool.used_bytes())

    orig_stamp = timing.stamp

    def stamp(ctx, label):
        prev = cur["label"]
        card_sample(prev)
        base = str(label).split("/")[0]
        cur["label"] = f"{cur['phase']}:{base}"
        u = pool.used_bytes()
        unit_start[cur["label"]] = u
        if u > unit_start_live[cur["label"]]:
            unit_start_live[cur["label"]] = u
        return orig_stamp(ctx, label)

    timing.stamp = stamp
    drvmod.timing.stamp = stamp

    orig_state_init = statemod.State.__init__

    def state_init(self, *aa, **kk):
        orig_state_init(self, *aa, **kk)
        self.fields = TrackDict(self.fields)
        holder["st"] = self

    statemod.State.__init__ = state_init
    orig_post = drvmod.Ctx.__post_init__

    def post_init(self):
        orig_post(self)
        self.tables = TrackDict(self.tables)
        holder["ctx"] = self

    drvmod.Ctx.__post_init__ = post_init

    from namgpu.init import init_case_driver as icd
    import importlib
    orig_import = importlib.import_module

    def wrap_run(mod, label):
        if getattr(mod, "_audit_wrapped", False) or not hasattr(mod, "run"):
            return mod
        f = mod.run

        def run(*aa, **kk):
            prev = cur["label"]
            card_sample(prev)
            cur["label"] = label
            unit_start[label] = pool.used_bytes()
            try:
                return f(*aa, **kk)
            finally:
                card_sample(label)
                cur["label"] = prev
        mod.run = run
        mod._audit_wrapped = True
        return mod

    units = icd._units()
    for uid in icd.CHAIN:
        try:
            wrap_run(orig_import(units[uid]["wrapper"]), f"init:{uid}")
        except Exception:  # noqa: BLE001
            pass
    from namgpu.steps import drv_dfi_tdfi_driver as tdfi
    for dep, (m, _fn) in tdfi.DEPS.items():
        wrap_run(orig_import(m), f"tdfi:{dep}")
    from namgpu.steps import drv_time_loop
    orig_tl = drv_time_loop._run

    def tl_run(ctx, hours, tdfi_mod, log=print):
        class T:
            @staticmethod
            def run(c):
                cur["phase"] = "tdfi"
                try:
                    return tdfi_mod.run(c)
                finally:
                    cur["phase"] = "fcst"
        cur["phase"] = "fcst"
        return orig_tl(ctx, hours, T, log=log)

    drv_time_loop._run = tl_run
    orig_hist = drvmod.Driver.write_history

    def write_history(self):
        ph = cur["phase"]
        cur["phase"] = "hist"
        try:
            return orig_hist(self)
        finally:
            card_sample(cur["label"])
            cur["phase"] = ph

    drvmod.Driver.write_history = write_history
    orig_drun = drvmod.Driver.run

    def drun(self, *aa, **kk):
        cur["phase"] = "fcst"
        return orig_drun(self, *aa, **kk)

    drvmod.Driver.run = drun
    cur["phase"] = "init"

    # this process's own device memory (NVML per process, every 10 ms): the card is shared, so memGetInfo (whole
    # card) can include other jobs. A 0.5 s nvidia-smi sampler tagged the peak with whichever unit ran when its
    # sample landed (three runs of one tip named three different units); the peak is a plateau of seconds across
    # many units, so the record keeps the plateau and the time each unit spends on it.
    import ctypes
    import os
    import threading
    proc = {"peak_mib": 0, "stop": False, "trace": []}
    # every sample: (MiB, pool total, pool used, stack bytes/thread). The peak is a plateau seconds long across
    # many units, so the breakdown is a statistic over the plateau samples, not the one sample that first hit it
    # (that split moved 16-48 % run to run at the same tip and input while the total held within 2.3 %).
    samp = array.array("q")
    rt = cp.cuda.runtime
    props = rt.getDeviceProperties(cp.cuda.Device().id)
    resident = int(props["multiProcessorCount"]) * int(props["maxThreadsPerMultiProcessor"])

    class _P(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong), ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]

    nv = ctypes.CDLL("libnvidia-ml.so.1")
    nv.nvmlInit_v2()
    nvh = ctypes.c_void_p()
    # NVML handle by the PCI bus id of the CUDA device in use: CUDA_VISIBLE_DEVICES may be a GPU UUID (card
    # leases set GPU-...; int() of it raised ValueError at start) and NVML index order need not match CUDA's
    bus = cp.cuda.Device().pci_bus_id
    if nv.nvmlDeviceGetHandleByPciBusId_v2(ctypes.c_char_p(bus.encode()), ctypes.byref(nvh)) != 0:
        raise SystemExit(f"vram_audit: NVML finds no device at PCI bus {bus} (CUDA_VISIBLE_DEVICES="
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    nvq = getattr(nv, "nvmlDeviceGetComputeRunningProcesses_v3", None) or nv.nvmlDeviceGetComputeRunningProcesses_v2
    nvbuf = (_P * 128)()
    me = os.getpid()

    def proc_mib():
        n = ctypes.c_uint(128)
        if nvq(nvh, ctypes.byref(n), nvbuf) != 0:
            return -1
        return sum(int(nvbuf[i].used) >> 20 for i in range(n.value) if nvbuf[i].pid == me)

    cp.cuda.runtime.memGetInfo()                 # context
    proc["context_mib"] = proc_mib()             # context only: nothing allocated, no kernel loaded yet

    def sampler():
        cp.cuda.Device(cp.cuda.Device().id).use()
        t0 = time.perf_counter()
        last = None
        while not proc["stop"]:
            try:
                mib = proc_mib()
                lab = cur["label"]
                if mib < 0:
                    time.sleep(0.01)
                    continue
                if (mib, lab) != last:
                    proc["trace"].append((round(time.perf_counter() - t0, 3), mib, lab))
                    last = (mib, lab)
                p = cp.get_default_memory_pool()
                samp.extend((mib, int(p.total_bytes()), int(p.used_bytes()),
                             int(rt.deviceGetLimit(rt.cudaLimitStackSize))))
                if mib > proc["peak_mib"]:
                    proc.update(peak_mib=mib, at=lab)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)

    sampler_th = threading.Thread(target=sampler, daemon=True)
    sampler_th.start()

    try:
        from namgpu import run_forecast           # the package (woof-nam)
    except ImportError:                           # the port tree: tools/run_forecast.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_forecast", ROOT / "tools" / "run_forecast.py")
        run_forecast = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_forecast)
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rc = run_forecast.main(["--case", a.case, "--source", a.source, "--hours", str(a.hours), "--out", str(out / "run"),
                            "--no-maps"] + (["--init-from", str(a.init_from)] if a.init_from is not None else []))
    wall = time.perf_counter() - t0
    proc["stop"] = True
    sampler_th.join(timeout=5)                    # samp is read below: no append in flight (free-threaded build)
    g = holder["ctx"].grid if holder["ctx"] is not None else None
    ncols = int(g.im) * int(g.jm) if g is not None else 954 * 835
    rec = {"rc": rc, "wall_s": round(wall, 1), "ncols": ncols, "card_peak_gib": round(card["peak"] / 2**30, 3),
           "card_peak_at": card.get("label"), "card_peak_phase": card.get("phase"),
           "card_peak_pool_total_gib": round(card["pool_total"] / 2**30, 3),
           "card_peak_pool_used_gib": round(card["pool_used"] / 2**30, 3),
           "card_peak_outside_pool_gib": round((card["peak"] - card["pool_total"]) / 2**30, 3),
           "process_peak_mib": proc["peak_mib"], "process_peak_first_unit": proc.get("at"),
           "pool_highwater_gib": round(hw["used"] / 2**30, 3), "pool_highwater_at": hw["label"],
           "pool_highwater_phase": hw["phase"]}
    pk = proc["peak_mib"] * 2**20
    if pk > 0:
        # the plateau: every sample within tol of the peak (tol covers the run-to-run noise of the total, 2.3 %)
        tol = max(64, proc["peak_mib"] * 3 // 100)
        plat = [tuple(samp[i:i + 4]) for i in range(0, len(samp), 4) if samp[i] >= proc["peak_mib"] - tol]
        rows_s = {"pool live": [u for _, _, u, _ in plat],
                  "pool slack (reserved, not live)": [t - u for _, t, u, _ in plat],
                  "device local-memory reservation": [lm * resident for _, _, _, lm in plat],
                  "CUDA context + modules + other": [m * 2**20 - t - lm * resident for m, t, _, lm in plat]}

        def _med(v):
            v = sorted(v)
            return v[len(v) // 2] if v else 0

        # the process peak, whole: pool live + pool slack + local-memory reservation + the rest outside the pool
        # (CUDA context, kernel modules, cuBLAS/cuFFT handles). Each row is its median over the plateau samples
        # (rows sum to about the plateau's median total); the range is min..max over the plateau.
        rows_pk = {k: _med(v) for k, v in rows_s.items()}
        rec["process_peak_breakdown_gib"] = {k: round(v / 2**30, 3) for k, v in rows_pk.items()}
        rec["process_peak_breakdown_bytes_per_col"] = {k: round(v / ncols, 1) for k, v in rows_pk.items()}
        rec["process_peak_breakdown_range_gib"] = {k: [round(min(v) / 2**30, 3), round(max(v) / 2**30, 3)]
                                                   for k, v in rows_s.items() if v}
        rec["process_peak_plateau_median_mib"] = _med([m for m, _, _, _ in plat])
        rec["process_peak_lmem_bytes_per_thread"] = sorted({lm for _, _, _, lm in plat})
        rec["process_peak_breakdown_samples"] = len(plat)
        rec["context_mib"] = proc.get("context_mib")
        tr = proc["trace"]
        on = collections.defaultdict(float)
        ts = []
        for i, (t, mib, lab) in enumerate(tr):
            if mib >= proc["peak_mib"] - tol:
                dt = (tr[i + 1][0] if i + 1 < len(tr) else t) - t
                on[lab] += dt
                ts.append(t)
        rec["process_peak_plateau"] = {
            "tol_mib": tol, "start_s": ts[0] if ts else None, "end_s": ts[-1] if ts else None,
            "units_s": {k: round(v, 3) for k, v in sorted(on.items(), key=lambda r: -r[1])}}
        (out / "process-trace.json").write_text(json.dumps(tr))
    rec["phase_max_gib"] = {ph: {"card_used": round(v[0] / 2**30, 3), "pool_reserved": round(v[1] / 2**30, 3),
                                 "pool_live": round(v[2] / 2**30, 3)} for ph, v in phase_max.items()}
    try:
        from namgpu import devmem
        rec["pool"] = devmem.stats()
    except ImportError:
        pass
    snap = hw["snap"]
    if snap is not None:
        used, lab, phase, items, ptot = snap
        named = sum(i[2] for i in items)
        rec.update(snapshot_used_gib=round(used / 2**30, 3), snapshot_at=lab, snapshot_phase=phase,
                   snapshot_pool_total_gib=round(ptot / 2**30, 3),
                   snapshot_named_gib=round(named / 2**30, 3),
                   snapshot_temporaries_gib=round((used - named) / 2**30, 3))
        by = collections.defaultdict(int)
        rows = []
        for hold, path, b, dt, sh, org in items:
            grp = ("NAM state (tables)" if hold == "st" and org == "pre" else
                   "TDFI" if hold == "filt" else "halo/boundary" if hold == "bc" else
                   "history writer" if hold == "history_fields" else group_of(org))
            by[grp] += b
            rows.append({"holder": hold, "name": path, "bytes": b, "dtype": dt, "shape": sh, "creator": org,
                         "group": grp})
        by[f"temporaries of {lab}"] += used - named
        rec["groups_gib"] = {k: round(v / 2**30, 3) for k, v in sorted(by.items(), key=lambda r: -r[1])}
        rec["groups_bytes_per_col"] = {k: round(v / ncols, 1) for k, v in sorted(by.items(), key=lambda r: -r[1])}
        rows.sort(key=lambda r: -r["bytes"])
        rec["arrays"] = rows
        dts = collections.defaultdict(float)
        for r in rows:
            dts[r["dtype"]] += r["bytes"] / 2**30
        rec["dtype_gib"] = {k: round(v, 3) for k, v in dts.items()}
    rec["unit_transient_gib"] = {k: round(v / 2**30, 3) for k, v in sorted(unit_peak_tr.items(), key=lambda r: -r[1])
                                 if v > 16 << 20}
    rec["unit_start_live_gib"] = {k: round(v / 2**30, 3) for k, v in sorted(unit_start_live.items(),
                                                                            key=lambda r: -r[1])[:40]}
    (out / "vram-audit.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("arrays", "unit_transient_gib", "unit_start_live_gib")},
                     indent=1), flush=True)
    print("unit transient top:", list(rec["unit_transient_gib"].items())[:25], flush=True)
    # groups_gib is the pool census at the pool high-water mark; this is the process peak, whole (rows sum to it)
    for k, v in rec.get("process_peak_breakdown_gib", {}).items():
        lo, hi = rec["process_peak_breakdown_range_gib"].get(k, [v, v])
        print(f"  process peak: {k:34s} {v:7.3f} GiB {rec['process_peak_breakdown_bytes_per_col'][k]:10.1f} B/col"
              f"  (plateau {lo:.3f}..{hi:.3f})", flush=True)
    for r in rec.get("arrays", [])[:60]:
        print(f"  {r['bytes']/2**20:8.1f} MiB {r['holder']:7s} {r['name'][:40]:40s} {r['dtype']:8s} {str(r['shape']):24s} "
              f"{r['group']:18s} {r['creator']}", flush=True)
    return rc


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(int(rc or 0))
