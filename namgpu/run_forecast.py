"""Run one WOOF-NAM forecast through the driver entry (Driver.run): hourly history, then maps
(io-maps, rw_wrfbatch).

    woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z
    (same as: python -m namgpu.run_forecast ...; python tools/run_forecast.py ... in a checkout)
Writes <out>/forecast-receipt.json (init s, per-hour wall, ms/step, nonfinite census at the end,
peak VRAM, units left out) plus the driver's timing_receipt.json, history files and maps/.

Our own box lanes source env-box.sh, which sets WOOF_NAM_BOX=1: only then does the run hand back the
card lock it inherited (CARD3.lock / NAMGPU_CARD_LOCK) before the maps. Without it no lock is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from namgpu import paths

PKG_PARENT = str(paths.PKG.parent)      # put back on sys.path for the maps-only tail


def release_card(ctx) -> dict:
    """Free this run's device memory. With WOOF_NAM_BOX=1 (our own box lanes) also unlock the card lock
    this process inherited (rl_run.sh and the lane runners hold it on an inherited fd; LOCK_UN on that fd
    releases it for the parent too). The lock file is NAMGPU_CARD_LOCK (default: any open file named
    CARD3.lock). Without WOOF_NAM_BOX no lock is touched. Returns what was done."""
    import fcntl
    import gc
    import os

    import cupy as cp
    ctx.stream.synchronize()
    ctx.st.fields.clear()
    ctx.tables.clear()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    used = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    want = os.environ.get("NAMGPU_CARD_LOCK", "")
    dropped = []
    for fd in (os.listdir("/proc/self/fd") if paths.box_mode() else ()):
        try:
            tgt = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if (want and os.path.realpath(want) == tgt) or (not want and os.path.basename(tgt) == "CARD3.lock"):
            fcntl.flock(int(fd), fcntl.LOCK_UN)
            os.close(int(fd))
            dropped.append(tgt)
    print(f"card released: pool freed, card used {used} GiB, lock dropped {dropped or 'none held'}", flush=True)
    return {"card_used_gib_after": used, "locks_dropped": dropped}


def main(argv=None):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Run one WOOF-NAM forecast on the NAM 12 km CONUS grid (954 x 835 x 60, all physics) on "
                    "one NVIDIA card. Writes hourly history (wrfout_d01_<valid time>), maps in <out>/maps/ and "
                    "<out>/forecast-receipt.json. Memory today: the published code needs a card with more "
                    "than 76 GB.",
        epilog="example (after woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z):\n"
               "  woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z\n"
               "A GFS start (--source gfs) also needs the static fields once: woof-nam static --help.",
        formatter_class=HelpFormatter)
    ap.add_argument("--case", required=True,
                    help="the case folder woof-nam fetch wrote (cases/2025-03-14_12z), or its bare name, looked up "
                         "under <work root>/cases/; the name gives the start time (YYYY-MM-DD_HHz)")
    ap.add_argument("--source", default="native",
                    help="start from the archived NAM analysis (native, cycles before 2026-11-03 12Z) or GFS (gfs)")
    ap.add_argument("--hours", type=float, default=1.0, help="forecast length in hours (default 1)")
    ap.add_argument("--out", type=Path, required=True, help="output folder (relative paths are under the work root)")
    ap.add_argument("--skip-init", default="init-fetch-case",
                    help="comma list of init units to skip (default: the fetch, already done by woof-nam fetch)")
    ap.add_argument("--leave-out", default="", help="comma list of plan units set to mode skip (receipt lists them)")
    ap.add_argument("--no-maps", action="store_true", help="write history only, no maps")
    ap.add_argument("--maps-only", action="store_true", help="internal: the post-run maps tail (no CUDA)")
    ap.add_argument("--cards", type=int, default=1,
                    help="number of cards (default 1). N > 1 runs one process per card, each owning a band of whole "
                         "rows of the grid, with halo rows exchanged over NCCL; the history files are the same as "
                         "one card's, bit for bit")
    ap.add_argument("--card-list", default=os.environ.get("NAM_CARDS", ""),
                    help="comma list of card indices for --cards N (default 0..N-1; env NAM_CARDS)")
    ap.add_argument("--hist-workers", type=int, default=4, help="history writer processes for --cards N (default 4)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a saved initial state (python -m namgpu.init.slab_init save ...) to load instead of running "
                         "the init; every card keeps its own rows. With --cards N and no --init-from, the run saves "
                         "one first, on the first card, in <out>/init (deleted at the end unless --keep-init)")
    ap.add_argument("--keep-init", action="store_true", help="keep the saved initial state <out>/init (--cards N)")
    ap.add_argument("--managed-init", action="store_true",
                    help="--cards N: build the saved initial state in CUDA managed memory (a domain whose init does "
                         "not fit one card)")
    a = ap.parse_args(argv)
    if a.maps_only:
        return maps_only(a.case, a.out if a.out.is_absolute() else paths.work_root() / a.out)
    if a.cards > 1 and "NAMGPU_RANK" not in os.environ:
        return launch(a, list(argv) if argv is not None else sys.argv[1:])
    if "NAMGPU_RANK" in os.environ:
        # kill -USR1 <rank pid> prints every thread stack to the rank log (no debugger needed)
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver

    out = a.out if a.out.is_absolute() else paths.work_root() / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    multi = rank_setup(g) if "NAMGPU_RANK" in os.environ else None   # several cards: this process is one rank
    if multi is not None and a.init_from is None:
        raise SystemExit("--cards N needs --init-from <saved init>: the init chain sums with float atomics "
                         "(init-gwd-orog, init-vcoord-recover), so ranks that each ran it would start from different "
                         "bits; the init runs once (namgpu.init.slab_init save) and every rank loads its rows")
    if multi is not None:
        g = multi["grid"]                                     # the rank's grid (slab.local_grid)
    if a.init_from is not None:
        # the saved init replaces every field: allocate only the scalars here (no 2-D/3-D/4-D arrays twice)
        st = statemod.State(g, names={k for k, v in statemod.all_specs().items() if v.rank == 0})
    else:
        st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    if multi is not None:
        ctx.slab, ctx.comm, ctx.hist_geo_global = multi["slab"], multi["comm"], multi["geo"]
        ctx.decomp = (ctx.slab.rank, 1, ctx.slab.nranks)
        print(f"rank {ctx.slab.rank}/{ctx.slab.nranks}: rows {ctx.slab.jts}..{ctx.slab.jte} of {ctx.slab.jmg}, "
              f"memory {ctx.slab.jms}..{ctx.slab.jme} ({g.jm} rows), card {os.environ.get('CUDA_VISIBLE_DEVICES')}",
              flush=True)
    ctx.case_dir = paths.case_dir(a.case)
    ctx.start = datetime.strptime(ctx.case_dir.name[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.out_dir = out if multi is None else out / f"rank{ctx.slab.rank}"   # history goes via NAMGPU_HIST_SHM
    Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    skip = [s for s in a.skip_init.split(",") if s]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    rec = {"case": a.case, "source": a.source, "hours": a.hours, "dt": cfg.dt, "skipped_init": skip}
    mp = statemod.memory_path()
    rec["memory_path"] = mp
    print(f"memory path: {mp['path']} ({mp['how']}: {mp['reason']})", flush=True)
    t0 = time.perf_counter()
    if a.init_from is not None:
        from namgpu.init import slab_init
        rec["init"] = {"loaded_from": str(a.init_from), "load": slab_init.load(ctx, a.init_from)}
        rec["skipped_init"] = "all (loaded)"
        # The saved ctx attributes are the init process's (one card: comm/slab/decomp None); this rank's win.
        if multi is not None:
            ctx.slab, ctx.comm, ctx.hist_geo_global = multi["slab"], multi["comm"], multi["geo"]
            ctx.decomp = (ctx.slab.rank, 1, ctx.slab.nranks)
        else:
            ctx.slab = ctx.comm = ctx.decomp = None
        # Per-rank tables are dropped by the load (slab_init.PER_RANK_TABLES); drv-loop-limits rebuilds this rank's
        # NMMB_LIMITS (consumers such as dyn-pgforce-divcor read it and do not build it).
        from namgpu.steps import drv_loop_limits
        with ctx.stream:
            drv_loop_limits.run(ctx)
    else:
        with ctx.stream:
            rec["init"] = init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    rec["init_s"] = round(time.perf_counter() - t0, 2)
    print(f"init done {rec['init_s']} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    # Init temporaries (GWD terrain, table builds) stay cached in the CuPy pool and showed up as 40-55 GB on
    # card 3 for a forecast whose step loop needs ~17 GB. Return them.
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    rec["after_init_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    rec["pool_after_init_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)

    leave = [u for u in a.leave_out.split(",") if u]
    for u in leave:
        drv.units[u]["mode"] = "skip"
    rec["left_out"] = leave

    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()

    hours_log = []

    def log(msg):
        hours_log.append(msg)
        print(msg, flush=True)

    t1 = time.perf_counter()
    try:
        # TDFI first, as the NAM forecast does (filter_method 3, exnam_fcst.sh), then the forecast loop: the same
        # order as drv-time-loop / tests/smoke_full.py. Driver.run alone skipped the filter (every run_forecast
        # run before 2026-11-04 19:00Z was a no-DFI start).
        from types import SimpleNamespace
        from namgpu.steps import drv_dfi_tdfi_driver, drv_time_loop

        def _tdfi(c):
            tf = time.perf_counter()
            out = drv_dfi_tdfi_driver.run(c)
            c.stream.synchronize()
            rec["tdfi_s"] = round(time.perf_counter() - tf, 2)
            print(f"tdfi done {rec['tdfi_s']} s {out}", flush=True)
            return out
        rec["filter_method"] = int(cfg.filter_method)
        with ctx.stream:
            n = drv_time_loop._run(ctx, a.hours, SimpleNamespace(run=_tdfi), log=log)
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        n = ctx.ntimestep
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-3000:]
        print("RUN ERROR", rec["error"], flush=True)
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    rec["steps_done"] = int(ctx.ntimestep)
    rec["run_s"] = round(wall, 2)
    rec["ms_per_step"] = round(wall / max(1, ctx.ntimestep) * 1e3, 1)
    rec["hour_lines"] = hours_log
    bad = init_case_driver.census(ctx)
    if multi is not None:
        # Several cards: the answer is the owned rows (history, maps, sums). Halo rows of fields NMMB never
        # exchanges (physics scratch, accumulators) are not kept consistent and may hold NaN; they are never read for
        # an owned value (the 1 vs N card history is bitwise). The pass/fail census is on the owned rows; the halo
        # count is kept beside it.
        rec["nonfinite_end_incl_halo"] = bad
        bad = owned_census(ctx)
    rec["nonfinite_end"] = bad
    SANE = ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "T2", "TSKIN", "ACPREC", "PREC", "CUPREC")
    sx = {}
    for nm in SANE:
        v = st.fields.get(nm)
        if v is not None:
            sx[nm] = [round(float(v.min()), 5), round(float(v.max()), 5), round(float(v.mean()), 6)]
    rec["stats_end"] = sx
    rec["pool_end_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)
    rec["card_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    print("end stats", json.dumps(sx), "nonfinite", bad, flush=True)
    rd.__exit__(None, None, None)
    if not a.no_maps and rec["error"] is None and multi is None:
        # Maps are host work. Before them: drop the card lock and give back ALL of the card, so the next
        # run starts while the frames render (run lead 11:05Z: the old post-run maps held CARD3.lock and
        # 62 GB for ~50 min on one core). Freeing the pool still left 10.7 GB in unit caches, so this process
        # re-executes itself as the maps-only tail (same PID for the runner, no CUDA context left).
        from namgpu.io.io_history import drain
        drain(ctx)                                   # the last history frame is on disk
        rec["card_released"] = release_card(ctx)
        rec["exit_code"] = 1 if (rec["error"] or bad) else 0
        (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, "-c",
                                  "import sys; sys.path.insert(0, %r); from namgpu.run_forecast import main; "
                                  "sys.exit(main(sys.argv[1:]))" % PKG_PARENT,
                                  "--maps-only", "--case", str(ctx.case_dir), "--out", str(out)])
    rp = out / "forecast-receipt.json" if multi is None else out / f"forecast-receipt.r{ctx.slab.rank}.json"
    if multi is not None:
        rec["rank"] = {"rank": ctx.slab.rank, "nranks": ctx.slab.nranks, "jts": ctx.slab.jts, "jte": ctx.slab.jte,
                       "jms": ctx.slab.jms, "jme": ctx.slab.jme, "card": os.environ.get("CUDA_VISIBLE_DEVICES")}
        from namgpu.io.io_history import drain as _drain
        _drain(ctx)
    rp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {rp}", flush=True)
    return 1 if (rec["error"] or bad) else 0


def owned_census(ctx) -> dict:
    """Non-finite count per float field over this rank's OWNED rows (axis -2 of [.., JM, X] fields). One host sync."""
    import cupy as cp
    s = ctx.slab
    jm = int(ctx.grid.jm)
    out, seen = {}, set()
    for name, a in ctx.st.fields.items():
        if not isinstance(a, cp.ndarray) or a.dtype.kind != "f" or a.ndim < 2 or a.shape[-2] != jm:
            continue
        if a.data.ptr in seen and a.ndim == 2:
            continue
        seen.add(a.data.ptr)
        n = int(cp.count_nonzero(~cp.isfinite(a[..., s.own0:s.own0 + s.nown, :])))
        if n:
            out[name] = n
    return out


def rank_setup(g) -> dict:
    """This process is rank NAMGPU_RANK of NAMGPU_NRANKS (set by launch()). Keeps the global history geometry, then
    gives the rank's grid (slab.local_grid: jm = the slab's memory rows, per-row arrays cut). Opens the NCCL
    communicator."""
    import numpy as np

    from namgpu.parallel import Comm
    from namgpu.slab import Slab, local_grid
    r, n = int(os.environ["NAMGPU_RANK"]), int(os.environ["NAMGPU_NRANKS"])
    jmg, im = int(g.jm), int(g.im)
    cen = (jmg // 2, im // 2)
    geo = {"xlat": np.degrees(np.asarray(g.glat_h)).astype(np.float32),
           "xlon": np.degrees(np.asarray(g.glon_h)).astype(np.float32),
           "cen_lat": float(np.degrees(g.glat_h[cen])), "cen_lon": float(np.degrees(g.glon_h[cen])),
           "dx": float(np.asarray(g.dxh)[jmg // 2])}
    slab = Slab.for_rank(r, n, jmg)
    comm = Comm(r, n, os.environ["NAMGPU_NCCL_ID"])
    return {"slab": slab, "comm": comm, "geo": geo, "grid": local_grid(g, slab)}


def _py_module(mod: str) -> list:
    """argv that runs `mod`'s main() with this package importable (a source checkout or an installed wheel)."""
    return [sys.executable, "-c",
            "import sys; sys.path.insert(0, %r); from %s import main; sys.exit(main(sys.argv[1:]))" % (PKG_PARENT, mod)]


def _drop_opt(argv: list, opt: str) -> list:
    return [x for x in argv if x != opt]


def launch(a, argv) -> int:
    """--cards N: one process per card (CUDA_VISIBLE_DEVICES pinned; one Python thread driving N cards would be a
    single-core stage on the step path), plus the history writer pool. Every rank runs this same module with
    NAMGPU_RANK set. If one rank fails the others are stopped (NCCL would wait forever). Without --init-from the
    initial state is built once on the first card and saved to <out>/init first."""
    import shutil
    import signal
    import subprocess

    out = a.out if a.out.is_absolute() else paths.work_root() / a.out
    out.mkdir(parents=True, exist_ok=True)
    cards = [c.strip() for c in a.card_list.split(",") if c.strip()] or [str(i) for i in range(a.cards)]
    if len(cards) < a.cards:
        raise SystemExit(f"--cards {a.cards} but --card-list has {len(cards)} cards: {cards}")
    cards = cards[:a.cards]
    t0 = time.perf_counter()
    child_argv = _drop_opt(_drop_opt(list(argv), "--keep-init"), "--managed-init")
    rec = {"cards": cards}
    made_init = None
    if a.init_from is None:
        made_init = out / "init"
        cmd = _py_module("namgpu.init.slab_init") + [
            "save", "--case", str(a.case), "--source", a.source, "--hours", str(a.hours), "--out", str(made_init),
            "--skip-init", a.skip_init] + (["--managed"] if a.managed_init else [])
        print(f"saving the initial state once on card {cards[0]} -> {made_init} (log {out / 'init.log'})", flush=True)
        with open(out / "init.log", "w") as logf:
            rc = subprocess.call(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=cards[0]), stdout=logf,
                                 stderr=subprocess.STDOUT)
        rec["init_save_s"] = round(time.perf_counter() - t0, 1)
        if rc != 0:
            rec["exit_code"] = 1
            rec["error"] = f"initial state save failed (exit {rc}); see {out / 'init.log'}"
            (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
            print(rec["error"], flush=True)
            return 1
        child_argv += ["--init-from", str(made_init)]
    idfile = out / "nccl.id"
    idfile.unlink(missing_ok=True)
    shm = Path("/dev/shm") / f"namgpu-hist-{os.getpid()}"
    shutil.rmtree(shm, ignore_errors=True)
    shm.mkdir(parents=True)
    writer = subprocess.Popen(_py_module("namgpu.io.history_merge") + [
        "--shm", str(shm), "--out", str(out), "--nranks", str(a.cards), "--workers", str(a.hist_workers)])
    if "--no-maps" not in child_argv:
        child_argv.append("--no-maps")
    procs = []
    t_run = time.perf_counter()
    for r, card in enumerate(cards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card, NAMGPU_RANK=str(r), NAMGPU_NRANKS=str(a.cards),
                   NAMGPU_NCCL_ID=str(idfile), NAMGPU_HIST_SHM=str(shm))
        logf = open(out / f"rank{r}.log", "w")
        procs.append((subprocess.Popen(_py_module("namgpu.run_forecast") + child_argv, env=env,
                                       stdout=logf, stderr=subprocess.STDOUT), logf))
    print(f"launched {a.cards} ranks on cards {cards}; logs {out}/rank*.log; history pool {a.hist_workers}",
          flush=True)
    codes = [None] * len(procs)
    failed = False
    while any(c is None for c in codes):
        for i, (p, _) in enumerate(procs):
            if codes[i] is None and p.poll() is not None:
                codes[i] = p.returncode
                if p.returncode != 0 and not failed:
                    failed = True
                    print(f"rank {i} exited {p.returncode}: stopping the other ranks", flush=True)
                    for q, _ in procs:
                        if q.poll() is None:
                            q.send_signal(signal.SIGTERM)
        time.sleep(0.5)
    for _, f in procs:
        f.close()
    (shm / "END").write_text("1")
    wcode = writer.wait()
    shutil.rmtree(shm, ignore_errors=True)
    rec.update({"rank_exit": codes, "history_pool_exit": wcode, "run_wall_s": round(time.perf_counter() - t_run, 1),
                "wall_s": round(time.perf_counter() - t0, 1),
                "ranks": [json.loads(p.read_text()) if p.exists() else None
                          for p in (out / f"forecast-receipt.r{r}.json" for r in range(a.cards))]})
    # integer counters read at the end (nonfinite census per rank on the owned rows): one list, no step-path work
    rec["nonfinite_end_by_rank"] = [(r or {}).get("nonfinite_end") for r in rec["ranks"]]
    rec["ms_per_step_by_rank"] = [(r or {}).get("ms_per_step") for r in rec["ranks"]]
    rec["card_used_gib_by_rank"] = [(r or {}).get("card_used_gib") for r in rec["ranks"]]
    ok = all(c == 0 for c in codes) and wcode == 0 and not any(rec["nonfinite_end_by_rank"])
    rec["exit_code"] = 0 if ok else 1
    if made_init is not None and not a.keep_init:
        shutil.rmtree(made_init, ignore_errors=True)
        rec["init_deleted"] = str(made_init)
    (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"ranks exit {codes}, history pool {wcode}, wall {rec['wall_s']} s -> {out / 'forecast-receipt.json'}",
          flush=True)
    if ok and not a.no_maps:
        return maps_only(str(a.case), out)
    return 0 if ok else 1


def maps_only(case: str, out: Path) -> int:
    """Post-run maps in a process with no CUDA context: render every history frame of <out> not yet drawn
    (frames an hourly watcher rendered are skipped) across all cores, then finish the receipt."""
    from types import SimpleNamespace

    from namgpu.io import io_maps
    rp = out / "forecast-receipt.json"
    rec = json.loads(rp.read_text())
    cdir = paths.case_dir(case)
    ctx = SimpleNamespace(out_dir=out, case_dir=cdir, start=datetime.strptime(cdir.name[:14], "%Y-%m-%d_%Hz"))
    t2 = time.perf_counter()
    try:
        rec["maps"] = str(io_maps.run(ctx))[:2000]
    except Exception as e:  # noqa: BLE001
        rec["maps_error"] = f"{type(e).__name__}: {e}"
        print("MAPS ERROR", rec["maps_error"], flush=True)
    rec["maps_s"] = round(time.perf_counter() - t2, 2)
    print(f"maps {rec.get('maps', '')} in {rec['maps_s']} s", flush=True)
    rp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {rp}", flush=True)
    return int(rec.get("exit_code", 0)) or (1 if "maps_error" in rec else 0)


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # hard exit: after a device OOM the CUDA/CuPy teardown hung for minutes holding the card and its lock
    # (seen on a 32 GB card, 2026-11-04)
    os._exit(int(_rc or 0))
