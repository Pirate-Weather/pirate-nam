"""woof-nam nest --two-way | --concurrent: a 12 km parent and a nest inside it, run together in one process on one card.

    woof-nam nest --two-way --center 24N,83W --size-km 1900 --size-y-km 1400 --res-km 3 \
        --parent-size-km 3000 --parent-size-y-km 2600 --source gfs --start 2022092612 --hours 84 --out runs/ian-2way

A live parent (--concurrent, or any --parent-* option) runs one-way by default: the nest takes its boundaries from the
live parent every parent step (25 s), as the operational NAM ran its nests (nest_mode: 1-way). --two-way adds NMMB's
two-way feedback (the nest's T, Q, CW, U, V box means replace the parent's in the overlap, weight --2way-wgt, default
1.0 as NMMB's own two-way test). Without any of these `woof-nam nest` is the one-domain door, and --source parent
--parent-run is the one-way hand-off door, both unchanged.

Stages: domain (parent configfile, nest configfile on the parent's grid: domain.nest_grid_associated), preflight (both
domains in one process), renderer, warm, static (both), fetch (the parent's area holds the nest), wait-vram, init (each
domain's start saved once with namgpu.init.slab_init in its own process), forecast (namgpu.nest.twoway), maps (both),
sheets (the nest). Output: <out>/parent/ and <out>/nest/ (ordinary run folders: wrfout_d01_*, maps/), receipt
<out>/nest-receipt.json with the coupled run's twoway-receipt.json inside.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from namgpu.nest import cli, domain


def add_args(n) -> None:
    n.add_argument("--two-way", action="store_true", help="a 12 km parent and this nest run together in one "
                   "process, with NMMB's two-way feedback (the nest's T, Q, CW, U, V replace the parent's in the "
                   "overlap); off by default: a live parent runs one-way (--concurrent)")
    n.add_argument("--concurrent", action="store_true", help="a live parent with the feedback off: the nest takes "
                   "its boundaries from the live parent every parent step, as the operational NAM nests ran; the default "
                   "whenever a live parent is asked for (any --parent-* option)")
    n.add_argument("--parent-center", default=None, help="--two-way/--concurrent: the parent's centre (default "
                   "--center)")
    n.add_argument("--parent-size-km", type=float, default=None, help="--two-way/--concurrent: parent east-west "
                   "size (default --size-km + 800)")
    n.add_argument("--parent-size-y-km", type=float, default=None, help="parent north-south size (default the nest's "
                   "north-south size + 800)")
    n.add_argument("--2way-wgt", dest="twoway_wgt", type=float, default=1.0, help="weight of the nest in the parent's "
                   "overlap, 0-1 (NMMB 2way_wgt; default 1.0)")


def _combined_preflight(pg: dict, ng: dict, rows_given, index: str) -> dict:
    """Both domains in one CUDA context: the two pool parts add, the local-memory reservation and the context count
    once (domain.vram_estimate model)."""
    free = domain.free_gib(index)
    tried = []
    for r in ((rows_given,) if rows_given else domain.RRTM_ROW_CHOICES):
        ep = domain.vram_estimate(pg, r, 1, index)
        ec = domain.vram_estimate(ng, r, 1, index)
        pool = (ep["pool_limit_gib"] or ep["total_gib"] - ep["lmem_gib"] - ep["fixed_gib"]) + \
               (ec["pool_limit_gib"] or ec["total_gib"] - ec["lmem_gib"] - ec["fixed_gib"])
        tot = pool + ec["lmem_gib"] + ec["fixed_gib"]
        e = dict(ec, total_gib=round(tot, 2), pool_limit_gib=round(pool, 2) if ec["pool_limit_bytes"] else None,
                 pool_limit_bytes=int(pool * 2**30) if ec["pool_limit_bytes"] else None, parent=ep, nest=ec)
        tried.append(e)
        if tot + domain.VRAM["margin_gib"] <= free:
            return {"ok": True, "free_gib": round(free, 2), "pick": e, "tried": tried, "card": domain.card_name(index)}
    return {"ok": False, "free_gib": round(free, 2), "tried": tried, "card": domain.card_name(index)}


def main(a, argv) -> int:
    if a.two_way and a.concurrent:
        raise SystemExit("--two-way and --concurrent: pick one (--two-way is --concurrent plus the feedback)")
    if a.cards != 1:
        raise SystemExit("--two-way/--concurrent run both domains on one card in this release (--cards 1)")
    if a.source == "parent":
        raise SystemExit("--two-way/--concurrent start both domains from a source analysis (gfs, ifs, ...); "
                         "--source parent is the one-way hand-off door")
    t_all = time.perf_counter()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = cli.parse_center(a.center)
    plat, plon = cli.parse_center(a.parent_center) if a.parent_center else (lat, lon)
    cycle = datetime.strptime(a.start, "%Y%m%d%H")
    hours = float(a.hours)
    source, region = (domain.default_source(lat, lon, cycle) if a.source == "auto" else (a.source, "given"))
    boco_age, boco_why = cli.boco_age_for(source, cycle, a.boco_age_h)
    credit, credit_lines = cli.data_credits(source)
    sampler = cli.Sampler().start()
    mode = "two-way" if a.two_way else "concurrent"
    stages, rec = {}, {"door": f"woof-nam nest --{mode}", "argv": sys.argv[1:] if argv is None else list(argv),
                       "start": cycle.isoformat(), "hours": hours, "source": source, "source_region": region,
                       "boco_age_h": boco_age, "boco_age_why": boco_why, "data_credit": credit,
                       "data_licences": credit_lines, "host": os.uname().nodename, "mode": mode}

    def stage(name):
        sampler.mark(name)
        stages[name] = {"t0": time.perf_counter()}

    def done(name, **kw):
        s = stages[name]
        s["wall_s"] = round(time.perf_counter() - s.pop("t0"), 2)
        s.update(kw)
        print(f"woof-nam: {name} {s['wall_s']} s", flush=True)

    def finish(code):
        cli._write(out, rec, stages, sampler, t_all)
        return code

    # domain: the parent (own pole at its centre) and the nest on the parent's grid
    stage("domain")
    psx = a.parent_size_km or (a.size_km + 800.0)
    psy = a.parent_size_y_km or ((a.size_y_km or a.size_km) + 800.0)
    pg = domain.nest_grid(plat, plon, psx, 12, psy)
    ng = domain.nest_grid_associated(pg, lat, lon, a.size_km, a.res_km, a.size_y_km)
    pname = domain.domain_name(pg, psx, psy)
    name = domain.domain_name(ng, a.size_km, a.size_y_km) + f"_in{domain.grid_key(pg)[:6]}"
    from namgpu.init import static_build as _sb
    cli.STATIC_CACHE.mkdir(parents=True, exist_ok=True)
    pstatic = cli.STATIC_CACHE / f"{pname}-{domain.grid_key(pg)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    nstatic = cli.STATIC_CACHE / f"{name}-{domain.grid_key(ng)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    pcfg, ncfg = out / f"nam_configfile_{pname}", out / f"nam_configfile_{name}"
    pkeys = domain.write_configfile(pcfg, pg, pname, pstatic, False, handoff=False)
    nkeys = domain.write_configfile(ncfg, ng, name, nstatic, a.spec_adv, handoff=False)
    rec.update(parent={"domain": pname, "grid": pg, "configfile": str(pcfg), "keys": pkeys},
               nest={"domain": name, "grid": ng, "configfile": str(ncfg), "keys": nkeys}, domain=name, grid=ng,
               twoway_wgt=a.twoway_wgt if a.two_way else None)
    done("domain")
    print(f"woof-nam: parent {pname} {pg['im']} x {pg['jm']}; nest {name} {ng['im']} x {ng['jm']} x {ng['lm']} "
          f"({ng['extent_km']} km, ratio {ng['parent_child_space_ratio']}, SW corner on parent H "
          f"({ng['i_parent_start']}, {ng['j_parent_start']})), {mode}, source {source}", flush=True)

    stage("preflight")
    index = domain._visible_index()
    pf = _combined_preflight(pg, ng, a.rrtm_rows, index)
    rec["preflight"] = pf
    done("preflight")
    if not pf["ok"] and not a.dry_run and a.wait_min <= 0:
        rec["refused"] = (f"woof-nam: refused: parent + nest need about {pf['tried'][-1]['total_gib']} GiB and the card "
                          f"has {pf['free_gib']} GiB free ({pf['card']})")
        print(rec["refused"], flush=True)
        return finish(3)
    rows = pf["pick"]["rrtm_rows"] if pf["ok"] else (a.rrtm_rows or 8)

    if not (a.no_maps and a.no_sheets):
        stage("renderer")
        from namgpu.io import maps as _maps
        try:
            rw = _maps.renderer()
        except SystemExit as e:
            done("renderer", ok=False)
            rec["failed_stage"] = f"renderer: {e}"
            return finish(6)
        done("renderer", ok=True, path=str(rw))
    if not a.no_warm:
        stage("warm")
        from namgpu import kernels
        w = kernels.warm()
        done("warm", **{k: w[k] for k in ("workers", "modules", "compile_s_sum", "failed", "slowest")})

    env = dict(os.environ, NAMGPU_RRTM_ROWS=str(rows), PYTHONUNBUFFERED="1", NAMGPU_DATA_CREDIT=credit,
               NAMGPU_DATA_LICENCES=json.dumps(credit_lines), NAMGPU_START_SOURCE=source,
               NAMGPU_BOCO_AGE_H=str(boco_age),
               PYTHONPATH=str(cli.ROOT) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    a.boco_age_h = boco_age

    stage("static")
    for cfgfile, static in ((pcfg, pstatic), (ncfg, nstatic)):
        stale = cli._static_stale(static) if static.is_file() else None
        if static.is_file() and not stale:
            continue
        code, _ = cli.run_logged([sys.executable, "-m", "namgpu.init.static_build", "--config", str(cfgfile), "--date",
                                  f"{cycle:%Y-%m-%d}", "--out", str(static)], dict(env, NAMGPU_CONFIGFILE=str(cfgfile)),
                                 out / "static.log", sampler)
        if code:
            done("static", rc=code)
            return finish(code)
    done("static", parent=str(pstatic), nest=str(nstatic))
    rec["static_meta"] = {"parent": cli._static_meta(pstatic), "nest": cli._static_meta(nstatic)}
    if a.dry_run:
        return finish(0)

    stage("fetch")
    case = f"{cycle:%Y-%m-%d_%H}z"
    case_dir = (a.cases_root / case).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    fetch, lic = cli.resolve_fetch(source)
    from namgpu import config as configmod
    a.source_name = source
    fr = fetch(cycle, case_dir, max(6, int(hours + 0.999)), configmod.load(pcfg), a)   # the parent's area holds the nest
    rec["source_licence"] = lic
    done("fetch", **fr)
    if fr.get("failed"):
        return finish(4)

    stage("wait-vram")
    t_lim = time.perf_counter() + a.wait_min * 60
    while True:
        pf = _combined_preflight(pg, ng, a.rrtm_rows, index)
        if pf["ok"] or time.perf_counter() > t_lim:
            break
        time.sleep(30)
    done("wait-vram", free_gib=pf["free_gib"], ok=pf["ok"])
    rec["preflight_at_forecast"] = pf
    if not pf["ok"]:
        rec["refused"] = f"waited {a.wait_min} min: {pf['free_gib']} GiB free, need {pf['tried'][-1]['total_gib']} GiB"
        return finish(3)
    env["NAMGPU_RRTM_ROWS"] = str(pf["pick"]["rrtm_rows"])

    # init: each domain's start saved once, in its own process (its own configfile), then loaded by the coupled run
    shm = Path("/dev/shm")
    base = Path(os.environ.get("NAMGPU_INIT_DIR", str(shm if shm.is_dir() else out)))
    inits = {"parent": base / f"{pname}-{cycle:%Y%m%d%H}-init", "nest": base / f"{name}-{cycle:%Y%m%d%H}-init"}
    stage("init")
    for role, cfgfile in (("parent", pcfg), ("nest", ncfg)):
        # the nest's own boundaries serve only its TDFI (boco.0000): 1 h asks for the 6 h minimum
        h = hours if role == "parent" else 1.0
        code, _ = cli.run_logged([sys.executable, "-m", "namgpu.init.slab_init", "save", "--case", case, "--case-dir",
                                  str(case_dir), "--source", source, "--hours", str(h), "--out", str(inits[role])],
                                 dict(env, NAMGPU_CONFIGFILE=str(cfgfile)), out / f"init-{role}.log", sampler)
        if code:
            done("init", rc=code, failed=role)
            _rm(inits, rec)
            return finish(code)
    done("init", dirs={k: str(v) for k, v in inits.items()})

    stage("forecast")
    if pf.get("pick", {}).get("pool_limit_bytes"):
        env["CUPY_GPU_MEMORY_LIMIT"] = str(pf["pick"]["pool_limit_bytes"])
        rec["pool_limit_gib"] = pf["pick"]["pool_limit_gib"]
    cmd = [sys.executable, "-m", "namgpu.nest.twoway", "--parent-config", str(pcfg), "--nest-config", str(ncfg),
           "--parent-init", str(inits["parent"]), "--nest-init", str(inits["nest"]), "--case", case, "--case-dir",
           str(case_dir), "--source", source, "--hours", str(hours), "--out", str(out)]
    if a.two_way:
        cmd += ["--two-way", "--wgt", str(a.twoway_wgt)]
    code, _ = cli.run_logged(cmd, env, out / "run.log", sampler)
    _rm(inits, rec)
    tr = out / "twoway-receipt.json"
    if tr.is_file():
        rec["forecast"] = json.loads(tr.read_text())
        rec["forecast"].pop("trace", None) if not rec["forecast"].get("error") else None
    done("forecast", rc=code)
    rec["forecast_rc"] = code
    if code:
        return finish(code)

    if not a.no_maps:
        stage("maps")
        for role, cfgfile in (("parent", pcfg), ("nest", ncfg)):
            mc, _ = cli.run_logged([sys.executable, "-m", "namgpu.run_forecast", "--maps-only", "--case", case,
                                    "--case-dir", str(case_dir), "--out", str(out / role)],
                                   dict(env, NAMGPU_CONFIGFILE=str(cfgfile)), out / f"maps-{role}.log", sampler)
            if mc:
                rec["failed_stage"] = f"maps {role} rc {mc}"
        done("maps")
    if not a.no_sheets:
        stage("sheets")
        sheets_tool = cli.ROOT / "tools" / "nam_domain_sheets.py"
        run_dir = out / "nest"
        if sheets_tool.is_file():
            sc_cmd = [sys.executable, str(sheets_tool), "--run", str(run_dir), "--nest", source, "--out",
                      str(out / "sheets"), "--hours", f"1-{int(hours)}", "--start", f"{cycle:%Y-%m-%dT%H}",
                      "--cache", str(case_dir / "referee-cache")]
        else:
            sc_cmd = [sys.executable, "-m", "namgpu.io.io_referee_sheets", "nest", "--history",
                      *sorted(str(p) for p in run_dir.glob("wrfout_d01_*")), "--source", source, "--out",
                      str(out / "sheets"), "--leads", f"1-{int(hours)}", "--cache", str(case_dir / "referee-cache")]
        sc, _ = cli.run_logged(sc_cmd, dict(env, NAMGPU_CONFIGFILE=str(ncfg)), out / "sheets.log", sampler)
        done("sheets", rc=sc)
        if sc:
            rec["failed_stage"] = f"sheets rc {sc}"
    return finish(5 if rec.get("failed_stage") else 0)


def _rm(inits: dict, rec: dict) -> None:
    for d in inits.values():
        shutil.rmtree(d, ignore_errors=True)
    rec["init_dirs_removed"] = [str(d) for d in inits.values()]
