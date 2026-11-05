"""The door's forecast stage with --follow (woof-nam nest --follow): segments, controller and preparer together.

The controller (namgpu/nest/follow.py, the tracker) runs in the door process. It reads every closed hourly frame,
commits each landable time into <out>/follow/control.json, asks the preparer (namgpu.nest.prepare, the preparer)
for the next windows ahead of time, and starts each segment (namgpu.run_forecast, the move step's hand-off):
  generation 0   the plain door start (init, TDFI) on the reference window (the door's own configfile);
  generation g   the preparation's fresh start of the new window at the landing time, with segment g-1's restart
                 set laid over it shifted by (dj, di) (--init-from / --move-from), alignment checked (--move-check).
Every segment runs to the end of the forecast and stops early, with its restart set written, at the landable time
where control.json says stop (namgpu/nest/segment.py). Segments write <out>/follow/seg-<g>/wrfout_d01_*; the door's
<out>/run/ holds one link per valid time to the generation that owns it, so sheets and scores read one series.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def prepared(prep: Path) -> dict:
    """What a finished preparation (READY.json) gives the next segment: the window's configfile, the saved fresh
    start and its record."""
    rec = json.loads((Path(prep) / "READY.json").read_text())
    paths = rec.get("paths") or {}
    cfg = paths.get("configfile") or rec.get("configfile")
    if not cfg:
        cands = sorted(Path(prep).glob("nam_configfile*"))
        cfg = str(cands[0]) if cands else None
    init = paths.get("init_dir") or rec.get("init_dir") or paths.get("fresh")
    if not cfg or not init:
        raise SystemExit(f"--follow: {Path(prep) / 'READY.json'} names no configfile or fresh start ({sorted(rec)})")
    return {"configfile": str(cfg), "init_dir": str(init), "record": rec}


def segment_cmd(door: dict, out: Path, generation: int, prep: Path | None):
    seg = out / f"seg-{generation}"
    cmd = [sys.executable, "-m", "namgpu.run_forecast", "--case", door["case"], "--case-dir", str(door["case_dir"]),
           "--source", door["source"], "--hours", f"{door['hours']:g}", "--out", str(seg), "--no-maps",
           "--follow-control", str(out / "control.json"), "--restart-dir", str(seg / "restart")]
    if door.get("parent_run"):
        cmd += ["--parent-run", str(door["parent_run"])]
    env = {"NAMGPU_CONFIGFILE": str(door["cfgfile"])}
    if generation > 0:
        p = prepared(prep)
        cmd += ["--init-from", p["init_dir"], "--move-from", str(out / f"seg-{generation - 1}" / "restart"),
                "--move-check"]
        env["NAMGPU_CONFIGFILE"] = p["configfile"]
    return cmd, env


def prepare_argv(door: dict):
    def argv(w, landing, d):
        return [sys.executable, "-m", "namgpu.nest.prepare", "--window", str(Path(d) / "window.json"), "--landing-s",
                f"{landing:.0f}", "--start", f"{door['cycle']:%Y%m%d%H}", "--hours", f"{door['hours']:g}",
                "--source", door["source"], "--out", str(d)]
    return argv


def history_view(out: Path, run_dir: Path) -> dict:
    """run_dir/wrfout_d01_<valid>: a link per valid time to the newest generation that wrote it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    best: dict = {}
    segs = sorted((p for p in out.glob("seg-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    for seg in segs:
        for f in seg.glob("wrfout_d01_*"):
            best[f.name] = f
    for name, f in best.items():
        link = run_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    return {"frames": len(best), "by_generation": {s.name: sum(1 for f in best.values() if f.parent == s)
                                                   for s in segs}}


def run(spec: dict, door: dict, log=print) -> tuple[int, dict]:
    """Returns (rc, the receipt's follow block)."""
    from namgpu.nest import follow, track
    import os
    from namgpu.nest import domain, window as winmod
    out = Path(door["out"])            # control.json, follow-state.json, seg-<g>/, gen-<key>/ (tracker/preparer layout)
    g = door["grid"]
    # the run record the preparer reads (the preparer, <out>/follow/ref-grid.json)
    winmod.write_ref(out, g, door["name"], door["static"], door["cfgfile"], f"{door['cycle']:%Y%m%d%H}",
                     float(door["hours"]), door["source"], door["case_dir"], int(door.get("boco_age_h") or 0),
                     parent_run=door.get("parent_run"), boco_interval_h=door.get("boco_interval_h"),
                     rrtm_rows=int(door["env"].get("NAMGPU_RRTM_ROWS") or 8))
    # generation 0: the door's own configfile plus follow_meta, so its history carries the FOLLOW_* attributes too
    gj = winmod.write_gen_meta(out, winmod.gen_meta(g, door["name"], 0, 0, 0, 0.0, [], f"{door['cycle']:%Y%m%d%H}"))
    cfg0 = winmod.gen_dir(out, 0) / Path(door["cfgfile"]).name
    cfg0.write_text(domain.render(Path(door["cfgfile"]).read_text(), {"follow_meta": str(gj)}, door["name"]))
    door = dict(door, cfgfile=cfg0)
    c = spec.get("config") or {}
    path_rows = [[(datetime.fromisoformat(t) - door["cycle"]).total_seconds(), la, lo]
                 for t, la, lo in (spec.get("path") or [])]
    extra = {k: c[k] for k in ("fix_error_km", "motion_error_kmh", "plan_hours", "prestage_moves", "min_move_km",
                               "land_every_h", "max_lead_minutes") if k in c}
    cfg = follow.FollowConfig(target=spec["target"], source=door["source"], hours=float(door["hours"]),
                              check_minutes=float(spec["check_minutes"]), dt_s=float(door["keys"]["dt"]),
                              tracker=dict(c.get("tracker") or {}), path_rows=path_rows,
                              core_margin_km=c.get("core_margin_km"), **extra)
    track.refuse_history(door.get("history_fields"), spec["target"])
    w0 = follow.Window.from_grid(g, lnsh=int(door.get("lnsh", 5)))
    env = dict(door["env"])

    class Launcher(follow.ProcessLauncher):
        """The live launcher, with the segment's configfile per generation (env) and a failed segment named."""

        def handoff(self, frm, to, L, generation):
            if self.segment is not None and self.segment.poll() is None:
                self.segment.wait()
            if self.segment is not None and self.segment.returncode not in (0, None):
                raise SystemExit(f"--follow: segment {generation - 1} ended rc {self.segment.returncode} (see "
                                 f"{out / f'seg-{generation - 1}.log'})")
            d = self.gen_dir(f"{to.key}@{L:.0f}")
            if generation > 0 and (d / "GEN.json").is_file():
                # the prepared window becomes this generation (its history's FOLLOW_* attributes, the preparer)
                moves = list(self.ctl.state["moves"]) + [dict(at_s=L, frm=frm.key, to=to.key, i0=to.i0, j0=to.j0)]
                winmod.commit_generation(d, generation, moves)
            cmd, e = segment_cmd(door, out, generation, d if generation > 0 else None)
            self.env = dict(env, **e)
            log(f"follow: segment {generation} starts at {L / 3600:g} h in window {to.key}")
            self.segment = self._start(cmd, out / f"seg-{generation}.log")
            self.env = env

    launcher = Launcher(out, prepare_argv(door), None, env=env)
    t_all = time.perf_counter()
    ctl = follow.Controller(cfg, w0, door["cycle"], tuple(spec["storm_start"]), launcher, out, log=log)
    launcher.ctl = ctl
    ctl.write_control()
    launcher.handoff(ctl.window0, ctl.window0, 0.0, 0)
    rc, err = 0, None
    try:
        for t in ctl.frames_left():
            gen = ctl.state["generation"]
            path = follow.frame_path(out / f"seg-{gen}", ctl.start + timedelta(seconds=t))
            t_wait = time.time()
            while not path.is_file():
                p = launcher.segment
                if p is not None and p.poll() is not None:
                    raise SystemExit(f"--follow: segment {gen} ended (rc {p.returncode}) before frame {path.name} "
                                     f"(see {out / f'seg-{gen}.log'})")
                if time.time() - t_wait > 3600:
                    raise SystemExit(f"--follow: no frame {path} for an hour")
                time.sleep(1.0)
            follow.wait_closed(path, timeout_s=600)
            follow.observe_frame(ctl, path, t, {})
        if launcher.segment is not None:
            launcher.segment.wait()
            if launcher.segment.returncode:
                rc, err = launcher.segment.returncode, f"last segment rc {launcher.segment.returncode}"
    except SystemExit as e:
        rc, err = 7, str(e)
        log(f"follow: {err}")
    finally:
        launcher.stop()
        _rm_fresh_starts(out)
    block = summarise(out, ctl, time.perf_counter() - t_all)
    block["error"] = err
    try:
        from namgpu.nest import follow_receipt       # the preparer: the per-piece records (preparations, hand-offs)
        block["assembled"] = follow_receipt.follow_block(out)
    except Exception as e:  # noqa: BLE001
        block["assembled"] = {"error": f"{type(e).__name__}: {e}"}
    block["history_view"] = history_view(out, Path(door["run_dir"]))
    (out / "moves.json").write_text(json.dumps(ctl.move_list(), indent=1))
    return rc, block


def _rm_fresh_starts(out: Path) -> None:
    """Remove every saved fresh start of a preparation in /dev/shm (host RAM) once the run is over, used or not
    (the preparer: the first whole 12 h run left five of them, one per prepared window)."""
    import shutil
    for p in list(Path(out).glob("gen-*/PREPARED.json")) + list(Path(out).glob("gen-*/READY.json")):
        try:
            d = (json.loads(p.read_text()).get("paths") or {}).get("init_dir")
        except (OSError, ValueError):
            continue
        if d and str(d).startswith("/dev/shm/"):
            shutil.rmtree(d, ignore_errors=True)


def summarise(out: Path, ctl, wall_s: float) -> dict:
    """Fixes per frame, moves, holds, per-segment step time, hand-off overlay, alignment checks and shock numbers,
    and the preparation records."""
    segs = []
    for seg in sorted((p for p in out.glob("seg-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1])):
        rp = seg / "forecast-receipt.json"
        r = json.loads(rp.read_text()) if rp.is_file() else {}
        mv = r.get("move") or {}
        ov = mv.get("overlay") or {}
        segs.append({"segment": seg.name, "steps_done": r.get("steps_done"), "ms_per_step": r.get("ms_per_step"),
                     "run_s": r.get("run_s"), "init_s": r.get("init_s"), "tdfi_s": r.get("tdfi_s"),
                     "nonfinite_end": r.get("nonfinite_end"), "error": r.get("error"),
                     "shift_dj_di": mv.get("shift_dj_di"), "overlay": {k: v for k, v in ov.items()
                                                                       if k not in ("carried", "globals_from_set")},
                     "check": mv.get("check"), "monitors": mv.get("monitors"),
                     "follow_segment": r.get("follow_segment"), "stats_end": r.get("stats_end"),
                     "card_used_gib": r.get("card_used_gib")})
    preps = {}
    for d in out.glob("gen-*"):
        rp = d / "READY.json"
        if rp.is_file():
            r = json.loads(rp.read_text())
            preps[d.name] = {k: r[k] for k in ("stages", "walls", "wall_s", "alignment", "single_core_windows",
                                               "waited_s") if k in r}
    st = ctl.state
    fixes = []
    for f in st["frames"]:
        fx = f.get("fix") or {}
        fixes.append({"t_h": round(f["t"] / 3600.0, 2), "utc": f["utc"], "window": f["window"],
                      "generation": f.get("generation"), "room_km": f.get("room_km"),
                      **{k: fx[k] for k in ("ok", "lat", "lon", "mslp_hpa", "min_mslp_hpa", "vort850", "method",
                                            "methods", "reason", "intensity") if k in fx}})
    return {"target": ctl.cfg.target, "source": ctl.cfg.source, "landing_every_h": ctl.cfg.land_s / 3600.0,
            "window0": ctl.window0.as_dict(), "moves": st["moves"], "holds": st["holds"],
            "committed": st["committed"], "fixes": fixes, "segments": segs, "preparations": preps,
            "events": st["events"][-300:], "wall_s": round(wall_s, 1),
            "ms_per_step_by_segment": [s["ms_per_step"] for s in segs]}
