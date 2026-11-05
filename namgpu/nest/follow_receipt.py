"""The "follow" block of a moving run's nest-receipt.json, assembled from the records each piece writes.

    python -m namgpu.nest.follow_receipt --run <door out> [--write]     (--write: merge into <run>/nest-receipt.json)

Sources (each optional; a missing piece is named in "missing", never guessed):
  <run>/follow/ref-grid.json             the run (window.write_ref)
  <run>/follow/follow-state.json         the controller (namgpu.nest.follow): target, fixes, moves, holds
  <run>/follow/g*/PREPARED.json, <run>/gen-*/PREPARED.json   every preparation (namgpu.nest.prepare): walls per
                                         stage, alignment counts, single-core windows, card co-tenants at start/end
  <run>/**/move-receipt.json             every hand-off (the move step, namgpu.nest.move): overlay wall, carried arrays, the
                                         alignment checks (geometry, static) and the overlap mismatches
Totals: moves, preparations (used and spare), preparation wall by stage and in all, how much of it ran while a
forecast segment held the same card (co-tenant at start and end), hand-off wall, single-core windows over 30 s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _j(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def follow_block(run: Path) -> dict:
    run = Path(run)
    out, missing = {}, []
    ref = _j(run / "follow" / "ref-grid.json")
    if ref:
        out["run"] = {k: ref.get(k) for k in ("name", "start", "hours", "source", "boco_age_h", "landable_every_h")}
    else:
        missing.append("follow/ref-grid.json (window.write_ref)")
    state = _j(run / "follow" / "follow-state.json") or _j(run / "follow-state.json")
    if state:
        out["controller"] = {k: state.get(k) for k in ("target", "storm", "current", "generation") if k in state}
        out["fixes"] = state.get("frames") or state.get("fixes") or []
        out["moves"] = state.get("moves") or []
        out["holds"] = state.get("holds") or []
    else:
        missing.append("follow/follow-state.json (namgpu.nest.follow controller: fixes, moves)")
    preps = []
    for p in sorted(list(run.glob("follow/g*/PREPARED.json")) + list(run.glob("gen-*/PREPARED.json"))):
        r = _j(p) or {}
        preps.append({"dir": p.parent.name, "ok": r.get("ok"), "window": r.get("window"), "offset": r.get("offset"),
                      "land_hour": r.get("land_hour"), "walls_s": r.get("walls_s"), "wall_s": r.get("wall_s"),
                      "alignment": r.get("alignment"), "single_core_windows_over_30s":
                      r.get("single_core_windows_over_30s"), "vram_peak_gib": r.get("vram_peak_gib"),
                      "card": r.get("card"), "started": r.get("started"), "finished": r.get("finished"),
                      "card_cotenants_start": r.get("card_cotenants_start"),
                      "card_cotenants_end": r.get("card_cotenants_end"), "refused": r.get("refused"),
                      "cores_by_stage": {k: v.get("cores_mean") for k, v in (r.get("stages") or {}).items()}})
    out["preparations"] = preps
    if not preps:
        missing.append("PREPARED.json (namgpu.nest.prepare)")
    hand = []
    for p in sorted(list(run.rglob("move-receipt.json")) + list(run.rglob("HANDOFF.json"))):
        r = _j(p)
        if r is not None:
            keep = ("shift_dj_di", "overlay_s", "carried_n", "strip_cells", "interior_cells", "geometry", "static",
                    "reshaped_at_run", "refused", "wall_s", "monitors", "check")
            hand.append(dict({k: r[k] for k in keep if k in r}, file=str(p.relative_to(run))))
    out["handoffs"] = hand
    if not hand:
        missing.append("move-receipt.json (the move step namgpu.nest.move: overlay, alignment, shock numbers)")
    ok = [p for p in preps if p["ok"]]
    stages = {}
    for p in ok:
        for k, v in (p["walls_s"] or {}).items():
            stages[k] = round(stages.get(k, 0.0) + (v or 0.0), 2)
    beside = [p for p in ok if p.get("card_cotenants_start") or p.get("card_cotenants_end")]
    out["totals"] = {
        "moves": len(out.get("moves") or []) or len(hand) or None,
        "preparations": len(preps), "preparations_ok": len(ok),
        "prep_wall_s": round(sum(p["wall_s"] or 0.0 for p in ok), 2), "prep_wall_by_stage_s": stages,
        "prep_wall_max_s": max((p["wall_s"] or 0.0 for p in ok), default=None),
        "preparations_beside_a_forecast": len(beside),
        "handoff_overlay_s": round(sum(float(h.get("overlay_s") or h.get("wall_s") or 0.0) for h in hand), 2)
        if hand else None,
        "single_core_windows_over_30s": [w for p in preps for w in (p["single_core_windows_over_30s"] or [])],
    }
    out["missing"] = missing
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--write", action="store_true", help="merge the block into <run>/nest-receipt.json")
    a = ap.parse_args(argv)
    blk = follow_block(a.run)
    if a.write:
        rp = a.run / "nest-receipt.json"
        rec = _j(rp) or {}
        rec["follow"] = blk
        rp.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(blk["totals"], indent=1))
    if blk["missing"]:
        print("missing:", "; ".join(blk["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
