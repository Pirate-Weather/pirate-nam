"""Restart-shock check of a moving WOOF-NAM nest: the numbers each segment recorded around every move, as a table
and a time-series chart (an analysis chart, matplotlib; not a weather field).

    python tools/follow_shock.py --run runs/P1-ampil-follow [--out runs/P1-ampil-follow/shock]

Reads <run>/nest-receipt.json (follow.segments[*].monitors, the move step's namgpu/nest/move.py Monitor): every step for
the 30 min before each landable time and the 60 min after each move, the window-mean |dPD/dt| (Pa/s) and the max
|W| in the strip (new ground), along the seam (overlap cells next to the strip) and in the interior. A move is
clean when its mean |dPD/dt| falls back to the level of the 30 min before within the hour and the seam W stays at
the interior's level (no wave running in from the seam).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect(rec: dict) -> dict:
    segs = (rec.get("follow") or {}).get("segments") or []
    before, after = [], []
    for s in segs:
        steps = int(s.get("steps_done") or 0)
        for m in s.get("monitors") or []:
            if int(m.get("from_step", 0)) > steps:
                continue                       # a landing the segment never reached (it stopped earlier)
            row = dict(m, segment=s["segment"], shift=s.get("shift_dj_di"))
            (after if m["label"].startswith("after move") else before).append(row)
    return {"before": before, "after": after}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rec = json.loads((a.run / "nest-receipt.json").read_text())
    out = a.out or (a.run / "shock")
    out.mkdir(parents=True, exist_ok=True)
    c = collect(rec)
    ref = [m["mean_dpddt_pa_s"] for m in c["before"] if m.get("mean_dpddt_pa_s")]
    ref_mean = sum(ref) / len(ref) if ref else None
    lines = ["# Restart-shock check, " + a.run.name, "",
             f"Reference (30 min before each landable time, {len(ref)} windows): mean |dPD/dt| "
             f"{ref_mean:.4f} Pa/s" if ref_mean else "Reference: none", "",
             "| move (segment, shift dj/di) | mean dPD/dt 1 / 5 / 15 / 30 / 60 min, Pa/s | x reference at 60 min | "
             "max W strip / seam / interior at 1 min, m/s | max W seam over the hour |", "|---|---|---|---|---|"]
    for m in c["after"]:
        at = [m.get(k) for k in ("at_1min", "at_5min", "at_15min", "at_30min", "at_60min")]
        dp = " / ".join(f"{x[0]:.3f}" if x else "-" for x in at)
        x60 = f"{at[4][0] / ref_mean:.1f}" if at[4] and ref_mean else "-"
        w1 = " / ".join(f"{v:.1f}" for v in (at[0] or [0, 0, 0, 0])[1:])
        lines.append(f"| {m['segment']} {m.get('shift')} | {dp} | {x60} | {w1} | {m.get('max_w_seam')} |")
    (out / "SHOCK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
    for m in c["before"]:
        ser = m.get("series_every_6_steps") or []
        t = [(-30 + 6 * k * m["dt"] / 60.0) for k in range(len(ser))]
        ax.plot(t, [v[0] for v in ser], color="0.6", lw=0.8)
        bx.plot(t, [v[2] for v in ser], color="0.6", lw=0.8)
    for k, m in enumerate(c["after"]):
        ser = m.get("series_every_6_steps") or []
        t = [6 * j * m["dt"] / 60.0 for j in range(len(ser))]
        lab = f"move into {m['segment']} (dj, di {m.get('shift')})"
        ax.plot(t, [v[0] for v in ser], lw=1.4, label=lab)
        bx.plot(t, [v[2] for v in ser], lw=1.4, label=lab + ", seam")
        bx.plot(t, [v[1] for v in ser], lw=0.8, ls="--", color=bx.lines[-1].get_color())
    ax.set_yscale("log")
    ax.set_xlabel("minutes from the landing (grey: the 30 min before each landable time)")
    ax.set_ylabel("window-mean |dPD/dt| (Pa/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    bx.set_xlabel("minutes from the landing")
    bx.set_ylabel("max |W| (m/s): seam solid, strip dashed")
    bx.grid(alpha=0.3)
    fig.suptitle(f"WOOF-NAM moving nest {a.run.name}: pressure tendency and vertical motion around each move", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "shock.png", dpi=110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
