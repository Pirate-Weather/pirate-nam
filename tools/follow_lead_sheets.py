"""Lead sheets of a storm-following WOOF-NAM nest: the weather maps at a lead beside the move history up to it.

    python tools/follow_lead_sheets.py --run <door --out> --sheets <sheets dir> --storm-id SID \
        [--leads 12,24,36,48] [--products composite_reflectivity,mslp] [--title-name "Typhoon Ampil"] --out <dir>

Top: the WOOF-NAM panels at fNNN as rw_wrfbatch rendered them (<sheets>/panels/woof_nam/<product>/), each on that
lead's own window (render law: the weather fields are the Rust renderer's PNGs, only placed here). Bottom: the
move history to that lead (namgpu.nest.track_chart: every window the nest has taken so far with the hour it took
over, the nest's own hourly MSLP-minimum track, the IBTrACS best track, track error and minimum MSLP against
lead). Output: <out>/fNNN_lead_sheet.png. Composed with Pillow.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from PIL import Image, ImageDraw
    from namgpu.io import best_track as bt
    from namgpu.nest import track_chart as tc

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="the door's --out folder (its run/ holds every frame)")
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--storm-id", required=True)
    ap.add_argument("--leads", default="12,24,36,48")
    ap.add_argument("--products", default="composite_reflectivity,mslp")
    ap.add_argument("--title-name", default=None)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "woof-nam" / "ibtracs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    hist = a.run / "run" if (a.run / "run").is_dir() else a.run
    fx = tc.run_fixes(hist)
    if not fx:
        raise SystemExit(f"follow_lead_sheets: {hist}: no history frames with MSLP")
    tr = bt.resolve(a.cache, a.storm_id, None, None)
    try:
        from namgpu.io import maps
        basemap = maps.basemap_dir(maps.renderer())
    except (SystemExit, Exception):  # noqa: BLE001  -- no renderer: the chart has no coastline
        basemap = None
    init = fx[0]["time"] - timedelta(hours=fx[0]["lead_h"])
    name = a.title_name or tc.track_label(tr)
    a.out.mkdir(parents=True, exist_ok=True)
    for L in (int(x) for x in a.leads.split(",")):
        sub = [f for f in fx if f["lead_h"] <= L]
        moves = sorted({f.get("land_hour", 0.0) for f in sub} - {0.0})
        tmp = a.out / f".f{L:03d}_track.png"
        tc.draw([("WOOF-NAM 3 km moving nest (hourly MSLP-minimum fix)", sub)], tr, tmp,
                f"Move history to +{L} h: {len(moves)} move(s)"
                + (" at +" + ", +".join(f"{m:g}" for m in moves) + " h" if moves else ""), basemap)
        chart = Image.open(tmp).convert("RGB")
        tmp.unlink()
        panels = []
        for p in a.products.split(","):
            hits = sorted((a.sheets / "panels" / "woof_nam" / p).glob(f"*_f{L:03d}_{p}.png"))
            if hits:
                panels.append(Image.open(hits[0]).convert("RGB"))
        W = chart.width
        row = None
        if panels:
            h = min(im.height for im in panels)
            ims = [im.resize((round(im.width * h / im.height), h)) for im in panels]
            rw = sum(im.width for im in ims)
            k = W / rw
            ims = [im.resize((max(1, round(im.width * k)), max(1, round(im.height * k)))) for im in ims]
            row = Image.new("RGB", (W, max(im.height for im in ims)), "white")
            x = 0
            for im in ims:
                row.paste(im, (x, 0))
                x += im.width
        top = 60
        H = top + (row.height if row else 0) + chart.height
        out = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(out)
        valid = init + timedelta(hours=L)
        d.text((30, 14), f"WOOF-NAM 3 km storm-following nest: {name}, init {init:%Y-%m-%d %HZ}, +{L} h "
                         f"(valid {valid:%Y-%m-%d %HZ})", fill="black", font=tc._font(34))
        if row:
            out.paste(row, (0, top))
        out.paste(chart, (0, top + (row.height if row else 0)))
        dst = a.out / f"f{L:03d}_lead_sheet.png"
        out.save(dst)
        print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
