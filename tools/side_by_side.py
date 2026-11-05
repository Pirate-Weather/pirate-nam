"""Side-by-side montages: nam-gpu (ours) | operational NAM (archive, same cycle) | MRMS, from existing Rust-renderer PNGs.
Panels are only scaled to a common height and placed side by side, with a label strip; nothing is re-rendered.
usage: montage.py <maps_dir> <referee_png_dir> <out_dir> [leads...]"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

maps, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
args = sys.argv[4:]
start = datetime(2025, 3, 14, 12)
if "--start" in args:
    k = args.index("--start"); start = datetime.strptime(args[k + 1], "%Y-%m-%dT%H"); del args[k:k + 2]
leads = [int(x) for x in args] or [6, 12, 24]
NONAM = "no NAM: retired 2026-11-03 12Z" if start >= datetime(2026, 11, 3, 12) else "not in the referee set"
out.mkdir(parents=True, exist_ok=True)
FIELDS = {  # ours product -> NAM referee key
    "composite_reflectivity": "refc", "t2m": "t2m", "d2m": "d2m", "wind10": "ws10m", "qpf_1h": "qpf1h",
    "mslp": "mslp", "wind850": "ws850",
}
SHORT = {"composite_reflectivity": "refc", "wind10": "ws10m", "wind850": "ws850", "qpf_1h": "qpf1h"}
H = 900
try:
    font = ImageFont.load_default(size=28)
except TypeError:
    font = ImageFont.load_default()


def panel(p, label, missing="not in the referee set"):
    if p is not None and p.exists():
        im = Image.open(p).convert("RGB")
        im = im.resize((int(im.width * H / im.height), H))
    else:
        im = Image.new("RGB", (1200, H), (235, 235, 235))
        ImageDraw.Draw(im).text((40, H // 2), missing, fill=(90, 90, 90), font=font)
    strip = Image.new("RGB", (im.width, 44), (255, 255, 255))
    ImageDraw.Draw(strip).text((12, 6), label, fill=(0, 0, 0), font=font)
    c = Image.new("RGB", (im.width, H + 44), (255, 255, 255))
    c.paste(strip, (0, 0)); c.paste(im, (0, 44))
    return c


made = []
for f in leads:
    valid = start + timedelta(hours=f)
    vs = valid.strftime("%Y%m%d_%H%Mz")
    mrms = ref / f"f{f:02d}_observed_composite_reflectivity_{vs}.png"
    if not mrms.exists():   # no referee set (live case): our own map's MRMS panel, drawn by io-maps when MRMS has the hour
        mrms = maps / "composite_reflectivity" / f"namgpu_{start:%Y%m%d_%H}z_f{f:03d}_composite_reflectivity_observed.png"
    for prod, key in FIELDS.items():
        ours = maps / prod / f"namgpu_{start:%Y%m%d_%H}z_f{f:03d}_{prod}.png"
        if not ours.exists():
            continue
        nam = ref / f"f{f:02d}_cf_{key}_{vs}.png"
        ps = [panel(ours, f"nam-gpu (ours) +{f} h  {prod}"), panel(nam, f"operational NAM +{f} h (archive)", NONAM),
              panel(mrms, f"MRMS observed {valid:%Y-%m-%d %H}Z", "MRMS: hour not observed yet")]
        W = sum(p.width for p in ps) + 20 * (len(ps) - 1)
        m = Image.new("RGB", (W, H + 44), (255, 255, 255))
        x = 0
        for p in ps:
            m.paste(p, (x, 0)); x += p.width + 20
        o = out / f"f{f:03d}_{SHORT.get(prod, prod)}_ours_nam_mrms.png"
        m.save(o, optimize=True)
        made.append(o.name)
print("montages", len(made), made)
