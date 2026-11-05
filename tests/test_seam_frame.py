"""A frame across 180 is not drawn by the installed gpuwm 2.9.12 renderer, which smears it; other frames are.

Origin: the published gpuwm 2.9.12 rw_wrfbatch rendered a CONUS window moved across 180 with every row smeared east-west,
while the same window unmoved rendered correctly (2026-11-05). The pinned build (tools/renderer patch 01) and a renderer
named in NAMGPU_RENDERER are trusted. CPU only."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.io import maps  # noqa: E402

os.environ.pop(maps.RENDERER_ENV, None)
installed = Path("/nonexistent/gpuwm/libexec/bridges/rw_wrfbatch")
across, conus = (146.0, 211.0, 20.0, 60.0), (-126.0, -66.0, 22.0, 52.0)
maps._gpuwm_version = lambda: (2, 9, 12)
assert maps.seam_defect(installed, conus) is None
why = maps.seam_defect(installed, across)
assert why and "crosses 180" in why and maps.RENDERER_ENV in why, why
assert maps.seam_defect(maps.PINNED_RENDERER, across) is None
os.environ[maps.RENDERER_ENV] = "/some/patched/rw_wrfbatch"
assert maps.seam_defect(installed, across) is None
os.environ.pop(maps.RENDERER_ENV)
maps._gpuwm_version = lambda: (2, 9, 13)
assert maps.seam_defect(installed, across) is None
src = Path(maps.__file__).read_text()
assert "bad = seam_defect(rw, fr)" in src
print("test_seam_frame: PASS")
