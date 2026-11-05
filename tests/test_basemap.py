"""The wheel carries the map basemap, and a renderer with none of its own is pointed at it (no refusal).

Origin: the released gpuwm wheel ships rw_wrfbatch with no assets/basemap beside it; woof-nam's maps step then
refused the renderer and every `pip install woof-nam` run ended rc 1 at maps after a finished forecast.
CPU only, no gpuwm needed. Run: python tests/test_basemap.py
"""
import os
import sys
import tempfile
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from namgpu.io import maps  # noqa: E402


def main() -> int:
    for k in ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR"):
        os.environ.pop(k, None)
    root = maps.PACKAGED_BASEMAP
    for f in ("natural_earth_10m/ne_10m_coastline.shp", "natural_earth_10m/ne_10m_admin_1_states_provinces_lines.shp",
              "natural_earth_110m/ne_110m_coastline.shp", "us_counties_5m/cb_2023_us_county_5m.shp"):
        assert (root / f).is_file(), f"packaged basemap missing {f} under {root}"
    with tempfile.TemporaryDirectory() as d:
        rw = Path(d) / "bin" / "rw_wrfbatch"          # a renderer with no assets/basemap beside it
        rw.parent.mkdir()
        rw.write_text("")
        got = maps.ensure_basemap(rw)
        assert got is not None, "ensure_basemap found nothing"
        assert os.environ.get("RUSTWX_BASEMAP_DIR") == str(got), "RUSTWX_BASEMAP_DIR not pinned"
        assert maps.basemap_dir(rw) is not None, "renderer search order still finds no basemap"
        env = maps.renderer_env(rw)
        assert env["RUSTWX_BASEMAP_DIR"] == str(got)
    # the 12 km parent's own frame (holds the pole: whole circle from its centre) -> the CONUS frame
    assert maps.legacy_frame((73.0, 433.0, -8.0, 89.9)) == (-126.0, -66.0, 22.0, 52.0)
    assert maps.legacy_frame((170.0, 200.0, 40.0, 60.0)) is None          # a nest across 180 off CONUS
    # nothing anywhere: one warning, None, no exception (the maps are drawn without coastlines)
    os.environ.pop("RUSTWX_BASEMAP_DIR", None)
    keep = maps.PACKAGED_BASEMAP
    with tempfile.TemporaryDirectory() as d:
        maps.PACKAGED_BASEMAP = Path(d) / "none"
        rw = Path(d) / "rw_wrfbatch"
        try:
            none = maps.ensure_basemap(rw)
        finally:
            maps.PACKAGED_BASEMAP = keep
        assert none is None or "rustwx" in str(none), f"expected no basemap, got {none}"
    print(f"test_basemap: PASS (basemap {got})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
