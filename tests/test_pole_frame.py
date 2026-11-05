"""A domain that holds the pole (the 12 km parent) is drawn on the CONUS frame, not the whole longitude circle.

Origin: on a fresh pip install with gpuwm 2.9.12 the 12 km maps were drawn as a world map with the polar rows smeared
across Eurasia (2026-11-05); a nest away from the pole keeps its own frame. CPU only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.io import maps  # noqa: E402

# the parent: rotated lat-lon around 54N 106W reaching past the pole
jj, ii = np.meshgrid(np.linspace(-45, 45, 60), np.linspace(-50, 50, 70), indexing="ij")
lat0 = np.radians(54.0)
la, lo = np.radians(jj), np.radians(ii)
lat = np.degrees(np.arcsin(np.sin(la) * np.cos(lat0) + np.cos(la) * np.cos(lo) * np.sin(lat0)))
lon = -106.0 + np.degrees(np.arctan2(np.cos(la) * np.sin(lo),
                                     np.cos(la) * np.cos(lo) * np.cos(lat0) - np.sin(la) * np.sin(lat0)))
frame = maps.frame_bounds(lat, lon)
assert frame[1] - frame[0] >= 359.0, frame
assert maps.legacy_frame(frame) == tuple(float(v) for v in maps.BOUNDS.split(",")), maps.legacy_frame(frame)
src = (Path(maps.__file__)).read_text()
assert "frame = legacy_frame(frame)" in src
# a 3 km nest over Japan keeps its own frame
nlat, nlon = np.meshgrid(np.linspace(31, 40, 20), np.linspace(134, 145, 20), indexing="ij")
nf = maps.frame_bounds(nlat, nlon)
assert nf[1] - nf[0] < 359.0 and maps.legacy_frame(nf) is None, nf
print("test_pole_frame: PASS", frame)
