"""A dry rain window is an expected skip in the forecast's own map stage (io_maps), not a failed run.

Origin: woof-nam nest, 3 km 333 x 333 at 38.6N 98.9W, GFS 2025-03-14 12Z, 1 h: the forecast finished and the door
exited rc 1 at maps with "unexpected SKIPPED products: qpf_1h: ... dry window" (2026-11-05). CPU only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.io import io_maps  # noqa: E402

first = "wrfout_d01_2025-03-14_12:00:00"
assert io_maps._expected_skip("qpf_1h", "wrfout_d01_2025-03-14_13:00:00: dry window, no point at 0.01 in in 1 h", first)
assert io_maps._expected_skip("qpf_1h", "wrfout_d01_2025-03-14_12:00:00: no history frame 1 h earlier to difference", first)
assert not io_maps._expected_skip("qpf_1h", "wrfout_d01_2025-03-14_13:00:00: read failed", first)
print("test_maps_dry_window_skip: PASS")
