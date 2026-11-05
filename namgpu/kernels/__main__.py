"""python -m namgpu.kernels warm [--workers N]: parallel NVRTC warm-up of every kernel module (kernels.warm)."""
import argparse
import json

from namgpu.kernels import warm

ap = argparse.ArgumentParser(description="parallel NVRTC warm-up of every namgpu kernel module")
ap.add_argument("cmd", choices=["warm"])
ap.add_argument("--workers", type=int, default=None)
a = ap.parse_args()
print(json.dumps(warm(a.workers), default=str))
