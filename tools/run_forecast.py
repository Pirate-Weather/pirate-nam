#!/usr/bin/env python3
"""Checkout shim: the code is namgpu/run_forecast.py (shipped in the wheel; `woof-nam run`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from namgpu.run_forecast import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
