#!/usr/bin/env python3
"""Checkout shim: the code is namgpu/run_init_case.py (shipped in the wheel; `woof-nam init`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from namgpu.run_init_case import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
