#!/usr/bin/env python3
"""Checkout shim: the code is namgpu/io/hist_compare.py (woof-nam compare, shipped in the wheel)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from namgpu.io.hist_compare import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
