#!/usr/bin/env python3
"""Checkout shim: the code is namgpu/io/history_merge.py (shipped in the wheel)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from namgpu.io.history_merge import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
