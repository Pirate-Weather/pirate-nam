#!/usr/bin/env python3
"""Checkout shim: the code is namgpu/licence_check.py (shipped in the wheel; `woof-nam licence-check`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from namgpu.licence_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
