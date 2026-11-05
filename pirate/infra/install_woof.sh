#!/bin/bash
# Install this checkout's WOOF-NAM into a venv on a GPU machine. Usage: install_woof.sh VENV [CONSTRAINTS]
# The V100S cards need CuPy's CUDA 12 build: CUDA 13 no longer supports them. Upstream's pyproject asks for the CUDA 13 build,
# so the pin is applied to a temporary copy here and namgpu and pyproject stay as upstream has them.
# Needs python3-venv.
set -euo pipefail
VENV=$1; CONSTRAINTS=${2:-}; SRC=$(cd "$(dirname "$0")/../.." && pwd); T=$(mktemp -d); trap 'rm -rf $T' EXIT
cp -a $SRC/namgpu $SRC/pyproject.toml $T/; cp -a $SRC/README.md $SRC/LICENSE $SRC/NOTICE $T/
sed -i 's/cupy-cuda13x\[ctk\]/cupy-cuda12x[ctk]/' $T/pyproject.toml; grep -q cupy-cuda12x $T/pyproject.toml
python3 -m venv $VENV
$VENV/bin/pip install -q --upgrade pip
# NCCL is what a forecast on several cards exchanges its halo rows over; run_on_gpu.sh finds the library inside the venv.
$VENV/bin/pip install -q ${CONSTRAINTS:+-c $CONSTRAINTS} $T nvidia-nccl-cu12
$VENV/bin/python -c "import namgpu, cupy; print('namgpu', namgpu.__version__, 'cards', cupy.cuda.runtime.getDeviceCount())"
