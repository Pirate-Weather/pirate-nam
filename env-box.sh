# source env-box.sh   (a dedicated GPU box with the tree at /work/nam-gpu and the venv at /work/nam-venv)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NAMGPU_NO_LEASE=1        # dedicated card: drv_gpu_lease is a no-op
export NAMGPU_ROOT="${NAMGPU_ROOT:-/work/nam-gpu}"
export PATH="${NAMGPU_VENV:-/work/nam-venv}/bin:$PATH"
export PYTHONPATH="$NAMGPU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ARWEN_SRC="$NAMGPU_ROOT/third_party/arwen"
export GPUWM_BRIDGES="${GPUWM_BRIDGES:-$(python -c 'import gpuwm, pathlib; print(pathlib.Path(gpuwm.__file__).parent / "libexec" / "bridges")' 2>/dev/null)}"
export NAMGPU_GEOG="${NAMGPU_GEOG:-/work/geog}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/work/nam-cupy-cache}"
