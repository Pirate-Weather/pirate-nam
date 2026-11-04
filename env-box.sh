# source env-box.sh   (a dedicated GPU box; the tree is this folder, the venv NAMGPU_VENV or .venv in it)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WOOF_NAM_BOX=1          # our own box plumbing on (card lock hand-off, card lease); off for everyone else
export NAMGPU_NO_LEASE=1        # dedicated card: drv_gpu_lease is a no-op
export NAMGPU_ROOT="${NAMGPU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PATH="${NAMGPU_VENV:-$NAMGPU_ROOT/.venv}/bin:$PATH"
export PYTHONPATH="$NAMGPU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ARWEN_SRC="$NAMGPU_ROOT/namgpu/third_party/arwen"
export GPUWM_BRIDGES="${GPUWM_BRIDGES:-$(python -c 'import gpuwm, pathlib; print(pathlib.Path(gpuwm.__file__).parent / "libexec" / "bridges")' 2>/dev/null)}"
export NAMGPU_GEOG="${NAMGPU_GEOG:-$HOME/WPS_GEOG}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$NAMGPU_ROOT/scratch/cupy-cache}"
