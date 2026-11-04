# source env.sh   (from the repository root)
export NAMGPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$NAMGPU_ROOT/.venv" ] && export PATH="$NAMGPU_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$NAMGPU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ARWEN_SRC="$NAMGPU_ROOT/third_party/arwen"
# grib2_inventory, grib2_dump, rw_wrfbatch, libnetcdf_writer.so: the copy inside the installed gpuwm wheel
export GPUWM_BRIDGES="${GPUWM_BRIDGES:-$(python3 -c 'import pathlib, gpuwm; print(pathlib.Path(gpuwm.__file__).parent / "libexec" / "bridges")' 2>/dev/null || echo "$HOME/.gpuwm/bridges")}"
export NAMGPU_GEOG="${NAMGPU_GEOG:-$HOME/WPS_GEOG}"                 # WPS geog set with topo_gmted2010_30s
# Optional, for the check tools only: a clone of the NAM source (nam.v4.2.11).
# export NAM_SRC=/path/to/NAM; export NMMB_SRC="$NAM_SRC/sorc/nam_nems_nmmb.fd/src/atmos"
