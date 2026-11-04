"""Record every kernel module (unit) fetched during the physics-A real-state run (init + N calls)."""
import sys, json, atexit
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from namgpu import kernels
USED = set()
_get, _mod = kernels.get, kernels.module
def get(unit, kernel):
    USED.add(unit); return _get(unit, kernel)
def module(unit):
    USED.add(unit); return _mod(unit)
kernels.get, kernels.module = get, module
@atexit.register
def _dump():
    (ROOT / "logs/physa/kernels_used.json").write_text(json.dumps(sorted(USED), indent=0))
    print("KERNEL UNITS USED:", len(USED))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
sys.exit(R.main())
