"""Guard for the review finding: a sliced (non-contiguous) H strip must raise, not be read by address."""
import sys
from types import SimpleNamespace as NS

import cupy as cp

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from namgpu.boundary import bnd_bocoh_update as unit  # noqa: E402

IM, JM, LM, LNSH = 954, 835, 60, 5
S_SHAPE = (2, LNSH, IM)
W_SHAPE = (2, JM, LNSH)

# a padded S strip sliced back to IM columns: float32 and the right element count, not C-contiguous
padded = cp.zeros((2, LNSH, IM + 2), cp.float32)
sliced = padded[:, :, :IM]
assert not sliced.flags.c_contiguous and sliced.size == 2 * LNSH * IM


def dense(shape):
    return cp.zeros(shape, cp.float32)


v = NS(south=sliced, north=dense(S_SHAPE), west=dense(W_SHAPE), east=dense(W_SHAPE))
bc = NS(h2d=[v], h3d=[], h4d=[])
ctx = NS(cfg=NS(lnsh=LNSH), grid=NS(im=IM, jm=JM, lm=LM), st=None, stream=None,
         dt=25.0, bc=bc)
try:
    unit.run(ctx)
except ValueError as e:
    msg = str(e)
    assert "C-contiguous" in msg and "south h2d[0]" in msg, msg
    print("GUARD OK:", msg[:160])
else:
    raise SystemExit("GUARD FAIL: sliced strip was accepted")
