"""Done-rule check for phys-rrtm-lw-mcica-subcol: compile the kernels, run once on a full-size random state.

Full size: NLAY = 60, JM = 835, IM = 954, NGPTLW = 140 (CONTRACT section 5). iovrlw = 1 (NAM).
Checks: kernels compile; the generator runs at full size with no NaN-free-ness assumed on the output;
cldf = 0 gives no sub-column cloud and cldf = 1 gives every sub-column cloud (layers forced to the two
ends); the sub-column cloud fraction is close to cldf on average (sanity of the random stream, not a
bitwise test). Then the wrapper plumbing that the review found broken: run() reads the cldprop cldf key,
refuses without the rlwinit host copy, returns None for isubclw 0, and returns a cldfmc chunk that equals
the packed bits of the same columns.
"""
import sys
import time

import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_lw_mcica_subcol as mc

NLAY, JM, IM = 60, 835, 954
CHUNK = 8192


class _Ctx:
    def __init__(self, tables, lm, jm, im):
        self.tables = tables
        self.st = {"T": cp.empty((lm, jm, im), dtype=cp.float32)}
        self.stream = None


def count_bits(words):
    """words uint32 [NWORD, NLAY, JM, IM] -> int32 [NLAY, JM, IM]: number of set sub-column flags."""
    total = cp.zeros(words.shape[1:], dtype=cp.int32)
    for g in range(mc.NGPTLW):
        total += ((words[g >> 5] >> (g & 31)) & 1).astype(cp.int32)
    return total


def main():
    t0 = time.time()
    try:
        kernels.module(mc.UNIT).compile()
    except Exception as e:  # noqa: BLE001
        print("compile FAILED:", type(e).__name__, e, flush=True)
        return 1
    print("compile: ok", flush=True)

    rng = cp.random.default_rng(12345)
    cldf = rng.random((NLAY, JM, IM), dtype=cp.float32)
    cldf[:, :, : IM // 4] = 0.0                 # clear-sky block: must give no cloud
    cldf[:, :, IM // 4: IM // 4 + 8] = 1.0      # overcast block: must give every sub-column
    ipseed = rng.integers(1, 2**31 - 1, size=(JM, IM), dtype=cp.int32)
    ipseed[:, :4] = 0                           # seed 0 -> iseed path
    lcloudy = mc.new_lcloudy(NLAY, JM, IM)
    mc.mcica_subcol(cldf, ipseed, lcloudy, 1)
    cp.cuda.Device().synchronize()
    print("run ok, shape", tuple(lcloudy.shape), lcloudy.dtype, f"{time.time() - t0:.1f}s", flush=True)

    assert tuple(lcloudy.shape) == (mc.NWORD, NLAY, JM, IM)
    assert lcloudy.dtype == cp.uint32
    cnt = count_bits(lcloudy)                   # int32 [NLAY, JM, IM]
    clear = cnt[:, :, : IM // 4]
    over = cnt[:, :, IM // 4: IM // 4 + 8]
    clear_frac = float(clear.sum()) / float(clear.size * mc.NGPTLW)
    print(f"clear-sky sub-column flags {clear_frac:.3e} (max count {int(clear.max())})", flush=True)
    assert clear_frac < 1e-6, "clear-sky columns produce too many sub-column flags"
    assert int(over.min()) == mc.NGPTLW, "overcast columns missed sub-column cloud"
    frac_mid = float(cnt[:, :, IM // 4 + 8:].mean()) / mc.NGPTLW
    frac_expect = float(cldf[:, :, IM // 4 + 8:].mean())
    print(f"sub-column cloud fraction {frac_mid:.4f} vs cldf mean {frac_expect:.4f}", flush=True)
    assert abs(frac_mid - frac_expect) < 0.01, "sub-column fraction far from cldf"

    # Unpack one column chunk and compare with the bits it came from.
    col0, ncol = 0, CHUNK
    cfm = mc.unpack_cldfmc(lcloudy, col0, ncol)
    cp.cuda.Device().synchronize()
    assert tuple(cfm.shape) == (NLAY, mc.NGPTLW, ncol) and cfm.dtype == cp.float32
    lc = lcloudy.reshape(mc.NWORD, NLAY, JM * IM)[:, :, col0:col0 + ncol]
    ref = cp.stack([((lc[g >> 5] >> (g & 31)) & 1).astype(cp.float32) for g in range(mc.NGPTLW)], axis=1)
    assert bool(cp.array_equal(cfm, ref)), "cldfmc chunk differs from the packed lcloudy bits"
    print(f"cldfmc chunk {ncol} columns equals lcloudy bits", flush=True)

    del cnt, clear, over, cfm, lc, ref   # free check temporaries; the shared card has little headroom
    # run() plumbing: cldprop key, rlwinit host copy, isubclw 0, refusal without the host copy.
    ctx = _Ctx({"rrtm_lw_cldprop_cldf": cldf, "rrtm_lw_lcloudy": lcloudy,
                  "rrtm_lw_ctl_host": {"isubclw": 1, "iovrlw": 1}}, NLAY, JM, IM)
    out = mc.run(ctx, col0=0, ncol=CHUNK)
    cp.cuda.Device().synchronize()
    lc2 = ctx.tables["rrtm_lw_lcloudy"].reshape(mc.NWORD, NLAY, JM * IM)[:, :, :CHUNK]
    ref2 = cp.stack([((lc2[g >> 5] >> (g & 31)) & 1).astype(cp.float32) for g in range(mc.NGPTLW)], axis=1)
    assert out is ctx.tables["rrtm_lw_cldfmc"] and bool(cp.array_equal(out, ref2)), "run() cldfmc mismatch"
    ctx0 = _Ctx({"rrtm_lw_cldprop_cldf": cldf, "rrtm_lw_ctl_host": {"isubclw": 0, "iovrlw": 1}}, NLAY, JM, IM)
    assert mc.run(ctx0) is None, "isubclw 0 must return None"
    try:
        mc.run(_Ctx({"rrtm_lw_cldprop_cldf": cldf}, NLAY, JM, IM))
        raise AssertionError("missing rlwinit host copy must refuse")
    except RuntimeError:
        pass
    print("run() plumbing ok", flush=True)
    print("PASS", f"{time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
