"""check-phys-noah-tbnd: ctx path of phys-noah-tbnd against a host transcription of TBND.

Full size 954 x 835 x 4. Binds STC, TBOT (state), ZSOIL (absent, so registered), and ZBOT from the
GENPARM host table, then compares TBND from run(ctx) with a numpy float32 transcription of the Fortran
TBND (module_LS_NOAHLSM.F90:4945-4985, called as at :2890, :2958, :2989). Also checks that a missing
TBOT raises instead of defaulting.
"""
import sys
from types import SimpleNamespace

import numpy as np

JM, IM, NSOIL = 835, 954, 4
ZBOT_EXPECT = -8.0  # GENPARM.TBL ZBOT_DATA


class _State:
    def __init__(self):
        self.fields = {}

    def add(self, name, shape, dtype="float32"):
        import cupy as cp
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def host_tbnd(stc, tbot, zsoil, zbot):
    out = np.zeros_like(stc)
    for k in range(NSOIL):
        tu = stc[k]
        tb = stc[k + 1] if k < NSOIL - 1 else tbot
        zup = np.float32(0.0) if k == 0 else zsoil[k - 1]
        zb = (np.float32(2.0) * np.float32(zbot) - zsoil[k]) if k == NSOIL - 1 else zsoil[k + 1]
        out[k] = tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb)
    return out


def main():
    import cupy as cp
    from namgpu.physics.noah import phys_noah_parm_tables as parm
    from namgpu.physics.noah import phys_noah_tbnd as tb

    rng = np.random.default_rng(7)
    stc_h = rng.uniform(250.0, 285.0, size=(NSOIL, JM, IM)).astype(np.float32)
    tbot_h = rng.uniform(265.0, 285.0, size=(JM, IM)).astype(np.float32)
    host = SimpleNamespace(gen={"ZBOT_DATA": ZBOT_EXPECT})
    st = _State()
    st.fields["STC"] = cp.asarray(stc_h)
    st.fields["TBOT"] = cp.asarray(tbot_h)
    ctx = SimpleNamespace(st=st, tables={parm.KEY_HOST: host})

    out = tb.run(ctx)["TBND"]
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(out)
    zsoil = np.array(tb.ZSOIL_NAM, dtype=np.float32)
    want = host_tbnd(stc_h, tbot_h, zsoil, ZBOT_EXPECT)
    finite = bool(np.isfinite(got).all())
    diff = np.abs(got - want)
    print(f"finite={finite} max_abs_diff={float(diff.max()):.3e} "
          f"n_diff_gt_1e-5={int((diff > 1e-5).sum())} of {got.size}")
    alt = host_tbnd(stc_h, tbot_h, zsoil, -2.0)
    print(f"bottom layer differs between ZBOT -8 and -2 by max {float(np.abs(want[-1]-alt[-1]).max()):.3e} K "
          f"(ZBOT now read from the table)")

    # Missing TBOT must raise, not default.
    st2 = _State()
    st2.fields["STC"] = cp.asarray(stc_h)
    try:
        tb.run(SimpleNamespace(st=st2, tables={parm.KEY_HOST: host}))
        print("MISSING TBOT: no error (FAIL)")
        return 1
    except RuntimeError as e:
        print(f"MISSING TBOT raised: {e}")

    ok = finite and int((diff > 1e-5).sum()) == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    from namgpu.steps.drv_gpu_lease import gpu_lease
    with gpu_lease(vram_mib=4096, est_s=600, cards=1, what="phys-noah-tbnd ctx-path check"):
        sys.exit(main())
