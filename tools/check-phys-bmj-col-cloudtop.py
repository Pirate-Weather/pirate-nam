"""Done-rule check for phys-bmj-col-cloudtop after the review fixes.

    source env.sh
    python tools/check-phys-bmj-col-cloudtop.py

Full size (954 x 835 x 60). Only the chain's inputs are built (the full NAM state is about 14 GiB on
the shared card). The chain runs in order: phys-bmj-init tables, phys-bmj-col-prep, then
phys-bmj-col-cloudbase (which supplies BMJ_PSK, BMJ_RXNERSK, BMJ_THESP), then phys-bmj-col-cloudtop on
those real outputs. Checks: no NaN/Inf in the cloud-top outputs. Then one refusal check: with
BMJ_PSK removed from ctx.st, cloudtop must raise instead of building a stand-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.physics.bmj import (  # noqa: E402
    phys_bmj_col_cloudbase as CB, phys_bmj_col_cloudtop as CT, phys_bmj_col_prep as PR, phys_bmj_init as BI)


class _Fields:
    def __init__(self):
        self.fields = {}

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]

    def __getitem__(self, name):
        return self.fields[name]


def main():
    cfg = config.load()
    g = gridmod.build(cfg)
    IM, JM, LM = g.im, g.jm, g.lm
    rng = np.random.default_rng(3)
    prof = np.linspace(2.0e4, 1.0e5, LM + 1, dtype=np.float32)
    pint = (prof[:, None, None] * (1.0 + 0.01 * rng.random((LM + 1, JM, IM)))).astype(np.float32)
    pint = np.maximum.accumulate(pint, axis=0)
    st = _Fields()
    st.fields["PINT"] = cp.asarray(pint)
    st.fields["PD"] = cp.asarray((pint[-1] - pint[0]).astype(np.float32))
    st.fields["T"] = cp.asarray((220.0 + 85.0 * rng.random((LM, JM, IM))).astype(np.float32))
    st.fields["Q"] = cp.asarray((1.0e-6 + 0.02 * rng.random((LM, JM, IM)) ** 2).astype(np.float32))
    st.fields["XLAND"] = cp.asarray(rng.integers(1, 3, size=(JM, IM)).astype(np.float32))
    st.fields["CLDEFI"] = cp.asarray((0.5 + 0.5 * rng.random((JM, IM))).astype(np.float32))
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.tables = {}
    BI.run(ctx)
    PR.run(ctx)
    CB.run(ctx)
    CT.run(ctx)
    ctx.stream.synchronize()
    outs = {k: st[k] for k in CT.OUT_2D + CT.OUT_2D_I32 + CT.OUT_3D}
    bad = {k: int(cp.count_nonzero(~cp.isfinite(v))) for k, v in outs.items() if v.dtype.kind == "f"}
    bad = {k: v for k, v in bad.items() if v}
    print("chain full size:", (LM, JM, IM), "non-finite outputs:", bad or 0)
    psk = st.fields.pop("BMJ_PSK")
    try:
        CT.run(ctx)
    except RuntimeError as e:
        print("refusal without BMJ_PSK ok:", str(e)[:120])
    else:
        raise SystemExit("cloudtop ran without BMJ_PSK (stand-in path still present)")
    finally:
        st.fields["BMJ_PSK"] = psk
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
