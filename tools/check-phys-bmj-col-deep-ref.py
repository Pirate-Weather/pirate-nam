"""Done-rule check for phys-bmj-col-deep-ref after the review fixes.

    source ~/nam-gpu/env.sh
    python tools/check-phys-bmj-col-deep-ref.py

Full size (954 x 835 x 60). The inputs are written here, explicitly, for the upstream fields the
chain would supply, so the kernel runs on them. Checks: no NaN/Inf in the outputs; the thin-cloud,
no-CAPE and ltop<1 columns take the quick exit (DEEP=0, PLUME=0, SHALLOW=0, cldefi replaced, lbot=0,
ltop=LM); every deep column has lb >= 3. Then one refusal check: run() with the upstream fields
absent must raise instead of filling inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.physics.bmj import phys_bmj_col_deep_ref as U  # noqa: E402


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
    st = _Fields()
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx.tables = {}

    prof = np.linspace(2.0e2, 1.0e5, LM, dtype=np.float32)
    pm = np.maximum.accumulate(
        (prof[:, None, None] * (1.0 + 0.01 * rng.standard_normal((LM, JM, IM)))).astype(np.float32), axis=0)
    psfc = (1.0e5 * (1.0 + 0.01 * rng.random((JM, IM)))).astype(np.float32)
    exner = (pm / 1.0e5) ** (287.04 / 1004.6)
    lbot = rng.integers(LM // 2, LM - 1, size=(JM, IM)).astype(np.int32) + 1
    ltop = rng.integers(1, LM // 4, size=(JM, IM)).astype(np.int32)
    kind = rng.integers(0, 4, size=(JM, IM))          # 0 deep, 1 thin, 2 no cape, 3 ltop=0
    thin = kind == 1
    ltop = np.where(thin, lbot - 1, ltop).astype(np.int32)   # ltop > lbot-2: quick exit
    ltop = np.where(kind == 3, 0, ltop).astype(np.int32)     # ltop < 1: quick exit
    capecnv = np.where(kind == 2, 0.0, 500.0 + 1000.0 * rng.random((JM, IM))).astype(np.float32)
    jj, ii = np.meshgrid(np.arange(JM), np.arange(IM), indexing="ij")
    lb_idx = np.clip(lbot - 1, 0, LM - 1)
    lt_idx = np.clip(ltop - 1, 0, LM - 1)
    pbot = pm[lb_idx, jj, ii].astype(np.float32)
    ptop = pm[lt_idx, jj, ii].astype(np.float32)
    ins = {
        "T": (250.0 + 40.0 * rng.random((LM, JM, IM))).astype(np.float32),
        "BMJ_QCOL": (1.0e-3 + 1.0e-2 * rng.random((LM, JM, IM))).astype(np.float32),
        "BMJ_PRSMID": pm,
        "BMJ_RXNER": (1.0 / exner).astype(np.float32),
        "BMJ_THES": (320.0 + 20.0 * rng.random((LM, JM, IM))).astype(np.float32),
        "BMJ_CT_LBOT": lbot, "BMJ_CT_LTOP": ltop, "BMJ_CT_PBOT": pbot, "BMJ_CT_PTOP": ptop,
        "BMJ_CT_CAPE": capecnv, "SM": rng.integers(0, 2, size=(JM, IM)).astype(np.float32),
        "CLDEFI": (0.5 + 0.3 * rng.random((JM, IM))).astype(np.float32),
    }
    pint = np.zeros((LM + 1, JM, IM), dtype=np.float32)
    pint[:LM] = pm
    pint[LM] = psfc
    st.fields.update({k: cp.asarray(v) for k, v in ins.items()})
    st.fields["PINT"] = cp.asarray(pint)
    cldefi_in = ins["CLDEFI"].copy()
    tab = {
        "bmj_sthe": (0.5 + rng.random(U.ITB)).astype(np.float32),
        "bmj_the0": np.linspace(200.0, 400.0, U.ITB, dtype=np.float32),
        "bmj_ttbl": (200.0 + 150.0 * rng.random((U.ITB, U.JTB))).astype(np.float32),
        "bmj_stheq": (0.5 + rng.random(U.ITBQ)).astype(np.float32),
        "bmj_the0q": np.linspace(200.0, 400.0, U.ITBQ, dtype=np.float32),
        "bmj_ttblq": (200.0 + 150.0 * rng.random((U.ITBQ, U.JTBQ))).astype(np.float32),
    }
    ctx.tables.update({k: cp.asarray(v) for k, v in tab.items()})

    out = U.run(ctx)
    cp.cuda.Stream.null.synchronize()
    bad = []
    for n, a in out.items():
        h = cp.asnumpy(a)
        if np.issubdtype(h.dtype, np.floating) and not np.isfinite(h).all():
            bad.append(n)
    print("nonfinite outputs:", bad or "none")

    qe = (kind == 1) | (kind == 2) | (kind == 3)
    deep = cp.asnumpy(out["BMJ_DR_DEEP"]).astype(bool)
    plume = cp.asnumpy(out["BMJ_DR_PLUME"]).astype(bool)
    shallow = cp.asnumpy(out["BMJ_DR_SHALLOW"]).astype(bool)
    lb_o = cp.asnumpy(out["BMJ_DR_LB"])
    lbot_a = cp.asnumpy(st["BMJ_CT_LBOT"])
    ltop_a = cp.asnumpy(st["BMJ_CT_LTOP"])
    cl_a = cp.asnumpy(st["CLDEFI"])
    sm = ins["SM"]
    cl_expect = (0.6 * sm + 1.0 * (1.0 - sm)).astype(np.float32)
    print("quick-exit columns:", int(qe.sum()), " deep flags on them:", int(deep[qe].sum()),
          " plume:", int(plume[qe].sum()), " shallow:", int(shallow[qe].sum()))
    print("quick-exit lbot==0:", bool((lbot_a[qe] == 0).all()), " ltop==LM:", bool((ltop_a[qe] == LM).all()),
          " cldefi replaced:", bool(np.allclose(cl_a[qe], cl_expect[qe], atol=1e-6)))
    print("deep columns:", int(deep.sum()), " min lb on deep:", int(lb_o[deep].min()) if deep.any() else -1)
    print("untouched cldefi on non-quick-exit:", bool(np.array_equal(cl_a[~qe], cldefi_in[~qe])))

    # refusal: upstream absent must raise, not fabricate
    st2 = _Fields()
    ctx2 = Ctx(cfg=cfg, grid=g, st=st2, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    ctx2.tables = dict(ctx.tables)
    try:
        U.run(ctx2)
        print("REFUSAL: no error raised")
    except RuntimeError as e:
        print("refusal ok:", str(e)[:120])


if __name__ == "__main__":
    main()
