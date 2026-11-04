import time
import numpy as np
import cupy as cp
from namgpu.physics.noah import phys_noah_snowpack as m

# Full-size 954 x 835 batch (run() with no ctx), ESD spans negative values so both gate branches run.
t0 = time.time()
rng_esd_before = None
s = m._random_batch(835, 954, seed=1)
esd0 = cp.asarray(s["esd"]).copy()
out = m.snowpack(s["esd"], m.DTSEC_SNOWPACK, s["snowh"], s["sndens"], s["tsnow"], s["tsoil"])
cp.cuda.Device().synchronize()
dt = time.time() - t0
esd, snowh, sndens = s["esd"], s["snowh"], s["sndens"]
bad = {k: int((~cp.isfinite(v)).sum()) for k, v in dict(esd=esd, snowh=snowh, sndens=sndens).items()}
off = esd0 <= 0
print("full-size shape", tuple(esd.shape), "wall incl compile %.2f s" % dt)
print("nonfinite counts", bad)
print("gate cells (ESD<=0):", int(off.sum()), "ESD reset to 0:", bool((esd[off] == 0).all()),
      "SNOWH 0:", bool((snowh[off] == 0).all()), "SNDENS 0:", bool((sndens[off] == 0).all()))
on = ~off
print("active cells:", int(on.sum()), "SNDENS in [0.05,0.9]:", bool(((sndens[on] >= 0.05) & (sndens[on] <= 0.9)).all()),
      "SNOWH>=0:", bool((snowh[on] >= 0).all()))
print("ESD unchanged for active cells:", bool((esd[on] == esd0[on]).all()))
# timing of one warm launch
s2 = m._random_batch(835, 954, seed=3)
cp.cuda.Device().synchronize(); t1 = time.time()
m.snowpack(s2["esd"], m.DTSEC_SNOWPACK, s2["snowh"], s2["sndens"], s2["tsnow"], s2["tsoil"])
cp.cuda.Device().synchronize()
print("warm launch %.4f s" % (time.time() - t1))
