# woof-nam CONTRACT

Every unit (UNITS.json) is written to this contract. Source of truth for numerics is the NAM
clone at `$NMMB_SRC` (`<NAM clone>/sorc/nam_nems_nmmb.fd/src/atmos`)
(the dynamics source is `$NMMB_SRC/nmm/module_DYNAMICS_ROUTINES.F90`, in the `nmm/` subfolder)
(nam.v4.2.11, 83df4e13c1). Read logic from the uppercase `.F90` files; cpp keeps the
`ENABLE_SMP` branch only for OpenMP, which we ignore.

## 0. Project rules (2026-11-04)

Not a bitwise port: everything is ported, and every run is a full wet simulation. No bitwise campaigns, no column tests, no dry/adiabatic runs. The
proof of the port is the whole 12 km forecast with all physics on, judged by maps beside the
archived NAM forecast of the same cycle and MRMS.

## 1. Arrays: layout and naming

- Names are the NMMB names (`T`, `Q`, `CW`, `Q2`, `U`, `V`, `W`, `DWDT`, `PINT`, `PD`, `OMGALF`,
  `DIV`, `TCU`, `TCV`, `TCT`, `TRACERS`, ...), exactly as in `namgpu/parm/nam_solver_state.txt` and
  `namgpu/parm/nam_phy_state.txt`. `namgpu.state.State` allocates them; a unit that needs an array that
  is not in those tables (scratch, Adams-Bashforth history, kernel-local tables) adds it with
  `st.add(name, shape)` and lists it in its wrapper docstring.
- Fortran `A(i,j,l)` -> CuPy `A[l, j, i]`, C order, `float32` (`int32` for integer fields).
  Zero-based: `i0 = i-1`, `j0 = j-1`, `l0 = l-1`. Use `IDX3(i0, j0, l0, IM, JM)` from
  `kernels/nmmb_common.cuh`.
- Vertical: `l = 1` (index 0) is the MODEL TOP, `l = LM` the lowest layer, exactly as NMMB.
  - LW and SW cloud exception (cloud and McICA units, written 2026-11-03, SW units added 2026-11-04): the
    longwave units that run inside radlw on its internal arrays (phys-rrtm-lw-setcoef, phys-rrtm-lw-cldprop,
    phys-rrtm-lw-mcica-subcol, phys-rrtm-lw-taugb07
    and the rtrn units that consume taucld, cldf, cldfmc, lcloudy; and phys-rrtm-lw-taugb09, whose
    per-layer inputs and taug/fracs outputs are surface-first) and the shortwave cloud units inside radsw
    (phys-rrtm-sw-cldprop, phys-rrtm-sw-mcica-subcol; NAM iflip = 0, module_SOLVER_GRID_COMP.F90:10714, so
    radsw reverses the top-first input to its internal array, index 0 = the LOWEST layer) use the radiation
    INTERNAL order (LW: radlw; SW: radsw), index 0 = the LOWEST layer
    (RRTM convention, radlw_main_nmmb.f ivflip = 0 input reversal at grrad.f:1234-1304). lwrad_prep
    (phys-rrtm-lw-lwrad, ivflip 0 branch) reverses the NMMB top-first arrays once, when it writes its
    scratch, and the cloud fraction input is supplied bottom-up. This is
    not the NMMB array order; no other unit may use it.
  - SW exception (phys-rrtm-sw-swflux, 2026-11-03): ZREFB, ZREFD, ZTRAB, ZTRAD, ZLDBT, ZTDBT, ZFU, ZFD are
    `[NLP1, JM, IM]` with index 0 = the LOWEST interface (surface side), as the Fortran k = 1 (zrupb(1) =
    zrefb(1) is the surface albedo). The caller of spcvrc/spcvrm must pass surface-first arrays. These are
    RRTM-internal arrays; no other unit may use this order.
  Soil fields (`STC`, `SMC`, `SH2O`) are `[NSOIL=4, JM, IM]`, layer 1 at the top.
  `PINT` and other interface fields have `LM+1` levels (`PINT[0] = PT`).
- 2-D fields `[j, i]`; 4-D `TRACERS[n, l, j, i]` (n = tracer index; `INDX_Q`, `INDX_CW`, ...
  are set by unit drv-tracer-index from NMMB's tracer order).
- Staggering: B grid. H (mass) points carry T, Q, CW, Q2, PD, PINT, W, DWDT, physics. V points
  carry U, V (and TCU, TCV, UP, VP, DUDT, DVDT, U10, V10). `V(i,j)` sits NE of `H(i,j)`: the
  four H neighbours of V(i,j) are H(i,j), H(i+1,j), H(i,j+1), H(i+1,j+1). V arrays have the H
  shape; the last row/column is unused. H<->V averaging is `module_H_TO_V.F90`.
- Halo-free single-process layout: one card holds the whole 954 x 835 x 60 domain, so every MPI
  halo exchange (`HALO_EXCH`, `exch*`) is dropped. Where NMMB loops run `its_b1:ite_b1` style
  limits (interior minus boundary rows), the kernel reproduces those limits from the full-domain
  indices: `its_b1 = 2` (i.e. `i0 >= 1`) etc. `module_DM_PARALLEL.F90 looplimits` is the
  reference for every `_b1`, `_b2`, `_h1`, `_h2` limit.
- Global reductions (MONO mass sums) are device reductions (`cupy.sum` or a CUB block reduce)
  with the result left on the device.
- VRAM: the full state (48 3-D fields at 60 x 835 x 954 x 4 B = 191 MB each) is about 10.6 GiB (all fields of the two NAM state tables, measured from the shapes). Units
  allocate scratch once (first call) and reuse it; no per-step allocation.

## 1a. Boundary strips (BOCOH / READ_BC working objects)

- Strip arrays are dense, C-order, float32 device arrays. Fortran `SOUTH(ib,jb,[l,][k],[nl])`
  maps to CuPy `[nl, k, l, jb-1, ib-1]`, with `k` the Fortran time slot (0 = working, 1 =
  tendency). `NORTH` is the same shape. `WEST(ib,jb,...)` and `EAST` have `ib` in 1..lnsh and
  `jb` in 1..JM, so their shape ends `[..., JM, lnsh]`.
- Sizes: 2-d `[2, lnsh, IM]` (S/N) or `[2, JM, lnsh]` (W/E); 3-d the same with `LM` in front of
  the strip; 4-d `[NL, 2, LM, span, lnsh]`-shaped (NL outermost). A 2-d strip may also be staged
  as `[2, 1, span, lnsh]` by bnd-read-bc; the element count is the same. bnd-bocoh-update checks
  each strip is float32, dense C-contiguous and of the expected element count, and raises
  otherwise.
- Halo: Fortran allocates `SOUTH(IMS:IME,1:LNSH,1:2)` and `WEST(1:LNSH,JMS:JME,1:2)`, with halo
  columns outside 1..IM (or 1..JM). The port stores only `1..IM` and `1..JM`. One card gives
  `its_h2 = max(its-2, ids) = 1` and `ite_h2 = min(ite+2, ide) = IM`, so bocoh never reads a
  halo entry. Multi-card (row slabs, section 1c): the S/N strips keep this shape and are used only on the
  edge ranks; the W/E strips hold the rank's memory rows, `[.., jm_local, lnsh]`, Fortran jb = jms..jme.
- Unit split of Fortran `bocoh` (module_FLTBNDS.F90): 5064-5354 working update, bnd-bocoh-update;
  5355-5962 strip blend, corners, outer PINT, bnd-bocoh-blend (reads `wh(k)`); 5963-6387 H-row
  smoothing, bnd-bocoh-smooth (dead while `nsmud = 0`). `wq(k)` is initialised and never read.
- Known open gap (no NAM path uses it): bnd-read-bc stages a 4-d H-point class as one entry per
  `(nv, nl)`, while bnd-bocoh-update reads one NL-outermost strip per variable. The two agree only
  if NL = 1. `bnd-write-bc` has no 4-d H class (`h4d = []`), so no live path is affected. Fix
  before any 4-d H class is added.

## 1b. Band-local optical-depth planes (RRTM SW/LW taugNN, taurNN units)

The Fortran taug and taur (SW: `taug(CHK,nlay,ngptsw)`, radsw_main_nmmb.f:2257 and 3231) carry every
spectral plane. A band unit allocates only its own planes: `taugNN[NG, NLAY, JM, IM]` and
`taurNN[NG, NLAY, JM, IM]` (C order, float32), with plane `g0` (zero-based) = Fortran plane `NSNN + g0 + 1`.
NS and NG come from radsw_param_nmmb.f:163 and 171-175 (for band 26: NS26 = 80, NG26 = 6, so the planes are
81..86). Taumol units that write taug and taur for one g-point use the same plane convention.

- Consumer: the SW spectral sum reads the Fortran `taur(j1,k,jg)` and `taug(j1,k,jg)` at
  radsw_main_nmmb.f:2419 with jg = 1..ngptsw. A band-local consumer reads plane `jg - NSNN - 1` of
  that band's array. The spectral-sum unit is not yet ported (no consumer in namgpu/ as of 2026-11-04).
- Precision: the Fortran taug, taur and rayl are `kind_phys`. In the NAM configure we hold, SINGLE_PREC
  is off, so `kind_phys = 8` (gate0/nmmb/src/atmos/phys/machine.f). The port stores these planes as
  float32. This is a declared divergence, bounded per unit (see each kernel header), not a verified
  identity with the 64-bit build.

## 1c. Row slabs (multi-card; namgpu/slab.py, MULTICARD-PLAN.md; lanes E1/E2, 2026-11-04)

NMMB's own decomposition with INPES = 1, JNPES = N. Rank r owns global rows jts..jte (1-based, balanced split) and the
whole i range. It holds memory rows jms = max(jts-3, 1) .. jme = min(jte+3, JMG): JHALO = 3 halo rows on every side
that is not a domain edge, none past a domain edge. So the first and last ranks see their domain edge exactly where
one card does. `ctx.slab` (slab.Slab) describes it; `ctx.grid` is `slab.local_grid(g, slab)` (jm = local rows,
per-row host arrays cut, g.joff/g.jmg set); `ctx.cfg.jm` stays GLOBAL. No ctx.slab (or one rank) = one card.

- Kernel rule. A kernel with a domain test that is not a plain local `j0 >= k` / `j0 <= JM-1-k` with k <= 3 takes two
  trailing int arguments `JOFF, JMG` (`*slab.kargs(ctx)`; JOFF = jms-1, JMG = global JM): domain tests use the global
  row `jg = j0 + JOFF` and JMG; memory guards use j0 and the local JM; rows whose stencil of S rows would leave memory
  at an inner halo edge are skipped with `NMMB_SLAB_EDGE(j0, JM, jg, JMG, S)`. A plain k <= 3 local test gives the
  one-card answer on every owned row and is the right memory guard in the halo, so it stays. On one card JOFF = 0 and
  JMG = JM: every test is the one-card test, the arithmetic and the output are unchanged.
- Tile limits. ctx.tables["NMMB_LIMITS"] holds GLOBAL 1-based Fortran values per rank (drv-loop-limits with the
  rank's decomp); a kernel comparing it with rows uses jg. `slab.local_limits(ctx)` is the same table with every j
  entry shifted to local rows, for kernels written against local j (phys-jsfc).
- Host row limits: `slab.local_lims(ctx, lo_g, hi_g, reach)` (1-based global -> 1-based local, clipped to memory and
  pulled in by `reach` at an inner halo edge).
- Per-row 1-D tables (FAH, FAD, DARE, DXH, DXV, WPDAR, HDAC*, F, RDXH ...) are LOCAL: cut from the global table on
  the host, halo rows included.
- Boundaries: S strips and S corners are applied only where `slab.edges(ctx)[0]` (rank 0), N ones only where
  `slab.edges(ctx)[1]` (rank N-1). W/E strips are [.., jm_local, lnsh]. Boundary files hold the GLOBAL strips;
  bnd-read-bc scatters rows jms..jme of the W/E records.
- Init: run once over the whole domain, saved and cut per rank (namgpu/init/slab_init.py), because the chain is
  not reproducible run to run (float atomics in init-gwd-orog and init-vcoord-recover).
- Exchanges: drv-halo-exch rows X1-X21 (E1) at the Fortran HALO_EXCH sites; iterative units exchange inside their
  loop (`slab.halo_exchange`: io-quasipost relax4e); domain-wide flags and sums go through `slab.allreduce` or the
  deterministic MONO per-row partials. Only owned rows enter a sum, a history file or a map.
- Not built for slabs (refused with the reason): nsmud >= 1 smoothing (bocoh/bocov; NAM nsmud = 0), the 4th-order
  ADV1 uv branch (secadv false; NAM secadv true), global = true.

## 2. Step order (SOLVER_RUN, module_SOLVER_GRID_COMP.F90:2453-7168)

Driver: `namgpu/driver.py` (FIRST_STEP, DYNAMICS, PHYSICS, DIAGNOSTICS lists). Fortran line
numbers are the call sites in SOLVER_RUN.

One fundamental step `DT = 25 s` (configfile `dt_int: 25`, `dt_num: 0`). No acoustic sub-cycling:
NMMB is forward-backward for horizontal fast waves plus ONE implicit vertical sound-wave solve
(`VSOUND`) per step; Adams-Bashforth for horizontal advection (`cfc = 1.533`, `bfc = 1-cfc` in
PGFORCE), off-centred Crank-Nicolson for vertical advection (ADV1/ADV2 `w1`, `w2`).

**First step only** (`firststep` block, 3390-3542): `PGFORCE` (3425, first-pass switch: no u/v
update), `DHT` (3457).

**Every later step** (`not_firststep`, 3546-4228):
1. `HDIFF` (3578) if `HDIFF_ON>0` (Smagorinsky, `smag2 = 0.4`).
2. Lateral BCs (regional, 3672-3974): `UPDATE_BC_TENDS`/`READ_BC` -> our boundary arrays from
   GFS; `BOCOH` (3961) on H rows, `lnsh = 5` blending rows.
3. `PGFORCE` (3982): geopotential by upward recursion, pressure-gradient force components
   (x, y, NE, NW), divergence correction (Janjic 1974/1979), u/v update.
4. `UPDATEUV` (4020), `BOCOV` (4062, `lnsv = 5`).
5. `DHT` (4108), `DDAMP` (4177; external mode then divergence damping, `codamp = 9.0`,
   `wcor = 0.18`).

**Every step** (4245-4857):
6. `PDTSDT` (4245): surface-pressure tendency, PSDT, PSGDT.
7. `ADV1` (4296): T, U, V horizontal (A-B) and vertical (C-N) advection, Coriolis, boundary
   upstream rows.
8. If `mod(ntimestep, IDTADTQ) == 0` (configfile `idtad = 2`): `ADV2` (4325) on Q, `MONO` (4370),
   `UPDATES` (4392).
9. If `mod(ntimestep, IDTADT) == 0` (`idtadt = 2`): `ADV2` (4468) on the other tracers (CW, Q2 via
   interface->mid interpolation), `MONO` (4519), `UPDATES` (4543), `RIME_FACTOR_UPDATE` (4590).
   The two RIME_FACTOR_UPDATE calls (before at 4448, RIME_FACTOR_INPUT=.TRUE.; after at 4590) run
   only when RIME_FACTOR_ADVECT: MICROPHYSICS == 'fer_hires' AND F_QG AND SPEC_ADV
   (module_SOLVER_GRID_COMP.F90:2885-2890). namgpu/parm/nam_configfile sets spec_adv: .false., so both
   are dormant in the 12 km config. Wrapper: namgpu/steps/drv_rime_factor_update.py `advect_enabled`.
10. `VTOA` (4611), `UPDATET` (4648).
11. `CDZDT` (4694): omega and nonhydrostatic w; `CDWDT` (4750): dw/dt; `VSOUND` (4807): implicit
    vertical sound waves (one tridiagonal per column).

Global-only calls (FFTFHN/FFTFWN/FFTFUVN, POAVHN, POLEHN/POLEWN, SWAPHN/SWAPWN, PREFFT) are mode
"skip": `global: false` in the configfile.

**Physics** (5190-6077, `gfs: false` branch):
12. `UPDATE_WATER` (5202): CW + F_ICE/F_RAIN/F_RIMEF <-> species.
13. `RADIATION` (5312) is called with `NTIMESTEP_RAD` (0 at step 0, ntimestep+1 after; 4938-4942),
    not ntimestep. Shortwave fires when `mod(NTIMESTEP_RAD, NRADS) == 0`, longwave when
    `mod(NTIMESTEP_RAD, NRADL) == 0`, `nrads = nradl = 48` (every 20 min), so SW and LW run at steps
    0, 47, 95, ... (the wrapper reads ctx.ntimestep_rad; the plan cadence is "radiation"). NMMB RRTM via `module_RA_RRTM.F90:RRTM` ->
    `grrad_nmmb` -> `lwrad`/`swrad` (`*_nmmb.f`). `np3d = 3` (Ferrier stratiform clouds),
    `CHNK_RRTM = 8` chunking is a CPU detail and is dropped.
14. `RDTEMP` (5384) every step: apply RLWTT + RSWTT (with zenith-angle scaling between calls).
15. Accumulator/bucket resets: the NCLOD reset (5367-5376) runs before RDTEMP; the NRDLW, NRDSW, NSRFC
    and NPREC resets (5418-5467) run after RDTEMP, each on its own cadence. The plan runs
    drv-bucket-reset twice (blocks "nclod", then "post_rdtemp").
16. `TURBL` (5490) every `NPHS = 2` steps, `DTPHS = 50 s`: `JSFC` (Janjic surface layer, 787),
    `NOAHLSM` (879), `SFCDIAGS` (925), `MYJPBL` (1013), `GWD_DRIVER` (1291, `gwdflg: true`),
    then `H_TO_V_TEND` (5586/5588, NSTEPS = NPHS) for DUDT/DVDT. Scope exclusion: the GFS-physics
    call at 6998/7000 (NPHS) is not ported; `gfs: false`, and the wrapper refuses if gfs is set.
17. `CLTEND` (5708) cloud tendency spreading over NPRECIP steps.
18. `CUCNVC` (5732) every `NPRECIP = 2` steps: BMJ (`bmjdrv`) with the configfile shallow
    switches (fres 0.75, fr 1.0, fsl 0.85, fss 0.85, newswap true). Its DUDT_PHY/DVDT_PHY are the
    arrays DUDT/DVDT (5768), then `H_TO_V_TEND` (5827/5830, NSTEPS = NPRECIP) reads them for the
    wind tendency. The bmj glue aliases the two names (phys-bmj-driver).
19. `GSMDRIVE` (5880) every `NPRECIP = 2`: `FER_HIRES` (Ferrier-Aligo), `rhgrd = 0.98`,
    `spec_adv = false` (CW total condensate advected, species by fractions).
20. `CLTEND` mode 0 (5928: inside CALL_PRECIP, NPRECIP>1), `CLTEND` mode 1 (6008: NPRECIP>1 only, every
    step), then `TQADJUST` (6022). Mode -1 (5708) is item 17.
21. Diagnostics: `MAX_FIELDS` (5000-5001, first in the physics block) runs only when
    `mod(NTSD, NSTEPS_PER_CHECK) == 0` (NSTEPS_PER_CHECK = MAX(2, NINT(40/DT)), line 818), FILTER_METHOD
    is 0 and MICROPHYSICS is 'fer' (not 'fer_hires'). With filter_method 3 it never runs. History every
    `minutes_history = 60`.

**Initialisation** (SOLVER_INITIALIZE 263-2447, PHYSICS_INITIALIZE 9936-11062): `CONSTS`,
grid constants (`module_CONTROL.F90 grid_consts`), `boundary_init`, look-up tables
(`exptbl`, `tablep`, `tablet`, `psitbl`), then `rad_initialize_nmmb` + `RRTM_INIT`,
`JSFC_INIT`, `MYJPBL_INIT`, `NOAH_LSM_INIT`, `BMJ_INIT`, `FERRIER_INIT_HR`, `GWD_init`.

**DFI**: `filter_method = 3` (TDFI) for every cold-start forecast (exnam_fcst.sh:280-285):
backward `nsecs_bcktdfi = 600 s` then forward `nsecs_fwdtdfi = 600 s` at `filt_dt = 8 s`
(`filt_dt_int: 8`), Dolph-Chebyshev window (`dolph`, `cheby`), filtered variables from
`namgpu/parm/nam_filt_vars_norad.txt` (PD, Q, Q2, PINT, DWDT, T, U, V filtered; CW and soil restored).
`RUN_DIGITAL_FILTER_NMM` (module_NMM_GRID_COMP.F90:3624-5418) drives it.
**TDFI in the run (2026-11-04, dyn integrator):** drv-time-loop runs drv-dfi-tdfi-driver before the
forecast (filter_method 3): 75+75 backward and 75+75 forward filter steps at 8 s through `Driver.step`
(integrate_filter_step sets signed `ctx.dt`, `physics_on = dt > 0`, `hdiff_on`). The dt-scaled constants
follow SOLVER_RUN FIRST_PASS / changedir / end_filt through **drv-filter-consts**: FAD, FAH, FCP, DDMPU,
DDMPV, WPDAR, HDACX/Y, HDACVX/Y, EF4T. **One array per Fortran constant:** the canonical arrays are
`ctx.tables["grid"]` (init-grid-consts); `drv_filter_consts.bind(ctx)` (called by Driver.step once)
aliases `st.fields` FAH, FAD, HDACX, HDACY, HDACVX, HDACVY to them. A unit must read those names and
never cache its own copy of a dt-scaled constant or compute one from `ctx.dt` (that froze FAD at -8 s
and left HDAC* zero before this). Named gap (boundary group): the changedir WRITE_BC (3264-3277,
boundary tendencies recomputed after the backward leg) is not called; bnd-write-bc's key layout is not
reconciled with bnd-boundary-init. The physics block is gated on `ctx.physics_on` (IF(INTEGER_DT>0))
and HDIFF/DDAMP on `ctx.hdiff_on` (HDIFF_ON>0); both are True for the forecast.

**Run directory (2026-11-04):** the Fortran opens its fix files by name in the run directory. Ours is
`ctx.fix_dir`, else `$NAMGPU_FIX_DIR`, else `<repo>/fix` (tools/fetch_fix.sh: global_o3clim.txt,
co2historicaldata_YYYY.txt, sfc_emissivity_idx.txt, solarconstant_*.txt from NOAA's public fix/am
bucket; tools/build_noah_fix.py: namgpu/data/fix/noah VEGPARM/SOILPARM/GENPARM, see namgpu/data/fix/noah/PROVENANCE.txt).
drv-physics-initialize and drv-time-loop run inside `drv_physics_initialize.in_run_dir(ctx)`, because
gas_update reopens the CO2 file during the forecast. Paths a unit writes must be absolute.

## 3. Kernel calling convention

- Exception: a unit with no GPU computation (constant Fortran DATA tables, e.g.
  phys-rrtm-lw-table-avplank) has no `.cu`; its UNITS.json `kernel` is `""` and the wrapper
  uploads the table once into `ctx.tables`. Recorded 2026-11-04 (review of phys-rrtm-lw-table-avplank).
- One `.cu` per unit in `namgpu/kernels/<unit_id with '-' -> '_'>.cu`, starting with the NOAA
  header block of the Fortran routine as a notice (NOTICE) and the line
  `// Ported to CUDA for WOOF-NAM from <path>:<lines> (nam.v4.2.11, 83df4e13c1).`
- `#include "nmmb_common.cuh"`; kernels are `extern "C" __global__`.
- One launch per Fortran routine (or per routine section, as the unit is split). All arguments
  are device pointers (`float*`, `const float* __restrict__`) or scalars (`int`, `float`).
  No host sync, no device malloc, no printf inside kernels.
- Point kernels: one thread per (i, j) with the level in `blockIdx.z` (`kernels.grid_2d`).
  Column recurrences (geopotential in PGFORCE, CN sweeps in ADV1/ADV2, VSOUND tridiagonal, every
  physics column) run one thread per column with an l-loop (`kernels.grid_cols`).
- Wrapper: `namgpu/<group dir>/<unit_id_underscored>.py` with `run(ctx)`; it gets arrays from
  `ctx.st`, constants from `ctx.grid`/`ctx.cfg`/`ctx.tables`, and launches on `ctx.stream`.
  Group dirs: dynamics -> `namgpu/dynamics`, boundary -> `namgpu/boundary`,
  physics-<scheme> -> `namgpu/physics/<scheme>`, init -> `namgpu/init`, io -> `namgpu/io`,
  driver -> `namgpu/steps`.
- Fast numerics: NVRTC with `--fmad=true` (FMA on), IEEE division/sqrt, no `--use_fast_math`.
  `float` everywhere the Fortran is `kfpt`; `double` only where it declares `kdbl`.
- Tables (Ferrier `MY_GROWTH`, `MASSR`, `VENTR`; BMJ `ttblex`; MYJ/JSFC `PSIM/PSIH`; RRTM k-tables)
  are built once at init by the scheme's `*-init` unit, uploaded once into `ctx.tables`.

## 4. Where each Fortran module's outputs go

| Fortran module / routine | writes | in woof-nam |
|---|---|---|
| module_DYNAMICS_ROUTINES (pgforce, dht, ddamp, pdtsdt, adv1, adv2, mono, vtoa, updates/t/uv, hdiff, cdzdt, cdwdt, vsound) | T, U, V, Q, CW, Q2, TRACERS, PD, PINT, W, DWDT, OMGALF, DIV, PSDT, PSGDT, TCT/TCU/TCV, RTOP, Z | `ctx.st` same names |
| module_FLTBNDS (read_bc, write_bc, bocoh, bocov) | boundary strips PDB, TB, QB, CWB, UB, VB and tendencies | `ctx.bc` (boundary group) and `ctx.st` edge rows |
| module_RADIATION + module_RA_RRTM + grrad_nmmb / radlw_main_nmmb / radsw_main_nmmb | RLWTT, RSWTT, RLWIN, RSWIN, RSWOUT, RLWTOA, RSWTOA, CFRACL/M/H, CZEN, CZMEAN, SIGT4, TOTSWDN, ... | `ctx.st` |
| module_TURBULENCE (JSFC, NOAHLSM, SFCDIAGS, MYJPBL, GWD_DRIVER) | DUDT, DVDT, RTHBLTEN (T), RQVBLTEN (Q), Q2, EL_PBL, PBLH, USTAR, Z0, THS, QSH, TSKIN, STC, SMC, SH2O, CMC, SNO, SI, SNOWC, TSHLTR, QSHLTR, U10, V10, TH10, Q10, SFCEXC, potential evap, runoff, SFCSHX, SFCLHX, SUBSHX, GRNFLX, ... | `ctx.st` |
| module_CONVECTION (CUCNVC -> bmjdrv) | T, Q tendencies (via CLTEND), CUPREC, ACPREC, CPRATE, CNVTOP, CNVBOT, HTOP, HBOT, CUPPT, TCUCN | `ctx.st` |
| module_MICROPHYSICS (GSMDRIVE -> FER_HIRES) | T, Q, CW, F_ICE, F_RAIN, F_RIMEF, QC/QR/QS, PREC, ACPREC, SR, TRAIN, REFL_10CM, AVRAIN | `ctx.st` |
| module_DIAGNOSE (MAX_FIELDS_HR), module_QUASIPOST | hourly maxima (REFDMAX, UPHLMAX, ...), MSLP, ... | `ctx.st`, then history |
| module_DIGITAL_FILTER_NMM | filtered initial state | `ctx.st` |
| io-wind-rotate (port of PROF_NMMB.f rotation) | UEARTH, VEARTH (earth-relative H-point winds, 3-D); UE10, VE10 (10 m) | `ctx.st` |

## 5. NAM 12 km CONUS parent configuration (namgpu/parm/nam_configfile)

| item | value | source |
|---|---|---|
| grid | IM = 954, JM = 835, LM = 60, rotated lat-lon B grid | configfile `im`, `jm`, `lm` |
| centre | tph0d = 54.0 N, tlm0d = -106.0 E | configfile |
| extent | wbd = -60.039, sbd = -45.036 rotated deg -> dlmd = 0.126, dphd = 0.108 deg | configfile; nps `dx`, `dy` |
| top | PT = 200 Pa (etop = 2 hPa), PTSGM = 30000 Pa, 61 interface `coord_levs` | nam_namelist.nps, nam_modtop.parm |
| dt | 25 s; DFI filt_dt 8 s | `dt_int`, `filt_dt_int` |
| physics cadence | NPHS = 2 (turbulence/LSM), NPRECIP = 2 (convection/microphysics), NRADS = NRADL = 48 | configfile |
| passive advection | idtad = 2, idtadt = 2 | configfile |
| dynamics | secadv = true (2nd-order), smag2 = 0.4, codamp = 9.0, wcor = 0.18, lnsh = lnsv = 5, hydro = false, global = false | configfile |
| physics | microphysics fer_hires (rhgrd 0.98), convection bmj, turbulence myj, sfc_layer myj, land_surface noah (ivegsrc 1 = IGBP), shortwave rrtm, longwave rrtm, np3d 3, co2tf 1, gwdflg true (cdmb 0.1, cleffamp 1.0, sigfac 3.0, factop 0.5, rlolev 50, dpmin 0) | configfile |
| DFI | filter_method 3 (TDFI), nsecs_bcktdfi = nsecs_fwdtdfi = 600 s, radar_init 0 | configfile + exnam_fcst.sh |
| history | 60 min | `minutes_history` |
| boundaries | GFS every 3 h (`interval_seconds = 10800`), 6-h-old GFS cycle | nam_namelist.nps_bocos |

## 6. Done rule

A unit is DONE when:
1. its kernel compiles (`namgpu.kernels.module(<unit>).compile()` without error);
2. its wrapper runs once on a FULL-SIZE random state (`State(grid).randomize()`, 954 x 835 x 60)
   without producing a NaN or Inf in any array it writes;
3. its Fortran-to-CUDA diff review found no missing term: every assignment in the Fortran
   line range has a counterpart in the kernel, or a one-line reason in the kernel comment
   (MPI/halo/OpenMP/print/global-only).
Then the unit's line in UNITS.json gets `"status": "done"` and a dated line goes in STATUS.md.
No unit gets a column test, a bitwise comparison or a dry run. The integration proof is
`tests/smoke_full.py`: the whole forecast, wet, all physics, maps beside NAM and MRMS.

## 6a. H-point boundary layout (bnd-boundary-init to bnd-bocoh-blend)

- `ctx.bc` holds the H-point strips as a dict keyed `"<SIDE>_<VAR>"` (SIDE in S, N, W, E; VAR in
  PD, T, Q, CW), written by bnd-boundary-init. Each value is float32 with a leading slot axis of
  length 2. Slot 0 is the value that bnd-bocoh-blend reads; slot 1 is the tendency slot, left at
  zero by bnd-boundary-init.
- Value-slot shapes, slot excluded: PD S/N `[lnsh, IM]`, W/E `[JM, lnsh]`; T, Q, CW S/N
  `[LM, lnsh, IM]`, W/E `[LM, JM, lnsh]`. There is no 4-D H key (NVARS_BC_4D_H = 0).
- Q2 is not a parent H boundary variable (module_SOLVER_GRID_COMP.F90:7658-7664, NVARS_BC_2D_H = 1,
  NVARS_BC_3D_H = 3 for T, Q, CW). It is not in this layout and bnd-bocoh-blend does not blend it.
- Not reconciled here (owned by other units, a named follow-up for their owners): bnd-write-bc
  writes tuple keys `(cls, name, side)` with shape `[2, nspec, LM, lines, along]`, and bnd-read-bc
  and bnd-bocoh-update work on an attribute object (`ctx.bc.pending`, `ctx.bc.h2d`). bnd-bocoh-blend
  reads neither.

## 7. Modes

- `port`: write CUDA from the NAM Fortran line range.
- `copy-arwen`: start from the ArWen kernel in `namgpu/third_party/arwen/` (Apache-2.0, keep its header),
  then bring it to the NAM Fortran (top-down layout, NCEP changes, NAM constants).
- `new`: no Fortran counterpart to port (driver glue, our I/O, GRIB2 decode, init that replaces
  NPS).
- `skip`: not run in the 12 km regional configuration; the unit states the reason.

## Init after NAM retirement (2026-11-04)
NAM was discontinued 2026-11-03 12Z, so NO new NAM analyses (bgrd3d00 native B-grid files, grid 218 output) exist after that cycle.
- The archived native analysis is valid ONLY for past cases (our 2025-03-14 12Z proof case and any date before 2026-11-03 12Z).
- A live or future-date NAM run MUST start from a source that still runs: GFS 0.25 (default), HRRR, or RRFS analyses, interpolated
  onto the rotated B grid and NMMB hybrid levels by the init group (the NPS replacement), with GFS boundaries. This is a table row per
  source (arbitrary acceptance test), never a per-source adapter.
- The init integrator builds BOTH paths: (a) native-analysis start for archived cases, (b) GFS/HRRR/RRFS start for any date, and proves
  (b) on the same 2025-03-14 case next to (a) (maps side by side). The judge phase reports both.

## 8. Physics A on the real state (physics A integrator, 2026-11-04 ~09:50Z)

What every group must know after physics A first ran on the real 2025-03-14 12Z state:
- TURBL dummy names bound to the state, not copied (phys-turbl-driver ARG_ALIASES, the SOLVER_RUN call
  at 5490-5560): `TSFC` IS `TSKIN`, `PSFC_OUT` IS `PSFC` (same device array under two names).
- TURBL's local interface-height array Z(LM+1) (module_TURBULENCE.F90:339) is `TURBL_Z` in ctx.st.
  The state `Z` belongs to CDZDT/HDIFF/io; physics never writes it.
- GWD_DRIVER reads TURBL's working arrays: U_PHY, V_PHY (H-point winds), DELP, PHINT, PHMID, EXNER,
  TURBL_Z, LPBL (module_TURBULENCE.F90:1291-1305), not U, V, DP, PINT, PMID, EXNR, Z, KPBL.
- Noah's DZS is the MODULE_LS_NOAHLSM constant (0.1, 0.3, 0.6, 1.0) uploaded once to
  ctx.tables['noah_module_sldpth']; phys-noah-init also copies it into st SLDPTH (SOLVER_INITIALIZE 966).
  A zero SLDPTH made NaN soil on every land point.
- init-static-missing now fills EPSR where the analysis has none: 0.97 sea (SM > 0.9 or SICE > 0.5),
  1.0 land (NPS module_assorted_definitions.F90:134-155). The bgrd3d00 analysis has no EPSR.
- The NAM physics plan has NO bucket reset after TQADJUST: SOLVER_RUN 6348-6395 is the GFS-physics
  branch. The NAM resets are drv-bucket-reset "nclod" and "post_rdtemp" only.
- phys-rad-driver runs with cwd = the fix dir (ctx.fix_dir, $NAMGPU_FIX_DIR, else <repo>/fix), the
  NMMB run directory where co2historicaldata_YYYY.txt and solarconstant files are opened by bare name.
- `NAMGPU_NO_LEASE=1` (env-box.sh) makes drv_gpu_lease.gpu_lease a no-op on the dedicated box card.
- GWD orography source: NAMGPU_GWD_DEM = gmted2010.30sec.int (NOAA fix/orog 20231027,
  raw big-endian int16 43200 x 21600, row 0 north, column 0 at -180; ocean = 0 m, not -9999: declared
  difference from NPS). tools/run_physics_a.py loads it into ctx.tables['gwd_dem'] before init
  ($NAMGPU_GWD_DEM overrides); the init chain itself does not load it yet (init group).
Open, owned elsewhere (found by this run):
- init: after the init chain, Q2 = 0.02 (EPSQ2) at every point of the lowest 10 layers, so the
  analysis TKE (warm Q2 = max(2 TKE, 0.02) in init-native-atm) does not reach ctx.st Q2.
- physics B: phys-rrtm-driver-prep needs ctx.tables['rrtm_axsd']; phys-rrtm-init writes the table
  under its own dict ('axsd'). Until RRTM runs, RLWIN/RSWIN/SIGT4 are 0, so Noah sees no downward
  longwave and land skin temperature drops up to ~30 K in one call (no NaN, but not a usable result).
- init: init-gfs-bocos took 287 s for 16 boundary times on the box (check for a single-core stage).

## 9. Init and io conventions (init+io integrator, 2026-11-04)

- Shelter / 10 m state: TSHLTR and TH10 are POTENTIAL temperatures (P0 = 1e5 Pa), QSHLTR/Q10 specific
  humidity, PSHLTR Pa, U10/V10 grid-relative at H points (nam_phy_state column H). Init fills them from
  the source's 2 m / 10 m analysis (sources.shelter_state), as NMMB reads them from its input file.
  Anyone writing T2 must use T2 = TSHLTR * (PSHLTR * 1e-5) ** 0.28589641.
- Vertical coordinate: any interpolation onto model levels runs after gfs_start.ensure_vcoord (the NPS
  vcgenerator header, vcoord = 1), never on the grid's default SG.
- Chain order: init-gfs-bocos runs right after init-grid-geo, before drv-consts; boco files live in
  <case>/boco and ctx.boco_dir points there.
- History: HISTORY_UNITS = io-wind-rotate, io-quasipost-driver, io-history-fields, io-history. History
  carries T2 (temperature), MSLP (Pa, membrane MSLP) and, after the first microphysics call, COMPOSITE.
- Maps: rw_wrfbatch --cf on XLAT/XLONG (io/maps.py). The wrfout route misplaces MAP_PROJ 203; do not use
  it for woof-nam output. Products: composite_reflectivity, t2m, d2m, wind10, wind850, mslp, qpf_1h.
- GWD terrain: NAMGPU_GWD_DEM (NPS raw file) or NAMGPU_GEOG/topo_gmted2010_30s (else ~/WPS_GEOG).

## 10. Physics B (RRTM) on the real state (physics B integrator, 2026-11-04 ~10:10Z)

Radiation runs whole on the real 2025-03-14 12Z state: `tools/run_rrtm_real.py` (init chain, then
phys-rad-driver -> rrtm_call, SW + LW, at 12Z and 18Z), no NaN/Inf, 2.8-3.9 s GPU per call on the shared card 3.
What every group must know:
- STDH is NOT read by NAM (absent from NAM's own parm/nam_phy_state.txt); it keeps the NMMB cold-start
  value -1.E6 (module_SOLVER_GRID_COMP.F90:1075). init-static-missing now fills STDH = -1.E6 where it was
  never loaded. Init group: keep that (setemis refuses an all-zero STDH).
- config: `ialbflg = 2` (IALB), `iovrlw = 1` (IOVR_LW), `isubclw = 0` (ISUBCLW) added (SOLVER_GRID_COMP
  10676/10695/10702). `ivflip = 0` is the NAM runtime value everywhere in RRTM (progcld2 now takes it).
- rrtm_call row chunks: `NAMGPU_RRTM_ROWS` default 32 (was 8; launch-bound), about 6 GB of chain scratch on
  top of the State. Inside a chunk, the driver-call 2-D outputs that are GRRAD actual arguments (XLON, XLAT,
  SLMSK, TSEA, ...; rrtm_call.GRRAD_SFC_ARGS) are chunk-state names. P_PHY is computed full-domain before
  the chunk loop (the module_RA_RRTM prologue). Never cache row-dependent data in ctx.tables keyed on
  id(ctx.st): the chunk ChunkState object is reused for every row chunk (that bug drifted the SW day list).
- SW: swrad's cosz input is SW_COSZEN (coszen_loc, the SW-interval mean), not the state CZEN; solcon is the
  radupdate value of the call. swrad loops over the NDAY day columns only (one host read of the device
  day count per row chunk, 27 per radiation call), chunks of `NC_DEFAULT = 32768` day columns.
  Seam units that take the CONTRACT layout run through `namgpu/physics/rrtm/sw_seam.py` (layer-major
  [.., nlay, 1, nc] copies of the chunk scratch, outputs scattered back to the g-point planes NSnn + g).
- LW setcoef: the COLAMT macro now reads the value (it returned the flat index, which made every LW
  optical depth ~1e3-1e4 too large). taugb08's index guard now allows jp 1..12 (lower) and 13..58 (upper).
- Not done: radiation inside the run lead's forecast loop (1 h and 6 h); speed (spcvrtc is 112 launches per
  chunk, LW rtrnmr ~1 s per call) is a named gap for the speed pass.
