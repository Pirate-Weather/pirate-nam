"""Generate UNITS.json: the whole NAM 12 km parent split into swarm units.

Line ranges were read from the NAM clone (nam.v4.2.11, 83df4e13c1), 2026-11-04.
fortran paths are relative to $NMMB_SRC = sorc/nam_nems_nmmb.fd/src/atmos.
Run:  python tools/make_units.py  (writes namgpu/UNITS.json, shipped in the package)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DR = "nmm/module_DYNAMICS_ROUTINES.F90"
FB = "nmm/module_FLTBNDS.F90"
SV = "nmm/module_SOLVER_GRID_COMP.F90"
CT = "nmm/module_CONTROL.F90"
DF = "nmm/module_DIGITAL_FILTER_NMM.F90"
NG = "nmm/module_NMM_GRID_COMP.F90"
DG = "nmm/module_DIAGNOSE.F90"
QP = "nmm/module_QUASIPOST.F90"
RA = "nmm/module_RADIATION.F90"
RR = "phys/module_RA_RRTM.F90"
GR = "phys/grrad_nmmb.f"
LW = "phys/radlw_main_nmmb.f"
SW = "phys/radsw_main_nmmb.f"
LT = "phys/radlw_datatb_nmmb.f"
ST = "phys/radsw_datatb_nmmb.f"
FH = "phys/module_MP_FER_HIRES.F90"
BJ = "phys/module_CU_BMJ.F90"
MY = "phys/module_BL_MYJPBL.F90"
JS = "phys/module_SF_JSFC.F90"
NO = "phys/module_LS_NOAHLSM.F90"
GW = "nmm/module_GWD.F90"

U: list[dict] = []


def u(id_, group, fortran, purpose, mode="port", deps=(), **extra):
    U.append(dict(id=id_, group=group, fortran=fortran, purpose=purpose, mode=mode,
                  deps=list(deps), **extra))


BASE = ("init-grid-consts", "init-derive-state")

# ------------------------------------------------------------------ dynamics
D = "dynamics"
u("dyn-pgforce-geopot", D, f"{DR}:51-248",
  "PGFORCE setup and vertical grand loops: geopotential FIM by upward recursion from FIS, APEL, "
  "DFI per level (one thread per column). In: T, Q, CW, PD, PINT, FIS -> fim_3d, apel_3d, dfi_3d.", deps=BASE)
u("dyn-pgforce-pgf", D, f"{DR}:249-289",
  "Pressure-gradient force components behind H points (pcx, pcy, pcne, pcnw) with the "
  "nonhydrostatic correction. In: fim_3d, apel_3d, PINT, DWDT -> pgf scratch arrays.",
  deps=["dyn-pgforce-geopot"])
u("dyn-pgforce-divcor", D, f"{DR}:290-451",
  "Divergence correction (Janjic 1974/1979) of the PGF with wpdar. In: pgf scratch "
  "(PGX/PGY/PGNE/PGNW_3D), wpdar (init-grid-consts), tile limits (drv-loop-limits) -> DIV.",
  deps=["dyn-pgforce-pgf", "init-grid-consts", "drv-loop-limits"])
u("dyn-pgforce-uvupdate", D, f"{DR}:452-493",
  "PGF u/v tendency update: TCU, TCV += pressure-gradient terms from PGX/PGY/PGNE/PGNW "
  "(or TCU, TCV zeroed on a fresh first step). In: PGX_3D, PGY_3D, PGNE_3D, PGNW_3D, PD, "
  "DSG2, PDSG1, RDXV, RDYV -> TCU, TCV. Does not use cfc/bfc; U, V advance in dyn-updateuv.",
  deps=["dyn-pgforce-pgf", "dyn-pgforce-divcor"])
u("dyn-dht", D, f"{DR}:498-796",
  "Horizontal divergence and pressure advection term in the T equation: mass and pressure "
  "fluxes on H points, zero divergence on the regional boundary. In: U, V, PD, PINT -> DIV, RTOP, TDIV.",
  deps=BASE)
u("dyn-ddamp-extmode", D, f"{DR}:801-999",
  "DDAMP setup and external-mode divergence (vertically integrated). In: DIV, PD, DSG2, PDSG1 -> "
  "external-mode divergence scratch.", deps=["dyn-dht"])
u("dyn-ddamp-damp", D, f"{DR}:1000-1040",
  "Divergence damping of U, V with ddmpu/ddmpv (codamp 9.0), internal plus external mode. "
  "In: DIV, ext-mode scratch -> U, V.", deps=["dyn-ddamp-extmode"])
u("dyn-pdtsdt", D, f"{DR}:1045-1228",
  "Surface-pressure tendency PSDT and vertical mass flux PSGDT on interfaces, PD update, regional "
  "boundary conditions. In: DIV, PD, DSG2, PDSG1 -> PD, PDO, PSDT, PSGDT.", deps=["dyn-dht"])
u("dyn-adv1-vert-t", D, f"{DR}:1233-1538",
  "ADV1 setup and Crank-Nicolson (uncentred w1/w2) vertical advection of T: forward sweep on "
  "rcmt/crt, back substitution, one thread per column. In: T, PSGDT, PD -> vertical T tendency.",
  deps=["dyn-pdtsdt"])
u("dyn-adv1-horiz-t", D, f"{DR}:1539-1570",
  "Temperature fluxes on H points and Adams-Bashforth horizontal advection of T (secadv, 2nd "
  "order). In: T, U, V, PD, TP -> T tendency, TP.", deps=["dyn-adv1-vert-t"])
u("dyn-adv1-bndry-t", D, f"{DR}:1571-1705",
  "Regional branch: upstream advection of T along the southern, northern, western and eastern "
  "rows. In: T, U, V -> T tendency on boundary rows.", deps=["dyn-adv1-horiz-t"])
u("dyn-adv1-vert-uv", D, f"{DR}:1706-1837",
  "Crank-Nicolson vertical advection of U and V on V points (cmw), column sweeps. In: U, V, "
  "PSGDT -> vertical U/V tendencies.", deps=["dyn-pdtsdt"])
u("dyn-adv1-coriolis", D, f"{DR}:1838-1866",
  "Grand vertical loop start: Coriolis and curvature tendency (f, curv) and diagonally averaged "
  "fluxes on V points. In: U, V, PD -> U, V tendencies, flux scratch.", deps=["dyn-adv1-vert-uv"])
u("dyn-adv1-horiz-uv", D, f"{DR}:1867-2021",
  "Momentum advection (2nd order since secadv=true; 4th-order branch kept) and advection of "
  "u1d/v1d with Adams-Bashforth. In: U, V, flux scratch, UP, VP -> U, V, UP, VP.",
  deps=["dyn-adv1-coriolis"])
u("dyn-adv1-bndry-uv", D, f"{DR}:2022-2186",
  "Regional branch: upstream advection of momentum along the four lateral boundaries, final "
  "U, V, T update of ADV1. In: U, V, tendencies -> U, V, T.", deps=["dyn-adv1-horiz-uv", "dyn-adv1-bndry-t"])
u("dyn-vtoa", D, f"{DR}:2191-2347",
  "Omega-alpha term (vertical advection/conversion) of the thermodynamic equation: OMGALF and TCT "
  "(+= toa) and the interface pressure PINT from PSDT, TDIV, DWDT, RTOP, SG2 and EF4T (int_state%EF4T, "
  "rescaled only on DFI steps). In: PSDT, TDIV, DWDT, RTOP, SG2, PINT -> OMGALF, TCT, PINT "
  "(T tendency applied by dyn-updatet).", deps=["dyn-dht", "dyn-pdtsdt", "dyn-pgforce-geopot"])
u("dyn-updates", D, f"{DR}:2353-2405",
  "Apply the tracer tendency to every tracer in kss..kse (q_tracer and tracers calls). In: TRACERS, TRACERS_TEND -> TRACERS, zeroed TRACERS_TEND.",
  deps=["dyn-mono-fix"])
u("dyn-updatet", D, f"{DR}:2410-2456",
  "Apply the T tendency (advection + omega-alpha). In: T, TCT -> T.", deps=["dyn-vtoa"])
u("dyn-updateuv", D, f"{DR}:2461-2526",
  "Apply the U, V tendencies TCU, TCV. In: U, V, TCU, TCV -> U, V.", deps=["dyn-pgforce-uvupdate"])
u("dyn-hdiff-coef", D, f"{DR}:2531-2833",
  "HDIFF setup and grand vertical loop: deformation and nonlinear Smagorinsky coefficients "
  "(smag2 0.4, 4x for Q and CW), boundary zeroing. In: U, V, T, Q, CW, Q2 -> diffusion coefficients.",
  deps=BASE)
u("dyn-hdiff-apply", D, f"{DR}:2834-2915",
  "Apply second-order horizontal diffusion contributions behind mass points to T, Q, CW, Q2, U, "
  "V. In: coefficients, fields -> T, Q, CW, Q2, U, V.", deps=["dyn-hdiff-coef"])
u("dyn-cdzdt-nonhydro", D, f"{DR}:2920-3139",
  "CDZDT setup and nonhydrostatic equation: height Z of interfaces and local dz/dt from the mass "
  "field. In: T, Q, PINT, PD, FIS -> Z, W (local part).", deps=["dyn-pdtsdt"])
u("dyn-cdzdt-advect-z", D, f"{DR}:3140-3235",
  "Grand horizontal loop: mass fluxes on H points, horizontal advection of height, w=0 on the "
  "boundaries. In: Z, U, V, PD -> W.", deps=["dyn-cdzdt-nonhydro"])
u("dyn-cdzdt-extmode", D, f"{DR}:3236-3273",
  "Take the external (barotropic) mode out of W. In: W, PD, DSG2 -> W.", deps=["dyn-cdzdt-advect-z"])
u("dyn-cdwdt-boundary-smooth", D, f"{DR}:3278-3532",
  "CDWDT setup; no w diffusion in the backward DFI step; smoothing of w on H points along the "
  "boundaries. In: W, DWDT -> W.", deps=["dyn-cdzdt-extmode"])
u("dyn-cdwdt-local-deriv", D, f"{DR}:3533-3552",
  "Local time derivative of w on H points. In: W, W old -> DWDT.", deps=["dyn-cdwdt-boundary-smooth"])
u("dyn-cdwdt-diffusion", D, f"{DR}:3553-3592",
  "Lateral diffusion of w. In: W, coefficients -> DWDT.",
  deps=["dyn-cdwdt-local-deriv", "dyn-hdiff-coef"])
u("dyn-cdwdt-vert-adv", D, f"{DR}:3593-3652",
  "Vertical advection of w (column). In: W, PSGDT -> DWDT.", deps=["dyn-cdwdt-diffusion"])
u("dyn-cdwdt-horiz-adv", D, f"{DR}:3653-3727",
  "Horizontal advection of w. In: W, U, V -> DWDT.", deps=["dyn-cdwdt-vert-adv"])
u("dyn-cdwdt-extmode", D, f"{DR}:3728-3772",
  "Take the external mode out of dw/dt, cos^2 attenuation in the top 15 hPa. In: DWDT -> DWDT.",
  deps=["dyn-cdwdt-horiz-adv"])
u("dyn-cdwdt-filter", D, f"{DR}:3773-3899",
  "Spatial filtering of DWDT, DWDT set to 1 on boundary H points. In: DWDT -> DWDT.",
  deps=["dyn-cdwdt-extmode"])
u("dyn-vsound", D, f"{DR}:3904-4234",
  "Implicit vertically propagating sound waves: one tridiagonal per column (b1, b2, b3, c0, chi), "
  "updates PINT, W, T; DWDT=1 on boundaries. In: PINT, W, DWDT, T, PD -> PINT, W, T.",
  deps=["dyn-cdwdt-filter"])
u("dyn-adv2-q2-mid", D, f"{DR}:4238-4462",
  "ADV2 setup for a tracer group, interpolation of Q2 (2*TKE) from interfaces to midlayers, sqrt "
  "transform for positive definiteness. In: TRACERS, Q2 -> advected-variable scratch.",
  deps=["dyn-pdtsdt"])
u("dyn-adv2-vert", D, f"{DR}:4463-4569",
  "Crank-Nicolson vertical advection of the tracer group (column sweeps). In: scratch, PSGDT -> "
  "tendencies.", deps=["dyn-adv2-q2-mid"])
u("dyn-adv2-horiz", D, f"{DR}:4570-4608",
  "Tracer fluxes on H points and horizontal advection of species. In: scratch, U, V, PD -> "
  "tendencies.", deps=["dyn-adv2-vert"])
u("dyn-adv2-bndry", D, f"{DR}:4609-4763",
  "Regional branch: upstream advection of species along the four boundaries. In: scratch, U, V "
  "-> tendencies.", deps=["dyn-adv2-horiz"])
u("dyn-mono-fix", D, f"{DR}:4768-5075",
  "MONO setup and per-species monotonisation (local min/max limiter, sum of positive and "
  "negative changes). In: tracer, tendency -> limited tendency, local sums.", deps=["dyn-adv2-bndry"])
u("dyn-mono-global-sum", D, f"{DR}:5076-5203",
  "Global mass sums per species and the conservation rescale factor, as a device reduction "
  "(replaces MPI_Allreduce). In: local sums -> scale factors (device).", deps=["dyn-mono-fix"])
u("dyn-mono-q2-back", D, f"{DR}:5204-5235",
  "Interpolate Q2 tendencies and Q2 from midlayers back to interfaces. In: scratch -> Q2.",
  deps=["dyn-mono-global-sum"])

# ------------------------------------------------------------------ boundary
B = "boundary"
SKIP_GLOBAL = "skip: global-only polar code, nam_configfile global: false (regional parent)."
u("bnd-boundary-init", B, f"{CT}:1145-1774",
  "Boundary-zone strip copies of the parent edges (PD, T, Q, CW, U, V; S, N, W, E) into ctx.bc as keys SIDE_NAME with a [2, ...] slot axis (slot 0 = value). "
  "The lnsh/lnsv blending weights are applied in bnd-bocoh-blend and bnd-bocov-blend, not here. In: grid, state -> ctx.bc arrays.", deps=["init-grid-consts"])
u("bnd-read-bc", B, f"{FB}:3579-3925",
  "When a new 3-h boundary interval starts: load the next GFS-derived boundary values and form "
  "the per-step tendencies (PDB, TB, QB, CWB, Q2B, UB, VB). In: ctx.bc intervals -> ctx.bc tendencies.",
  deps=["init-boco-tendencies", "bnd-boundary-init"])
u("bnd-write-bc", B, f"{FB}:3928-5059",
  "Save boundary values from the model state at NTIMESTEP 0 (used so DFI starts from the "
  "analysis edges) and on demand. In: state -> ctx.bc.", deps=["bnd-boundary-init"])
u("bnd-bocoh-update", B, f"{FB}:5064-5354",
  "BOCOH setup: advance the boundary working objects at H points on the S/N and W/E strips by "
  "their tendencies. In: ctx.bc -> ctx.bc values.", deps=["bnd-read-bc"])
u("bnd-bocoh-blend", B, f"{FB}:5355-5962",
  "Write H-point boundary values into PD, T, Q, CW, Q2 and PINT on each side and the corners, "
  "blended over the lnsh rows. In: ctx.bc, state -> PD, T, Q, CW, Q2, PINT.", deps=["bnd-bocoh-update"])
u("bnd-bocoh-smooth", B, f"{FB}:5963-6387",
  "Optional spatial smoothing of the H boundary rows (S, N, W, E). In: state -> state.",
  deps=["bnd-bocoh-blend"])
u("bnd-bocov-update", B, f"{FB}:6392-6589",
  "BOCOV setup: advance the V-point boundary working objects (S/N, W/E). In: ctx.bc -> ctx.bc.",
  deps=["bnd-read-bc"])
u("bnd-bocov-blend", B, f"{FB}:6590-7010",
  "Write V-point boundary values into U, V blended over lnsv rows, corners. In: ctx.bc -> U, V.",
  deps=["bnd-bocov-update"])
u("bnd-update-bc-tends", B, f"{SV}:8155-9298",
  "Nest boundary tendencies from the parent (I_AM_A_NEST branch).", mode="skip",
  reason="skip: nest-only; the 12 km parent reads its boundaries (bnd-read-bc).")
u("bnd-save-bc-data", B, f"{SV}:9304-9930", "Boundary data saved into restart files.",
  mode="skip", reason="skip: restart writing only; this port writes history, not NMMB restarts.")
u("bnd-build-bc-bundle", B, f"{SV}:7270-8149", "ESMF bundle of boundary arrays.",
  mode="skip", reason="skip: ESMF framework plumbing, replaced by ctx.bc.")
u("bnd-boundary-data-to-domain", B, f"{NG}:5565-6062", "Coupler hand-off of nest boundary data.",
  mode="skip", reason="skip: ESMF parent-child coupler, nests out of the parent run.")
for nm, rng in (("prefft", "72-1034"), ("fftfhn", "1040-1336"), ("fftfwn", "1342-1515"),
                ("fftfuvn", "1521-1834"), ("fftpack", "1858-2862"), ("presmud", "2868-2907"),
                ("poavhn", "2913-3174"), ("swaphn", "3182-3290"), ("polehn", "3298-3372"),
                ("swapwn", "3380-3484"), ("polewn", "3492-3571")):
    u(f"bnd-{nm}", B, f"{FB}:{rng}", f"Global polar operator {nm.upper()}.", mode="skip",
      reason=SKIP_GLOBAL)
u("bnd-halo-exchange", B, "nmm/module_EXCHANGE.F90:1-4572", "MPI halo exchanges (12 signatures).",
  mode="skip", reason="skip: halo-free single-card layout (CONTRACT.md section 1).")
u("bnd-nesting", B, "nmm/module_NESTING.F90:1-13176", "Nest setup and interpolation.",
  mode="skip", reason="skip: one-way nests are not part of the 12 km parent run.")
u("bnd-parent-child-cpl", B, "nmm/module_PARENT_CHILD_CPL_COMP.F90:1-24839",
  "ESMF parent-child coupler.", mode="skip", reason="skip: ESMF coupler; no nests in this run.")

# ------------------------------------------------------------------ driver
R = "driver"
u("drv-time-loop", R, "nmm/module_NMM_INTEGRATE.F90:79-1666",
  "Time loop: clock, forecast length, history alarms, step counter; replaces the ESMF loop. "
  "In: config -> driver.Driver.run.", mode="new", deps=["drv-step-plan"])
u("drv-step-plan", R, f"{SV}:2453-7168",
  "SOLVER_RUN call order and cadences as data (driver.py FIRST_STEP/DYNAMICS/PHYSICS lists); "
  "checks every unit exists. In: UNITS.json -> step plan.", mode="new")
u("drv-loop-limits", R, "nmm/module_DM_PARALLEL.F90:1941-1969",
  "looplimits and the its_b1/ite_h2 style limits as full-domain index ranges in "
  "nmmb_common.cuh macros. In: IM, JM -> limit constants.", mode="port")
u("drv-consts", R, f"{CT}:123-345",
  "CONSTS: derived dynamics constants (dt factors, smag, codamp, wcor scalings) into ctx. "
  "In: config, grid -> ctx.tables['consts'].", deps=["init-grid-consts"])
u("drv-exptbl", R, f"{CT}:1780-1820", "EXPTBL exponential look-up table. In: - -> table.")
u("drv-tablep", R, f"{CT}:1868-1964", "TABLEP pressure/theta table for moist processes. In: - -> table.",
  deps=["drv-spline"])
u("drv-tablet", R, f"{CT}:1970-2068", "TABLET temperature table. In: - -> table.", deps=["drv-spline"])
u("drv-spline", R, f"{CT}:2074-2188", "Cubic spline used to build the tables (host, init only).")
u("drv-psitbl", R, f"{CT}:2189-2300", "PSITBL stability-function table. In: - -> table.")
u("drv-tracer-index", R, f"{SV}:263-2447",
  "Tracer slots of TRACERS (INDX_Q, INDX_CW, INDX_Q2, INDX_O3 ...) as SOLVER_INITIALIZE sets "
  "them for num_tracers_met=4. In: config -> ctx.tables['indx'].", mode="new")
u("drv-update-water", R, f"{SV}:11069-11530",
  "UPDATE_WATER: total condensate CW and F_ICE/F_RAIN/F_RIMEF <-> QC, QR, QS (spec_adv false). "
  "In: CW, F_*, T -> QC, QR, QS, CW.", deps=["drv-tracer-index"])
u("drv-calc-rh-radar-dfi", R, f"{SV}:11537-11590", "RH hold for radar DFI.", mode="skip",
  reason="skip: radar_init 0 (no GSI radar heating in the free forecast).")
u("drv-cltend", R, f"{SV}:11597-11681",
  "CLTEND: spread the microphysics/convection temperature change over NPRECIP steps (Told, "
  "Tadj). In: T, Told, Tadj -> T.")
u("drv-rime-factor-update", R, f"{SV}:11688-11771",
  "RIME_FACTOR_UPDATE after species advection. In: F_RIMEF, QS -> F_RIMEF.")
u("drv-tqadjust", R, f"{SV}:11777-12113",
  "TQADJUST: consistency of T, Q, CW and fractions after physics. In: T, Q, QC, CW, F_* -> same.")
u("drv-rdtemp", R, "phys/module_RA_GFDL.F90:116-233",
  "RDTEMP every step: apply RLWTT + RSWTT to T with the zenith-angle scaling of SW between "
  "radiation calls. In: RLWTT, RSWTT, CZMEAN, GLAT/GLON, time -> T.", deps=["phys-rrtm-coszmn", "phys-rrtm-sol-update"])
u("drv-bucket-reset", R, f"{SV}:5367-5470,6348-6384",
  "Accumulator resets by NCLOD, NRDLW, NRDSW, NSRFC, NPREC (bucket hours from the config). "
  "In: counters -> accumulators.", mode="port")
u("drv-h-to-v", R, "nmm/module_H_TO_V.F90:42-122", "H_TO_V_2D/3D: average H-point fields to V points.")
u("drv-h-to-v-tend", R, "nmm/module_H_TO_V.F90:128-224",
  "H_TO_V_TEND_2D/3D: average DUDT/DVDT from H to V points and apply them to U, V over NPHS. "
  "In: DUDT, DVDT -> U, V.", deps=["drv-h-to-v"])
u("drv-dfi-tdfi-driver", R, f"{NG}:3624-5418",
  "RUN_DIGITAL_FILTER_NMM, TDFI branch (filter_method 3): backward 600 s then forward 600 s at "
  "filt_dt 8 s, state save/restore, filtered state. In: state -> filtered state.",
  deps=["drv-dfi-dyn-init", "drv-dfi-dyn-sum", "drv-dfi-dyn-average", "drv-dfi-phy-save",
        "drv-dfi-phy-restore", "drv-dfi-dolph", "drv-dfi-filt-vars", "drv-step-plan"])
u("drv-dfi-dyn-init", R, f"{DF}:46-203", "DIGITAL_FILTER_DYN_INIT: zero the filter sums. In: - -> sums.")
u("drv-dfi-dyn-sum", R, f"{DF}:208-383", "DIGITAL_FILTER_DYN_SUM: add the weighted state each step.",
  deps=["drv-dfi-dyn-init"])
u("drv-dfi-dyn-average", R, f"{DF}:388-646", "DIGITAL_FILTER_DYN_AVERAGE: filtered state from sums.",
  deps=["drv-dfi-dyn-sum"])
u("drv-dfi-phy-init", R, f"{DF}:650-736", "DIGITAL_FILTER_PHY_INIT.")
u("drv-dfi-phy-save", R, f"{DF}:741-824", "Save physics state (soil, CW etc. 'R' variables) before DFI.")
u("drv-dfi-phy-restore", R, f"{DF}:829-921", "Restore the saved physics state after DFI.",
  deps=["drv-dfi-phy-save"])
u("drv-dfi-dolph", R, f"{DF}:929-1039", "DOLPH + CHEBY: Dolph-Chebyshev filter weights. In: window -> weights.")
u("drv-dfi-filt-vars", R, "namgpu/parm/nam_filt_vars_norad.txt",
  "Read the filter/restore table (F/R/B per variable) into the DFI variable lists.", mode="new")
u("drv-physics-initialize", R, f"{SV}:9936-11062",
  "PHYSICS_INITIALIZE orchestration for the selected schemes (rad_initialize_nmmb, RRTM_INIT, "
  "JSFC_INIT, MYJPBL_INIT, NOAH_LSM_INIT, BMJ_INIT, FERRIER_INIT_HR, GWD_init).",
  deps=["phys-rrtm-init", "phys-jsfc-init", "phys-myj-init", "phys-noah-init", "phys-bmj-init",
        "phys-fer-init", "phys-gwd-init"])
u("drv-timing-receipt", R, "-", "Per-stage wall-clock stamps (no added syncs) into the run receipt.",
  mode="new")
u("drv-gpu-lease", R, "-", "Take and release the shared-card pool lease (OWNER line) around a run.", mode="new")
u("drv-moving-domain", R, f"{SV}:3053-3125", "Moving-domain tracker hooks and SET_DOMAIN_SPECS.",
  mode="skip", reason="skip: my_domain_moves false, ntrack 0.")
u("drv-timeseries", R, "nmm/module_TIMESERIES.F90:1-612", "Point time series output.",
  mode="skip", reason="skip: not part of the parent history; soundings come from history files.")
u("drv-gfs-physics", R, f"{SV}:6077-7000", "GFS physics package branch (gbphys, grrad_gfs).",
  mode="skip", reason="skip: nam_configfile gfs: false.")

# ------------------------------------------------------------------ init (NPS replacement)
I = "init"
u("init-fetch-case", I, "-", "Fetch the case from NODD (NAM native B-grid analysis, NAM 218 "
  "referee, GFS 0.25 boundaries, MRMS). In: cycle -> cases/<case>/raw.", mode="new")
u("init-grib2-decode", I, "-",
  "GRIB2 decode to device arrays through the gpuwm Rust bridges (grib2_inventory/grib2_dump), "
  "incl. GDT 3.32769 (rotated B grid) and GFS complex packing. In: grib2 -> arrays.", mode="new",
  deps=["init-fetch-case"])
u("init-grid-geo", I, f"{CT}:351-1139",
  "Rotated lat-lon geometry (GLAT/GLON of H and V points) checked against NLAT/ELON of the "
  "analysis; fixes grid.rotated_to_geographic if they differ. In: config -> grid lat/lon.",
  deps=["init-grib2-decode"])
u("init-grid-consts", I, f"{CT}:351-1139",
  "GRID_CONSTS in full: dxh, dxv, dyh, wpdar, fcp, fdiv, fad, fah, ddmpu, ddmpv, hdac*, curv, f "
  "row arrays on device. In: grid, config -> ctx.tables['grid'].", deps=["init-grid-geo"])
u("init-vcoord-recover", I, "-",
  "Fit PSGML1(l), SGML2(l) from the analysis PRES on 60 hybrid levels against PD (linear in PD, "
  "per level, on device), then SG1/SG2/DSG*/PDTOP exactly (grid.vcoord_from_mid). In: PRES, PSFC -> grid vcoord.",
  mode="new", deps=["init-grib2-decode"])
u("init-native-atm", I, "-",
  "Map nam bgrd3d00 hybrid fields to the state: TMP->T, SPFH->Q, UGRD/VGRD (V points, grid-relative)"
  "->U/V, CLMR+CICE+RWMR+SNMR->CW (NPS CWM gate: do_clouds and maxval(CLMR)>0), F_ICE/F_RAIN/F_RIMEF=0, Q2=0.02, W=0, PSFC->PD (vcoord 1: PSFC-PT). In: grib -> state.",
  mode="new", deps=["init-vcoord-recover"])
u("init-native-sfc", I, "-",
  "Surface/soil state from bgrd3d00: TSOIL(4)->STC, SOILW(4)->SMC, SH2O from SMC/STC (frozen "
  "fraction), TMP sfc->TSKIN, WTMP->SST, ICEC->SICE, LAND->SM, VGTYP/SOTYP, VEG, ALBDO, SNOD/WEASD, "
  "CNWAT, SFCR, SNOWC, TSOIL 3 m->TG. In: grib -> state.", mode="new", deps=["init-grib2-decode"])
u("init-static-missing", I, "-",
  "Static fields the analysis lacks (snow-free albedo ALBASE, max snow albedo MXSNAL, emissivity, "
  "slope type, ISLOPE, soil/veg table indices) from tables or our static assets. In: types -> state.",
  mode="new", deps=["init-native-sfc"])
u("init-gwd-orog", I, "-",
  "GWD orographic statistics on the B grid (HSTDV, HCNVX, HASYW, HASYS, HASYSW, HASYNW, HLENW, "
  "HLENS, HLENSW, HLENNW, HANGL, HANIS, HSLOP, HZMAX) from 30-arcsec terrain, as NPS geogrid do_gwd. In: DEM -> state.",
  mode="new", deps=["init-grid-geo"])
u("init-derive-state", I, "-",
  "Derived prognostics: PINT from PD and SG1/SG2, FIS, Z, DWDT=1, OMGALF, RTOP, THS/QSH, Told, "
  "TRACERS assembly (INDX_*), UP/VP/TP = current. In: state -> state.", mode="new",
  deps=["init-native-atm", "init-native-sfc", "drv-tracer-index"])
u("init-gfs-decode", I, "-",
  "GFS 0.25 deg pressure-level HGT, TMP, SPFH, UGRD, VGRD, CLMR, ICMR, RWMR, SNMR + surface PRES/HGT "
  "for each 3-h boundary time to device. In: gfs grib -> arrays.", mode="new", deps=["init-grib2-decode"])
u("init-gfs-hinterp", I, "-",
  "Horizontal interpolation of GFS fields to the B-grid boundary strips (lnsh/lnsv = 5 rows, H and V "
  "points), bilinear on the GPU (METGRID.TBL.bocos rules). In: GFS arrays -> strip arrays.",
  mode="new", deps=["init-gfs-decode", "init-grid-geo"])
u("init-gfs-vinterp", I, "-",
  "Vertical interpolation to NMMB hybrid levels as nemsinterp does: PD from hydrostatic integration "
  "to model terrain (FIS), T, Q, CW, U, V at mid levels, ptsgm split. In: strips -> boundary values.",
  mode="new", deps=["init-gfs-hinterp", "init-vcoord-recover"])
u("init-gfs-wind-rotate", I, "-",
  "Rotate GFS earth-relative winds to rotated-grid-relative at V points. In: U, V earth -> grid.",
  mode="new", deps=["init-gfs-hinterp"])
u("init-boco-tendencies", I, "-",
  "Boundary values for every 3-h interval and their tendencies (the boco.#### content with "
  "boundary_flux true), first interval from the native analysis edges. In: values -> ctx.bc.",
  mode="new", deps=["init-gfs-vinterp", "init-gfs-wind-rotate", "init-derive-state"])
u("init-case-driver", I, "-",
  "namgpu/init/case_init.run: whole init chain in order with stage timings. In: case dir -> ctx.",
  mode="new", deps=["init-boco-tendencies", "init-static-missing", "init-gwd-orog", "drv-physics-initialize"])
u("init-read-nemsio", I, "nmm/module_INIT_READ_NEMSIO.F90:37-4169", "NEMSIO input reader.",
  mode="skip", reason="skip: inputs are GRIB2 from NODD; the NEMSIO field list is the checklist "
  "for init-native-atm/sfc.")

# ------------------------------------------------------------------ io
O = "io"
u("io-history-fields", O, "-",
  "Build the wrfout-shaped history fields on device: PSFC, theta, P, PH, QVAPOR, QCLOUD, QRAIN, "
  "QICE, QSNOW, earth-relative U10/V10, RAINNC/RAINC from NMMB arrays. In: state -> fields.",
  mode="new", deps=["io-wind-rotate"])
u("io-wind-rotate", O, "nam_post0.fd/PROF_NMMB.f:2115-2150,2232-2237",
  "Rotate grid-relative V-point winds (3-D and 10 m) to earth-relative H-point winds: each in-domain V neighbour rotated with its exact inverse of init-gfs-wind-rotate's map, then averaged (decision: pole inside domain, see kernel). In: U, V, U10, V10 -> UEARTH, VEARTH, UE10, VE10.",
  mode="port")
u("io-history", O, "-", "Write one history NetCDF per output hour (io/history.py). In: fields -> file.",
  mode="new", deps=["io-history-fields"])
u("io-history-rust", O, "-", "Bind the gpuwm Rust NetCDF writer (libnetcdf_writer.so) for history.",
  mode="new", deps=["io-history"])
u("io-map-regrid", O, "-",
  "If rw_wrfbatch refuses MAP_PROJ 203, regrid history to the NAM 218 Lambert frame on the GPU "
  "for maps. In: history -> companion file.", mode="new", deps=["io-history"])
u("io-maps", O, "-", "Maps with rw_wrfbatch (io/maps.py) beside NAM 218 and MRMS sheets.", mode="new",
  deps=["io-history", "io-map-regrid"])
u("io-referee-sheets", O, "-",
  "Three-panel sheets per hour: MRMS | WOOF-NAM | archived NAM (composite reflectivity, 2 m T, "
  "precip) with the Rust renderer. In: case raw + history -> PNGs.", mode="new", deps=["io-maps"])
u("io-max-fields", O, f"{DG}:2190-2518",
  "MAX_FIELDS_HR: hourly maxima (REFDMAX, UPHLMAX, wind max, T2 max/min ...). In: state -> maxima.",
  deps=["io-uphlcy"])
u("io-uphlcy", O, f"{DG}:2114-2186", "CALC_UPHLCY: updraft helicity 2-5 km. In: W, U, V, Z -> UPHL.")
u("io-quasipost-driver", O, f"{QP}:68-255", "QUASIPOST driver: in-model post fields for history.",
  deps=["io-quasipost-3d", "io-quasipost-interp", "io-quasipost-mslp"])
u("io-quasipost-3d", O, f"{QP}:257-686", "calculate_3D: 3-D diagnostic fields (heights, RH ...).")
u("io-quasipost-interp", O, f"{QP}:688-815", "calculate_interP: interpolation to pressure levels.")
u("io-quasipost-mslp", O, f"{QP}:817-852", "smoothMSLP: Shuell/NAM MSLP smoothing.")
u("io-wrt-pcp", O, f"{DG}:1454-1584", "Write precip for assimilation.", mode="skip",
  reason="skip: write_prec_adj false.")
u("io-nemsio-write", O, "nmm/module_WRITE_ROUTINES.F90:1-5247", "NEMSIO history/restart writer.",
  mode="skip", reason="skip: history is NetCDF for our renderer.")

# ------------------------------------------------------------------ physics: driver glue
PD = "physics-driver"
u("phys-rad-driver", PD, f"{RA}:66-995",
  "RADIATION: per-call setup, cloud fractions (np3d 3), RRTM call, outputs RLWTT, RSWTT, RLWIN, "
  "RSWIN, TOA fluxes, cloud cover. In: state -> radiation fields.",
  deps=["phys-rrtm-driver-prep", "phys-rad-cal-cldfra"])
u("phys-rad-radconst", PD, f"{RA}:1000-1042", "RADCONST: DEAD IN NAM (no caller in nam.v4.2.11; definition only). Faithful port, no consumer.")
u("phys-rad-cal-cldfra", PD, f"{RA}:1045-1134", "CAL_CLDFRA: cloud fraction from condensate.")
u("phys-turbl-driver", PD, "nmm/module_TURBULENCE.F90:89-1420",
  "TURBL glue every NPHS: column prep (theta, exner, Z), calls JSFC, NOAHLSM, SFCDIAGS, MYJPBL, "
  "GWD_DRIVER, accumulates fluxes, DUDT/DVDT. In: state -> tendencies and surface fields.",
  deps=["phys-jsfc-jsfc", "phys-noah-driver", "phys-noah-sfcdiags", "phys-myj-myjpbl", "phys-gwd-driver"])
u("phys-bmj-driver", PD, "nmm/module_CONVECTION.F90:48-658",
  "CUCNVC glue every NPRECIP: column prep, BMJ call, CUPREC/ACPREC, CNVTOP/BOT, T/Q tendency. In: "
  "state -> tendencies.", deps=["phys-bmj-bmjdrv"])
u("phys-fer-driver", PD, "nmm/module_MICROPHYSICS.F90:60-558",
  "GSMDRIVE glue every NPRECIP: column prep, FER_HIRES call, precip accumulation, T/Q/CW update. "
  "In: state -> state.", deps=["phys-fer-fer-hires"])
u("phys-precip-adjust", PD, "nmm/module_PRECIP_ADJUST.F90:26-334", "READPCP/CHKSNOW/ADJPPT.",
  mode="skip", reason="skip: pcpflg false in the free forecast (exnam_fcst.sh:276).")
for nm, rng, why in (("etamp-new", "phys/module_MP_ETANEW.F90:1-2627", "microphysics fer_hires, not fer"),
                     ("wsm6", "phys/module_MP_WSM6.F90:1-1815", "microphysics fer_hires"),
                     ("thompson", "phys/module_mp_thompson.F90:1-4753", "microphysics fer_hires"),
                     ("gfsmp", "phys/module_MP_GFS.F90:1-327", "microphysics fer_hires"),
                     ("sas", "phys/module_CU_SAS.F90:1-449", "convection bmj"),
                     ("sashur", "phys/module_CU_SASHUR.F90:1-5882", "convection bmj"),
                     ("gfspbl", "phys/module_BL_GFSPBL.F90:1-387", "turbulence myj"),
                     ("liss", "phys/module_LS_LISS.F90:1-2130", "land_surface noah"),
                     ("sf-gfdl", "phys/module_SF_GFDL.F90:1-2145", "sfc_layer myj"),
                     ("ra-gfdl", "phys/module_RA_GFDL.F90:1-115", "shortwave/longwave rrtm")):
    u(f"phys-alt-{nm}", PD, rng, f"Alternative scheme {nm}, not selected.", mode="skip",
      reason=f"skip: nam_configfile selects {why}.")

# ------------------------------------------------------------------ physics: Ferrier-Aligo
F = "physics-fer_hires"
u("phys-fer-module-data", F, f"{FH}:1-207",
  "Module constants and table declarations of MP_FER_HIRES (NLImax, NSI_max, RHgrd, table sizes) "
  "into fer_hires.cuh.", deps=[])
u("phys-fer-init", F, f"{FH}:2391-2744",
  "FERRIER_INIT_hr: build MASSR, VENTR1/2, ACCRR, MASSI, VSNOWI, VEL_RF and the growth tables; "
  "upload once. In: dt, dx -> ctx.tables['fer'].", deps=["phys-fer-module-data", "phys-fer-growth-rates", "phys-fer-gpvs"])
u("phys-fer-growth-rates", F, f"{FH}:2746-2785", "MY_GROWTH_RATES_NMM_hr: ice growth table for DTPH.")
u("phys-fer-gpvs", F, f"{FH}:2791-2836", "GPVS_hr: saturation vapour pressure table.")
u("phys-fer-fpvs", F, f"{FH}:2840-2989", "FPVS, FPVS0, FPVSX, FPVSX0 device functions.", deps=["phys-fer-gpvs"])
u("phys-fer-fer-hires", F, f"{FH}:208-604",
  "FER_HIRES column driver: species from CW and fractions, per-column call of EGCP01COLUMN_hr, "
  "write back T, Q, CW, F_*, precip, reflectivity inputs. In: state -> state (one thread per column).",
  deps=["phys-fer-col-setup", "phys-fer-col-satcheck", "phys-fer-col-ice-props",
        "phys-fer-col-processes", "phys-fer-col-melt-evap", "phys-fer-col-rain-sedim"])
u("phys-fer-col-setup", F, f"{FH}:650-910",
  "EGCP01COLUMN_hr arguments, local tables, top-down level loop, precip falling in from above "
  "(rain, ice). Device function on one column.", deps=["phys-fer-init"])
u("phys-fer-col-satcheck", F, f"{FH}:911-1028",
  "Grid-scale saturation checks, effective saturation mixing ratios, RHO, THICK and process "
  "initialisation.", deps=["phys-fer-col-setup"])
u("phys-fer-col-ice-props", F, f"{FH}:1029-1320",
  "Decision tree, time-averaged properties, large-ice number fraction, mean size (Houze), rime "
  "factor limits, NLICE bounds.", deps=["phys-fer-col-satcheck"])
u("phys-fer-col-processes", F, f"{FH}:1321-1478",
  "Autoconversion and collection, homogeneous freezing, ice deposition/sublimation, condensation, "
  "rime freezing limits.", deps=["phys-fer-col-ice-props", "phys-fer-condense", "phys-fer-deposit"])
u("phys-fer-col-melt-evap", F, f"{FH}:1479-1587",
  "Melting and evaporation of snow/ice, shedding, melt limits.", deps=["phys-fer-col-processes"])
u("phys-fer-col-rain-sedim", F, f"{FH}:1588-2159",
  "Rain evaporation, accretion, rain/ice fall-out and sedimentation, final updates of the column.",
  deps=["phys-fer-col-melt-evap", "phys-fer-get-indexr"])
u("phys-fer-condense", F, f"{FH}:2160-2225", "CONDENSE: cloud water condensation function.", deps=["phys-fer-fpvs"])
u("phys-fer-deposit", F, f"{FH}:2231-2289", "DEPOSIT: ice deposition function.", deps=["phys-fer-fpvs"])
u("phys-fer-get-indexr", F, f"{FH}:2297-2380", "GET_INDEXR: rain-size table index.")
u("phys-fer-calmict", F, f"{DG}:1589-1798",
  "CALMICT: Ferrier radar reflectivity (REFL_10CM, composite) and species diagnostics for history.",
  deps=["phys-fer-init"])

# ------------------------------------------------------------------ physics: BMJ
J = "physics-bmj"
u("phys-bmj-init", J, f"{BJ}:2158-2425", "BMJ_INIT: tables (ptbl, ttbl, qs0, sqs, the, sthe). In: - -> tables.",
  deps=["phys-bmj-spline"])
u("phys-bmj-spline", J, f"{BJ}:2429-2547", "SPLINE for the BMJ tables (init only).")
u("phys-bmj-ttblex", J, f"{BJ}:2071-2154", "TTBLEX: table look-up device function.", deps=["phys-bmj-init"])
u("phys-bmj-bmjdrv", J, f"{BJ}:117-395",
  "BMJDRV: per-column setup, call BMJ, tendencies and precipitation out. One thread per column.",
  deps=["phys-bmj-col-prep", "phys-bmj-col-cloudbase", "phys-bmj-col-cloudtop", "phys-bmj-col-deep-ref",
        "phys-bmj-col-deep-iter", "phys-bmj-col-deep-tend", "phys-bmj-col-shallow-top", "phys-bmj-col-shallow-adjust"])
u("phys-bmj-col-prep", J, f"{BJ}:399-597", "BMJ arguments, locals, preparations of the column.", deps=["phys-bmj-ttblex"])
u("phys-bmj-col-cloudbase", J, f"{BJ}:598-696",
  "Search for the max-buoyancy parcel, table indices, saturation point, cloud base.", deps=["phys-bmj-col-prep"])
u("phys-bmj-col-cloudtop", J, f"{BJ}:697-939",
  "Entrainment, entropy check, updraft T along the moist adiabat, CAPE, cloud top, quick exits.",
  deps=["phys-bmj-col-cloudbase"])
u("phys-bmj-col-deep-ref", J, f"{BJ}:940-1084",
  "Deep convection: reference temperature and humidity profiles, freezing level, first adjustment.",
  deps=["phys-bmj-col-cloudtop"])
u("phys-bmj-col-deep-iter", J, f"{BJ}:1085-1179",
  "Cloud-efficiency iteration, enthalpy conservation, lqm, T/Q corrections.", deps=["phys-bmj-col-deep-ref"])
u("phys-bmj-col-deep-tend", J, f"{BJ}:1180-1316",
  "Deep convection heating, moistening and precipitation.", deps=["phys-bmj-col-deep-iter"])
u("phys-bmj-col-shallow-top", J, f"{BJ}:1317-1449",
  "Shallow cloud top search (virtual-temperature excess), gather shallow points.", deps=["phys-bmj-col-cloudtop"])
u("phys-bmj-col-shallow-adjust", J, f"{BJ}:1450-2067",
  "Shallow convection adjustment with the configfile switches (fres, fr, fsl, fss, newswap), swap "
  "points, final tendencies.", deps=["phys-bmj-col-shallow-top"])

# ------------------------------------------------------------------ physics: MYJ PBL (copy-arwen)
M = "physics-myj"
AW = "namgpu/third_party/arwen/core/kernels/myjpbl.cu"
for nm, rng, what in (("myjpbl", "180-753", "MYJPBL column driver: theta, Q2, mixing, tendencies"),
                      ("mixlen", "758-965", "MIXLEN: master length scale"),
                      ("prodq2", "969-1173", "PRODQ2: TKE production/dissipation"),
                      ("difcof", "1178-1337", "DIFCOF: exchange coefficients"),
                      ("vdifq", "1342-1420", "VDIFQ: vertical diffusion of Q2"),
                      ("vdifh", "1425-1522", "VDIFH: vertical diffusion of heat/moisture"),
                      ("vdifv", "1527-1607", "VDIFV: vertical diffusion of momentum")):
    u(f"phys-myj-{nm}", M, f"{MY}:{rng}",
      f"{what}. Start from ArWen {AW} (bottom-up, WRF v4.6.1) and bring it to the NAM file "
      "(top-down, NAM constants; NAM copy is ~26% line-identical to WRF).",
      mode="copy-arwen",
      deps=(["phys-myj-init", "phys-myj-mixlen"] if nm == "prodq2" else ["phys-myj-init"]), arwen=AW)
u("phys-myj-init", M, f"{MY}:1612-1657", "MYJPBL_INIT: initial Q2, EL_PBL, tables.", mode="port")
u("phys-myj-tables", M, f"{MY}:1661-2029", "HERFTBL/HERF, TABLEPT/TABLETT, SPLINE: MYJ look-up tables.",
  deps=["phys-myj-init"])

# ------------------------------------------------------------------ physics: JSFC surface layer
S = "physics-jsfc"
AJ = "namgpu/third_party/arwen/core/kernels/myjsfc.cu"
u("phys-jsfc-jsfc", S, f"{JS}:80-354",
  f"JSFC driver: surface layer per column (USTAR, Z0, AKHS, AKMS, THZ0, QZ0, U10, V10, TH10, Q10). "
  f"Start from ArWen {AJ}.", mode="copy-arwen", deps=["phys-jsfc-sfcdif"], arwen=AJ)
u("phys-jsfc-sfcdif", S, f"{JS}:357-1011",
  f"SFCDIF: Janjic similarity, viscous sublayer over water, iterations. Start from ArWen {AJ}; NAM "
  "copy ~80% identical to WRF.", mode="copy-arwen", deps=["phys-jsfc-init"], arwen=AJ)
u("phys-jsfc-init", S, f"{JS}:1014-1251", "JSFC_INIT: PSIM/PSIH tables, initial USTAR, Z0. Ref ArWen "
  "namgpu/third_party/arwen/core/myjsfc_tables.py.", mode="port")

# ------------------------------------------------------------------ physics: Noah (copy-arwen)
N = "physics-noah"
AN = "namgpu/third_party/arwen/core/kernels/noah.cu"
u("phys-noah-init", N, f"{NO}:1164-1293", "NOAH_LSM_INIT: soil/veg parameters per point, SH2O, initial state.",
  deps=["phys-noah-parm-tables"])
u("phys-noah-parm-tables", N, f"{NO}:1298-1547",
  "LSM_PARM_INIT + SOIL_VEG_GEN_PARM: IGBP veg (ivegsrc 1) and soil parameter tables as the NAM sets them.")
u("phys-noah-driver", N, f"{NO}:188-1161",
  "NOAHLSM driver: per land point prep (forcing from the lowest layer, radiation, precip), call "
  "SFLX, write back STC, SMC, SH2O, TSKIN, SNO, CMC, fluxes, runoff. One thread per column.",
  deps=["phys-noah-sflx"])
u("phys-noah-sflx", N, f"{NO}:1550-2278",
  f"SFLX column physics with the NCEP v4 changes (frozen-soil transpiration, five-veg roughness). "
  f"Start from ArWen {AN}.", mode="copy-arwen", arwen=AN,
  deps=["phys-noah-redprm", "phys-noah-nopac", "phys-noah-snopac", "phys-noah-penman",
        "phys-noah-canres", "phys-noah-alcalc", "phys-noah-csnow", "phys-noah-snfrac", "phys-noah-tdfcnd"])
for nm, rng, what in (("alcalc", "2281-2328", "ALCALC: albedo with snow"),
                      ("canres", "2331-2469", "CANRES: canopy resistance"),
                      ("csnow", "2472-2511", "CSNOW: snow thermal conductivity"),
                      ("devap", "2514-2554", "DEVAP: bare-soil evaporation"),
                      ("evapo", "2557-2641", "EVAPO: total evaporation"),
                      ("frh2o", "2644-2782", "FRH2O: unfrozen soil water"),
                      ("hrt", "2785-3035", "HRT: soil heat equation RHS"),
                      ("hrtice", "3038-3150", "HRTICE: sea-ice/glacier heat RHS"),
                      ("hstep", "3153-3201", "HSTEP: soil temperature step"),
                      ("nopac", "3204-3387", "NOPAC: no-snow energy/water balance"),
                      ("penman", "3390-3468", "PENMAN: potential evaporation"),
                      ("redprm", "3471-3664", "REDPRM: per-point parameters, NCEP roughness change"),
                      ("rosr12", "3666-3725", "ROSR12: tridiagonal solver"),
                      ("shflx", "3729-3796", "SHFLX: soil heat flux"),
                      ("smflx", "3799-3916", "SMFLX: soil moisture flux"),
                      ("snfrac", "3920-3957", "SNFRAC: snow cover fraction"),
                      ("snksrc", "3960-4047", "SNKSRC: phase-change sink/source"),
                      ("snopac", "4050-4374", "SNOPAC: snow energy/water balance"),
                      ("snowpack", "4378-4505", "SNOWPACK: snow compaction"),
                      ("snowz0", "4508-4526", "SNOWZ0: roughness with snow"),
                      ("snow-new", "4530-4600", "SNOW_NEW: new-snow density"),
                      ("srt", "4603-4852", "SRT: soil water RHS, frozen-soil transpiration change"),
                      ("sstep", "4855-4942", "SSTEP: soil moisture step"),
                      ("tbnd", "4945-4985", "TBND: boundary-layer temperature interpolation"),
                      ("tdfcnd", "4989-5092", "TDFCND: soil thermal diffusivity"),
                      ("tmpavg", "5095-5197", "TMPAVG: layer-mean temperature"),
                      ("transp", "5200-5309", "TRANSP: transpiration"),
                      ("wdfcnd", "5312-5369", "WDFCND: soil water diffusivity")):
    u(f"phys-noah-{nm}", N, f"{NO}:{rng}", f"{what}. Device function; start from ArWen {AN}.",
      mode="copy-arwen", arwen=AN, deps=["phys-noah-parm-tables"])
u("phys-noah-sfcdiags", N, f"{NO}:5609-5656", "SFCDIAGS: 2 m T and Q (TSHLTR, QSHLTR) from fluxes.")
u("phys-noah-sfcdif-off", N, f"{NO}:5372-5606", "Noah's own SFCDIF.", mode="skip",
  reason="skip: '_off' copy, not called; JSFC provides the exchange coefficients.")

# ------------------------------------------------------------------ physics: GWD
G = "physics-gwd"
u("phys-gwd-init", G, f"{GW}:32-119", "GWD_init: constants and orography statistics into device arrays.",
  deps=["init-gwd-orog"])
u("phys-gwd-driver", G, f"{GW}:125-362",
  "GWD_DRIVER: column prep (rotate winds, pressure, exner), call GWD_col, DUDT/DVDT out (cdmb 0.1, "
  "cleffamp 1.0, sigfac 3.0, factop 0.5, rlolev 50, dpmin 0).", deps=["phys-gwd-col-blocking", "phys-gwd-col-wave"])
u("phys-gwd-col-blocking", G, f"{GW}:368-897", "GWD_col part 1: setup, Lott-Miller mountain blocking, mountain-induced drag terms.",
  deps=["phys-gwd-init"])
u("phys-gwd-col-wave", G, f"{GW}:898-1132", "GWD_col part 2: low-level wind, base stress, Ri-limited stress profile, top de-acceleration limit.",
  deps=["phys-gwd-col-blocking"])

# ------------------------------------------------------------------ physics: NMMB RRTM
X = "physics-rrtm"
u("phys-rrtm-init", X, f"{RR}:1424-1517", "RRTM_INIT + GAUSIN: driver init, Gaussian angles.",
  deps=["phys-rrtm-rad-initialize"])
u("phys-rrtm-rad-initialize", X, "phys/rad_initialize_nmmb.f:2-224",
  "rad_initialize_nmmb: physparam flags (icldflg, ioznflg, icmphys), calls radinit.", deps=["phys-rrtm-radinit"])
u("phys-rrtm-radinit", X, f"{GR}:226-451", "radinit_nmmb: init of astronomy, aerosols, gases, clouds, LW, SW.",
  deps=["phys-rrtm-sol-init", "phys-rrtm-aer-init", "phys-rrtm-clim-aerinit", "phys-rrtm-gas-init",
        "phys-rrtm-cld-init", "phys-rrtm-sfc-init", "phys-rrtm-lw-rlwinit", "phys-rrtm-sw-rswinit"])
u("phys-rrtm-radupdate", X, f"{GR}:456-602", "radupdate_nmmb: date-dependent solar constant, aerosols, gases.",
  deps=["phys-rrtm-sol-update", "phys-rrtm-aer-update", "phys-rrtm-gas-update"])
u("phys-rrtm-dayparts", X, f"{GR}:606-656", "dayparts: sunlit-point selection.")
u("phys-rrtm-driver-prep", X, f"{RR}:98-878",
  "RRTM: column inputs from the NMMB state (plyr, plvl, tlyr, qlyr, o3, clouds from Ferrier "
  "fractions), CHNK_RRTM chunking dropped (all columns at once).", deps=["phys-rrtm-dayparts"])
u("phys-rrtm-driver-call", X, f"{RR}:879-1112", "RRTM: dayparts call, zenith angles, call setup.",
  deps=["phys-rrtm-driver-prep", "phys-rrtm-coszmn"])
u("phys-rrtm-driver-post", X, f"{RR}:1113-1418",
  "RRTM: grrad_nmmb call and outputs RLWTT, RSWTT, RLWIN, RSWIN, RSWOUT, TOA, CZMEAN, cloud cover.",
  deps=["phys-rrtm-grrad-setup", "phys-rrtm-grrad-gases", "phys-rrtm-grrad-aerosols",
        "phys-rrtm-grrad-clouds", "phys-rrtm-grrad-sw", "phys-rrtm-grrad-lw"])
u("phys-rrtm-grrad-setup", X, f"{GR}:660-1396", "grrad_nmmb arguments, layer arrays, unit conversions.")
u("phys-rrtm-grrad-gases", X, f"{GR}:1397-1539", "grrad_nmmb: ozone (getozn), coszmn, getgases.",
  deps=["phys-rrtm-getozn", "phys-rrtm-getgases", "phys-rrtm-coszmn"])
u("phys-rrtm-grrad-aerosols", X, f"{GR}:1540-1576", "grrad_nmmb: setaer call.", deps=["phys-rrtm-setaer"])
u("phys-rrtm-grrad-clouds", X, f"{GR}:1577-1817", "grrad_nmmb: cloud properties, np3d 3 -> progcld2.",
  deps=["phys-rrtm-progcld2"])
u("phys-rrtm-grrad-sw", X, f"{GR}:1818-1962", "grrad_nmmb: setalb and swrad calls, SW heating/fluxes.",
  deps=["phys-rrtm-setalb", "phys-rrtm-sw-swrad"])
u("phys-rrtm-grrad-lw", X, f"{GR}:1963-2207", "grrad_nmmb: setemis and lwrad calls, LW heating/fluxes, outputs.",
  deps=["phys-rrtm-setemis", "phys-rrtm-lw-lwrad"])
# astronomy, aerosols, gases, clouds, surface
for nm, path, rng, what in (
        ("sol-init", "phys/radiation_astronomy_nmmb.f", "113-259", "sol_init_nmmb: solar constant tables"),
        ("sol-update", "phys/radiation_astronomy_nmmb.f", "265-570", "sol_update_nmmb + solar: declination, equation of time"),
        ("coszmn", "phys/radiation_astronomy_nmmb.f", "744-848", "coszmn_nmmb: mean cos zenith over the call period"),
        ("aer-init", "phys/radiation_aerosols_nmmb.f", "445-816", "aer_init (+wrt_aerlog, set_spectrum, set_volcaer)"),
        ("clim-aerinit", "phys/radiation_aerosols_nmmb.f", "821-1498", "clim_aerinit, set_aercoef, optavg: aerosol optics; data namgpu/parm/nam_climaeropac_global.txt"),
        ("aer-update", "phys/radiation_aerosols_nmmb.f", "1504-1839", "aer_update, trop_update, volc_update"),
        ("setaer", "phys/radiation_aerosols_nmmb.f", "1845-2371", "setaer: aerosol optical properties per layer"),
        ("aer-property", "phys/radiation_aerosols_nmmb.f", "2377-3107", "aer_property + radclimaer"),
        ("gas-init", "phys/radiation_gases_nmmb.f", "186-483", "gas_init: CO2/trace gas setup (co2tf 1)"),
        ("gas-update", "phys/radiation_gases_nmmb.f", "488-838", "gas_update: monthly CO2"),
        ("getgases", "phys/radiation_gases_nmmb.f", "843-979", "getgases: gas profiles per column"),
        ("getozn", "phys/radiation_gases_nmmb.f", "984-1094", "getozn: climatological ozone"),
        ("cld-init", "phys/radiation_clouds_nmmb.f", "225-349", "cld_init: cloud scheme flags"),
        ("progcld2", "phys/radiation_clouds_nmmb.f", "787-1323", "progcld2: Ferrier cloud optical properties (np3d 3)"),
        ("gethml", "phys/radiation_clouds_nmmb.f", "2167-2494", "gethml: high/mid/low cloud layers"),
        ("rsipath2", "phys/radiation_clouds_nmmb.f", "2779-2964", "rsipath2_tmp: rain/snow/ice paths"),
        ("sfc-init", "phys/radiation_surface_nmmb.f", "117-401", "sfc_init + sfcinit: surface albedo scheme init"),
        ("setalb", "phys/radiation_surface_nmmb.f", "407-725", "setalb: surface albedo (diurnal variation removed, Ferrier 2005)"),
        ("setemis", "phys/radiation_surface_nmmb.f", "730-897", "setemis: surface emissivity")):
    u(f"phys-rrtm-{nm}", X, f"{path}:{rng}", f"{what}.",
      deps=["phys-rrtm-aer-init"] if nm == "clim-aerinit" else ())
for nm, path, rng, why in (("progcld1", "phys/radiation_clouds_nmmb.f", "354-782", "np3d 4 (Zhao-Carr)"),
                           ("progcld8", "phys/radiation_clouds_nmmb.f", "1329-1542", "np3d 8 (Thompson)"),
                           ("diagcld1", "phys/radiation_clouds_nmmb.f", "1548-2162", "diagnostic-cloud branch; np3d 3 is prognostic"),
                           ("rhtable", "phys/radiation_clouds_nmmb.f", "2499-2769", "used by diagcld1 only")):
    u(f"phys-rrtm-{nm}", X, f"{path}:{rng}", f"Cloud branch {nm}.", mode="skip", reason=f"skip: {why}.")
# LW
for nm, rng, what in (("lwrad", "352-1139", "lwrad: LW driver per column"),
                      ("rlwinit", "1145-1339", "rlwinit: LW init (Planck, transmittance tables)"),
                      ("cldprop", "1344-1697", "cldprop: LW cloud optical depth"),
                      ("mcica-subcol", "1702-1858", "mcica_subcol: McICA sub-columns (if isubcol>0)"),
                      ("setcoef", "1863-2121", "setcoef: per-layer interpolation coefficients"),
                      ("rtrn", "2127-2484", "rtrn: clear/cloudy radiative transfer"),
                      ("rtrnmr", "2489-3054", "rtrnmr: max-random overlap transfer"),
                      ("rtrnmc", "3059-3418", "rtrnmc: McICA transfer"),
                      ("taumol", "3423-3609", "taumol: LW band dispatcher")):
    u(f"phys-rrtm-lw-{nm}", X, f"{LW}:{rng}", f"{what}.", deps=["phys-rrtm-lw-params"])
u("phys-rrtm-lw-params", X, "phys/radlw_param_nmmb.f:57-175", "LW control and band parameters (NBANDS, NGPTLW, NPLNK).")
u("phys-rrtm-lw-table-avplank", X, f"{LT}:57-727", "Extract module_radlw_avplank_nmmb to data/rrtm_lw/avplank.npz.")
u("phys-rrtm-lw-table-cldprlw", X, f"{LT}:733-1144", "Extract module_radlw_cldprlw_nmmb to data/rrtm_lw/cldprlw.npz.")
LWK = {1: (1150, 1747), 2: (1753, 2777), 3: (2783, 8935), 4: (8941, 14760), 5: (14766, 20657),
       6: (20663, 20853), 7: (20859, 22991), 8: (22997, 23638), 9: (23644, 26117),
       10: (26123, 26544), 11: (26550, 27126), 12: (27132, 28169), 13: (28175, 28715),
       14: (28721, 28903), 15: (28909, 29201), 16: (29207, 29499)}
LWG = {1: (3610, 3712), 2: (3716, 3788), 3: (3792, 4094), 4: (4098, 4337), 5: (4341, 4598),
       6: (4602, 4683), 7: (4687, 4942), 8: (4946, 5065), 9: (5069, 5313), 10: (5317, 5387),
       11: (5391, 5478), 12: (5482, 5662), 13: (5666, 5908), 14: (5912, 5976), 15: (5980, 6184),
       16: (6188, 6376)}
for b in range(1, 17):
    a, z = LWG[b]
    ka, kz = LWK[b]
    u(f"phys-rrtm-lw-taugb{b:02d}", X, f"{LW}:{a}-{z}",
      f"taugb{b:02d}: LW band {b} optical depths and Planck fractions. Also extract its k-table "
      f"module_radlw_kgb{b:02d}_nmmb ({LT}:{ka}-{kz}) to data/rrtm_lw/kgb{b:02d}.npz.",
      deps=["phys-rrtm-lw-setcoef", "phys-rrtm-lw-params"])
# SW
for nm, rng, what in (("swrad", "374-1239", "swrad: SW driver per column"),
                      ("rswinit", "1244-1395", "rswinit: SW init"),
                      ("cldprop", "1400-1822", "cldprop: SW cloud optics"),
                      ("mcica-subcol", "1827-1984", "mcica_subcol: SW McICA sub-columns (if isubcol>0)"),
                      ("setcoef", "1989-2139", "setcoef: SW interpolation coefficients"),
                      ("spcvrtc", "2144-2964", "spcvrtc: delta-Eddington two-stream, clear/cloudy"),
                      ("swflux", "2977-3102", "swflux: SW flux/heating outputs"),
                      ("taumol", "3107-3383", "taumol: SW band dispatcher")):
    u(f"phys-rrtm-sw-{nm}", X, f"{SW}:{rng}", f"{what}.", deps=["phys-rrtm-sw-params"])
u("phys-rrtm-sw-spcvrtm", X, f"{SW}:2970-2971", "spcvrtm stub.", mode="skip", reason="skip: empty stub in NAM source.")
u("phys-rrtm-sw-params", X, "phys/radsw_param_nmmb.f:51-189", "SW control and band parameters.")
u("phys-rrtm-sw-table-cldprtb", X, f"{ST}:52-1064", "Extract module_radsw_cldprtb_nmmb to data/rrtm_sw/cldprtb.npz.")
u("phys-rrtm-sw-table-sflux", X, f"{ST}:1070-1378", "Extract module_radsw_sflux_nmmb to data/rrtm_sw/sflux.npz.")
SWK = {16: (1384, 2528), 17: (2533, 7128), 18: (7133, 8634), 19: (8639, 10141), 20: (10146, 10910),
       21: (10915, 14755), 22: (14760, 15190), 23: (15195, 15436), 24: (15441, 16967),
       25: (16972, 17119), 26: (17124, 17154), 27: (17159, 17749), 28: (17754, 20060),
       29: (20065, 20973)}
SWG = {16: (3384, 3490), 17: (3495, 3627), 18: (3632, 3738), 19: (3743, 3847), 20: (3852, 3943),
       21: (3948, 4080), 22: (4085, 4201), 23: (4206, 4272), 24: (4277, 4381), 25: (4386, 4445),
       26: (4450, 4481), 27: (4486, 4554), 28: (4559, 4678), 29: (4683, 4767)}
for b in range(16, 30):
    a, z = SWG[b]
    ka, kz = SWK[b]
    u(f"phys-rrtm-sw-taumol{b}", X, f"{SW}:{a}-{z}",
      f"taumol{b}: SW band {b} optical depths and solar source. Also extract its k-table "
      f"module_radsw_kgb{b}_nmmb ({ST}:{ka}-{kz}) to data/rrtm_sw/kgb{b}.npz.",
      deps=["phys-rrtm-sw-setcoef", "phys-rrtm-sw-params", "phys-rrtm-sw-table-sflux"])


# ------------------------------------------------------------------ wrappers and kernels
# Pure-I/O units: no kernel is expected (kernel ""), proven by a fresh full-case fetch.
NO_KERNEL = {"init-fetch-case"}
PKG = {"dynamics": "namgpu.dynamics", "boundary": "namgpu.boundary", "driver": "namgpu.steps",
       "init": "namgpu.init", "io": "namgpu.io", "physics-driver": "namgpu.physics.driver"}


def finish():
    seen = set()
    for x in U:
        assert x["id"] not in seen, x["id"]
        seen.add(x["id"])
        g = x["group"]
        pkg = PKG.get(g) or "namgpu.physics." + g.split("-", 1)[1]
        mod = x["id"].replace("-", "_")
        x["wrapper"] = f"{pkg}.{mod}"
        x["kernel"] = "" if x["mode"] in ("skip",) or x["id"] in NO_KERNEL else f"namgpu/kernels/{mod}.cu"
        x["status"] = "skip" if x["mode"] == "skip" else ("done" if x["id"] in NO_KERNEL else "todo")
    ids = {x["id"] for x in U}
    for x in U:
        for d in x["deps"]:
            assert d in ids, (x["id"], d)
    doc = {
        "repo": "woof-nam",
        "fortran_root": "$NAM_SRC/sorc/nam_nems_nmmb.fd/src/atmos "
                        "(MatthewPyle-NOAA/NAM @ 83df4e13c115d197db67b972cb3befec8511291a, nam.v4.2.11)",
        "contract": "CONTRACT.md",
        "case": "cases/2025-03-14_12z",
        "done_rule": "kernel compiles; wrapper runs once on a full-size random State without NaN/Inf; "
                     "Fortran-to-CUDA diff review finds no missing term (CONTRACT.md section 6)",
        "counts": {},
        "units": U,
    }
    for x in U:
        doc["counts"][x["mode"]] = doc["counts"].get(x["mode"], 0) + 1
    doc["counts"]["total"] = len(U)
    (ROOT / "namgpu" / "UNITS.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


if __name__ == "__main__":
    d = finish()
    print(d["counts"])
