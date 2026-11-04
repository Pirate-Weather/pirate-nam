// phys_rrtm_gas_init.cu -- GAS_INIT: ozone and CO2 setup (module_radiation_gases_nmmb).
//
// NOTICE: NAM source file phys/radiation_gases_nmmb.f (NOAA/NCEP/EMC, US Government work, no licence
// file). The subroutine header is kept here as the notice:
//   subroutine gas_init (me): sets up ozone, co2, etc. parameters. If climatology ozone then reads
//   monthly ozone data. Outputs (module variables): lclimoz, ico2flg, ictmflg, ivflip, pkstr, o3r,
//   co2_glb, co2vmr_sav, co2cyc_sav, gco2cyc.
//   ICTM -2/-1/0/1/yyyy# data time control; ICO2 0/1/2 CO2 source; IOZN ozone source (0 climatology,
//   >0 interactive); iflip vertical order.
// Ported to CUDA for nam-gpu from phys/radiation_gases_nmmb.f:186-483 (nam.v4.2.11, 83df4e13c1).
//
// Split (CONTRACT.md section 3, one launch per Fortran section):
//   The file reads (ozone climatology NIO3CLM, CO2 user/cycle files NICO2CN) and every branch
//   selection are host I/O in namgpu/physics/rrtm/phys_rrtm_gas_init.py (Fortran READ/OPEN/INQUIRE
//   have no device form). The arithmetic on the file data is done here, on the device:
//
// Fortran line-by-line (every assignment with arithmetic has a kernel counterpart):
//   225-226   if (me==0) print VTAGGAS                  host diagnostic, not ported
//   ioznflg>0 (interactive ozone): print only           host, no state written
//   ioznflg==0 (climatology):
//     timeozc /= 12 abort                               host check (wrapper)
//     read pstr4(k), o3clim4(j,k,imo)                   host (wrapper), float32 (kind_io4)
//     o3r(j,k,imo) = o3clim4(j,k,imo) * 1.655e-6        gas_init_o3r  (layout [imo][k][j])
//     pstr(k) = pstr4(k)                                host (kind_phys = float32 single build)
//     pkstr(k) = fpkap(pstr(k)*100)   if pstr(k) >= 0.1  gas_init_pkstr (fpkap table, see below)
//     pkstr(k) = (pstr(k)/1000.)**con_rocp  otherwise   gas_init_pkstr
//   co2:
//     co2_glb = co2vmr_def (350e-6)                     gas_init_co2_glb (use_file = 0)
//     ico2flg==0: prescribed global mean, nothing else  (co2_glb default)
//     ictmflg==-1 (user file): header line read (host), co2_glb = co2g1*1.0e-6
//                                                       gas_init_co2_glb (use_file = 1)
//       ico2flg==2: co2vmr_sav(i,j,imo) = co2dat(i,j)*1.0e-6   gas_init_co2_scale (layout [imo][j][i])
//     ictmflg==-2 (observed + seasonal cycle, file co2cyc_file):
//       gco2cyc(imo) = gco2cyc(imo)*1.0e-6              gas_init_co2_scale (n = 12)
//       co2cyc_sav(i,j,imo) = co2dat(i,j)*1.0e-6        gas_init_co2_scale (layout [imo][j][i])
//     ictmflg>=0 or yyyy: observed, co2vmr_sav allocated but not filled (NMMB behaviour; zeros here)
//
// fpkap (phys/funcphys.f:2224-2268, table gpkap/fpkapx at 2198-2220): the table holds
//   tbpkap(jx) = fpkapx(x) = (x/1e5)**con_rocp with x = xmin + (jx-1)*xinc, xmin = 0, xinc =
//   110000/(nxpkap-1) = 10 Pa, nxpkap = 11001, so the table is not stored: each lookup evaluates the
//   two table nodes with the identical formula. c1xpkap = 1 - xmin/xinc = 1, c2xpkap = 1/xinc.
//   xj = min(max(c1 + c2*p, 1), nxpkap); jx = min(xj, nxpkap-1) (truncated to integer, 1-based);
//   fpkap = tb(jx) + (xj-jx)*(tb(jx+1)-tb(jx)).
// con_rocp = con_rd/con_cp (phys/physcons.f:76), kind_phys = 4 for the single-precision build.
// Precision: float everywhere (kind_phys = 4, SINGLE_PREC, machine.f). Launches use no host sync.

#define NXPKAP        11001
#define CON_ROCP      (2.8705e+2f / 1.0046e+3f)  // physcons.f con_rd/con_cp
#define CO2VMR_DEF    (350.0e-6f)                // radiation_gases_nmmb.f co2vmr_def
#define O3_SCALE      (1.655e-6f)                // gas_init, o3r = o3clim4 * 1.655e-6
#define CO2_SCALE     (1.0e-6f)                  // gas_init, co2dat * 1.0e-6

__device__ __forceinline__ float gas_fpkapx(float p) {
    // fpkapx (funcphys.f:2407): (p/1.e5)**con_rocp
    return powf(p / 1.e5f, CON_ROCP);
}

__device__ __forceinline__ float gas_tbpkap(int jx1) {
    // tbpkap(jx), 1-based, gpkap (funcphys.f:2213-2219): x = xmin + (jx-1)*xinc, xmin = 0
    const float xinc = 110000.0f / (float)(NXPKAP - 1);
    return gas_fpkapx(0.0f + (float)(jx1 - 1) * xinc);
}

__device__ __forceinline__ float gas_fpkap(float p) {
    // fpkap (funcphys.f:2224-2268), table linear interpolation
    const float xinc = 110000.0f / (float)(NXPKAP - 1);
    const float c1 = 1.0f - 0.0f / xinc;          // c1xpkap
    const float c2 = 1.0f / xinc;                  // c2xpkap
    float xj = fminf(fmaxf(c1 + c2 * p, 1.0f), (float)NXPKAP);
    int jx = (int)fminf(xj, (float)(NXPKAP - 1));
    float t0 = gas_tbpkap(jx);
    float t1 = gas_tbpkap(jx + 1);
    return t0 + (xj - (float)jx) * (t1 - t0);
}

// pkstr(k), k = 1..LOZ (climatology ozone pressure levels, hPa).
extern "C" __global__ void gas_init_pkstr(
    const float* __restrict__ pstr,   // pstr(LOZ) in hPa (host: pstr4 read, kind_phys)
    float* __restrict__ pkstr,        // pkstr(LOZ) output (module variable)
    const int loz)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= loz) return;
    const float ps = pstr[k];
    if (ps >= 0.1f) {
        pkstr[k] = gas_fpkap(ps * 100.0f);        // line: pkstr(k) = fpkap(pstr(k)*100.0)
    } else {
        pkstr[k] = powf(ps / 1000.0f, CON_ROCP);  // line: (pstr(k)/1000.)**(con_rocp)
    }
}

// o3r(j,k,imo) = o3clim4(j,k,imo) * 1.655e-6. Both arrays hold (12, LOZ, JMR) in the same C order,
// so the flat index is the same for both (Fortran o3r(JMR,LOZ,12) -> [imo][k][j]).
extern "C" __global__ void gas_init_o3r(
    const float* __restrict__ o3clim4,   // (12, LOZ, JMR) float32 from the climatology file
    float* __restrict__ o3r,             // (12, LOZ, JMR) output
    const int n)                         // 12 * LOZ * JMR
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    o3r[idx] = o3clim4[idx] * O3_SCALE;
}

// dst = src * 1.0e-6 (co2vmr_sav, co2cyc_sav, gco2cyc). Element-wise; layout is the caller's.
extern "C" __global__ void gas_init_co2_scale(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx] * CO2_SCALE;
}

// co2_glb: default co2vmr_def, or co2g1*1.0e-6 when the user CO2 file (ictmflg == -1) supplies it.
extern "C" __global__ void gas_init_co2_glb(
    const int use_file,
    const float co2g1,                 // user file global annual mean (f7.2 header field)
    float* __restrict__ co2_glb)       // co2_glb (module variable), length 1
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    co2_glb[0] = (use_file != 0) ? co2g1 * CO2_SCALE : CO2VMR_DEF;
}
