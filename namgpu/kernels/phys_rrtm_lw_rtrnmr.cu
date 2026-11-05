// phys_rrtm_lw_rtrnmr.cu -- rtrnmr: maximum-random overlap longwave flux and heating-rate transfer.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
// BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
// Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
//
// NOTICE: NAM source file phys/radlw_main_nmmb.f (NCEP/EMC, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE, no licence file).
// The routine header (module_radlw_main_nmmb, rtrnmr) is kept as the notice:
//
//   subroutine rtrnmr ( semiss,delp,cldfrc,taucld,tautot,pklay,pklev,
//                       fracs,secdif, nlay,nlp1,
//                       totuflux,totdflux,htr, totuclfl,totdclfl,htrcl, htrb )
//   purpose: compute the upward/downward radiative fluxes, and heating rates for both clear or
//   cloudy atmosphere.  clouds are assumed as in maximum-randomly overlaping in a vertical colum.
//   NCEP lw-rrtm3 package (rrtm-lw from AER, NCEP modifications, NMMB version, Nov 2012).
//
// Ported to CUDA for WOOF-NAM from phys/radlw_main_nmmb.f:2489-3054 (nam.v4.2.11, 83df4e13c1).
//
// Index conventions (kept from the Fortran call; see the wrapper docstring):
//   Fortran k = 1 is the lowest layer (surface side), k = nlay the top; the interface arrays
//   pklay/pklev/totuflux/... run 0..nlay with 0 the surface interface.
//   The arrays are NOT in that order. They are in NAM slot order (slot 0 = model top), the same
//   convention as phys-rrtm-lw-rtrn: Fortran layer k is slot nlay-k, Fortran interface m is slot
//   nlay-m, Fortran cldfrc(m) is slot nlay+1-m. The kernel maps every index with LAY/IFC/CFX below,
//   so the caller passes NAM top-down arrays exactly as lwrad builds them (k1 = nlp1 - k). Memory: a column array
//   with a leading band or g index and a level index is [x, level, JM, IM]; the column is
//   col = j0*IM + i0 and element (x, m) sits at (x*L + m)*N + col with N = JM*IM.
//
// Structure (one thread per column, one launch; statement order follows the Fortran routine):
//   argument list                          -> kernel arguments.
//   overlap factors, upward (k = 1..nlay-1) -> facs-up block (facclr*/faccld*/faccmb*, lstu).
//   overlap factors, downward (k = nlay..2) -> facs-down block (same, lstd).
//   zeroing of the flux and band accumulators -> tuf/tdf/tuc/tdc and the per-band arrays.
//   g loop, downward k = nlay..1           -> odepth, gas transmittance, lookup tables, planck
//                                             fractions, cloud branches (lstcldd, clfr >= eps).
//   surface emission and reflection        -> rad0, radtotu/radclru, toturad(0), clrurad(0).
//   g loop, upward k = 1..nlay             -> totradu/clrradu/rad, toturad/clrurad.
//   totals over bands, times flxfac        -> tuf/tdf/tuc/tdc, written to the flux outputs.
//   htr (total sky), htrcl (lhlw0), htrb (lhlwb).
//
// Band blocking: the Fortran ig loop runs ig = 1..ngptlw with ngb(ig) the band. Here the loop runs
// band by band over the same g-points in the same order, with per-band toturad/totdrad/clrurad/
// clrdrad accumulators, and the band totals are added into tuf/tdf/tuc/tdc in band order. Every
// per-band sum is the same sequence of float additions as the Fortran, and the totals are the same
// sum over bands. The band table is the Fortran band start (NSxx) and count (NGxx) from
// radlw_param.f:58-137 (module_radlw_parameters, the table radlw_main_nmmb.f rtrnmr uses at its
// ib = ngb(ig) reference), which gives ngb(ig) exactly. The wrapper asserts the table against the
// NGB data statement of radlw_param.f:120-122.
//
// Persistent scalars (radtotd, radclrd, totradd, clrradd, rad, radtotu, radclru, totradu, clrradu,
// rat1, rat2) are declared once per column and live across the k, ig and band loops exactly as the
// Fortran subroutine locals do.

#include "nmmb_common.cuh"

#define LW_NBANDS 16
#define LW_NGPT   140
#define LW_NTBL   10000
#define LW_MAXL   64          /* maximum nlay (layers); the wrapper checks nlay <= LW_MAXL */

// radlw_param.f:58-137 (NSxx band start, 0-based g index; NGxx g-points per band). Band 1 has 10
// g-points here (NG01=10), not the 8 of radlw_param_nmmb.f, because radlw_main_nmmb.f rtrnmr uses
// module_radlw_parameters and the kgb01 gas optics of radlw_datatb.f.
__constant__ int LW_NS[LW_NBANDS] = {0, 10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138};
__constant__ int LW_NG[LW_NBANDS] = {10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2};

// Slot maps. The body (phys_rrtm_lw_rtrnmr_body.cuh) is compiled twice; the two kernels differ only in where
// they read and write a layer, never in arithmetic.
//  phys_rrtm_lw_rtrnmr     NAM slot order (slot 0 = model top): inputs/outputs reversed on the layer axis,
//                          pklay/pklev as [NBANDS, NLAY+1, JM, IM]. Fortran layer index l0 = k-1, interface m,
//                          cldfrc index m.
//  phys_rrtm_lw_rtrnmr_sf  radlw internal order (slot 0 = the surface layer, as lwrad and setcoef hold them):
//                          every layer array is read and written where the LW seam already has it, and
//                          pklay/pklev in setcoef's own [NLAY+1, NBANDS, JM, IM] layout. The seam (lw_core) then
//                          needs no reversed or transposed copies of tautot, fracs, taucld, delp, cldfrc, pklay,
//                          pklev, nor of the outputs (VRAM audit 2026-11-04: two 140-g-point copies per chunk).
#define RTRNMR_NAME phys_rrtm_lw_rtrnmr
#define LAY(l0)  (nlay - 1 - (l0))
#define IFC(m)   (nlay - (m))
#define CFX(m)   (nlay + 1 - (m))
#define PKX(ib, s) ((size_t)(ib) * (nlay + 1) + (s))
#include "phys_rrtm_lw_rtrnmr_body.cuh"
#undef RTRNMR_NAME
#undef LAY
#undef IFC
#undef CFX
#undef PKX

#define RTRNMR_NAME phys_rrtm_lw_rtrnmr_sf
#define LAY(l0)  (l0)
#define IFC(m)   (m)
#define CFX(m)   (m)
#define PKX(ib, s) ((size_t)(s) * LW_NBANDS + (ib))
#include "phys_rrtm_lw_rtrnmr_body.cuh"
#undef RTRNMR_NAME
#undef LAY
#undef IFC
#undef CFX
#undef PKX
