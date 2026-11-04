// phys_noah_sflx.cu -- unit phys-noah-sflx (SFLX, phys/module_LS_NOAHLSM.F90:1550-2278, nam.v4.2.11,
// 83df4e13c1). NOAA / NCEP NAM, US government work; MODULE_LS_NOAHLSM notice kept in the header.
//
// The SFLX column is the device function noah_sflx in phys_noah_sflx.cuh, written from the NAM Fortran
// (2026-11-04). It runs inside NOAHLSM (phys-noah-driver, one thread per land column). This file is the
// unit's compile target only. The earlier body here (ArWen WRF v4.6.1 noah.cu lineage, not reconciled to
// the NAM SFLX) is retired; it is in git history at f60da26.
#include "phys_noah_sflx.cuh"

// Launchable probe of the column: one land point per thread over a [JM, IM] block of columns is what
// phys-noah-driver does; this file adds no second launch path.
