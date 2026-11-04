// phys_rrtm_sfc_init.cu -- nam-gpu unit phys-rrtm-sfc-init.
//
// NOAA notice: phys/radiation_surface_nmmb.f (NCEP/EMC NAM nam.v4.2.11, 83df4e13c1), module
// module_radiation_surface_nmmb, subroutines sfc_init (lines 117-233) and sfcinit (lines 274-401).
// NOAA/NCEP US government work; the NAM clone carries no licence file. The module history block is
// kept as the notice of the source (VTAGSFC NCEP-Radiation_surface v5.1, Nov 2012):
//   1995 y.t. hou (albaer); 1997-2012 y.t. hou, s. moorthi, h. wei, c. marshall, y. yang (see source).
//
// Ported to CUDA for nam-gpu from phys/radiation_surface_nmmb.f:117-401 (nam.v4.2.11, 83df4e13c1).
//
// This is initialisation, not a step-path stage. It runs once per process on the card and does two
// things the Fortran does at init:
//   (1) sfc_init_flags: sets the surface-albedo flag, the surface-emissivity flag and iemslw (module
//       state of radiation_surface_nmmb, read by setalb and setemis later). One thread.
//   (2) sfc_init_idxems: copies the emissivity-type index table (IMXEMS x JMXEMS = 360 x 180 int) into
//       its device layout. The host reads and decodes the text file (inquire/open/read are host I/O);
//       the card receives the decoded digits in Fortran read order and writes them to [JMXEMS, IMXEMS].
//
// Fortran-to-CUDA review (CONTRACT section 6, part 3). Every term of 117-401 has a counterpart:
//   sfc_init: IALB check (stop on bad value)            -> host wrapper ValueError (Fortran stop)
//   sfc_init: ialbflg/iemsflg are the physparam values; sfc_init never assigns iemsflg. The host
//       wrapper reads IALB/IEMS from rad_physparam and passes them as ialb/iems     -> flags[0], flags[1]
//   sfc_init: iemslw = mod(iemsflg, 10)                  -> iemslw = iems % 10 (C and Fortran both truncate;
//       the wrapper rejects any value other than 0 or 1 before launch, as Fortran stops)
//   sfc_init: iemslw == 1 and file missing -> iemslw = 0 -> !file_ok branch (iemslw)
//   sfc_init: iemslw == 1 -> allocate idxems + read      -> sfc_init_idxems (host decoded digits)
//   sfc_init: iemslw other than 0/1 -> stop              -> host wrapper ValueError
//   sfc_init: print statements (me == 0)                 -> host wrapper log (diagnostic only)
//   sfcinit: ialbflg = IALB; iemsflg = IEMS             -> flags[0] = ialb, flags[1] = iems (mode 1)
//   sfcinit: IEMS == 1 and file missing -> iemsflg = 0   -> !file_ok branch (iemsflg)
//   sfcinit: does not assign iemslw (module value kept)  -> flags[2] = iemslw_prev
//   sfcinit: IEMS other than 0/1 -> stop                 -> host wrapper ValueError
//   read (NIRADSF,12) cline / read (NIRADSF,14) idxems   -> host decode: line 1 skipped, then
//       records of 80 characters read with format 80i1, filling idxems(i,j) column-major (i fastest).
//       Blank characters read as 0 (gfortran BLANK='NULL'); any other non-digit is a read error.
//
// Index convention (CONTRACT section 1): the table is [JMXEMS=180, IMXEMS=360] int32 in C order; the
// Fortran idxems(i,j) is table[j-1, i-1]. The stream position of idxems(i,j) in the file read order
// is k = (i-1) + IMXEMS*(j-1), so the copy below is the identity on the stream reshaped to [JM, IM].
// The unit has no state arrays and no vertical or column layout.
#include "nmmb_common.cuh"

#define SFC_IMXEMS 360   // radiation_surface_nmmb.f:  IMXEMS  lon points in global emissivity map
#define SFC_JMXEMS 180   // radiation_surface_nmmb.f:  JMXEMS  lat points in global emissivity map

// mode 0 = sfc_init (NMMB path, grrad_nmmb.f:413), mode 1 = sfcinit (GFS/FIM path, interface with IALB/IEMS args).
// out[0] = ialbflg, out[1] = iemsflg, out[2] = iemslw. Host validates ialb and iems first (Fortran stop).
extern "C" __global__ void sfc_init_flags(
    const int mode, const int ialb, const int iems, const int iemslw_prev, const int file_ok,
    int* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int ialbflg = ialb;                       // both routines: ialbflg = IALB (sfcinit) / physparam (sfc_init)
    int iemsflg;
    int iemslw;
    if (mode == 0) {
        iemsflg = iems;                             // physparam module value (sfc_init never assigns it)
        iemslw = iems % 10;                         // iemslw = mod(iemsflg, 10)
        if (iemslw == 1 && !file_ok) iemslw = 0;    // "Change to fixed surface emissivity = 1.0"
    } else {
        iemsflg = iems;                             // sfcinit: iemsflg = IEMS
        if (iems == 1 && !file_ok) iemsflg = 0;     // "Change to fixed surface emissivity = 1.0"
        iemslw = iemslw_prev;                       // sfcinit never assigns iemslw
    }
    out[0] = ialbflg;
    out[1] = iemsflg;
    out[2] = iemslw;
}

// Decoded emissivity-index digits in Fortran read order (uint8, 0..9), one per element, length
// SFC_IMXEMS*SFC_JMXEMS. One thread per element; element k goes to table[j, i] with k = i + IMXEMS*j.
extern "C" __global__ void sfc_init_idxems(
    const unsigned char* __restrict__ stream, int* __restrict__ idxems, const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    // (i, j) = (k % IMXEMS, k / IMXEMS); table row-major [JMXEMS, IMXEMS] index is j*IMXEMS + i == k.
    idxems[k] = (int)stream[k];
}
