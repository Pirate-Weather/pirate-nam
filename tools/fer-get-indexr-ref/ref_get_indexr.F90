! Referee for phys-fer-get-indexr. The FUNCTION below is copied verbatim from
! nam.v4.2.11 (83df4e13c1) sorc/nam_nems_nmmb.fd/src/atmos/phys/module_MP_FER_HIRES.F90 lines 2297-2380
! (NOAA/NCEP, US government work). The parameters are copied from lines 151-153 and 181-187; the seven
! RR_DR* breakpoints are module variables in the source (set by FER_HIRES_INIT from RRATE, which needs the
! ETAMPNEW_DATA table not present in the clone), so this harness reads them from the test's input file.
MODULE fer_ref
  IMPLICIT NONE
  REAL, PARAMETER :: DMRmin=.05e-3, DMRmax=1.0e-3, DelDMR=1.e-6, XMRmin=1.e6*DMRmin, XMRmax=1.e6*DMRmax
  INTEGER, PARAMETER :: MDRmin=XMRmin, MDRmax=XMRmax
  REAL, PARAMETER :: DMR1=.1E-3, DMR2=.2E-3, DMR3=.32E-3, DMR4=0.45E-3, DMR5=0.67E-3
  REAL, PARAMETER :: XMR1=1.e6*DMR1, XMR2=1.e6*DMR2, XMR3=1.e6*DMR3, XMR4=1.e6*DMR4, XMR5=1.e6*DMR5
  INTEGER, PARAMETER :: MDR1=XMR1, MDR2=XMR2, MDR3=XMR3, MDR4=XMR4, MDR5=XMR5
  REAL :: RR_DRmin, RR_DR1, RR_DR2, RR_DR3, RR_DR4, RR_DR5, RR_DRmax
CONTAINS
      INTEGER FUNCTION GET_INDEXR(RR)
      IMPLICIT NONE
      REAL, INTENT(IN) :: RR
      IF (RR .LE. RR_DRmin) THEN
!
!--- Assume fixed mean diameter of rain (0.05 mm) for low rain rates, 
!      instead vary N0r with rain rate
!
        GET_INDEXR=MDRmin
      ELSE IF (RR .LE. RR_DR1) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables 
!      for mean diameters (Dr) between 0.05 and 0.10 mm:
!      V(Dr)=5.6023e4*Dr**1.136, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*5.6023e4*Dr**(4+1.136) = 1.408e15*Dr**5.136,
!        RR in kg/(m**2*s)
!      Dr (m) = 1.123e-3*RR**.1947 -> Dr (microns) = 1.123e3*RR**.1947
!
        GET_INDEXR=INT( 1.123E3*RR**.1947 + .5 )
        GET_INDEXR=MAX( MDRmin, MIN(GET_INDEXR, MDR1) )
      ELSE IF (RR .LE. RR_DR2) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables 
!      for mean diameters (Dr) between 0.10 and 0.20 mm:
!      V(Dr)=1.0867e4*Dr**.958, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*1.0867e4*Dr**(4+.958) = 2.731e14*Dr**4.958,
!        RR in kg/(m**2*s)
!      Dr (m) = 1.225e-3*RR**.2017 -> Dr (microns) = 1.225e3*RR**.2017
!
        GET_INDEXR=INT( 1.225E3*RR**.2017 + .5 )
        GET_INDEXR=MAX( MDR1, MIN(GET_INDEXR, MDR2) )
      ELSE IF (RR .LE. RR_DR3) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables 
!      for mean diameters (Dr) between 0.20 and 0.32 mm:
!      V(Dr)=2831.*Dr**.80, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*2831.*Dr**(4+.80) = 7.115e13*Dr**4.80, 
!        RR in kg/(m**2*s)
!      Dr (m) = 1.3006e-3*RR**.2083 -> Dr (microns) = 1.3006e3*RR**.2083
!
        GET_INDEXR=INT( 1.3006E3*RR**.2083 + .5 )
        GET_INDEXR=MAX( MDR2, MIN(GET_INDEXR, MDR3) )
      ELSE IF (RR .LE. RR_DR4) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables
!      for mean diameters (Dr) between 0.32 and 0.45 mm:
!      V(Dr)=963.0*Dr**.666, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*963.0*Dr**(4+.666) = 2.4205e13*Dr**4.666,
!        RR in kg/(m**2*s)
!      Dr (m) = 1.354e-3*RR**.2143 -> Dr (microns) = 1.354e3*RR**.2143
!
        GET_INDEXR=INT( 1.354E3*RR**.2143 + .5 )
        GET_INDEXR=MAX( MDR3, MIN(GET_INDEXR, MDR4) )
      ELSE IF (RR .LE. RR_DR5) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables
!      for mean diameters (Dr) between 0.45 and 0.675 mm:
!      V(Dr)=309.0*Dr**.5185, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*309.0*Dr**(4+.5185) = 7.766e12*Dr**4.5185,
!        RR in kg/(m**2*s)
!      Dr (m) = 1.404e-3*RR**.2213 -> Dr (microns) = 1.404e3*RR**.2213
!
        GET_INDEXR=INT( 1.404E3*RR**.2213 + .5 )
        GET_INDEXR=MAX( MDR4, MIN(GET_INDEXR, MDR5) )
      ELSE IF (RR .LE. RR_DRmax) THEN
!
!--- Best fit to mass-weighted fall speeds (V) from rain lookup tables
!      for mean diameters (Dr) between 0.675 and 1.0 mm:
!      V(Dr)=85.81Dr**.343, V in m/s and Dr in m
!      RR = PI*1000.*N0r0*85.81*Dr**(4+.343) = 2.1566e12*Dr**4.343,
!        RR in kg/(m**2*s)
!      Dr (m) = 1.4457e-3*RR**.2303 -> Dr (microns) = 1.4457e3*RR**.2303
!
        GET_INDEXR=INT( 1.4457E3*RR**.2303 + .5 )
        GET_INDEXR=MAX( MDR5, MIN(GET_INDEXR, MDRmax) )
      ELSE 
!
!--- Assume fixed mean diameter of rain (1.0 mm) for high rain rates, 
!      instead vary N0r with rain rate
!
        GET_INDEXR=MDRmax
      ENDIF              ! End IF (RR .LE. RR_DRmin) etc. 
!
      END FUNCTION GET_INDEXR
END MODULE fer_ref

PROGRAM ref_get_indexr
  USE fer_ref
  IMPLICIT NONE
  INTEGER :: n, k, u_in, u_out
  REAL, ALLOCATABLE :: rr(:)
  REAL :: thr(7)
  INTEGER, ALLOCATABLE :: outv(:)
  CHARACTER(len=1024) :: fin, fout
  CALL GET_COMMAND_ARGUMENT(1, fin)
  CALL GET_COMMAND_ARGUMENT(2, fout)
  OPEN(NEWUNIT=u_in, FILE=TRIM(fin), FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD')
  READ(u_in) n
  READ(u_in) thr
  ALLOCATE(rr(n), outv(n))
  READ(u_in) rr
  CLOSE(u_in)
  RR_DRmin=thr(1); RR_DR1=thr(2); RR_DR2=thr(3); RR_DR3=thr(4)
  RR_DR4=thr(5); RR_DR5=thr(6); RR_DRmax=thr(7)
  DO k=1,n
    outv(k)=GET_INDEXR(rr(k))
  END DO
  OPEN(NEWUNIT=u_out, FILE=TRIM(fout), FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE')
  WRITE(u_out) outv
  CLOSE(u_out)
END PROGRAM ref_get_indexr
