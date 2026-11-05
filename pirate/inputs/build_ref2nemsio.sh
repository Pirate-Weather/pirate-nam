#!/bin/bash
source /nam/env.sh; cd /nam/nam.github/sorc/nam_ref2nemsio.fd
mpif90 -O0 -g -fbacktrace -fconvert=big-endian -std=legacy -fno-range-check -ffree-line-length-none -ffree-form ${NEMSIO_INC:+-I$NEMSIO_INC} ref2nemsio.f -o /nam/exec/nam_ref2nemsio $NEMSIO_LIB $BACIO_LIB4 $W3EMC_LIB4 $W3NCO_LIB4 -fopenmp 2>&1 | grep -E 'Error|undefined' | head -5 | cut -c1-110; ls -la /nam/exec/nam_ref2nemsio | cut -c25-
