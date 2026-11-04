"""Correctly rounded decimal-to-float32 for Fortran real*4 DATA literals.

gfortran rounds each single-precision literal straight to the nearest float32. Going through
float64 first (np.float32(float(text))) rounds twice and can land one float32 ulp away from the
compiler's value. This helper takes the exact rational value of the literal and picks the
nearest float32 among the float64-rounded candidate and its two neighbours.
"""
from fractions import Fraction

import numpy as np


def f32_literal(text):
    f = Fraction(text.strip().replace("d", "e").replace("D", "e"))
    c = np.float32(float(f))
    best, bestd = c, abs(Fraction(float(c)) - f)
    for cand in (np.nextafter(c, np.float32(np.inf)), np.nextafter(c, np.float32(-np.inf))):
        d = abs(Fraction(float(cand)) - f)
        if d < bestd:
            best, bestd = cand, d
    return best
