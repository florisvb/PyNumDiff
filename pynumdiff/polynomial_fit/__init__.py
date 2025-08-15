"""Find a smooth fit with some kind of polynomials, and then take derivative.
"""
from ._polynomial_fit import splinediff, polydiff, savgoldiff

__all__ = ['splinediff', 'polydiff', 'savgoldiff'] # So automodule from the .rst finds them