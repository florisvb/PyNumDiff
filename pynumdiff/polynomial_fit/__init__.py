"""Methods based on fitting data with polynomials
"""
from ._polynomial_fit import splinediff, polydiff, savgoldiff

__all__ = ['splinediff', 'polydiff', 'savgoldiff'] # So automodule from the .rst finds them