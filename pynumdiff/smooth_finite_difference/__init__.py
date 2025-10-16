"""Apply smoothing method before finite difference.
"""
from ._smooth_finite_difference import kerneldiff, mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
# splinediff is still included in the imports list so backwards compatibility isn't broken