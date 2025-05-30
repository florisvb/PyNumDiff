"""Apply smoothing method before finite difference.
"""
from ._smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff

__all__ = ['mediandiff', 'meandiff', 'gaussiandiff', 'friedrichsdiff', 'butterdiff', 'splinediff'] # So these get treated as direct members of the module by sphinx
