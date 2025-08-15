"""Apply smoothing method before finite difference.
"""
from ._smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff

__all__ = ['mediandiff', 'meandiff', 'gaussiandiff', 'friedrichsdiff', 'butterdiff'] # So automodule from the .rst finds them
 # splinediff is still included in the imports list so backwards compatibility isn't broken, but excluded
 # from the __all__ list so sphinx doesn't try to document it from this module.
