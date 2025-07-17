"""This module implements interpolation-based differentiation schemes.
"""
try:
    import cvxpy
    from ._linear_model import lineardiff
except:
    from warnings import warn
    warn("Limited Linear Model Support Detected! CVXPY is not installed. " +
        "Install CVXPY to use lineardiff derivatives. You can still use other methods.")

from ._linear_model import savgoldiff, polydiff, spectraldiff, chebydiff

__all__ = ['lineardiff', 'savgoldiff', 'polydiff', 'spectraldiff', 'chebydiff'] # So these get treated as direct members of the module by sphinx
