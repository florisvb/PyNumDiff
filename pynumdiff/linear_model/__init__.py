"""This module implements interpolation-based differentiation schemes.
"""
try:
    import cvxpy
    from ._linear_model import lineardiff
except:
    from warnings import warn
    warn("Limited Linear Model Support Detected! CVXPY is not installed. " +
        "Install CVXPY to use lineardiff derivatives. You can still use other methods.")

from ._linear_model import savgoldiff, polydiff, spectraldiff, rbfdiff

__all__ = ['lineardiff', 'spectraldiff', 'rbfdiff'] # So these get treated as direct members of the
# module by sphinx polydiff and savgoldiff are still imported for now for backwards compatibility
# but are not documented as part of this module, since they've moved
