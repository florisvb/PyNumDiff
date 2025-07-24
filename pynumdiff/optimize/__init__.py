"""Import functions from the optimize module
"""
try:
    import cvxpy
except ImportError:
    from warnings import warn
    warn("Limited support for total variation regularization and linear model detected! " + 
        "Some functions in the `total_variation_regularization` and `linear_model` modules require " +
        "CVXPY to be installed. You can still pynumdiff.optimize for other functions.")

from ._optimize import optimize, suggest_method

__all__ = ['optimize', 'suggest_method'] # So these get treated as direct members of the module by sphinx
