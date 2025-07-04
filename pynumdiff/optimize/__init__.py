"""Import functions from the optimize module
"""
try:
    import cvxpy
    from . import total_variation_regularization
except ImportError:
    from warnings import warn
    warn("Limited support for total variation regularization and linear model detected! " + 
        "Some functions in the `total_variation_regularization` and `linear_model` modules require " +
        "CVXPY to be installed. You can still pynumdiff.optimize for other functions.")

from . import finite_difference, smooth_finite_difference, linear_model, kalman_smooth

__all__ = ['finite_difference', 'smooth_finite_difference', 'linear_model', 'kalman_smooth', 'total_variation_regularization'] # So these get treated as direct members of the module by sphinx
