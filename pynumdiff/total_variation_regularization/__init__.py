"""This module implements some common total variation regularization methods
"""
try:
    import cvxpy
    from ._total_variation_regularization import tvrdiff, velocity, acceleration, jerk, jerk_sliding, smooth_acceleration
except:
    from warnings import warn
    warn("Limited Total Variation Regularization Support Detected! CVXPY is not installed. " +
        "Many Total Variation Methods require CVXPY including: velocity, acceleration, jerk, jerk_sliding, smooth_acceleration. " +
        "Please install CVXPY to use these methods. Recommended to also install MOSEK and obtain a MOSEK license. " +
        "You can still use: total_variation_regularization.iterative_velocity.") 

from ._total_variation_regularization import iterative_velocity
