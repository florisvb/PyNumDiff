"""This module implements some common total variation regularization methods
"""
try:
    import cvxpy
except:
    from warnings import warn
    warn("Limited Total Variation Regularization Support Detected! CVXPY is not installed. " +
        "Many Total Variation Methods require CVXPY including: velocity, acceleration, jerk, jerk_sliding, smooth_acceleration. " +
        "Please install CVXPY to use these methods. You can still use: total_variation_regularization.iterative_velocity.")
else: # executes if try is successful
    from ._total_variation_regularization import tvrdiff, velocity, acceleration, jerk, jerk_sliding, smooth_acceleration

from ._total_variation_regularization import iterative_velocity
