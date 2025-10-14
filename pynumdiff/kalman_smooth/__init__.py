"""This module implements constant-derivative model-based smoothers based on Kalman filtering and its generalization.
"""
try:
    import cvxpy
except:
    from warnings import warn
    warn("CVXPY is not installed; robustdiff and l1_solve not available.")
else: # runs if try was successful
    from ._kalman_smooth import robustdiff, convex_smooth

from ._kalman_smooth import kalman_filter, rts_smooth, rtsdiff, constant_velocity, constant_acceleration, constant_jerk