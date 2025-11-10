"""This module implements constant-derivative model-based smoothers based on Kalman filtering and its generalization.
"""
from ._kalman_smooth import kalman_filter, rts_smooth, rtsdiff, constant_velocity, constant_acceleration, constant_jerk, robustdiff, convex_smooth, robustdiffclassic
