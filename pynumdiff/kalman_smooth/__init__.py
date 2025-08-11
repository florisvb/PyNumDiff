"""This module implements Kalman filters
"""
from ._kalman_smooth import rts_const_deriv, constant_velocity, constant_acceleration, constant_jerk, known_dynamics

__all__ = ['rts_const_deriv', 'constant_velocity', 'constant_acceleration', 'constant_jerk', 'known_dynamics'] # So these get treated as direct members of the module by sphinx
