"""This module implements Kalman filters
"""
from ._kalman_smooth import constant_velocity, constant_acceleration, constant_jerk, known_dynamics

__all__ = ['constant_velocity', 'constant_acceleration', 'constant_jerk', 'known_dynamics'] # So these get treated as direct members of the module by sphinx
