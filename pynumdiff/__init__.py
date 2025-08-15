"""Import useful functions from all modules
"""
from ._version import __version__

from .finite_difference import finitediff, first_order, second_order, fourth_order
from .smooth_finite_difference import meandiff, mediandiff, gaussiandiff, friedrichsdiff, butterdiff
from .polynomial_fit import splinediff, polydiff, savgoldiff
from .total_variation_regularization import tvrdiff, velocity, acceleration, jerk, iterative_velocity, smooth_acceleration, jerk_sliding
from .kalman_smooth import rts_const_deriv, constant_velocity, constant_acceleration, constant_jerk, known_dynamics
from .linear_model import spectraldiff, lineardiff