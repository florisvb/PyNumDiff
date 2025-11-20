"""Import useful functions from all modules
"""
from ._version import __version__

try: # cvxpy dependencies
    import cvxpy
except ImportError:
    from warnings import warn
    warn("tvrdiff, robustdiff, and lineardiff not available due to lack of convex solver. To use those, install CVXPY.")
else: # executes if try is successful
    from .total_variation_regularization import tvrdiff, velocity, acceleration, jerk, smooth_acceleration
    from .kalman_smooth import robustdiff, convex_smooth
    from .linear_model import lineardiff

from .finite_difference import finitediff, first_order, second_order, fourth_order
from .smooth_finite_difference import kerneldiff, meandiff, mediandiff, gaussiandiff, friedrichsdiff, butterdiff
from .polynomial_fit import splinediff, polydiff, savgoldiff
from .basis_fit import spectraldiff, rbfdiff
from .total_variation_regularization import iterative_velocity
from .kalman_smooth import kalman_filter, rts_smooth, rtsdiff, constant_velocity, constant_acceleration, constant_jerk
