"""
Import useful functions from the optimize module
"""

import logging as _logging
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        _logging.FileHandler("debug.log"),
        _logging.StreamHandler()
    ]
)

try:
    import cvxpy
    __cvxpy_installed__ = True
except ImportError:
    _logging.info( 'Limited support for total variation regularization and linear model detected!\n\
                    ---> Some functions in pynumdiff.optimize.total_variation_regularization and require CVXPY to be installed.\n\
                    ---> Some functions in pynumdiff.linear_model and require CVXPY to be installed.\n\
                    You can still pynumdiff.optimize for other functions.')
    __cvxpy_installed__ = False

from pynumdiff.optimize import finite_difference
from pynumdiff.optimize import smooth_finite_difference
from pynumdiff.optimize import total_variation_regularization
from pynumdiff.optimize import linear_model
from pynumdiff.optimize import kalman_smooth




