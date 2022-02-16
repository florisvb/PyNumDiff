"""
Import useful functions from _total_variation_regularization module
"""
import warnings as _warnings

try:
    import cvxpy
    __cvxpy_installed__ = True
except:
    __cvxpy_installed__ = False

from pynumdiff.total_variation_regularization._total_variation_regularization import iterative_velocity

if __cvxpy_installed__:
    from pynumdiff.total_variation_regularization._total_variation_regularization import velocity
    from pynumdiff.total_variation_regularization._total_variation_regularization import acceleration
    from pynumdiff.total_variation_regularization._total_variation_regularization import jerk
    from pynumdiff.total_variation_regularization._total_variation_regularization import jerk_sliding
    from pynumdiff.total_variation_regularization._total_variation_regularization import smooth_acceleration
else:
    __warning__ =   'Limited Total Variation Regularization Support Detected! \n'\
                    'Limited Total Variation Regularization Support Detected! \n'\
                    '---> CVXPY is not installed. \n'\
                    '---> Many Total Variation Methods require CVXPY including: \n'\
                    '---> velocity, acceleration, jerk, jerk_sliding, smooth_acceleration\n'\
                    '---> Please install CVXPY to use these methods.\n'\
                    '---> Recommended to also install MOSEK and obtain a MOSEK license.\n'\
                    'You can still use: total_variation_regularization.iterative_velocity\n'
    _warnings.warn(__warning__)
