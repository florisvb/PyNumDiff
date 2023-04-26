"""
Import useful functions from _linear_model module
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

# try:
#     import pychebfun
#     __pychebfun_installed__ = True
# except:
#     __pychebfun_installed__ = False

try:
    import cvxpy
    __cvxpy_installed__ = True
except:
    __cvxpy_installed__ = False


# if __pychebfun_installed__:
#     from pynumdiff.linear_model._linear_model import chebydiff
# else:
#     __warning__ =   '\nLimited Linear Model Support Detected! \n'\
#                     '---> PYCHEBFUN is not installed. \n'\
#                     '---> Install pychebfun to use chebfun derivatives (https://github.com/pychebfun/pychebfun/) \n'\
#                     'You can still use other methods \n'
#     _logging.info(__warning__)

if __cvxpy_installed__:
    from pynumdiff.linear_model._linear_model import lineardiff
    from pynumdiff.linear_model._linear_model import __integrate_dxdt_hat_matrix__
    from pynumdiff.linear_model._linear_model import __solve_for_A_and_C_given_X_and_Xdot__
else:
    __warning__ =   '\nLimited Linear Model Support Detected! \n'\
                    '---> CVXPY is not installed. \n'\
                    '---> Install CVXPY to use lineardiff derivatives \n'\
                    'You can still use other methods \n'
    _logging.info(__warning__)

from pynumdiff.linear_model._linear_model import savgoldiff
from pynumdiff.linear_model._linear_model import spectraldiff
from pynumdiff.linear_model._linear_model import polydiff