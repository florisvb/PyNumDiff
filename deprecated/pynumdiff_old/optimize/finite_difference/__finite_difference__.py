import scipy.optimize
import numpy as np

from pynumdiff.utils import utility as utility
from pynumdiff.utils import evaluate as evaluate
import pynumdiff.finite_difference

from pynumdiff.optimize.__optimize__ import __optimize__

def first_order(x, dt, params=None, options={'iterate': True}, dxdt_truth=None, tvgamma=1e-2, padding='auto', metric='rmse'):
    # initial condition
    if params is None:
        if options['iterate'] is False:
            return []
        else:
            params = [[5], [10], [30], [50]]

    # param types and bounds
    params_types = [int,]
    params_low = [1,]
    params_high = [1e3,]

    # function
    function = pynumdiff.finite_difference.first_order

    # optimize
    args = [function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args) 

    return opt_params, opt_val

