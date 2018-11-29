import scipy.optimize
import numpy as np

from pynumdiff.utils import utility as utility
from pynumdiff.utils import evaluate as evaluate
import pynumdiff.finite_difference


def first_order(x, dt, params=None, options={'iterate': True}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    # initial condition
    if params is None:
        if options['iterate'] is False:
            return []
        else:
            iterations = [1, 5, 10]
            params = []
            for window_size in window_sizes:
                for iteration in iterations:
                    params.append([window_size, iteration])

    # param types and bounds
    if options['iterate'] is False:
        params_types = [int,]
        params_low = [1,]
        params_high = [1e6,]
    else:
        params_types = [int, int]
        params_low = [1, 1]
        params_high = [1e6, 1e2]

    # function
    function = pynumdiff.finite_difference.first_order

    # optimize
    args = [function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding]
    opt_params, opt_val = __optimize__(params, args) 

    return opt_params, opt_val

