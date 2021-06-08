"""
functions for optimizing finite difference
"""
import pynumdiff.finite_difference
from pynumdiff.optimize.__optimize__ import __optimize__


def first_order(x, dt, params=None, options={'iterate': True}, dxdt_truth=None,
                tvgamma=1e-2, padding='auto', metric='rmse'):
    """
    :param x:
    :param dt:
    :param params:
    :param options:
    :param dxdt_truth:
    :param tvgamma:
    :param padding:
    :param metric:
    :return:
    """
    # initial condition
    if params is None:
        if not options['iterate']:
            return []

        params = [[5], [10], [30], [50]]

    # param types and bounds
    params_types = [int, ]
    params_low = [1, ]
    params_high = [1e3, ]

    # function
    func = pynumdiff.finite_difference.first_order

    # optimize
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args)

    return opt_params, opt_val
