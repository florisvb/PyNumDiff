"""
functions for optimizing smoothed finite difference
"""
import pynumdiff.smooth_finite_difference
from pynumdiff.optimize.__optimize__ import __optimize__

####################
# Helper functions #
####################


def __kerneldiff__(func, x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2,
                   padding='auto', metric='rmse'):
    """
    :param func:
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
        if options['iterate'] is False:
            params = [[5], [15], [30], [50]]
        else:
            window_sizes = [5, 15, 30, 50]
            iterations = [1, 5, 10]
            params = []
            for window_size in window_sizes:
                for iteration in iterations:
                    params.append([window_size, iteration])

    # param types and bounds
    if options['iterate'] is False:
        params_types = [int, ]
        params_low = [1, ]
        params_high = [1e6, ]
    else:
        params_types = [int, int]
        params_low = [1, 1]
        params_high = [len(x)-1, 1e2]

    # optimize
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args)

    return opt_params, opt_val

######################
# Optimize functions #
######################


def mediandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2,
               padding='auto', metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.mediandiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.mediandiff for detailed documentation.
    """
    func = pynumdiff.smooth_finite_difference.mediandiff
    opt_params, opt_val = __kerneldiff__(func, x, dt, params, options, dxdt_truth, tvgamma, padding, metric)
    return opt_params, opt_val


def meandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding='auto', metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.meandiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.meandiff for detailed documentation.
    """
    func = pynumdiff.smooth_finite_difference.meandiff
    opt_params, opt_val = __kerneldiff__(func, x, dt, params, options, dxdt_truth, tvgamma, padding, metric)
    return opt_params, opt_val


def gaussiandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2,
                 padding='auto', metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.gaussiandiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.gaussiandiff for detailed documentation.
    """
    func = pynumdiff.smooth_finite_difference.gaussiandiff
    opt_params, opt_val = __kerneldiff__(func, x, dt, params, options, dxdt_truth, tvgamma, padding, metric)
    return opt_params, opt_val


def friedrichsdiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2,
                   padding='auto', metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.friedrichsdiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.friedrichsdiff for detailed documentation.
    """
    func = pynumdiff.smooth_finite_difference.friedrichsdiff
    opt_params, opt_val = __kerneldiff__(func, x, dt, params, options, dxdt_truth, tvgamma, padding, metric)
    return opt_params, opt_val


def butterdiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
               optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.butterdiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.butterdiff for detailed documentation.
    """
    # initial condition
    if params is None:
        ns = [1, 2, 3, 4, 5, 6, 7]
        wns = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5]
        if options['iterate'] is False:
            params = []
            for n in ns:
                for wn in wns:
                    params.append([n, wn])
        else:
            iterations = [1, 5, 10]
            params = []
            for n in ns:
                for wn in wns:
                    for i in iterations:
                        params.append([n, wn, i])

    # param types and bounds
    if options['iterate'] is False:
        params_types = [int, float]
        params_low = [1, 1e-4]
        params_high = [10, 1-1e-2]
    else:
        params_types = [int, float, int]
        params_low = [3, 1e-4, 1]
        params_high = [10, 1, 1e3]

    # optimize
    func = pynumdiff.smooth_finite_difference.butterdiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def splinediff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
               optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.smooth_finite_difference.splinediff
    See pynumdiff.optimize.__optimize__ and pynumdiff.smooth_finite_difference.splinediff for detailed documentation.
    """
    # initial condition
    if params is None:
        ks = [3, 5]
        ss = [0.5, 0.9, 0.95, 1, 10, 100]
        if options['iterate'] is False:
            params = []
            for s in ss:
                for k in ks:
                    params.append([k, s])
        else:
            iterations = [1, 5, 10]
            params = []
            for s in ss:
                for k in ks:
                    for i in iterations:
                        params.append([k, s, i])

    # param types and bounds
    if options['iterate'] is False:
        params_types = [int, float]
        params_low = [3, 1e-2]
        params_high = [5, 1e6]
    else:
        params_types = [int, float, int]
        params_low = [3, 1e-2, 1]
        params_high = [5, 1e6, 10]

    # optimize
    func = pynumdiff.smooth_finite_difference.splinediff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val
