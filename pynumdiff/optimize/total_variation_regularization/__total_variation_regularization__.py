"""
functions for optimizing total variation regularization
"""
import pynumdiff.total_variation_regularization
from pynumdiff.optimize.__optimize__ import __optimize__

####################
# Helper functions #
####################


def __optimize_tvr__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                     optimization_method, optimization_options, metric):
    """
    :param func:
    :param x:
    :param dt:
    :param params:
    :param options:
    :param dxdt_truth:
    :param tvgamma:
    :param padding:
    :param optimization_method:
    :param optimization_options:
    :param metric:
    :return:
    """
    # initial condition
    if params is None:
        params = [[1e-2], [1e-1], [1], [10], [100], [1000]]

    # param types and bounds
    params_types = [float, ]
    params_low = [1e-4, ]
    params_high = [1e7, ]

    # optimize
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val

######################
# Optimize functions #
######################


def iterative_velocity(x, dt, params=None, options={'cg_maxiter': 1000, 'scale': 'small'},
                       dxdt_truth=None, tvgamma=1e-2, padding='auto',
                       optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.iterative_velocity
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.iterative_velocity for detailed documentation.
    """

    # optimize
    func = pynumdiff.total_variation_regularization.iterative_velocity

    # initial condition
    if params is None:
        params = [[1, 1e-2], [1, 1e-1], [1, 1], [1, 10], [1, 100], [1, 1000]]

    # param types and bounds
    params_types = [int, float, ]
    params_low = [1, 1e-4, ]
    params_high = [100, 1e7, ]

    # optimize
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def velocity(x, dt, params=None, options={'solver': 'MOSEK'}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
             optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.velocity
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.velocity for detailed documentation.
    """

    # optimize
    func = pynumdiff.total_variation_regularization.velocity
    opt_params, opt_val = __optimize_tvr__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                           optimization_method, optimization_options, metric)

    return opt_params, opt_val


def acceleration(x, dt, params=None, options={'solver': 'MOSEK'}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
                 optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.acceleration
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.acceleration for detailed documentation.
    """

    # optimize
    func = pynumdiff.total_variation_regularization.acceleration
    opt_params, opt_val = __optimize_tvr__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                           optimization_method, optimization_options, metric)

    return opt_params, opt_val


def jerk(x, dt, params=None, options={'solver': 'MOSEK'}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
         optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.jerk
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.jerk for detailed documentation.
    """
    # optimize
    function = pynumdiff.total_variation_regularization.jerk
    opt_params, opt_val = __optimize_tvr__(function, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                           optimization_method, optimization_options, metric)

    return opt_params, opt_val


def jerk_sliding(x, dt, params=None, options={'solver': 'MOSEK'}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
         optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.jerk_sliding
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.jerk_sliding for detailed documentation.
    """

    # optimize
    func = pynumdiff.total_variation_regularization.jerk_sliding
    opt_params, opt_val = __optimize_tvr__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                           optimization_method, optimization_options, metric)

    return opt_params, opt_val


def smooth_acceleration(x, dt, params=None, options={'solver': 'MOSEK'}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
                        optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.total_variation_regularization.smooth_acceleration
    See pynumdiff.optimize.__optimize__ and pynumdiff.total_variation_regularization.smooth_acceleration for detailed documentation.
    """
    # initial condition
    if params is None:
        gammas = [1e-2, 1e-1, 1, 10, 100, 1000]
        window_sizes = [1, 10, 30, 50, 90, 130]
        params = []
        for gamma in gammas:
            for window_size in window_sizes:
                params.append([gamma, window_size])

    # param types and bounds
    params_types = [int, int]
    params_low = [1e-4, 1]
    params_high = [1e7, 1e3]

    # optimize
    func = pynumdiff.total_variation_regularization.smooth_acceleration
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val
