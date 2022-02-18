"""
functions for optimizing kalman smoothing approaches
"""
import numpy as np
import pynumdiff
from pynumdiff.optimize.__optimize__ import __optimize__
from pynumdiff.linear_model import polydiff

####################
# Helper functions #
####################


def __estimate_noise__(x, dt, window_size=50):
    """
    :param x:
    :param dt:
    :param window_size:
    :return:
    """
    x_hat, dxdt_hat = polydiff(x, dt, [5, window_size],
                               options={'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'})
    noise_hat = x-x_hat
    noise_std_hat = np.std(noise_hat)
    return noise_std_hat**2


def __optimize_kalman__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
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
    # r = estimate_noise__(x, dt) # estimate noise using a 5th order sliding polynomial smoother
    if params is None:
        rs = [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8]
        qs = [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8]
        params = []
        for r in rs:
            for q in qs:
                params.append([r, q])

    # param types and bounds
    params_types = [float, float]
    params_low = [1e-10, 1e-10]
    params_high = [1e10, 1e10]

    # optimize
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val

######################
# Optimize functions #
######################


def constant_velocity(x, dt, params=None, options={'forwardbackward': True}, dxdt_truth=None, tvgamma=1e-2,
                      padding='auto', optimization_method='Nelder-Mead', optimization_options={'maxiter': 10},
                      metric='rmse'):
    """
    Optimize the parameters for pynumdiff.kalman_smooth.constant_velocity
    See pynumdiff.optimize.__optimize__ and pynumdiff.kalman_smooth.constant_velocity for detailed documentation.
    """

    # optimize
    func = pynumdiff.kalman_smooth.constant_velocity
    opt_params, opt_val = __optimize_kalman__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                              optimization_method, optimization_options, metric)

    return opt_params, opt_val


def constant_acceleration(x, dt, params=None, options={'forwardbackward': True}, dxdt_truth=None, tvgamma=1e-2,
                          padding='auto', optimization_method='Nelder-Mead',
                          optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.kalman_smooth.constant_acceleration
    See pynumdiff.optimize.__optimize__ and pynumdiff.kalman_smooth.constant_acceleration for detailed documentation.
    """

    # optimize
    func = pynumdiff.kalman_smooth.constant_acceleration
    opt_params, opt_val = __optimize_kalman__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                              optimization_method, optimization_options, metric)

    return opt_params, opt_val


def constant_jerk(x, dt, params=None, options={'forwardbackward': True}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
                  optimization_method='Nelder-Mead', optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.kalman_smooth.constant_jerk
    See pynumdiff.optimize.__optimize__ and pynumdiff.kalman_smooth.constant_jerk for detailed documentation.
    """
    # optimize
    func = pynumdiff.kalman_smooth.constant_jerk
    opt_params, opt_val = __optimize_kalman__(func, x, dt, params, options, dxdt_truth, tvgamma, padding,
                                              optimization_method, optimization_options, metric)

    return opt_params, opt_val
