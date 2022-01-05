"""
This module implements some common finite difference schemes
"""
import numpy as np
from pynumdiff.utils import utility


def first_order(x, dt, params=None, options={}):
    """
    First-order centered difference method

    :param x: array of time series to differentiate
    :type x: np.array (floats)

    :param dt: time step size
    :type dt: float

    :param params: number of iterations (if 'iterate' is enabled in options)
    :type params: list (int) or int, optional

    :param options: a dictionary indicating whether to iteratively apply the difference method to smooth the estimates
    :type options: dict {'iterate': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if 'iterate' in options:
        assert params and isinstance(params, list), "params should be a non-empty list!"
        return __iterate_first_order__(x, dt, params)

    # Calculate the finite difference
    dxdt_hat = np.diff(x) / dt
    # Pad the data
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1]))
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0)

    return x, dxdt_hat


def second_order(x, dt):
    """
    Second-order centered difference method

    :param x: array of time series to differentiate
    :type x: np.array (floats)

    :param dt: time step size
    :type dt: float

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    dxdt_hat = (x[2:] - x[0:-2]) / (2 * dt)
    first_dxdt_hat = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * dt)
    last_dxdt_hat = (3 * x[-1] - 4 * x[-2] + x[-3]) / (2 * dt)
    dxdt_hat = np.hstack((first_dxdt_hat, dxdt_hat, last_dxdt_hat))
    return x, dxdt_hat


def __x_hat_using_finite_difference__(x, dt):
    """
    :param x:
    :param dt:
    :return:
    """
    x_hat, dxdt_hat = first_order(x, dt)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    return x_hat + x0


def __iterate_first_order__(x, dt, params):
    """
    Iterative first order centered finite difference.

    :param x: array of time series to differentiate
    :type x: np.array (floats)

    :param dt: time step size
    :type dt: float

    :param params: number of iterations (if 'iterate' is enabled in options)
    :type params: list (int) or int, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if isinstance(params, list):
        iterations = params[0]
    else:
        iterations = params

    # set up weights
    w = np.arange(0, len(x), 1)
    w = w / np.max(w)

    # forward backward passes
    for _ in range(iterations):
        xf = __x_hat_using_finite_difference__(x, dt)
        xb = __x_hat_using_finite_difference__(x[::-1], dt)
        x = xf * w + xb[::-1] * (1 - w)

    x_hat, dxdt_hat = first_order(x, dt)

    return x_hat, dxdt_hat
