import numpy as np
from pynumdiff.utils import utility


def first_order(x, dt, params=None, options={}, num_iterations=None):
    """First-order centered difference method

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size
    :param list[float] or float params: (**deprecated**, prefer :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) a dictionary consisting of {'iterate': (bool)}
    :param int num_iterations: If performing iterated FD to smooth the estimates, give the number of iterations.
            If ungiven, FD will not be iterated.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None and 'iterate' in options:
        raise DeprecationWarning("""`params` and `options` parameters will be removed in a future version. Use `num_iterations` instead.""")
        if isinstance(params, list): params = params[0]
        return _iterate_first_order(x, dt, params)
    elif num_iterations:
        return _iterate_first_order(x, dt, num_iterations)

    dxdt_hat = np.diff(x) / dt # Calculate the finite difference
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1])) # Pad the data
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0) # Re-finite dxdt_hat using linear interpolation

    return x, dxdt_hat


def second_order(x, dt):
    """Second-order centered difference method

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    dxdt_hat = (x[2:] - x[0:-2]) / (2 * dt)
    first_dxdt_hat = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * dt)
    last_dxdt_hat = (3 * x[-1] - 4 * x[-2] + x[-3]) / (2 * dt)
    dxdt_hat = np.hstack((first_dxdt_hat, dxdt_hat, last_dxdt_hat))
    return x, dxdt_hat


def _x_hat_using_finite_difference(x, dt):
    """Find a smoothed estimate of the true function by taking FD and then integrating with trapezoids
    """
    x_hat, dxdt_hat = first_order(x, dt)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    return x_hat + x0


def _iterate_first_order(x, dt, num_iterations):
    """Iterative first order centered finite difference.

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size
    :param int num_iterations: number of iterations

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    w = np.arange(len(x)) / (len(x) - 1) # set up weights, [0., ... 1.0]

    # forward backward passes
    for _ in range(num_iterations):
        xf = _x_hat_using_finite_difference(x, dt)
        xb = _x_hat_using_finite_difference(x[::-1], dt)
        x = xf * w + xb[::-1] * (1 - w)

    x_hat, dxdt_hat = first_order(x, dt)

    return x_hat, dxdt_hat
