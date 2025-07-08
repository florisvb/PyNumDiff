"""This is handy for this module https://web.media.mit.edu/~crtaylor/calculator.html"""
import numpy as np
from pynumdiff.utils import utility
from warnings import warn


def first_order(x, dt, params=None, options={}, num_iterations=None):
    """First-order centered difference method

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[float] or float params: (**deprecated**, prefer :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) a dictionary consisting of {'iterate': (bool)}
    :param int num_iterations: If performing iterated FD to smooth the estimates, give the number of iterations.
            If ungiven, FD will not be iterated.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None and 'iterate' in options:
        warn("`params` and `options` parameters will be removed in a future version. Use `num_iterations` instead.", DeprecationWarning)
        if isinstance(params, list): params = params[0]
        return _iterate_first_order(x, dt, params)
    elif num_iterations:
        return _iterate_first_order(x, dt, num_iterations)

    dxdt_hat = np.diff(x) / dt # Calculate the finite difference
    dxdt_hat = np.hstack((dxdt_hat, dxdt_hat[-1])) # using stencil -1,0, you get expression for previous value

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

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int num_iterations: number of iterations

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    w = np.linspace(0, 1, len(x)) # set up weights, [0., ... 1.0]

    # forward backward passes
    for _ in range(num_iterations):
        xf = _x_hat_using_finite_difference(x, dt)
        xb = _x_hat_using_finite_difference(x[::-1], dt)
        x = xf * w + xb[::-1] * (1 - w)

    x_hat, dxdt_hat = first_order(x, dt)

    return x_hat, dxdt_hat


def second_order(x, dt):
    """Second-order centered difference method, with special endpoint formulas.

    :param np.array[float] x: data to differentiate
    :param float dt: step size

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    dxdt_hat = np.zeros(x.shape)
    dxdt_hat[1:-1] = x[2:] - x[:-2]
    dxdt_hat[0] = -3 * x[0] + 4 * x[1] - x[2]
    dxdt_hat[-1] = 3 * x[-1] - 4 * x[-2] + x[-3]
    dxdt_hat /= 2*dt

    return x, dxdt_hat


def fourth_order(x, dt):
    """Fourth-order centered difference method, with special endpoint formulas.

    :param np.array[float] x: data to differentiate
    :param float dt: step size

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    dxdt_hat = np.zeros(x.shape)
    dxdt_hat[2:-2] = (8*(x[3:-1] - x[1:-3]) - x[4:] + x[:-4])
    dxdt_hat[0] = -25*x[0] + 48*x[1] - 36*x[2] + 16*x[3] - 3*x[4]
    dxdt_hat[1] = -3*x[0] - 10*x[1] + 18*x[2] - 6*x[3] + x[4]
    dxdt_hat[-2] = 3*x[-1] + 10*x[-2] - 18*x[-3] + 6*x[-4] - x[-5]
    dxdt_hat[-1] = 25*x[-1] - 48*x[-2] + 36*x[-3] - 16*x[-4] + 3*x[-5]
    dxdt_hat /= 12*dt

    return x, dxdt_hat
