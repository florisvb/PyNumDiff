"""This is handy for this module https://web.media.mit.edu/~crtaylor/calculator.html"""
import numpy as np
from pynumdiff.utils import utility
from warnings import warn

def _finite_difference(x, dt, num_iterations, order):
    """Helper for all finite difference methods, since their iteration structure is all the same.
    
    :param int order: 1, 2, or 4, controls which finite differencing scheme to employ
    For other parameters and return values, see public function docstrings
    """
    if num_iterations < 1: raise ValueError("num_iterations must be >0")
    if order not in [1, 2, 4]: raise ValueError("order must be 1, 2, or 4")

    x_hat = x # preserve a reference to x, because if iterating we need it to find the final constant of integration
    dxdt_hat = np.zeros(x.shape) # preallocate reusable memory

    for i in range(num_iterations):
        # calculate the derivative
        if order == 1:
            dxdt_hat[:-1] = np.diff(x_hat) / dt
            dxdt_hat[-1] = dxdt_hat[-2] # using stencil -1,0 vs stencil 0,1 you get an expression for the same value
        elif order == 2:
            dxdt_hat[1:-1] = x_hat[2:] - x_hat[:-2]
            dxdt_hat[0] = -3 * x_hat[0] + 4 * x_hat[1] - x_hat[2]
            dxdt_hat[-1] = 3 * x_hat[-1] - 4 * x_hat[-2] + x_hat[-3]
            dxdt_hat /= 2*dt
        elif order == 4:
            dxdt_hat[2:-2] = (8*(x_hat[3:-1] - x_hat[1:-3]) - x_hat[4:] + x_hat[:-4])
            dxdt_hat[0] = -25*x_hat[0] + 48*x_hat[1] - 36*x_hat[2] + 16*x_hat[3] - 3*x_hat[4]
            dxdt_hat[1] = -3*x_hat[0] - 10*x_hat[1] + 18*x_hat[2] - 6*x_hat[3] + x_hat[4]
            dxdt_hat[-2] = 3*x_hat[-1] + 10*x_hat[-2] - 18*x_hat[-3] + 6*x_hat[-4] - x_hat[-5]
            dxdt_hat[-1] = 25*x_hat[-1] - 48*x_hat[-2] + 36*x_hat[-3] - 16*x_hat[-4] + 3*x_hat[-5]
            dxdt_hat /= 12*dt

        if i < num_iterations - 1: # if not the last iteration
            x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt) # estimate new x_hat by integrating derivative
            # no need to find integration constant until the very end, because we just differentiate again

    if num_iterations > 1: # We've lost a constant of integration in the above
        x_hat += utility.estimate_integration_constant(x, x_hat) # uses least squares

    return x_hat, dxdt_hat


def first_order(x, dt, params=None, options={}, num_iterations=1):
    """First-order centered difference method

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[float] or float params: (**deprecated**, prefer :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) a dictionary consisting of {'iterate': (bool)}
    :param int num_iterations: number of iterations. If >1, the derivative is integrated with trapezoidal
            rule, that result is finite-differenced again, and the cycle is repeated num_iterations-1 times

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- original x if :code:`num_iterations=1`, else smoothed x that yielded dxdt_hat
             - **dxdt_hat** -- estimated derivative of x
    """
    warn("`first_order` in past releases was actually calculating a second-order FD. Use `second_order` to achieve " +
        "approximately the same behavior. Note that odd-order methods have asymmetrical stencils, which causes " +
        "horizontal drift in the answer, especially when iterating.", DeprecationWarning)
    if params != None and 'iterate' in options:
        warn("`params` and `options` parameters will be removed in a future version. Use `num_iterations` instead.", DeprecationWarning)
        num_iterations = params[0] if isinstance(params, list) else params

    return _finite_difference(x, dt, num_iterations, 1)


def second_order(x, dt, num_iterations=1):
    """Second-order centered difference method, with special endpoint formulas.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int num_iterations: number of iterations. If >1, the derivative is integrated with trapezoidal
            rule, that result is finite-differenced again, and the cycle is repeated num_iterations-1 times

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- original x if :code:`num_iterations=1`, else smoothed x that yielded dxdt_hat
             - **dxdt_hat** -- estimated derivative of x
    """
    return _finite_difference(x, dt, num_iterations, 2)


def fourth_order(x, dt, num_iterations=1):
    """Fourth-order centered difference method, with special endpoint formulas.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int num_iterations: number of iterations. If >1, the derivative is integrated with trapezoidal
            rule, that result is finite-differenced again, and the cycle is repeated num_iterations-1 times

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- original x if :code:`num_iterations=1`, else smoothed x that yielded dxdt_hat
             - **dxdt_hat** -- estimated derivative of x
    """
    return _finite_difference(x, dt, num_iterations, 4)
  