"""This module implements some common finite difference schemes.
This is handy for this module https://web.media.mit.edu/~crtaylor/calculator.html"""
import numpy as np
from pynumdiff.utils import utility


def finitediff(x, dt, num_iterations=1, order=2, axis=0):
    """Perform iterated finite difference of a given order. This serves as the common backing function for
    all other methods in this module.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int num_iterations: number of iterations. If >1, the derivative is integrated with trapezoidal
            rule, that result is finite-differenced again, and the cycle is repeated num_iterations-1 times
    :param int order: 1, 2, or 4, controls which finite differencing scheme to employ
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- original x if :code:`num_iterations=1`, else smoothed x that yielded dxdt_hat
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Differencing spreads a NaN to its neighbors, and iterating spreads it further.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. Difference stencils assume uniformly spaced samples.")
    if num_iterations < 1: raise ValueError("num_iterations must be >0")
    if order not in [1, 2, 4]: raise ValueError("order must be 1, 2, or 4")

    x = np.moveaxis(x, axis, 0) # move the axis of interest to the front to simplify differencing indexing
    x_hat = np.asarray(x) # allows for array-like. Preserve reference to x, for finding the final constant of integration
    dxdt_hat = np.zeros(x.shape) # preallocate reusable memory

    # For all but the last iteration, do the differentate->integrate smoothing loop, being careful with endpoints
    for i in range(num_iterations-1):
        if order == 1:
            dxdt_hat[:-1] = np.diff(x_hat, axis=0)
            dxdt_hat[-1] = dxdt_hat[-2] # using stencil -1,0 vs stencil 0,1 you get an expression for the same value
        elif order == 2:
            dxdt_hat[1:-1] = (x_hat[2:] - x_hat[:-2])/2 # second-order center-difference formula
            dxdt_hat[0] = x_hat[1] - x_hat[0]
            dxdt_hat[-1] = x_hat[-1] - x_hat[-2] # use first-order endpoint formulas so as not to amplify noise. See #104
        elif order == 4:
            dxdt_hat[2:-2] = (8*(x_hat[3:-1] - x_hat[1:-3]) - x_hat[4:] + x_hat[:-4])/12 # fourth-order center-difference
            dxdt_hat[1] = (x_hat[2] - x_hat[0])/2
            dxdt_hat[-2] = (x_hat[-1] - x_hat[-3])/2 # use second-order formula for next-to-endpoints so as not to amplify noise
            dxdt_hat[0] = x_hat[1] - x_hat[0]
            dxdt_hat[-1] = x_hat[-1] - x_hat[-2] # use first-order endpoint formulas so as not to amplify noise. See #104

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, 1, axis=0) # estimate new x_hat by integrating derivative
        # We can skip dividing by dt here and pass dt=1, because the integration multiplies dt back in.
        # No need to find integration constant until the very end, because we just differentiate again.
        # Note that I also tried integrating with Simpson's rule here, and it seems to do worse. See #104

    if order == 1:
        dxdt_hat[:-1] = np.diff(x_hat, axis=0)
        dxdt_hat[-1] = dxdt_hat[-2] # using stencil -1,0 vs stencil 0,1 you get an expression for the same value
    elif order == 2:
        dxdt_hat[1:-1] = x_hat[2:] - x_hat[:-2] # second-order center-difference formula
        dxdt_hat[0] = -3 * x_hat[0] + 4 * x_hat[1] - x_hat[2] # second-order endpoint formulas
        dxdt_hat[-1] = 3 * x_hat[-1] - 4 * x_hat[-2] + x_hat[-3]
        dxdt_hat /= 2
    elif order == 4:
        dxdt_hat[2:-2] = 8*(x_hat[3:-1] - x_hat[1:-3]) - x_hat[4:] + x_hat[:-4] # fourth-order center-difference
        dxdt_hat[0] = -25*x_hat[0] + 48*x_hat[1] - 36*x_hat[2] + 16*x_hat[3] - 3*x_hat[4]
        dxdt_hat[1] = -3*x_hat[0] - 10*x_hat[1] + 18*x_hat[2] - 6*x_hat[3] + x_hat[4]
        dxdt_hat[-2] = 3*x_hat[-1] + 10*x_hat[-2] - 18*x_hat[-3] + 6*x_hat[-4] - x_hat[-5]
        dxdt_hat[-1] = 25*x_hat[-1] - 48*x_hat[-2] + 36*x_hat[-3] - 16*x_hat[-4] + 3*x_hat[-5]
        dxdt_hat /= 12
    dxdt_hat /= dt # don't forget to scale by dt, can't skip it this time

    if num_iterations > 1: # We've lost a constant of integration in the above
        x_hat += utility.estimate_integration_constant(x, x_hat, axis=0)

    return np.moveaxis(x_hat, 0, axis), np.moveaxis(dxdt_hat, 0, axis) # reorder axes back to original

