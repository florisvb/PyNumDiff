"""
This module implements some common total variation regularization methods
"""
import logging
import numpy as np

from pynumdiff.total_variation_regularization import __chartrand_tvregdiff__
import pynumdiff.smooth_finite_difference
from pynumdiff.utils import utility
__gaussian_kernel__ = utility.__gaussian_kernel__

try:
    import cvxpy
except ImportError:
    pass

# Iterative total variation regularization
def iterative_velocity(x, dt, params, options=None):
    """
    Use an iterative solver to find the total variation regularized 1st derivative.
    See __chartrand_tvregdiff__.py for details, author info, and license
    Methods described in: Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data,"
    ISRN Applied Mathematics, Vol. 2011, Article ID 164564, 2011.
    Original code (MATLAB and python):  https://sites.google.com/site/dnartrahckcir/home/tvdiff-code

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list consisting of:

                    - iterations: Number of iterations to run the solver. More iterations results in blockier derivatives, which approach the convex result
                    - gamma: Regularization parameter.

    :type params: list (int, float)

    :param options: a dictionary with 2 key value pairs

                    - 'cg_maxiter': Max number of iterations to use in scipy.sparse.linalg.cg. Default is None, results in maxiter = len(x). This works well in our test examples.
                    - 'scale': This method has two different numerical options. From __chartrand_tvregdiff__.py: 'large' or 'small' (case insensitive).  Default is 'small'.  'small' has somewhat better boundary behavior, but becomes unwieldly for data larger than 1000 entries or so.  'large' has simpler numerics but is more efficient for large-scale problems. 'large' is more readily modified for higher-order derivatives, since the implicit differentiation matrix is square.

    :type options: dict {'cg_maxiter': (int), 'scale': (string)}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'cg_maxiter': 1000, 'scale': 'small'}

    iterations, gamma = params
    dxdt_hat = __chartrand_tvregdiff__.TVRegDiff(x, iterations, gamma, dx=dt,
                                                 maxit=options['cg_maxiter'], scale=options['scale'],
                                                 ep=1e-6, u0=None, plotflag=False, diagflag=1)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


# Generalized total variation regularized derivatives
def __total_variation_regularized_derivative__(x, dt, N, gamma, solver='MOSEK'):
    """
    Use convex optimization (cvxpy) to solve for the Nth total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: (np.array of floats, 1xN) time series to differentiate
    :param dt: (float) time step
    :param N: (int) 1, 2, or 3, the Nth derivative to regularize
    :param gamma: (float) regularization parameter
    :param solver: (string) Solver to use. Solver options include: 'MOSEK' and 'CVXOPT',
                            in testing, 'MOSEK' was the most robust.
    :return: x_hat    : estimated (smoothed) x
             dxdt_hat : estimated derivative of x
    """

    # Normalize
    mean = np.mean(x)
    std = np.std(x)
    x = (x-mean)/std

    # Define the variables for the highest order derivative and the integration constants
    var = cvxpy.Variable(len(x) + N)

    # Recursively integrate the highest order derivative to get back to the position
    derivatives = [var[N:]]
    for i in range(N):
        d = cvxpy.cumsum(derivatives[-1]) + var[i]
        derivatives.append(d)

    # Compare the recursively integration position to the noisy position
    sum_squared_error = cvxpy.sum_squares(derivatives[-1] - x)

    # Total variation regularization on the highest order derivative
    r = cvxpy.sum(gamma*cvxpy.tv(derivatives[0]))

    # Set up and solve the optimization problem
    obj = cvxpy.Minimize(sum_squared_error + r)
    prob = cvxpy.Problem(obj)
    prob.solve(solver=solver)

    # Recursively calculate the value of each derivative
    final_derivative = var.value[N:]
    derivative_values = [final_derivative]
    for i in range(N):
        d = np.cumsum(derivative_values[-1]) + var.value[i]
        derivative_values.append(d)
    for i, _ in enumerate(derivative_values):
        derivative_values[i] = derivative_values[i]/(dt**(N-i))

    # Extract the velocity and smoothed position
    dxdt_hat = derivative_values[-2]
    x_hat = derivative_values[-1]

    dxdt_hat = (dxdt_hat[0:-1] + dxdt_hat[1:])/2
    ddxdt_hat_f = dxdt_hat[-1] - dxdt_hat[-2]
    dxdt_hat_f = dxdt_hat[-1] + ddxdt_hat_f
    dxdt_hat = np.hstack((dxdt_hat, dxdt_hat_f))

    # fix first point
    d = dxdt_hat[2] - dxdt_hat[1]
    dxdt_hat[0] = dxdt_hat[1] - d

    return x_hat*std+mean, dxdt_hat*std


def velocity(x, dt, params, options=None):
    """
    Use convex optimization (cvxpy) to solve for the velocity total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [gamma], where gamma (float) is the regularization parameter
                    or if 'iterate' in options: [gamma, num_iterations]

    :type params: list (float) or float

    :param options: {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT',
                    in testing, 'MOSEK' was the most robust.

    :type options: dict {'solver': SOLVER}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'solver': 'MOSEK'}

    if isinstance(params, list):
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 1, gamma, solver=options['solver'])


def acceleration(x, dt, params, options=None):
    """
    Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [gamma], where gamma (float) is the regularization parameter
                    or if 'iterate' in options: [gamma, num_iterations]

    :type params: list (float) or float

    :param options: {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT',
                    in testing, 'MOSEK' was the most robust.

    :type options: dict {'solver': SOLVER}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'solver': 'MOSEK'}

    if isinstance(params, list):
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 2, gamma, solver=options['solver'])


def jerk(x, dt, params, options=None):
    """
    Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [gamma], where gamma (float) is the regularization parameter
                    or if 'iterate' in options: [gamma, num_iterations]

    :type params: list (float) or float

    :param options: {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT',
                    in testing, 'MOSEK' was the most robust.

    :type options: dict {'solver': SOLVER}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'solver': 'MOSEK'}

    if isinstance(params, list):
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 3, gamma, solver=options['solver'])


def smooth_acceleration(x, dt, params, options=None):
    """
    Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative.
    And then apply a convolutional gaussian smoother to the resulting derivative to smooth out the peaks.
    The end result is similar to the jerk method, but can be more time-efficient.

    Default solver is MOSEK: https://www.mosek.com/

    :param x: time series to differentiate
    :type x: np.array of floats, 1xN

    :param dt: time step
    :type dt: float

    :param params:  list with values [gamma, window_size], where gamma (float) is the regularization parameter, window_size (int) is the window_size to use for the gaussian kernel
    :type params: list -> [float, int]

    :param options: a dictionary indicating which SOLVER option to use, ie. 'MOSEK' or 'CVXOPT', in testing, 'MOSEK' was the most robust.
    :type options: dict {'solver': SOLVER}

    :return: a tuple consisting of:
            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x
    :rtype: tuple -> (np.array, np.array)

    """
    if options is None:
        options = {'solver': 'MOSEK'}

    gamma, window_size = params

    x_hat, dxdt_hat = acceleration(x, dt, [gamma], options=options)
    kernel = __gaussian_kernel__(window_size)
    dxdt_hat = pynumdiff.smooth_finite_difference.__convolutional_smoother__(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def jerk_sliding(x, dt, params, options=None):
    """
    Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [gamma], where gamma (float) is the regularization parameter
                    or if 'iterate' in options: [gamma, num_iterations]

    :type params: list (float) or float

    :param options: {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT',
                    in testing, 'MOSEK' was the most robust.

    :type options: dict {'solver': SOLVER}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'solver': 'MOSEK'}

    if isinstance(params, list):
        gamma = params[0]
    else:
        gamma = params

    window_size = 1000
    stride = 200

    if len(x) < window_size:
        return jerk(x, dt, params, options=options)

    # slide the jerk
    final_xsmooth = []
    final_xdot_hat = []
    first_idx = 0
    final_idx = first_idx + window_size
    last_loop = False

    final_weighting = []

    try:
        while not last_loop:
            xsmooth, xdot_hat = __total_variation_regularized_derivative__(x[first_idx:final_idx], dt, 3,
                                                                           gamma, solver=options['solver'])

            xsmooth = np.hstack(([0]*first_idx, xsmooth, [0]*(len(x)-final_idx)))
            final_xsmooth.append(xsmooth)

            xdot_hat = np.hstack(([0]*first_idx, xdot_hat, [0]*(len(x)-final_idx)))
            final_xdot_hat.append(xdot_hat)

            # blending
            w = np.hstack(([0]*first_idx,
                           np.arange(1, 201)/200,
                           [1]*(final_idx-first_idx-400),
                           (np.arange(1, 201)/200)[::-1],
                           [0]*(len(x)-final_idx)))
            final_weighting.append(w)

            if final_idx >= len(x):
                last_loop = True
            else:
                first_idx += stride
                final_idx += stride
                if final_idx > len(x):
                    final_idx = len(x)
                    if final_idx - first_idx < 200:
                        first_idx -= (200 - (final_idx - first_idx))

        # normalize columns
        weights = np.vstack(final_weighting)
        for c in range(weights.shape[1]):
            weights[:, c] /= np.sum(weights[:, c])

        # weighted sums
        xsmooth = np.vstack(final_xsmooth)
        xsmooth = np.sum(xsmooth*weights, axis=0)

        xdot_hat = np.vstack(final_xdot_hat)
        xdot_hat = np.sum(xdot_hat*weights, axis=0)

        return xsmooth, xdot_hat

    except ValueError:
        print('Solver failed, returning finite difference instead')
        return pynumdiff.utils.utility.finite_difference(x, dt)
