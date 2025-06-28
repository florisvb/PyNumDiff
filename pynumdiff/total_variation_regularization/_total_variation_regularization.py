import numpy as np
from warnings import warn

from pynumdiff.total_variation_regularization import _chartrand_tvregdiff
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


def iterative_velocity(x, dt, params=None, options=None, num_iterations=None, gamma=None, cg_maxiter=1000, scale='small'):
    """Use an iterative solver to find the total variation regularized 1st derivative. See
    _chartrand_tvregdiff.py for details, author info, and license. Methods described in:
    Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data," ISRN Applied Mathematics,
    Vol. 2011, Article ID 164564, 2011. Original code at https://sites.google.com/site/dnartrahckcir/home/tvdiff-code

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size
    :param list params: (**deprecated**, prefer :code:`num_iterations` and :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`cg_maxiter` and :code:`scale`)
        a dictionary consisting of {'cg_maxiter': (int), 'scale': (str)}
    :param int num_iterations: number of iterations to run the solver. More iterations results in
        blockier derivatives, which approach the convex result
    :param float gamma: regularization parameter
    :param int cg_maxiter: Max number of iterations to use in :code:`scipy.sparse.linalg.cg`. Default
        :code:`None` results in maxiter = len(x). This works well in our test examples.
    :param str scale: This method has two different numerical options. From :code:`_chartrand_tvregdiff.py`:
        :code:`'large'` or :code:`'small'` (case insensitive).  Default is :code:`'small'`. :code:`'small'`
        has somewhat better boundary behavior, but becomes unwieldly for data larger than 1000 entries or so.
        :code:`'large'` has simpler numerics but is more efficient for large-scale problems. :code:`'large'`
        is more readily modified for higher-order derivatives, since the implicit differentiation matrix is square.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `num_iterations`, " +
            "`gamma`, `cg_maxiter`, and `scale` instead.", DeprecationWarning)
        num_iterations, gamma = params
        if options != None:
            if 'cg_maxiter' in options: cg_maxiter = options['cg_maxiter']
            if 'scale' in options: scale = options['scale']
    elif num_iterations == None or gamma == None:
        raise ValueError("`num_iterations` and `gamma` must be given.")

    dxdt_hat = _chartrand_tvregdiff.TVRegDiff(x, num_iterations, gamma, dx=dt,
                                                maxit=cg_maxiter, scale=scale,
                                                ep=1e-6, u0=None, plotflag=False, diagflag=1)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def _total_variation_regularized_derivative(x, dt, order, gamma, solver=None):
    """Generalized total variation regularized derivatives. Use convex optimization (cvxpy) to solve for a
    total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: time step
    :param int order: 1, 2, or 3, the derivative to regularize
    :param float gamma: regularization parameter
    :param str solver: Solver to use. Solver options include: 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS'.
                    In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.
  
    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    # Normalize
    mean = np.mean(x)
    std = np.std(x)
    if std == 0: std = 1 # safety guard
    x = (x-mean)/std

    # Define the variables for the highest order derivative and the integration constants
    deriv_values = cvxpy.Variable(len(x)) # values of the order^th derivative, in which we're penalizing variation
    integration_constants = cvxpy.Variable(order) # constants of integration that help get us back to x

    # Recursively integrate the highest order derivative to get back to the position
    y = deriv_values
    for i in range(order):
        y = cvxpy.cumsum(y) + integration_constants[i]

    # Set up and solve the optimization problem
    prob = cvxpy.Problem(cvxpy.Minimize(
        # Compare the recursively integrated position to the noisy position, and add TVR penalty
        cvxpy.sum_squares(y - x) + cvxpy.sum(gamma*cvxpy.tv(deriv_values)) ))
    prob.solve(solver=solver)

    # Recursively integrate the final derivative values to get back to the function and derivative values
    y = deriv_values.value
    for i in range(order-1): # stop one short to get the first derivative
        y = np.cumsum(y) + integration_constants.value[i]
    dxdt_hat = y/dt # y only holds the dx values; to get deriv scale by dt
    x_hat = np.cumsum(y) + integration_constants.value[order-1] # smoothed data

    dxdt_hat = (dxdt_hat[0:-1] + dxdt_hat[1:])/2 # take first order FD to smooth a touch
    ddxdt_hat_f = dxdt_hat[-1] - dxdt_hat[-2]
    dxdt_hat_f = dxdt_hat[-1] + ddxdt_hat_f # What is this doing? Could we use a 2nd order FD above natively?
    dxdt_hat = np.hstack((dxdt_hat, dxdt_hat_f))

    # fix first point
    d = dxdt_hat[2] - dxdt_hat[1]
    dxdt_hat[0] = dxdt_hat[1] - d

    return x_hat*std+mean, dxdt_hat*std


def velocity(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the velocity total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: time step size
    :param params: (**deprecated**, prefer :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `gamma` " +
            "and `solver` instead.", DeprecationWarning)
        gamma = params[0] if isinstance(params, list) else params
        if options != None:
            if 'solver' in options: solver = options['solver']
    elif gamma == None:
        raise ValueError("`gamma` must be given.")

    return _total_variation_regularized_derivative(x, dt, 1, gamma, solver=solver)


def acceleration(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative.
    
    :param np.array[float] x: data to differentiate
    :param float dt: time step size
    :param params: (**deprecated**, prefer :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `gamma` " +
            "and `solver` instead.", DeprecationWarning)
        gamma = params[0] if isinstance(params, list) else params
        if options != None:
            if 'solver' in options: solver = options['solver']
    elif gamma == None:
        raise ValueError("`gamma` must be given.")

    return _total_variation_regularized_derivative(x, dt, 2, gamma, solver=solver)


def jerk(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: time step size
    :param params: (**deprecated**, prefer :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `gamma` " +
            "and `solver` instead.", DeprecationWarning)
        gamma = params[0] if isinstance(params, list) else params
        if options != None:
            if 'solver' in options: solver = options['solver']
    elif gamma == None:
        raise ValueError("`gamma` must be given.")

    return _total_variation_regularized_derivative(x, dt, 3, gamma, solver=solver)


def smooth_acceleration(x, dt, params=None, options=None, gamma=None, window_size=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative,
    and then apply a convolutional gaussian smoother to the resulting derivative to smooth out the peaks.
    The end result is similar to the jerk method, but can be more time-efficient.

    :param np.array[float] x: data to differentiate
    :param float dt: time step size
    :param params: (**deprecated**, prefer :code:`gamma` and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param int window_size: window size for gaussian kernel
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `gamma` " +
            "and `solver` instead.", DeprecationWarning)
        gamma, window_size = params
        if options != None:
            if 'solver' in options: solver = options['solver']
    elif gamma == None or window_size == None:
        raise ValueError("`gamma` and `window_size` must be given.")

    _, dxdt_hat = _total_variation_regularized_derivative(x, dt, 2, gamma, solver=solver)

    kernel = utility._gaussian_kernel(window_size)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def jerk_sliding(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: time step size
    :param params: (**deprecated**, prefer :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `gamma` " +
            "and `solver` instead.", DeprecationWarning)
        gamma = params[0] if isinstance(params, list) else params
        if options != None:
            if 'solver' in options: solver = options['solver']
    elif gamma == None:
        raise ValueError("`gamma` must be given.")

    window_size = 1000
    stride = 200

    if len(x) < window_size:
        return _total_variation_regularized_derivative(x, dt, 3, gamma, solver=solver)

    # slide the jerk
    final_xsmooth = []
    final_xdot_hat = []
    first_idx = 0
    final_idx = first_idx + window_size
    last_loop = False

    final_weighting = []

    try:
        while not last_loop:
            xsmooth, xdot_hat = _total_variation_regularized_derivative(x[first_idx:final_idx], dt, 3,
                                                                           gamma, solver=solver)

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
        warn('Solver failed, returning finite difference instead')
        from pynumdiff.finite_difference import second_order
        return second_order(x, dt)
