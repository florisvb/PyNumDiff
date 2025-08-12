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

    :param np.array[float] x: data to differentiate
    :param float dt: step size
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
    x0 = utility.estimate_integration_constant(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def tvrdiff(x, dt, order, gamma, solver=None):
    """Generalized total variation regularized derivatives. Use convex optimization (cvxpy) to solve for a
    total variation regularized derivative. Other convex-solver-based methods in this module call this function.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int order: 1, 2, or 3, the derivative to regularize
    :param float gamma: regularization parameter
    :param str solver: Solver to use. Solver options include: 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS'.
                    In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    # Normalize for numerical consistency with convex solver
    mean = np.mean(x)
    std = np.std(x)
    if std == 0: std = 1 # safety guard
    x = (x-mean)/std

    # Define the variables for the highest order derivative and the integration constants
    deriv_values = cvxpy.Variable(len(x)) # values of the order^th derivative, in which we're penalizing variation
    integration_constants = cvxpy.Variable(order) # constants of integration that help get us back to x

    # Recursively integrate the highest order derivative to get back to the position. This is a first-
    # order scheme, but it's very fast and tends to do not markedly worse than 2nd order. See #116
    # I also tried a trapezoidal integration rule here, and it works no better. See #116 too
    y = deriv_values
    for i in range(order):
        y = cvxpy.cumsum(y) + integration_constants[i]

    # Set up and solve the optimization problem
    prob = cvxpy.Problem(cvxpy.Minimize(
        # Compare the recursively integrated position to the noisy position, and add TVR penalty
        cvxpy.sum_squares(y - x) + gamma*cvxpy.sum(cvxpy.tv(deriv_values)) ))
    prob.solve(solver=solver)

    # Recursively integrate the final derivative values to get back to the function and derivative values
    y = deriv_values.value
    for i in range(order-1): # stop one short to get the first derivative
        y = np.cumsum(y) + integration_constants.value[i]
    dxdt_hat = y/dt # y only holds the dx values; to get deriv scale by dt
    x_hat = np.cumsum(y) + integration_constants.value[order-1] # smoothed data

    # Due to the first-order nature of the derivative, it has a slight lag. Average together every two values
    # to better center the answer. But this leaves us one-short, so devise a good last value.
    dxdt_hat = (dxdt_hat[:-1] + dxdt_hat[1:])/2
    dxdt_hat = np.hstack((dxdt_hat, 2*dxdt_hat[-1] - dxdt_hat[-2])) # last value = penultimate value [-1] + diff between [-1] and [-2]

    return x_hat*std+mean, dxdt_hat*std # derivative is linear, so scale derivative by std


def velocity(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the velocity total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
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

    return tvrdiff(x, dt, 1, gamma, solver=solver)


def acceleration(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative.
    
    :param np.array[float] x: data to differentiate
    :param float dt: step size
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

    return tvrdiff(x, dt, 2, gamma, solver=solver)


def jerk(x, dt, params=None, options=None, gamma=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
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

    return tvrdiff(x, dt, 3, gamma, solver=solver)


def smooth_acceleration(x, dt, params=None, options=None, gamma=None, window_size=None, solver=None):
    """Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative,
    and then apply a convolutional gaussian smoother to the resulting derivative to smooth out the peaks.
    The end result is similar to the jerk method, but can be more time-efficient.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
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

    _, dxdt_hat = tvrdiff(x, dt, 2, gamma, solver=solver)

    kernel = utility.gaussian_kernel(window_size)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def jerk_sliding(x, dt, params=None, options=None, gamma=None, solver=None, window_size=101):
    """Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param params: (**deprecated**, prefer :code:`gamma`)
    :param dict options: (**deprecated**, prefer :code:`solver`) a dictionary consisting of {'solver': (str)}
    :param float gamma: the regularization parameter
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.
    :param int window_size: how wide to make the kernel

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
            if 'window_size' in options: window_size = options['window_size']
    elif gamma == None:
        raise ValueError("`gamma` must be given.")

    if len(x) < window_size or window_size < 15:
        warn("len(x) should be > window_size >= 15, calling standard jerk() without sliding")
        return tvrdiff(x, dt, 3, gamma, solver=solver)

    if window_size % 2 == 0:
        window_size += 1 # has to be odd
        warn("Kernel window size should be odd. Added 1 to length.")
    ramp = window_size//5
    kernel = np.hstack((np.arange(1, ramp+1)/ramp, np.ones(window_size - 2*ramp), np.arange(ramp, 0, -1)/ramp))
    return utility.slide_function(tvrdiff, x, dt, kernel, 3, gamma, stride=ramp, solver=solver)
