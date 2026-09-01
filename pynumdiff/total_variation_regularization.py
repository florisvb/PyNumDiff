"""This module implements some common total variation regularization methods."""
import numpy as np
try: import cvxpy
except ImportError: pass

from pynumdiff.utils import _chartrand_tvregdiff, utility


def iterative_velocity(x, dt, num_iterations, gamma, cg_maxiter=1000, scale='small'):
    """Use an iterative solver to find the total variation regularized 1st derivative. See
    _chartrand_tvregdiff.py for details, author info, and license. Methods described in:
    Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data," ISRN Applied Mathematics,
    Vol. 2011, Article ID 164564, 2011. Original code at https://sites.google.com/site/dnartrahckcir/home/tvdiff-code

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int num_iterations: number of iterations to run the solver. More iterations results in
        blockier derivatives, which approach the convex result
    :param float gamma: regularization parameter
    :param int cg_maxiter: Max number of iterations to use in :code:`scipy.sparse.linalg.cg`. Default
        :code:`None` results in maxiter = len(x). This works well in our test examples.
    :param str scale: This method has two different numerical options. From :code:`_chartrand_tvregdiff.py`:
        :code:`'large'` or :code:`'small'` (case insensitive).  Default is :code:`'small'`. :code:`'small'`
        has somewhat better boundary behavior, but becomes unwieldly for data larger than 1000 entries or so.
        :code:`'large'` has simpler numerics and is more efficient for large-scale problems. :code:`'large'`
        is more readily modified for higher-order derivatives, since the implicit differentiation matrix is square.

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    dxdt_hat = _chartrand_tvregdiff.TVRegDiff(x, num_iterations, gamma, dx=dt,
                                                maxit=cg_maxiter, scale=scale,
                                                ep=1e-6, u0=None, plotflag=False)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


@np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: canonicalizing norm1/huber/tv builds sum atoms, which reduce over uninitialized
#  memory just to read off a shape, so they warn when it holds garbage. This wall can come down if they ever fix it upstream.
def tvrdiff(x, dt, order, gamma, huberM=float('inf'), solver=None, axis=0):
    """Generalized total variation regularized derivatives. Use convex optimization (cvxpy) to solve for a
    total variation regularized derivative. Other convex-solver-based methods in this module call this function.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int order: 1, 2, or 3, the derivative to regularize
    :param float gamma: regularization parameter
    :param float huberM: Huber loss parameter, in units of scaled median absolute deviation of input data.
                    :math:`M = \\infty` reduces to :math:`\\ell_2` loss squared on first, fidelity cost term, and
                    :math:`M = 0` reduces to :math:`\\ell_1` loss, which seeks sparse residuals.
    :param str solver: Solver to use. Solver options include: 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS'.
                    If not given, fall back to CVXPY's default.
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. CVXPY cannot form a problem with missing data.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The convex problem setup integrates with a cumulative "
        "sum and penalizes variation between consecutive samples, both of which assume uniform steps.")

    x_hat = np.empty(x.shape, dtype=float); dxdt_hat = np.empty(x.shape, dtype=float) # float explicitly, so inherited integer input type cannot silently truncate

    for vec_idx in np.ndindex(x.shape[:axis] + x.shape[axis+1:]):
        s = vec_idx[:axis] + (slice(None),) + vec_idx[axis:] # for indexing this iteration's vector in the overall array
        x_v = x[s]

        # Normalize for numerical consistency with convex solver
        mu = np.mean(x_v)
        sigma = utility.robust_data_scale(x_v)
        if sigma == 0: sigma = 1 # safety guard
        y = (x_v-mu)/sigma

        # Define the variables for the highest order derivative and the integration constants
        deriv_values = cvxpy.Variable(len(y)) # values of the order^th derivative, in which we're penalizing variation
        integration_constants = cvxpy.Variable(order) # constants of integration that help get us back to x

        # Recursively integrate the highest order derivative to get back to the position. This is a first-
        # order scheme, but it's very fast and tends to do not markedly worse than 2nd order. See #116
        # I also tried a trapezoidal integration rule here, and it works no better. See #116 too.
        hx = deriv_values # variables are integrated to produce the signal estimate variables, \hat{x} in the math
        for i in range(order):
            hx = cvxpy.cumsum(hx) + integration_constants[i] # cumsum is like integration assuming dt = 1

        # Compare the recursively integrated position to the noisy position. \ell_2 doesn't get scaled by 1/2 here,
        # so cvxpy's doubled Huber is already the right scale, and \ell_1 should be scaled by 2\sqrt{2} to match.
        fidelity_cost = cvxpy.sum_squares(y - hx) if huberM == float('inf') \
                else np.sqrt(8)*cvxpy.norm(y - hx, 1) if huberM == 0 \
                else utility.huber_const(huberM)*cvxpy.sum(cvxpy.huber(y - hx, huberM)) # data is already scaled, so M rather than M*sigma
        # Set up and solve the optimization problem
        prob = cvxpy.Problem(cvxpy.Minimize(fidelity_cost + gamma*cvxpy.sum(cvxpy.tv(deriv_values)) ))
        prob.solve(solver=solver)

        # Recursively integrate the final derivative values to get back to the function and derivative values
        v = deriv_values.value
        for i in range(order-1): # stop one short to get the first derivative
            v = np.cumsum(v) + integration_constants.value[i]
        dxdt_hat_v = v/dt # v only holds the dx values; to get deriv scale by dt
        x_hat_v = np.cumsum(v) + integration_constants.value[order-1] # smoothed data

        # Due to the first-order nature of the derivative, it has a slight lag. Average together every two values
        # to better center the answer. But this leaves us one-short, so devise a good last value.
        dxdt_hat_v = (dxdt_hat_v[:-1] + dxdt_hat_v[1:])/2
        dxdt_hat_v = np.hstack((dxdt_hat_v, 2*dxdt_hat_v[-1] - dxdt_hat_v[-2])) # last value = penultimate value [-1] + diff between [-1] and [-2]

        x_hat[s] = x_hat_v*sigma+mu
        dxdt_hat[s] = dxdt_hat_v*sigma # derivative is linear, so scale derivative by scatter

    return x_hat, dxdt_hat


def smooth_acceleration(x, dt, gamma, window_size, solver=None):
    """Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative,
    and then apply a convolutional gaussian smoother to the resulting derivative to smooth out the peaks.
    The end result is similar to the jerk method, but can be more time-efficient.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param float gamma: the regularization parameter
    :param int window_size: window size for gaussian kernel
    :param str solver: the solver CVXPY should use, 'MOSEK', 'CVXOPT', 'CLARABEL', 'ECOS', etc.
                In testing, 'MOSEK' was the most robust. If not given, fall back to CVXPY's default.

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    _, dxdt_hat = tvrdiff(x, dt, 2, gamma, solver=solver)

    kernel = utility.gaussian_kernel(window_size)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat
