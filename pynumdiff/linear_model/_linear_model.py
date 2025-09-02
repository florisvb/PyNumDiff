import math, scipy
import numpy as np
from warnings import warn

from pynumdiff.finite_difference import second_order as finite_difference
from pynumdiff.polynomial_fit import savgoldiff as _savgoldiff # patch through
from pynumdiff.polynomial_fit import polydiff as _polydiff # patch through
from pynumdiff.basis_fit import spectraldiff as _spectraldiff # patch through
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


def savgoldiff(*args, **kwargs):
    warn("`savgoldiff` has moved to `polynomial_fit.savgoldiff` and will be removed from "
        + "`linear_model` in a future release.", DeprecationWarning)
    return _savgoldiff(*args, **kwargs)

def polydiff(*args, **kwargs):
    warn("`polydiff` has moved to `polynomial_fit.polydiff` and will be removed from "
        + "`linear_model` in a future release.", DeprecationWarning)
    return _polydiff(*args, **kwargs)

def spectraldiff(*args, **kwargs):
    warn("`spectraldiff` has moved to `basis_fit.spectraldiff` and will be removed from "
        + "`linear_model` in a future release.", DeprecationWarning)
    return _spectraldiff(*args, **kwargs)


def _solve_for_A_and_C_given_X_and_Xdot(X, Xdot, num_integrations, dt, gammaC=1e-1, gammaA=1e-6,
                                           solver=None, A_known=None, epsilon=1e-6):
    """Given state and the derivative, find the system evolution and measurement matrices."""
    # Set up the variables
    A = cvxpy.Variable((X.shape[0], X.shape[0]))
    C = cvxpy.Variable((X.shape[0], num_integrations))

    # Integrate the integration constants
    Csum = 0
    t = np.arange(0, X.shape[1])*dt
    for n in range(num_integrations):
        C_subscript = n
        t_exponent = num_integrations - n -1
        den = math.factorial(t_exponent)
        Cn = cvxpy.vstack((1/den*C[i, C_subscript]*t**t_exponent for i in range(X.shape[0])))
        Csum = Csum + Cn

    # Define the objective function
    error = cvxpy.sum_squares(Xdot - ( cvxpy.matmul(A, X) + Csum))
    C_regularization = gammaC*cvxpy.sum(cvxpy.abs(C))
    A_regularization = gammaA*cvxpy.sum(cvxpy.abs(A))
    obj = cvxpy.Minimize(error + C_regularization + A_regularization)

    # constraints
    constraints = []
    if A_known is not None:
        for i in range(A_known.shape[0]):
            for j in range(A_known.shape[1]):
                if not np.isnan(A_known[i, j]):
                    constraint_lo = A[i, j] >= A_known[i, j]-epsilon
                    constraint_hi = A[i, j] <= A_known[i, j]+epsilon
                    constraints.extend([constraint_lo, constraint_hi])

    # Solve the problem
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver=solver)

    return np.array(A.value), np.array(C.value)

def lineardiff(x, dt, params=None, options=None, order=None, gamma=None, window_size=None,
    step_size=10, kernel='friedrichs', solver=None):
    """Slide a smoothing derivative function across data, with specified window size.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int, float, int] params: (**deprecated**, prefer :code:`order`, :code:`gamma`, and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`sliding`, :code:`step_size`, :code:`kernel`, and :code:`solver`
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str), 'solver': (str)}
    :param int>1 order: order of the polynomial
    :param float gamma: regularization term
    :param int window_size: size of the sliding window (ignored if not sliding)
    :param int step_size: step size for sliding
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param str solver: CVXPY solver to use, one of :code:`cvxpy.installed_solvers()`

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `order`, " +
            "`gamma`, and `window_size` instead.", DeprecationWarning)
        order, gamma = params[:2]
        if len(params) > 2: window_size = params[2]        
        if options != None:
            if 'sliding' in options and not options['sliding']: window_size = None
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
            if 'solver' in options: solver = options['solver']
    elif order == None or gamma == None or window_size == None:
        raise ValueError("`order`, `gamma`, and `window_size` must be given.")

    def _lineardiff(x, dt, order, gamma, solver=None):
        """Estimate the parameters for a system xdot = Ax, and use that to calculate the derivative"""
        mean = np.mean(x)
        x = x - mean

        # Generate the matrix of integrals of x
        X = [x]
        for n in range(1, order):
            X.append(utility.integrate_dxdt_hat(X[-1], dt))
        X = np.vstack(X[::-1])
        integral_Xdot = X
        integral_X = np.hstack((np.zeros((X.shape[0], 1)), scipy.integrate.cumulative_trapezoid(X, axis=1)))*dt

        # Solve for A and the integration constants
        A, C = _solve_for_A_and_C_given_X_and_Xdot(integral_X, integral_Xdot, order, dt, gamma, solver=solver)

        # Add the integration constants
        Csum = 0
        t = np.arange(0, X.shape[1])*dt
        for n in range(0, order - 1):
            C_subscript = n
            t_exponent = order - n - 2
            den = math.factorial(t_exponent)
            Cn = np.vstack([1/den*C[i, C_subscript]*t**t_exponent for i in range(X.shape[0])])
            Csum = Csum + Cn
        Csum = np.array(Csum)

        # Use A and C to calculate the derivative
        Xdot_reconstructed = (A@X + Csum)
        dxdt_hat = np.ravel(Xdot_reconstructed[-1, :])

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        x_hat = x_hat + utility.estimate_integration_constant(x+mean, x_hat)

        return x_hat, dxdt_hat

    if not window_size:
        return _lineardiff(x, dt, order, gamma, solver)
    elif window_size % 2 == 0:
        window_size += 1
        warn("Kernel window size should be odd. Added 1 to length.")

    kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)

    x_hat_forward, _ = utility.slide_function(_lineardiff, x, dt, kernel, order, gamma, stride=step_size, solver=solver)
    x_hat_backward, _ = utility.slide_function(_lineardiff, x[::-1], dt, kernel, order, gamma, stride=step_size, solver=solver)

    # weights
    w = np.arange(1, len(x_hat_forward)+1,1)[::-1]
    w = np.pad(w, [0, len(x)-len(w)], mode='constant')
    wfb = np.vstack((w, w[::-1]))
    norm = np.sum(wfb, axis=0)

    # orient and pad
    x_hat_forward = np.pad(x_hat_forward, [0, len(x)-len(x_hat_forward)], mode='constant')
    x_hat_backward = np.pad(x_hat_backward[::-1], [len(x)-len(x_hat_backward), 0], mode='constant')

    # merge
    x_hat = x_hat_forward*w/norm + x_hat_backward*w[::-1]/norm
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat
