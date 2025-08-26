import math, scipy
import numpy as np
from scipy import sparse
from warnings import warn

from pynumdiff.finite_difference import second_order as finite_difference
from pynumdiff.polynomial_fit import savgoldiff as _savgoldiff # patch through
from pynumdiff.polynomial_fit import polydiff as _polydiff # patch through
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


def spectraldiff(x, dt, params=None, options=None, high_freq_cutoff=None, even_extension=True, pad_to_zero_dxdt=True):
    """Take a derivative in the fourier domain, with high frequency attentuation.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[float] or float params: (**deprecated**, prefer :code:`high_freq_cutoff`)
    :param dict options: (**deprecated**, prefer :code:`even_extension`
            and :code:`pad_to_zero_dxdt`) a dictionary consisting of {'even_extension': (bool), 'pad_to_zero_dxdt': (bool)}
    :param float high_freq_cutoff: The high frequency cutoff as a multiple of the Nyquist frequency: Should be between 0
            and 1. Frequencies below this threshold will be kept, and above will be zeroed.
    :param bool even_extension: if True, extend the data with an even extension so signal starts and ends at the same value.
    :param bool pad_to_zero_dxdt: if True, extend the data with extra regions that smoothly force the derivative to
            zero before taking FFT.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `high_freq_cutoff`, " +
            "`even_extension`, and `pad_to_zero_dxdt` instead.", DeprecationWarning)
        high_freq_cutoff = params[0] if isinstance(params, list) else params
        if options != None:
            if 'even_extension' in options: even_extension = options['even_extension']
            if 'pad_to_zero_dxdt' in options: pad_to_zero_dxdt = options['pad_to_zero_dxdt']
    elif high_freq_cutoff == None:
        raise ValueError("`high_freq_cutoff` must be given.")

    L = len(x)

    # make derivative go to zero at ends (optional)
    if pad_to_zero_dxdt:
        padding = 100
        pre = x[0]*np.ones(padding) # extend the edges
        post = x[-1]*np.ones(padding)
        x = np.hstack((pre, x, post))
        kernel = utility.mean_kernel(padding//2)
        x_hat = utility.convolutional_smoother(x, kernel) # smooth the edges in
        x_hat[padding:-padding] = x[padding:-padding] # replace middle with original signal
        x = x_hat
    else:
        padding = 0

    # Do even extension (optional)
    if even_extension is True:
        x = np.hstack((x, x[::-1]))

    # If odd, make N even, and pad x
    N = len(x)

    # Define the frequency range.
    k = np.concatenate((np.arange(N//2 + 1), np.arange(-N//2 + 1, 0)))
    if N % 2 == 0: k[N//2] = 0 # odd derivatives get the Nyquist element zeroed out
    omega = k*2*np.pi/(dt*N) # turn wavenumbers into frequencies in radians/s

    # Frequency based smoothing: remove signals with a frequency higher than high_freq_cutoff
    discrete_cutoff = int(high_freq_cutoff*N/2) # Nyquist is at N/2 location, and we're cutting off as a fraction of that
    omega[discrete_cutoff:N-discrete_cutoff] = 0

    # Derivative = 90 deg phase shift
    dxdt_hat = np.real(np.fft.ifft(1.0j * omega * np.fft.fft(x)))
    dxdt_hat = dxdt_hat[padding:L+padding]

    # Integrate to get x_hat
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x[padding:L+padding], x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


def rbfdiff(x, _t, sigma=1, lmbd=0.01):
    """Find smoothed function and derivative estimates by fitting noisy data with radial-basis-functions. Naively,
    fill a matrix with basis function samples, similar to the implicit inverse problem of spectral methods, but
    truncate tiny values to make columns sparse. Each basis function "hill" is topped with a "tower" of height
    :code:`lmbd` to reach noisy data samples, and the final smoothed reconstruction is found by razing these and only
    keeping the hills.

    :param np.array[float] x: data to differentiate
    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param float sigma: controls width of radial basis functions
    :param float lmbd: controls smoothness

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if np.isscalar(_t):
        t = np.arange(len(x))*_t
    else: # support variable step size for this function
        if len(x) != len(_t): raise ValueError("If `_t` is given as array-like, must have same length as `x`.")
        t = _t

    # The below does the approximate equivalent of this code, but sparsely in O(N sigma^2), since the rbf falls off rapidly
    # t_i, t_j = np.meshgrid(t,t)
    # r = t_j - t_i # radius
    # rbf = np.exp(-(r**2) / (2 * sigma**2)) # radial basis function kernel, O(N^2) entries
    # drbfdt = -(r / sigma**2) * rbf # derivative of kernel
    # rbf_regularized = rbf + lmbd*np.eye(len(t))
    # alpha = np.linalg.solve(rbf_regularized, x) # O(N^3)

    cutoff = np.sqrt(-2 * sigma**2 * np.log(1e-4))
    rows, cols, vals, dvals = [], [], [], []
    for n in range(len(t)):
        # Only consider points within a cutoff. Gaussian drops below eps at distance ~ sqrt(-2*sigma^2 log eps)
        l = np.searchsorted(t, t[n] - cutoff) # O(log N) to find indices of points within cutoff
        r = np.searchsorted(t, t[n] + cutoff) # finds index where new value should be inserted
        for j in range(l, r): # width of this is dependent on sigma. [l, r) is correct inclusion/exclusion
            radius = t[n] - t[j]
            v = np.exp(-radius**2 / (2 * sigma**2))
            dv = -radius / sigma**2 * v
            rows.append(n); cols.append(j); vals.append(v); dvals.append(dv)

    rbf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(t), len(t))) # Build sparse kernels, O(N sigma) entries
    drbfdt = sparse.csr_matrix((dvals, (rows, cols)), shape=(len(t), len(t)))
    rbf_regularized = rbf + lmbd*sparse.eye(len(t), format="csr") # identity matrix gives a little extra height at the centers
    alpha = sparse.linalg.spsolve(rbf_regularized, x) # solve sparse system targeting the noisy data, O(N sigma^2)

    return rbf @ alpha, drbfdt @ alpha # find samples of reconstructions using the smooth bases
