import copy, math, logging, scipy
import numpy as np
from warnings import warn

from pynumdiff import smooth_finite_difference
from pynumdiff.finite_difference import first_order as finite_difference
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass

KERNELS = {'friedrichs': utility._friedrichs_kernel,
           'gaussian': utility._gaussian_kernel}

####################
# Helper functions #
####################
def _slide_function(func, x, dt, args, window_size, step_size, kernel_name):
    """Slide a smoothing derivative function across a timeseries with specified window size.

    :param callable func: name of the function to slide
    :param np.array[float] x: data to differentiate
    :param float dt: time step
    :param dict args: see func for requirements
    :param int window_size: size of the sliding window
    :param int step_size: step size for slide (e.g. 1 means slide by 1 step)
    :param str kernel_name: name of the smoothing kernel (e.g. 'friedrichs' or 'gaussian')

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """

    # get smoothing kernel
    if not window_size % 2: # then make odd
        window_size += 1
    ker = KERNELS[kernel_name](window_size)

    x_hat_list = []
    dxdt_hat_list = []
    weight_list = []

    for p in range(0, len(x), step_size):
        # deal with end points
        start = p - int((window_size-1)/2)
        end = p + int((window_size-1)/2)+1

        ker_start = 0
        ker_end = window_size

        if start < 0:
            ker_start = np.abs(start)
            start = 0
        if end > len(x):
            ker_end = window_size - (end-len(x))
            end = len(x)

        # weights
        w = ker[ker_start:ker_end]
        w = w/np.sum(w)

        # run the function on the window
        _x = x[start:end]
        x_hat, dxdt_hat = func(_x, dt, *args, weights=w)

        # stack results
        z_x_hat = np.zeros([len(x)])
        z_x_hat[start:end] = x_hat
        x_hat_list.append(z_x_hat)

        z_dxdt_hat = np.zeros([len(x)])
        z_dxdt_hat[start:end] = dxdt_hat
        dxdt_hat_list.append(z_dxdt_hat)

        z_weights = np.zeros([len(x)])
        z_weights[start:end] = w
        weight_list.append(z_weights)

    # column norm weights
    weights = np.vstack(weight_list)
    for col in range(weights.shape[1]):
        weights[:, col] = weights[:, col] / np.sum(weights[:, col])

    # stack and weight x_hat and dxdt_hat
    x_hat = np.vstack(x_hat_list)
    dxdt_hat = np.vstack(dxdt_hat_list)

    x_hat = np.sum(weights*x_hat, axis=0)
    dxdt_hat = np.sum(weights*dxdt_hat, axis=0)

    return x_hat, dxdt_hat


#########################
# Savitzky-Golay filter #
#########################
def savgoldiff(x, dt, params=None, options=None, polynomial_order=None, window_size=None, smoothing_win=None):
    """Use the Savitzky-Golay to smooth the data and calculate the first derivative. It wses scipy.signal.savgol_filter. The Savitzky-Golay is very similar to the sliding polynomial fit, but slightly noisier, and much faster

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list params: (**deprecated**, prefer :code:`polynomial_order`, :code:`window_size`, and :code:`smoothing_win`)
    :param dict options: (**deprecated**)
    :param int polynomial_order: order of the polynomial
    :param int window_size: size of the sliding window, must be odd (if not, 1 is added)
    :param int smoothing_win: size of the window used for gaussian smoothing, a good default is window_size, but smaller for high frequnecy data

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `polynomial_order`, " +
            "`window_size`, and `smoothing_win` instead.", DeprecationWarning)
        polynomial_order, window_size, smoothing_win = params
    elif polynomial_order == None or window_size == None or smoothing_win == None:
        raise ValueError("`polynomial_order`, `window_size`, and `smoothing_win` must be given.")

    if window_size > len(x)-1:
        window_size = len(x)-1

    if smoothing_win > len(x)-1:
        smoothing_win = len(x)-1

    if window_size <= polynomial_order:
        window_size = polynomial_order + 1

    if not window_size % 2:  # then make odd
        window_size += 1

    dxdt_hat = scipy.signal.savgol_filter(x, window_size, polynomial_order, deriv=1) / dt

    kernel = utility._gaussian_kernel(smoothing_win)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


######################
# Polynomial fitting #
######################
def _polydiff(x, dt, polynomial_order, weights=None):
    """Fit polynomials to the timeseries, and differentiate the polynomials.

    :param np.array[float] x: data to differentiate
    :param float dt: time step
    :param int polynomial_order: order of the polynomial
    :param np.array[float] weights: weights applied to each point in calculating the polynomial fit.
        Defaults to 1s if missing.

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if weights is None:
        weights = np.ones(x.shape)

    t = np.arange(1, len(x)+1)*dt

    r = np.polyfit(t, x, polynomial_order, w=weights) # polyfit returns highest order first
    dr = np.polyder(r) # power rule already implemented for us

    dxdt_hat = np.polyval(dr, t) # evaluate the derivative and original polynomials at points t
    x_hat = np.polyval(r, t) # smoothed x

    return x_hat, dxdt_hat


def polydiff(x, dt, params=None, options=None, polynomial_order=None, window_size=None,
    sliding=True, step_size=1, kernel='friedrichs'):
    """Fit polynomials to the data, and differentiate the polynomials.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`polynomial_order` and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`sliding`, :code:`step_size`, and :code:`kernel`)
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str)}
    :param int polynomial_order: order of the polynomial
    :param int window_size: size of the sliding window (ignored if not sliding)
    :param bool sliding: whether to use sliding approach
    :param int step_size: step size for sliding
    :param str kernel: kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `polynomial_order` " +
            "and `window_size` instead.", DeprecationWarning)
        polynomial_order, window_size = params
        if options != None:
            if 'sliding' in options: sliding = options['sliding']
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
    elif polynomial_order == None or window_size == None:
        raise ValueError("`polynomial_order` and `window_size` must be given.")

    if window_size < polynomial_order*3:
        window_size = polynomial_order*3+1

    if sliding:
        return _slide_function(_polydiff, x, dt, [polynomial_order], window_size, step_size, kernel)

    return _polydiff(x, dt, polynomial_order)


#############
# Chebychev #
# Removed - Not Useful and requires old package
#############


# def __chebydiff__(x, dt, params, options=None):
#     """
#     Fit the timeseries with chebyshev polynomials, and differentiate this model.

#     :param x: (np.array of floats, 1xN) data to differentiate
#     :param dt: (float) time step
#     :param params: (list) [N] : (int) order of the polynomial
#     :param options:
#     :return: x_hat    : estimated (smoothed) x
#              dxdt_hat : estimated derivative of x
#     """

#     if isinstance(params, list):
#         n = params[0]
#     else:
#         n = params

#     mean = np.mean(x)
#     x = x - mean

#     def f(y):
#         t = np.linspace(-1, 1, len(x))
#         return np.interp(y, t, x)

#     # Chebychev polynomial
#     poly = pychebfun.chebfun(f, N=n, domain=[-1, 1])
#     ts = np.linspace(poly.domain()[0], poly.domain()[-1], len(x))

#     x_hat = poly(ts) + mean
#     dxdt_hat = poly.differentiate()(ts)*(2/len(x))/dt

#     return x_hat, dxdt_hat


# def chebydiff(x, dt, params, options=None):
#     """
#     Slide a smoothing derivative function across a times eries with specified window size.

#     :param x: data to differentiate
#     :type x: np.array (float)

#     :param dt: step size
#     :type dt: float

#     :param params: a list of 2 elements:

#                     - N: order of the polynomial
#                     - window_size: size of the sliding window (ignored if not sliding)

#     :type params: list (int)

#     :param options: a dictionary consisting of 3 key value pairs:

#                     - 'sliding': whether to use sliding approach
#                     - 'step_size': step size for sliding
#                     - 'kernel_name': kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

#     :type options: dict {'sliding': (bool), 'step_size': (int), 'kernel_name': (string)}, optional

#     :return: a tuple consisting of:

#             - x_hat: estimated (smoothed) x
#             - dxdt_hat: estimated derivative of x

#     :rtype: tuple -> (np.array, np.array)
#     """

#     if options is None:
#         options = {'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'}

#     if 'sliding' in options.keys() and options['sliding']:
#         window_size = copy.copy(params[-1])
#         if window_size < params[0]*2:
#             window_size = params[0]*2+1
#             params[1] = window_size
#         return __slide_function__(__chebydiff__, x, dt, params, window_size,
#                                   options['step_size'], options['kernel_name'])

#     return __chebydiff__(x, dt, params)


def __solve_for_A_and_C_given_X_and_Xdot__(X, Xdot, num_integrations, dt, gammaC=1e-1, gammaA=1e-6,
                                           solver=None, A_known=None, epsilon=1e-6, rows_of_interest='all'):
    """Given state and the derivative, find the system evolution and measurement matrices.
    """

    if rows_of_interest == 'all':
        rows_of_interest = np.arange(0, X.shape[0])

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
    error = cvxpy.sum_squares(Xdot[rows_of_interest, :] - ( cvxpy.matmul(A, X) + Csum)[rows_of_interest, :])
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

    A = np.array(A.value)
    return A, np.array(C.value)


def __integrate_dxdt_hat_matrix__(dxdt_hat, dt):
    """Do integration analogous to integrate_dxdt_hat in the utilities, except on a 2D matrix.
    """
    if len(dxdt_hat.shape) == 1:
        dxdt_hat = np.reshape(dxdt_hat, [1, len(dxdt_hat)])
    x = np.array(scipy.integrate.cumulative_trapezoid(dxdt_hat, axis=1))
    first_value = x[:, 0:1] - np.mean(dxdt_hat[:, 0:1], axis=1).reshape(dxdt_hat.shape[0], 1)
    x = np.hstack((first_value, x))*dt
    return x


def _lineardiff(x, dt, N, gamma, solver=None, weights=None):
    """Estimate the parameters for a system xdot = Ax, and use that to calculate the derivative

    :param np.array[float] x: data to differentiate
    :param float dt: time step
    :param int > 1 N: order (e.g. 2: velocity; 3: acceleration)
    :param float gamma: regularization term
    :param np.array[float] weights: Bug? Currently not used here, although `lineardiff` takes a kernel
    :param str solver: which cvxpy solver to use

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    mean = np.mean(x)
    x = x - mean

    # Generate the matrix of integrals of x
    X = [x]
    for n in range(1, N):
        X.append(utility.integrate_dxdt_hat(X[-1], dt))
    X = (np.vstack(X[::-1]))
    integral_Xdot = X
    integral_X = __integrate_dxdt_hat_matrix__(X, dt)

    # Solve for A and the integration constants
    A, C = __solve_for_A_and_C_given_X_and_Xdot__(integral_X, integral_Xdot, N, dt, gamma, solver=solver)

    # Add the integration constants
    Csum = 0
    t = np.arange(0, X.shape[1])*dt
    for n in range(0, N - 1):
        C_subscript = n
        t_exponent = N - n - 2
        den = math.factorial(t_exponent)
        Cn = np.vstack([1/den*C[i, C_subscript]*t**t_exponent for i in range(X.shape[0])])
        Csum = Csum + Cn
    Csum = np.array(Csum)

    # Use A and C to calculate the derivative
    Xdot_reconstructed = (A@X + Csum)
    dxdt_hat = np.ravel(Xdot_reconstructed[-1, :])

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x_hat = x_hat + utility.estimate_initial_condition(x+mean, x_hat)

    return x_hat, dxdt_hat


def lineardiff(x, dt, params=None, options=None, order=None, gamma=None, window_size=None,
    sliding=True, step_size=10, kernel='friedrichs', solver=None):
    """Slide a smoothing derivative function across data, with specified window size.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int, float, int] params: (**deprecated**, prefer :code:`order`, :code:`gamma`, and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`sliding`, :code:`step_size`, :code:`kernel`, and :code:`solver`
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str), 'solver': (str)}
    :param int>1 order: order of the polynomial
    :param float gamma: regularization term
    :param int window_size: size of the sliding window (ignored if not sliding)
    :param bool sliding: whether to use sliding approach
    :param int step_size: step size for sliding
    :param str kernel_name: kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param str solver: CVXPY solver to use, one of :code:`cvxpy.installed_solvers()`

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `order`, " +
            "`gamma`, and `window_size` instead.", DeprecationWarning)
        order, gamma, window_size = params
        if options != None:
            if 'sliding' in options: sliding = options['sliding']
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
            if 'solver' in options: solver = options['solver']
    elif order == None or gamma == None or window_size == None:
        raise ValueError("`order`, `gamma`, and `window_size` must be given.")

    if sliding:
        # forward and backward
        x_hat_forward, _ = _slide_function(_lineardiff, x, dt, [order, gamma, solver], window_size, step_size,
                                              kernel)
        x_hat_backward, _ = _slide_function(_lineardiff, x[::-1], dt, [order, gamma, solver], window_size, step_size,
                                               kernel)

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

    return _lineardiff(x, dt, params, options)

###############################
# Fourier Spectral derivative #
###############################
def spectraldiff(x, dt, params=None, options=None, high_freq_cutoff=None, even_extension=True, pad_to_zero_dxdt=True):
    """Take a derivative in the fourier domain, with high frequency attentuation.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[float] or float params: (**deprecated**, prefer :code:`high_freq_cutoff`)
    :param dict options: (**deprecated**, prefer :code:`even_extension`
            and :code:`pad_to_zero_dxdt`) a dictionary consisting of {'even_extension': (bool), 'pad_to_zero_dxdt': (bool)}
    :param float high_freq_cutoff: The high frequency cutoff. Frequencies below this threshold will be kept, and above will be zeroed.
    :param bool even_extension: if True, extend the data with an even extension so signal starts and ends at the same value.
    :param bool pad_to_zero_dxdt: if True, extend the data with extensions that smoothly force the derivative to zero. This
            allows the spectral derivative to fit data which does not start and end with derivatives equal to zero.

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

    original_L = len(x)

    # make derivative go to zero at ends (optional)
    if pad_to_zero_dxdt:
        padding = 100
        pre = x[0]*np.ones(padding)
        post = x[-1]*np.ones(padding)
        x = np.hstack((pre, x, post))
        x_hat, _ = smooth_finite_difference.meandiff(x, dt, [int(padding/2)], options={'iterate': False})
        x_hat[padding:-padding] = x[padding:-padding]
        x = x_hat
    else:
        padding = 0

    # Do even extension (optional)
    if even_extension is True:
        x = np.hstack((x, x[::-1]))

    # If odd, make N even, and pad x
    L = len(x)
    if L % 2 != 0:
        N = L + 1
        x = np.hstack((x, x[-1] + dt*(x[-1]-x[-1])))
    else:
        N = L

    # Define the frequency range.
    k = np.asarray(list(range(0, int(N/2))) + [0] + list(range(int(-N/2) + 1,0)))
    k = k*2*np.pi/(dt*N)

    # Frequency based smoothing: remove signals with a frequency higher than high_freq_cutoff
    discrete_high_freq_cutoff = int(high_freq_cutoff*N)
    k[discrete_high_freq_cutoff:N-discrete_high_freq_cutoff] = 0

    # Derivative = 90 deg phase shift
    dxdt_hat = np.real(np.fft.ifft(1.0j * k * np.fft.fft(x)))
    dxdt_hat = dxdt_hat[padding:original_L+padding]

    # Integrate to get x_hat
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x[padding:original_L+padding], x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat
