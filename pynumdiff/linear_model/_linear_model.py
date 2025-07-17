import copy, math, logging, scipy
import numpy as np
from warnings import warn

from pynumdiff import smooth_finite_difference
from pynumdiff.finite_difference import first_order as finite_difference
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


#########################
# Savitzky-Golay filter #
#########################
def savgoldiff(x, dt, params=None, options=None, poly_order=None, window_size=None, smoothing_win=None):
    """Use the Savitzky-Golay to smooth the data and calculate the first derivative. It uses
    scipy.signal.savgol_filter. The Savitzky-Golay is very similar to the sliding polynomial fit,
    but slightly noisier, and much faster.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list params: (**deprecated**, prefer :code:`poly_order`, :code:`window_size`, and :code:`smoothing_win`)
    :param dict options: (**deprecated**)
    :param int poly_order: order of the polynomial
    :param int window_size: size of the sliding window, must be odd (if not, 1 is added)
    :param int smoothing_win: size of the window used for gaussian smoothing, a good default is
        window_size, but smaller for high frequnecy data

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `poly_order`, " +
            "`window_size`, and `smoothing_win` instead.", DeprecationWarning)
        poly_order, window_size, smoothing_win = params
    elif poly_order == None or window_size == None or smoothing_win == None:
        raise ValueError("`poly_order`, `window_size`, and `smoothing_win` must be given.")

    window_size = np.clip(window_size, poly_order + 1, len(x)-1)
    if window_size % 2 == 0: window_size += 1 # window_size needs to be odd
    smoothing_win = min(smoothing_win, len(x)-1)

    dxdt_hat = scipy.signal.savgol_filter(x, window_size, poly_order, deriv=1)/dt

    kernel = utility.gaussian_kernel(smoothing_win)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


######################
# Polynomial fitting #
######################
def polydiff(x, dt, params=None, options=None, poly_order=None, window_size=None, step_size=1,
    kernel='friedrichs'):
    """Fit polynomials to the data, and differentiate the polynomials.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`poly_order` and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`step_size` and :code:`kernel`)
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str)}
    :param int poly_order: order of the polynomial
    :param int window_size: size of the sliding window, if not given no sliding
    :param int step_size: step size for sliding
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `poly_order` " +
            "and `window_size` instead.", DeprecationWarning)
        poly_order = params[0]
        if len(params) > 1: window_size = params[1]
        if options != None:
            if 'sliding' in options and not options['sliding']: window_size = None
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
    elif poly_order == None or window_size == None:
        raise ValueError("`poly_order` and `window_size` must be given.")

    if window_size < poly_order*3:
        window_size = poly_order*3+1
    if window_size % 2 == 0:
        window_size += 1
        warn("Kernel window size should be odd. Added 1 to length.")

    def _polydiff(x, dt, poly_order, weights=None):
        t = np.arange(len(x))*dt

        r = np.polyfit(t, x, poly_order, w=weights) # polyfit returns highest order first
        dr = np.polyder(r) # power rule already implemented for us

        dxdt_hat = np.polyval(dr, t) # evaluate the derivative and original polynomials at points t
        x_hat = np.polyval(r, t) # smoothed x

        return x_hat, dxdt_hat

    if not window_size:
        return _polydiff(x, dt, poly_order)

    kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)
    return utility.slide_function(_polydiff, x, dt, kernel, poly_order, stride=step_size, pass_weights=True)


###############
# Linear diff #
###############
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
        x_hat, _ = smooth_finite_difference.meandiff(x, dt, window_size=int(padding/2))
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
    x0 = utility.estimate_integration_constant(x[padding:original_L+padding], x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat


#############
# Chebychev #
#############
def chebydiff(x, dt, poly_order, window_size=None, step_size=1, kernel='friedrichs'):
    """Fit Chebyshev polynomials to the data, and differentiate those

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param int poly_order: keep polynomials up to this order
    :param int window_size: size of the sliding window, if not given no sliding

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if window_size % 2 == 0:
        window_size += 1
        warn("Kernel window size should be odd. Added 1 to length.")

    def _chebdiff(x, dt, poly_order, weights=None):
        t = np.arange(len(x))*dt

        r = np.polynomial.chebyshev.chebfit(t, x, poly_order, w=weights) # chebfit returns lowest order first
        dr = np.polynomial.chebyshev.chebder(r) # series derivative rule already implemented for us

        dxdt_hat = np.polynomial.chebyshev.chebval(t, dr) # evaluate the derivative and original polynomials at points t
        x_hat = np.polynomial.chebyshev.chebval(t, r) # smoothed x

        return x_hat, dxdt_hat

    if not window_size:
        return _chebdiff(x, dt, poly_order)

    kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)
    return utility.slide_function(_chebdiff, x, dt, kernel, poly_order, stride=step_size, pass_weights=True)
