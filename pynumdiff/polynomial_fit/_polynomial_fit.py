import numpy as np
import scipy
from warnings import warn

from pynumdiff.utils import utility


def splinediff(x, dt_or_t, params=None, options=None, degree=3, s=None, num_iterations=1):
    """Find smoothed data and derivative estimates by fitting a smoothing spline to the data with
    scipy.interpolate.UnivariateSpline. Variable step size is supported with equal ease as uniform step size.

    :param np.array[float] x: data to differentiate
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param list params: (**deprecated**, prefer :code:`degree`, :code:`cutoff_freq`, and :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) a dictionary of {'iterate': (bool)}
    :param int degree: polynomial degree of the spline. A kth degree spline can be differentiated k times.
    :param float s: positive smoothing factor used to choose the number of knots. Number of knots will be increased
        until the smoothing condition is satisfied: :math:`\\sum_t (x[t] - \\text{spline}[t])^2 \\leq s`
    :param int num_iterations: how many times to apply smoothing

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `order`, `s`, and " +
            "`num_iterations` instead.", DeprecationWarning)
        degree, s = params[0:2]
        if options != None:
            if 'iterate' in options and options['iterate']: num_iterations = params[2]

    if np.isscalar(dt_or_t):
        t = np.arange(len(x))*dt_or_t
    else: # support variable step size for this function
        if len(x) != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        t = dt_or_t

    x_hat = x
    for _ in range(num_iterations):
        spline = scipy.interpolate.UnivariateSpline(t, x_hat, k=degree, s=s)
        x_hat = spline(t)

    dspline = spline.derivative()
    dxdt_hat = dspline(t)

    return x_hat, dxdt_hat


def polydiff(x, dt, params=None, options=None, degree=None, window_size=None, step_size=1,
    kernel='friedrichs'):
    """Fit polynomials to the data, and differentiate the polynomials.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`degree` and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`step_size` and :code:`kernel`)
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str)}
    :param int degree: degree of the polynomial
    :param int window_size: size of the sliding window, if not given no sliding
    :param int step_size: step size for sliding
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `degree` " +
            "and `window_size` instead.", DeprecationWarning)
        degree = params[0]
        if len(params) > 1: window_size = params[1]
        if options != None:
            if 'sliding' in options and not options['sliding']: window_size = None
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
    elif degree == None or window_size == None:
        raise ValueError("`degree` and `window_size` must be given.")

    if window_size < degree*3:
        window_size = degree*3+1
    if window_size % 2 == 0:
        window_size += 1
        warn("Kernel window size should be odd. Added 1 to length.")

    def _polydiff(x, dt, degree, weights=None):
        t = np.arange(len(x))*dt

        r = np.polyfit(t, x, degree, w=weights) # polyfit returns highest order first
        dr = np.polyder(r) # power rule already implemented for us

        dxdt_hat = np.polyval(dr, t) # evaluate the derivative and original polynomials at points t
        x_hat = np.polyval(r, t) # smoothed x

        return x_hat, dxdt_hat

    if not window_size:
        return _polydiff(x, dt, degree)

    kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)
    return utility.slide_function(_polydiff, x, dt, kernel, degree, stride=step_size, pass_weights=True)


def savgoldiff(x, dt, params=None, options=None, degree=None, window_size=None, smoothing_win=None):
    """Use the Savitzky-Golay to smooth the data and calculate the first derivative. It uses
    scipy.signal.savgol_filter. The Savitzky-Golay is very similar to the sliding polynomial fit,
    but slightly noisier, and much faster.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list params: (**deprecated**, prefer :code:`degree`, :code:`window_size`, and :code:`smoothing_win`)
    :param dict options: (**deprecated**)
    :param int degree: degree of the polynomial
    :param int window_size: size of the sliding window, must be odd (if not, 1 is added)
    :param int smoothing_win: size of the window used for gaussian smoothing, a good default is
        window_size, but smaller for high frequnecy data

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `degree`, " +
            "`window_size`, and `smoothing_win` instead.", DeprecationWarning)
        degree, window_size, smoothing_win = params
    elif degree == None or window_size == None or smoothing_win == None:
        raise ValueError("`degree`, `window_size`, and `smoothing_win` must be given.")

    window_size = np.clip(window_size, degree + 1, len(x)-1)
    if window_size % 2 == 0:
        window_size += 1 # window_size needs to be odd
        warn("Kernel window size should be odd. Added 1 to length.")
    smoothing_win = min(smoothing_win, len(x)-1)

    dxdt_hat = scipy.signal.savgol_filter(x, window_size, degree, deriv=1)/dt

    kernel = utility.gaussian_kernel(smoothing_win)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x_hat += utility.estimate_integration_constant(x, x_hat)

    return x_hat, dxdt_hat
