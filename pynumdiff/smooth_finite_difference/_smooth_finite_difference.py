import numpy as np
import scipy.signal
from warnings import warn

# included code
from pynumdiff.finite_difference import first_order as finite_difference
from pynumdiff.utils import utility


################################
# Smoothing finite differences #
################################
def mediandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform median smoothing using scipy.signal.medfilt followed by first order finite difference

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size
    :param list[int] params: (**deprecated**, prefer :code:`window_size` and :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int window_size: filter window size
    :param int num_iterations: how many times to apply median smoothing


    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("""`params` and `options` parameters will be removed in a future version. Use `window_size`
            and `num_iterations` instead.""", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    if not window_size % 2:
        window_size += 1 # make sure window_size is odd

    x_hat = x
    for _ in range(num_iterations):
        x_hat = scipy.signal.medfilt(x_hat, window_size)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def meandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform mean smoothing by convolving mean kernel with x followed by first order finite difference

    :param np.ndarray[float] x: array of time series to differentiate
    :param float dt: time step size

    :param list[int] params: (**deprecated**, prefer :code:`window_size` and :code:`num_iterations`)
        :code:`[window_size]` or, :code:`if 'iterate' in options`, :code:`[window_size, num_iterations]`
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int window_size: filter window size
    :param int num_iterations: how many times to apply mean smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("""`params` and `options` parameters will be removed in a future version. Use `window_size`
            and `num_iterations` instead.""", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    kernel = utility._mean_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def gaussiandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform gaussian smoothing by convolving gaussian kernel with x followed by first order finite difference

    :param np.array[float] x: array of time series to differentiate
    :param float dt: time step size
    :param list[int] params: (**deprecated**, prefer :code:`window_size` and :code:`num_iterations`)
        :code:`[window_size]` or, :code:`if 'iterate' in options`, :code:`[window_size, num_iterations]`
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int window_size: filter window size
    :param int num_iterations: how many times to apply gaussian smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("""`params` and `options` parameters will be removed in a future version. Use `window_size`
            and `num_iterations` instead.""", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    kernel = utility._gaussian_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def friedrichsdiff(x, dt, params, options={}):
    """
    Perform friedrichs smoothing by convolving friedrichs kernel with x
    followed by first order finite difference

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [filter_window_size] or if 'iterate' in options:
                    [filter_window_size, num_iterations]

    :type params: list (int)

    :param options: an empty dictionary or a dictionary with 1 key value pair

                    - 'iterate': whether to run multiple iterations of the smoother. Note: iterate does nothing for median smoother.

    :type options: dict {'iterate': (boolean)}

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """

    if 'iterate' in options.keys() and options['iterate'] is True:
        window_size, iterations = params
    else:
        iterations = 1
        if isinstance(params, list):
            window_size = params[0]
        else:
            window_size = params

    kernel = utility._friedrichs_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def butterdiff(x, dt, params, options={}):
    """
    Perform butterworth smoothing on x with scipy.signal.filtfilt
    followed by first order finite difference

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [n, wn], n  = order of the filter; wn = Cutoff frequency.
                    For a discrete timeseries, the value is normalized to the range 0-1,
                    where 1 is the Nyquist frequency.

    :type params: list (int)

    :param options: an empty dictionary or a dictionary with 2 key value pair

                    - 'iterate': whether to run multiple iterations of the smoother. Note: iterate does nothing for median smoother.
                    - 'padmethod': "pad" or "gust", see scipy.signal.filtfilt

    :type options: dict {'iterate': (boolean), 'padmethod': string}

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if 'iterate' in options.keys() and options['iterate'] is True:
        n, wn, iterations = params
    else:
        iterations = 1
        n, wn = params

    b, a = scipy.signal.butter(n, wn)

    x_hat = x
    for _ in range(iterations):
        if len(x) < 9:
            x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad", padlen=len(x)-1)
        else:
            x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad")

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    offset = np.mean(x) - np.mean(x_hat)
    x_hat = x_hat + offset

    return x_hat, dxdt_hat


def splinediff(x, dt, params, options={}):
    """
    Perform spline smoothing on x with scipy.interpolate.UnivariateSpline
    followed by first order finite difference

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: [k, s], k: Order of the spline. A kth order spline can be differentiated k times.
                    s: Positive smoothing factor used to choose the number of knots.
                    Number of knots will be increased until the smoothing condition is satisfied:
                    sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

    :type params: list (int)

    :param options: an empty dictionary or a dictionary with 1 key value pair

                    - 'iterate': whether to run multiple iterations of the smoother. Note: iterate does nothing for median smoother.

    :type options: dict {'iterate': (boolean)}

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if 'iterate' in options.keys() and options['iterate'] is True:
        k, s, iterations = params
    else:
        iterations = 1
        k, s = params

    t = np.arange(0, len(x)*dt, dt)

    x_hat = x
    for _ in range(iterations):
        spline = scipy.interpolate.UnivariateSpline(t, x_hat, k=k, s=s)
        x_hat = spline(t)

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat
