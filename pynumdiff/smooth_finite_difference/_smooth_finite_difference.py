import numpy as np
import scipy.signal
from warnings import warn

# included code
from pynumdiff.finite_difference import second_order as finite_difference
from pynumdiff.utils import utility


################################
# Smoothing finite differences #
################################
def mediandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform median smoothing using scipy.signal.medfilt followed by first order finite difference

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`window_size` and :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int window_size: filter window size
    :param int num_iterations: how many times to apply median smoothing


    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `window_size` " +
            "and `num_iterations` instead.", DeprecationWarning)
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

    :param np.ndarray[float] x: data to differentiate
    :param float dt: step size

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
        warn("`params` and `options` parameters will be removed in a future version. Use `window_size` " +
            "and `num_iterations` instead.", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    kernel = utility.mean_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def gaussiandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform gaussian smoothing by convolving gaussian kernel with x followed by first order finite difference

    :param np.array[float] x: data to differentiate
    :param float dt: step size
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
        warn("`params` and `options` parameters will be removed in a future version. Use `window_size` " +
            "and `num_iterations` instead.", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    kernel = utility.gaussian_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def friedrichsdiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform friedrichs smoothing by convolving friedrichs kernel with x followed by first order finite difference

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`window_size` and :code:`num_iterations`)
        :code:`[window_size]` or, :code:`if 'iterate' in options`, :code:`[window_size, num_iterations]`
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int window_size: filter window size
    :param int num_iterations: how many times to apply smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `window_size` " +
            "and `num_iterations` instead.", DeprecationWarning)
        window_size = params[0] if isinstance(params, list) else params
        if 'iterate' in options and options['iterate']:
            num_iterations = params[1]

    kernel = utility.friedrichs_kernel(window_size)
    x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def butterdiff(x, dt, params=None, options={}, filter_order=2, cutoff_freq=0.5, num_iterations=1):
    """Perform butterworth smoothing on x with scipy.signal.filtfilt followed by first order finite difference

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list[int] params: (**deprecated**, prefer :code:`filter_order`, :code:`cutoff_freq`,
        and :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int filter_order: order of the filter
    :param float cutoff_freq: cutoff frequency :math:`\\in [0, 1]`. For a discrete vector, the
        value is normalized to the range 0-1, where 1 is the Nyquist frequency.
    :param int num_iterations: how many times to apply smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `filter_order`, " +
            "`cutoff_freq`, and `num_iterations` instead.", DeprecationWarning)
        filter_order, cutoff_freq = params[0:2]
        if 'iterate' in options and options['iterate']:
            num_iterations = params[2]

    b, a = scipy.signal.butter(filter_order, cutoff_freq)

    x_hat = x
    padlen = len(x)-1 if len(x) < 9 else None
    for _ in range(num_iterations):
        x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad", padlen=padlen)

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    offset = np.mean(x) - np.mean(x_hat)
    x_hat = x_hat + offset

    return x_hat, dxdt_hat


def splinediff(x, dt, params=None, options={}, order=3, s=None, num_iterations=1):
    """Perform spline smoothing on x with scipy.interpolate.UnivariateSpline followed by first order finite difference

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param list params: (**deprecated**, prefer :code:`order`, :code:`cutoff_freq`, and :code:`num_iterations`)
    :param dict options: (**deprecated**, prefer :code:`num_iterations`) an empty dictionary or {'iterate': (bool)}
    :param int order: polynomial order of the spline. A kth order spline can be differentiated k times.
    :param float s: positive smoothing factor used to choose the number of knots. Number of knots will be increased
        until the smoothing condition is satisfied: :math:`\\sum_t (x[t] - \\text{spline}[t])^2 \\leq s`
    :param int num_iterations: how many times to apply smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `order`, `s`, and " +
            "`num_iterations` instead.", DeprecationWarning)
        order, s = params[0:2]
        if 'iterate' in options and options['iterate']:
            num_iterations = params[2]

    t = np.arange(0, len(x)*dt, dt)

    x_hat = x
    for _ in range(num_iterations):
        spline = scipy.interpolate.UnivariateSpline(t, x_hat, k=order, s=s)
        x_hat = spline(t)

    dspline = spline.derivative()
    dxdt_hat = dspline(t)

    return x_hat, dxdt_hat
