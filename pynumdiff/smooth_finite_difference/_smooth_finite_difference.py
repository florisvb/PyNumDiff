import numpy as np
import scipy.signal
from warnings import warn

# included code
from pynumdiff.finite_difference import second_order as finite_difference
from pynumdiff.polynomial_fit import splinediff as _splinediff # patch through
from pynumdiff.utils import utility


def kerneldiff(x, dt, kernel='friedrichs', window_size=5, num_iterations=1):
    """Differentiate by applying a smoothing kernel to the signal, then performing 2nd-order finite difference.
    :code:`meandiff`, :code:`mediandiff`, :code:`gaussiandiff`, and :code:`friedrichsdiff` call this function.

    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param str kernel: prefilter data, {:code:`'mean'`, :code:`'median'`, :code:`'gaussian'`,
        :code:`'friedrichs'`}
    :param int window_size: filtering kernel size
    :param int num_iterations: how many times to apply mean smoothing

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if kernel in ['mean', 'gaussian', 'friedrichs']:
        kernel = getattr(utility, f"{kernel}_kernel")(window_size)
        x_hat = utility.convolutional_smoother(x, kernel, num_iterations)
    elif kernel == 'median':
        if not window_size % 2: window_size += 1 # make sure window_size is odd, else medfilt throws error

        x_hat = x
        for _ in range(num_iterations):
            x_hat = scipy.signal.medfilt(x_hat, window_size)
    else:
        raise ValueError("filter_type must be mean, median, gaussian, or friedrichs")

    return finite_difference(x_hat, dt)


def meandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform mean smoothing by convolving mean kernel with x followed by second order finite difference\n
    **Deprecated**, prefer :code:`kerneldiff` with kernel :code:`'mean'` instead.

    :param np.array[float] x: data to differentiate
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

    warn("`meandiff` is deprecated. Call `kerneldiff` with kernel 'mean' instead.", DeprecationWarning)
    return kerneldiff(x, dt, kernel='mean', window_size=window_size, num_iterations=num_iterations)


def mediandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform median smoothing using scipy.signal.medfilt followed by second order finite difference\n
    **Deprecated**, prefer :code:`kerneldiff` with kernel :code:`'median'` instead.

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

    warn("`mediandiff` is deprecated. Call `kerneldiff` with kernel 'median' instead.", DeprecationWarning)
    return kerneldiff(x, dt, kernel='median', window_size=window_size, num_iterations=num_iterations)


def gaussiandiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform gaussian smoothing by convolving gaussian kernel with x followed by second order finite difference\n
    **Deprecated**, prefer :code:`kerneldiff` with kernel :code:`'gaussian'` instead.

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

    warn("`gaussiandiff` is deprecated. Call `kerneldiff` with kernel 'gaussian' instead.", DeprecationWarning)
    return kerneldiff(x, dt, kernel='gaussian', window_size=window_size, num_iterations=num_iterations)


def friedrichsdiff(x, dt, params=None, options={}, window_size=5, num_iterations=1):
    """Perform friedrichs smoothing by convolving friedrichs kernel with x followed by second order finite difference\n
    **Deprecated**, prefer :code:`kerneldiff` with kernel :code:`'friedrichs'` instead.

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

    warn("`friedrichsdiff` is deprecated. Call `kerneldiff` with kernel 'friedrichs' instead.", DeprecationWarning)
    return kerneldiff(x, dt, kernel='friedrichs', window_size=window_size, num_iterations=num_iterations)


def butterdiff(x, dt, params=None, options={}, filter_order=2, cutoff_freq=0.5, num_iterations=1):
    """Perform butterworth smoothing on x with scipy.signal.filtfilt followed by second order finite difference

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
        x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad", padlen=padlen) # applies forward and backward pass so zero phase

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    offset = np.mean(x) - np.mean(x_hat)
    x_hat = x_hat + offset

    return x_hat, dxdt_hat


def splinediff(*args, **kwargs):
    warn("`splindiff` has moved to `polynomial_fit.splinediff` and will be removed from "
        + "`smooth_finite_difference` in a future release.", DeprecationWarning)
    return _splinediff(*args, **kwargs)
