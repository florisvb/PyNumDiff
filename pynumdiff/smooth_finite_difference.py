"""Apply smoothing method before finite difference."""
from warnings import warn
import numpy as np
import scipy.signal

from pynumdiff.finite_difference import finitediff
from pynumdiff.utils import utility


def kerneldiff(x, dt, kernel='friedrichs', window_size=5, num_iterations=1, axis=0):
    """Differentiate by applying a smoothing kernel to the signal, then performing 2nd-order finite difference.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param str kernel: prefilter data, {:code:`'mean'`, :code:`'median'`, :code:`'gaussian'`,
        :code:`'friedrichs'`}
    :param int window_size: filtering kernel size
    :param int num_iterations: how many times to apply mean smoothing
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Convolution spreads a NaN across many windows.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. Convolving with a fixed-width kernel assumes uniformly sampled data.")

    if window_size % 2 == 0: window_size += 1; warn("Even-width kernels shift answers by half-samples. Added 1 to length.")

    if kernel in ['mean', 'gaussian', 'friedrichs']:
        kernel = getattr(utility, f"{kernel}_kernel")(window_size)
        x_hat = utility.convolutional_smoother(x, kernel, num_iterations, axis=axis)
    elif kernel == 'median':
        s = [1]*x.ndim; s[axis] = window_size
        x_hat = x
        for _ in range(num_iterations):
            x_hat = scipy.signal.medfilt(x_hat, s)
    else:
        raise ValueError("filter_type must be mean, median, gaussian, or friedrichs")

    return finitediff(x_hat, dt, order=2, axis=axis)


def butterdiff(x, dt, filter_order=2, cutoff_freq=0.5, num_iterations=1, axis=0):
    """Perform butterworth smoothing on x with scipy.signal.filtfilt followed by second order finite difference

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int filter_order: order of the filter
    :param float cutoff_freq: cutoff frequency as a fraction of Nyquist, in :math:`\\in [0, 1]`
    :param int num_iterations: how many times to apply smoothing
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Filtering carries a NaN through the whole signal.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. A Butterworth filter is designed against a fixed sample rate.")

    sos = scipy.signal.butter(filter_order, cutoff_freq, output='sos') # second-order sections rather than the (b, a) transfer
        # function, whose coefficients lose all precision when many poles bunch up near z = 1 at high order and low cutoff

    x_hat = x
    padlen = min(3*(filter_order + 1), x.shape[axis]-1) # scipy's own default, can overwhelm short data
    for _ in range(num_iterations):
        x_hat = scipy.signal.sosfiltfilt(sos, x_hat, axis=axis, padlen=padlen) # applies forward and backward pass so zero phase

    return finitediff(x_hat, dt, order=2, axis=axis)

