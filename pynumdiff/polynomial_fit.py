"""Methods based on fitting data with polynomials"""
from warnings import warn, catch_warnings
import numpy as np
import scipy

from pynumdiff.utils import utility


def splinediff(x, dt_or_t, degree=3, s=1, num_iterations=1, axis=0):
    """Find smoothed data and derivative estimates by fitting a smoothing spline to the data with
    scipy.interpolate.make_splrep. Variable step size is supported with equal ease as uniform step size.

    :param np.array[float] x: data to differentiate. May contain NaN values (missing data); NaNs are excluded from
        fitting and imputed by spline interpolation. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int degree: polynomial degree of the spline. A kth degree spline can be differentiated k times.
    :param float s: nonnegative smoothing factor. Number of knots will be increased until the smoothing condition
        :math:`\\sum_t (x[t] - \\text{spline}[t])^2 \\leq s N \\hat\\sigma^2` is met, where :math:`N` is data length and
        :math:`\\hat\\sigma` is a robust estimate of noise stddev. :math:`s = 1` leaves exactly that much energy for
        residuals and is a good default; larger over-smooths, smaller under-smooths, and 0 yields an interpolating spline.
    :param int num_iterations: how many times to apply smoothing
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if num_iterations < 1: raise ValueError("`num_iterations` should be >=1")
    if np.isscalar(dt_or_t):
        t = np.arange(x.shape[axis]) * dt_or_t
    else: # support variable step size for this function
        if x.shape[axis] != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        if np.any(np.diff(dt_or_t) <= 0): raise ValueError("`dt_or_t` must be strictly increasing. Out-of-order or repeated "
            "sample locations make neighbor differences and windows meaningless.")
        t = dt_or_t

    x_hat = np.empty(x.shape, dtype=float); dxdt_hat = np.empty(x.shape, dtype=float) # float explicitly, so inherited integer input type cannot silently truncate

    for vec_idx in np.ndindex(x.shape[:axis] + x.shape[axis+1:]):
        i = vec_idx[:axis] + (slice(None),) + vec_idx[axis:] # use i instead of s, becase s is already used as smoothness param

        obs = ~np.isnan(x[i]) # make_splrep can't handle NaN, so use only observed points for first fit
        # Scale the noise budget by N σ̂² so s stays meaningful across signal lengths and noise levels. If tiny sigma_hat
        # (relative to data scale), the spline fit chases overly fine residuals, so interpolate instead by setting budget to 0
        sigma_hat = utility.robust_noise_scale(x[i][obs])
        s_abs = s*np.sum(obs)*sigma_hat**2 if sigma_hat > 1e-12*np.max(np.abs(x[i][obs])) else 0

        with catch_warnings(action="ignore", category=RuntimeWarning): # FITPACK warns at knife-edge values of s, but still solves reliably
            spline = scipy.interpolate.make_splrep(t[obs], x[i][obs], k=degree, s=s_abs)
            x_hat[i] = spline(t) # interpolate at all t
            for _ in range(num_iterations-1):
                spline = scipy.interpolate.make_splrep(t, x_hat[i], k=degree, s=s_abs) # hold noise (drift) budget fixed across iterations
                x_hat[i] = spline(t)
        dxdt_hat[i] = spline.derivative()(t) # evaluate derivative at sample points

    return x_hat, dxdt_hat


def polydiff(x, dt_or_t, degree, window_size=None, stride=1, kernel='friedrichs', axis=0):
    """Fit polynomials to the data, and differentiate the polynomials.

    :param np.array[float] x: data to differentiate. May contain NaN values (missing data); NaNs are excluded from
        fitting and imputed by polynomial interpolation. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int degree: degree of the polynomial
    :param int window_size: number of samples in the sliding window, or number of average step sizes to use as window
        width if irregular sampling; if not given, no sliding
    :param int stride: step size for sliding
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if not np.isscalar(dt_or_t): # check once here rather than per window, since `slide_function` hands slices to `_polydiff`
        if x.shape[axis] != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        if np.any(np.diff(dt_or_t) <= 0): raise ValueError("`dt_or_t` must be strictly increasing. Out-of-order or repeated "
            "sample locations make neighbor differences and windows meaningless.")
    if window_size:
        if window_size < degree*3: window_size = degree*3 + 1 + degree%2 # parity term to keep this odd
        if window_size % 2 == 0: window_size += 1; warn("Kernel window size should be odd. Added 1 to length.")
        if stride > window_size: stride = window_size; warn("`stride` > `window_size` would skip samples between windows, reduced to `window_size`")
        kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel]

    def _polydiff(x, dt_or_t, degree, weights=None):
        t = dt_or_t if not np.isscalar(dt_or_t) else np.arange(len(x)) * dt_or_t # sample locations
        obs = ~np.isnan(x) # Filter out any NaN values so polyfit doesn't lose its mind in the event of missing data
        if obs.sum() <= degree: # too few points to pin down the coefficients, so polyfit will fail
            raise ValueError(f"Window encountered with only {obs.sum()} non-NaN samples < {degree+1} samples needed for degree "
                f"{degree} fit. Widen `window_size` or lower `degree`.")

        r = np.polyfit(t[obs], x[obs], degree, w=np.sqrt(weights[obs]) if weights is not None else None) # sqrt(weights), because (weights*residuals)^2 internally
        dr = np.polyder(r) # power rule already implemented for us

        dxdt_hat = np.polyval(dr, t) # evaluate the derivative and original polynomials at points t
        x_hat = np.polyval(r, t) # smoothed x

        return x_hat, dxdt_hat

    x_hat = np.empty(x.shape, dtype=float); dxdt_hat = np.empty(x.shape, dtype=float) # float explicitly, so inherited integer input type cannot silently truncate

    for vec_idx in np.ndindex(x.shape[:axis] + x.shape[axis+1:]):
        s = vec_idx[:axis] + (slice(None),) + vec_idx[axis:]
        x_hat[s], dxdt_hat[s] = _polydiff(x[s], dt_or_t, degree) if not window_size else \
            utility.slide_function(_polydiff, x[s], dt_or_t, kernel, window_size, stride=stride,
                min_samples=degree+1, pass_weights=True, degree=degree)
    
    return x_hat, dxdt_hat


def savgoldiff(x, dt, degree, window_size, smoothing_win, axis=0):
    """Use the Savitzky-Golay to smooth the data and calculate the first derivative. It uses
    scipy.signal.savgol_filter. The Savitzky-Golay is very similar to the sliding polynomial fit,
    but slightly noisier and much faster.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int degree: degree of the polynomial
    :param int window_size: size of the sliding window, must be odd (if not, 1 is added)
    :param int smoothing_win: size of the window used for gaussian smoothing, a good default is
        window_size, but smaller for high frequnecy data
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values spread through the filter.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. Savitzky-Golay assumes fixed-width windows with uniform sampling.")

    window_size = np.clip(window_size, degree + 1 + degree%2, x.shape[axis] - 1 + x.shape[axis]%2) # returns odd numbers
    if window_size % 2 == 0: window_size += 1; warn("Kernel window size should be odd. Added 1 to length.")
    smoothing_win = min(smoothing_win, x.shape[axis] - 1 + x.shape[axis]%2) # parity check so an odd window can't be clamped even
    if smoothing_win % 2 == 0: smoothing_win += 1; warn("Smoothing window size should be odd. Added 1 to length.")

    dxdt_hat = scipy.signal.savgol_filter(x, window_size, degree, deriv=1, axis=axis)/dt

    kernel = utility.gaussian_kernel(smoothing_win)
    dxdt_hat = utility.convolutional_smoother(dxdt_hat, kernel, axis=axis)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt, axis=axis)
    x_hat += utility.estimate_integration_constant(x, x_hat, axis=axis)

    return x_hat, dxdt_hat
