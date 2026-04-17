"""Methods based on fitting basis functions to data"""
from warnings import warn
import numpy as np
from scipy import sparse
import pywt

from pynumdiff.utils import utility

def spectraldiff(x, dt, params=None, options=None, high_freq_cutoff=None, even_extension=True,
    pad_to_zero_dxdt=True, axis=0):
    """Take a derivative in the Fourier domain, with high frequency attentuation.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param list[float] or float params: (**deprecated**, prefer :code:`high_freq_cutoff`)
    :param dict options: (**deprecated**, prefer :code:`even_extension`
            and :code:`pad_to_zero_dxdt`) a dictionary consisting of {'even_extension': (bool), 'pad_to_zero_dxdt': (bool)}
    :param float high_freq_cutoff: The high frequency cutoff as a multiple of the Nyquist frequency: Should be between 0
            and 1. Frequencies below this threshold will be kept, and above will be zeroed.
    :param bool even_extension: if True, extend the data with an even extension so signal starts and ends at the same value.
    :param bool pad_to_zero_dxdt: if True, extend the data with extra regions that smoothly force the derivative to
            zero before taking FFT.

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params is not None: # Warning to support old interface for a while. Remove these lines along with params in a future release.
        warn("`params` and `options` parameters will be removed in a future version. Use `high_freq_cutoff`, " +
            "`even_extension`, and `pad_to_zero_dxdt` instead.", DeprecationWarning)
        high_freq_cutoff = params[0] if isinstance(params, list) else params
        if options is not None:
            if 'even_extension' in options: even_extension = options['even_extension']
            if 'pad_to_zero_dxdt' in options: pad_to_zero_dxdt = options['pad_to_zero_dxdt']
    elif high_freq_cutoff is None:
        raise ValueError("`high_freq_cutoff` must be given.")

    L = x.shape[axis]

    # Make derivative go to zero at the ends (optional)
    if pad_to_zero_dxdt:
        padding = 100
        pre = np.repeat(np.take(x, [0], axis=axis), padding, axis=axis) # take keeps dimensions, unlike x[0]
        post = np.repeat(np.take(x, [-1], axis=axis), padding, axis=axis)
        x = np.concatenate((pre, x, post), axis=axis) # extend the edges
        kernel = utility.mean_kernel(padding//2)
        x_smoothed = utility.convolutional_smoother(x, kernel, axis=axis) # smooth the padded edges in
        m = (slice(None),)*axis + (slice(padding, L+padding),) + (slice(None),)*(x.ndim-axis-1) # middle
        x_smoothed[m] = x[m] # restore original signal in the middle
        x = x_smoothed
    else:
        m = (slice(None),)*axis + (slice(0, L),) + (slice(None),)*(x.ndim-axis-1) # indices where signal lives

    # Do even extension (optional)
    if even_extension is True:
        x = np.concatenate((x, np.flip(x, axis=axis)), axis=axis)

    s = [np.newaxis for dim in x.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

    # Form wavenumbers
    N = x.shape[axis]
    k = np.concatenate((np.arange(N//2 + 1), np.arange(-N//2 + 1, 0)))
    if N % 2 == 0: k[N//2] = 0 # odd derivatives get the Nyquist element zeroed out

    # Filter to zero out higher wavenumbers
    discrete_cutoff = int(high_freq_cutoff * N / 2) # Nyquist is at N/2 location, and we're cutting off as a fraction of that
    filt = np.ones(k.shape) # start with all frequencies passing
    filt[discrete_cutoff:-discrete_cutoff] = 0 # zero out high-frequency components

    # Smoothed signal
    X = np.fft.fft(x, axis=axis)
    x_hat = np.real(np.fft.ifft(filt[s] * X, axis=axis))

    # Derivative = 90 deg phase shift
    omega = 2*np.pi/(dt*N) # factor of 2pi/T turns wavenumbers into frequencies in radians/s
    dxdt_hat = np.real(np.fft.ifft(1j * k[s] * omega * filt[s] * X, axis=axis))

    return x_hat[m], dxdt_hat[m]


def rbfdiff(x, dt_or_t, sigma=1, lmbd=0.01, axis=0):
    """Find smoothed function and derivative estimates by fitting noisy data with radial-basis-functions. Naively,
    fill a matrix with basis function samples and solve a linear inverse problem against the data, but truncate tiny
    values to make columns sparse. Each basis function "hill" is topped with a "tower" of height :code:`lmbd` to reach
    noisy data samples, and the final smoothed reconstruction is found by razing these and only keeping the hills.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param float sigma: controls width of radial basis functions
    :param float lmbd: controls smoothness
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    N = x.shape[axis]
    x = np.moveaxis(x, axis, 0) # bring axis of differentiation to front so each N repeats comprise vector
    plump = x.shape
    x_flattened = x.reshape(N, -1) # (N, M) matrix where each column is a vector along the original axis

    if np.isscalar(dt_or_t):
        t = np.arange(N)*dt_or_t
    else: # support variable step size for this function
        if N != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        t = dt_or_t

    # For each vector along the axis of differentiation, the below does the approximate equivalent of this code,
    # but sparsely in O(N sigma^2), since the rbf falls off rapidly
    # t_i, t_j = np.meshgrid(t,t)
    # r = t_j - t_i # radius
    # rbf = np.exp(-(r**2) / (2 * sigma**2)) # radial basis function kernel, O(N^2) entries
    # drbfdt = -(r / sigma**2) * rbf # derivative of kernel
    # rbf_regularized = rbf + lmbd*np.eye(len(t))
    # alpha = np.linalg.solve(rbf_regularized, x) # O(N^3)

    cutoff = np.sqrt(-2 * sigma**2 * np.log(1e-4))
    rows, cols, vals, dvals = [], [], [], []
    for n in range(len(t)): # pylint: disable=consider-using-enumerate
        # Only consider points within a cutoff. Gaussian drops below eps at distance ~ sqrt(-2*sigma^2 log eps)
        l = np.searchsorted(t, t[n] - cutoff) # O(log N) to find indices of points within cutoff
        r = np.searchsorted(t, t[n] + cutoff) # finds index where new value should be inserted
        for j in range(l, r): # width of this is dependent on sigma. [l, r) is correct inclusion/exclusion
            radius = t[n] - t[j]
            v = np.exp(-radius**2 / (2 * sigma**2))
            dv = -radius / sigma**2 * v # take derivative of radial basis function, because d/dt coef*f(t) = coef*df/dt
            rows.append(n); cols.append(j); vals.append(v); dvals.append(dv)

    rbf = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N)) # Build sparse kernels, O(N sigma) entries
    drbfdt = sparse.csr_matrix((dvals, (rows, cols)), shape=(N, N))
    rbf_regularized = rbf + lmbd*sparse.eye(N, format="csr") # identity matrix gives a little extra height at the centers
    alpha = sparse.linalg.spsolve(rbf_regularized, x_flattened) # solve sparse system targeting the noisy data,
                                                                # can take matrix target, O(N sigma^2) for each vector
    x_hat_flattened = rbf @ alpha # find samples of reconstructions using the smooth bases
    dxdt_hat_flattened = drbfdt @ alpha

    return np.moveaxis(x_hat_flattened.reshape(plump), 0, axis), np.moveaxis(dxdt_hat_flattened.reshape(plump), 0, axis)


def waveletdiff(x, dt_or_t, wavelet='db4', level=None, threshold=1.0, axis=0, mode='periodization'):
    """Smooth and differentiate noisy data via discrete wavelet denoising.

    Decomposes x into wavelet detail and approximation coefficients, soft-thresholds
    the detail coefficients to remove noise using the Donoho & Johnstone (1994)
    universal threshold estimator, reconstructs a smoothed signal, then
    differentiates with finite differences via np.gradient.

    :param np.array x: data to differentiate
    :param float or array dt_or_t: scalar dt or array of sample times. If an
        array is provided it is passed directly to np.gradient, giving correct
        results for non-uniformly sampled data.
    :param str wavelet: PyWavelets wavelet name, e.g. 'db4', 'sym4', 'coif2'.
        'db4' is a solid general-purpose default. Biorthogonal wavelets such as
        'bior2.2' or 'bior4.4' are symmetric and designed for smooth reconstruction
        but may need a lower threshold value.
    :param int level: decomposition depth. None (default) resolves to
        min(pywt.dwt_max_level(N, wavelet), 5) to avoid over-decomposing short
        signals. Increase for heavily oversampled data.
    :param float threshold: soft-thresholding scale factor in [0, inf).
        Multiplies the universal threshold sigma * sqrt(2 * log(N)).
        threshold=1.0 is the classical Donoho & Johnstone universal threshold
        and is the recommended starting point. Values < 1.0 give less smoothing;
        values > 1.0 give more aggressive smoothing. This parameter maps onto
        tvgamma in the pynumdiff.optimize framework.
    :param int axis: axis along which to differentiate (default 0).
    :param str mode: PyWavelets signal extension mode passed to wavedec/waverec.
        'periodization' (default) keeps coefficient arrays exactly length N and
        is the most numerically stable choice for differentiation. 'reflect' is
        a good alternative for clearly non-periodic signals.
        See pywt.Modes.modes for all options.
    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    N = x.shape[axis]

    # Axis normalisation — bring target axis to front.
    # Skip moveaxis when axis is already 0 to avoid an unnecessary allocation.
    # When we do move, call ascontiguousarray immediately so the subsequent
    # reshape is guaranteed zero-copy.
    if axis == 0:
        x_work = x if x.flags['C_CONTIGUOUS'] else np.ascontiguousarray(x)
    else:
        x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0))

    shape = x_work.shape
    x_flat = x_work.reshape(N, -1)  # (N, M) contiguous, no hidden copy
    M = x_flat.shape[1]

    if np.isscalar(dt_or_t):
        grad_arg = dt_or_t
    else:
        if len(dt_or_t) != N:
            raise ValueError(
                "`dt_or_t` array must have the same length as x along `axis`."
            )
        grad_arg = dt_or_t  # np.gradient accepts a full coordinate array

    # Conservative level default avoids over-decomposing short signals
    # (pywt default uses the maximum possible level).
    if level is None:
        max_level = pywt.dwt_max_level(N, wavelet)
        level = min(max_level, 5)

    # Decompose all columns and stack coefficients into 2-D arrays of shape
    # (coeff_len_i, M). Probing column 0 first lets us pre-allocate correctly;
    # the probe result is reused for col 0 so we pay N+1 wavedec calls total.
    _probe = pywt.wavedec(x_flat[:, 0], wavelet, level=level, mode=mode)
    coeff_lengths = [len(c) for c in _probe]
    n_levels = len(_probe)

    coeffs_all = [
        np.empty((coeff_lengths[i], M), dtype=x_flat.dtype)
        for i in range(n_levels)
    ]
    for i, c in enumerate(_probe):
        coeffs_all[i][:, 0] = c

    for col in range(1, M):
        for i, c in enumerate(
            pywt.wavedec(x_flat[:, col], wavelet, level=level, mode=mode)
        ):
            coeffs_all[i][:, col] = c

    # Vectorised noise estimation and soft-thresholding over all columns at once.
    # sigma: robust MAD estimator from finest detail level, shape (M,).
    # thresh: per-column universal threshold, shape (M,).
    # Approximation coefficients (index 0) are left untouched; only detail
    # levels (indices 1..n_levels-1) are thresholded.
    sigma = np.median(np.abs(coeffs_all[-1]), axis=0) / 0.6745
    np.maximum(sigma, 1e-10, out=sigma)  # floor avoids zero threshold on clean signals

    thresh = threshold * sigma * np.sqrt(2 * np.log(N))  # shape (M,)

    coeffs_denoised = [coeffs_all[0]] + [
        pywt.threshold(c, thresh[np.newaxis, :], mode='soft')
        for c in coeffs_all[1:]
    ]

    # Reconstruct and differentiate — pywt.waverec is 1-D only so a column
    # loop remains, but all Python-level arithmetic has been moved out above.
    x_hat_flat    = np.empty_like(x_flat)
    dxdt_hat_flat = np.empty_like(x_flat)

    for col in range(M):
        col_coeffs = [coeffs_denoised[i][:, col] for i in range(n_levels)]
        x_hat_col = pywt.waverec(col_coeffs, wavelet, mode=mode)[:N]
        x_hat_flat[:, col]    = x_hat_col
        dxdt_hat_flat[:, col] = np.gradient(x_hat_col, grad_arg)

    # Restore original shape and axis order.
    # moveaxis on the way out is only needed when we moved on the way in.
    x_hat    = x_hat_flat.reshape(shape)
    dxdt_hat = dxdt_hat_flat.reshape(shape)

    if axis != 0:
        x_hat    = np.moveaxis(x_hat,    0, axis)
        dxdt_hat = np.moveaxis(dxdt_hat, 0, axis)

    return x_hat, dxdt_hat
