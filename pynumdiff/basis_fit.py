"""Methods based on fitting basis functions to data"""
from functools import lru_cache
from warnings import warn
import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
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


@lru_cache(maxsize=32)
def _wavelet_derivative_synthesis_matrix(N, dt, wavelet, level, mode):
    """Build sparse samples of d/dt of the inverse-DWT synthesis basis.

    For a fixed wavelet/level/mode/length, wavedec/waverec define a linear
    synthesis map

        x(t_n) = sum_k c_k phi_k(t_n).

    This routine samples phi'_k(t_n) once, stores those samples sparsely, and
    lets waveletdiff compute

        x'(t_n) = sum_k c_k phi'_k(t_n)

    without differentiating the reconstructed signal.  The derivative samples
    are obtained from a local cubic interpolant of each compactly supported
    synthesis basis vector; this is bookkeeping on the basis functions, not a
    finite-difference derivative of the data.
    """
    zero = np.zeros(N)
    template = pywt.wavedec(zero, wavelet, level=level, mode=mode)
    coeff_lengths = tuple(len(c) for c in template)
    coeff_offsets = np.cumsum((0,) + coeff_lengths[:-1])
    n_coeffs = sum(coeff_lengths)
    t = np.arange(N, dtype=float) * dt

    rows, cols, vals = [], [], []
    eps = 1e-12

    for band, (offset, length) in enumerate(zip(coeff_offsets, coeff_lengths)):
        for local_idx in range(length):
            coeffs = [np.zeros_like(c, dtype=float) for c in template]
            coeffs[band][local_idx] = 1.0
            basis = pywt.waverec(coeffs, wavelet, mode=mode)[:N]

            # Basis functions are compactly supported, but boundary extension can
            # split support across the two ends.  Differentiating only the active
            # samples keeps the matrix sparse and avoids global sinusoidal bases.
            active = np.flatnonzero(np.abs(basis) > eps)
            if active.size == 0:
                continue

            # Include one-sample padding around active support so the cubic has
            # enough context near the edges of the support.  If support wraps or
            # covers most of the signal, fall back to all samples.
            support = np.zeros(N, dtype=bool)
            support[active] = True
            support[np.maximum(active - 1, 0)] = True
            support[np.minimum(active + 1, N - 1)] = True
            idx = np.flatnonzero(support)
            if idx.size < 4 or (idx[-1] - idx[0] + 1) > 2 * idx.size:
                idx = np.arange(N)

            # CubicSpline requires strictly increasing x and at least two points.
            # With >=4 points the not-a-knot default is well-defined; with fewer,
            # fall back to clamped end slopes of zero.
            bc_type = 'not-a-knot' if idx.size >= 4 else ((1, 0.0), (1, 0.0))
            spline = CubicSpline(t[idx], basis[idx], bc_type=bc_type, extrapolate=False)
            deriv_vals = spline(t[idx], 1)
            keep = np.isfinite(deriv_vals) & (np.abs(deriv_vals) > eps)

            rows.extend(idx[keep])
            cols.extend(np.full(np.count_nonzero(keep), offset + local_idx))
            vals.extend(deriv_vals[keep])

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, n_coeffs)), coeff_lengths


def _flatten_wavelet_coeffs(coeffs):
    """Stack a wavedec coefficient list into a 2-D coefficient matrix."""
    return np.vstack([c for band in coeffs for c in band])


def waveletdiff(x, dt, wavelet='db4', level=None, threshold=1.0, axis=0, mode='periodization'):
    """Smooth and differentiate noisy data with a wavelet-basis derivative sum.

    Decomposes x into wavelet approximation/detail coefficients, soft-thresholds
    the detail coefficients to denoise, reconstructs a smoothed signal, and then
    estimates the derivative directly from the denoised wavelet coefficients:

        x(t_n)  = sum_k c_k phi_k(t_n)
        x'(t_n) = sum_k c_k phi'_k(t_n)

    The first sum is the ordinary inverse wavelet transform.  The second sum is
    evaluated by precomputing sparse samples of the derivative of each synthesis
    basis function and multiplying that sparse matrix by the denoised
    coefficients.  This avoids the previous reconstruct-then-FFT derivative path
    and does not call finite differences or np.gradient on the signal.

    Because the DWT requires uniform spacing, this method only accepts a scalar
    time step dt (not a vector of sample times). For non-uniformly sampled data,
    use :func:`rbfdiff` or :func:`splinediff` instead.

    :param np.array x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: uniform time step between samples.
    :param str wavelet: PyWavelets wavelet name, e.g. 'db4', 'sym4', 'coif2'.
    :param int level: decomposition depth. None (default) resolves to
        min(pywt.dwt_max_level(N, wavelet), 5) to avoid over-decomposing short signals.
    :param float threshold: soft-thresholding scale factor in [0, inf).
    :param int axis: axis along which to differentiate (default 0).
    :param str mode: PyWavelets signal extension mode passed to wavedec/waverec.
        'periodization' keeps coefficient arrays compact; 'reflect' is often a
        better choice for clearly non-periodic signals.
    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if not np.isscalar(dt):
        raise ValueError(
            "`dt` must be a scalar. The DWT requires uniformly sampled data. "
            "For variable step sizes, use rbfdiff or splinediff instead."
        )

    N = x.shape[axis]
    x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0))
    shape = x_work.shape
    x_flat = x_work.reshape(N, -1)
    M = x_flat.shape[1]

    if level is None:
        max_level = pywt.dwt_max_level(N, wavelet)
        level = min(max_level, 5)

    # Decompose all columns; probe column 0 first to learn coefficient lengths
    # and pre-allocate, reusing that result so we only pay N+1 wavedec calls.
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

    # Robust noise estimate from finest details, then Donoho-Johnstone
    # soft-thresholding on detail bands only.
    sigma = np.median(np.abs(coeffs_all[-1]), axis=0) / 0.6745
    np.maximum(sigma, 1e-10, out=sigma)
    thresh = threshold * sigma * np.sqrt(2 * np.log(N))
    coeffs_denoised = [coeffs_all[0]] + [
        pywt.threshold(c, thresh[np.newaxis, :], mode='soft')
        for c in coeffs_all[1:]
    ]

    Dphi, matrix_coeff_lengths = _wavelet_derivative_synthesis_matrix(
        N, float(dt), wavelet, int(level), mode
    )
    if tuple(coeff_lengths) != tuple(matrix_coeff_lengths):
        raise RuntimeError("Cached wavelet derivative matrix coefficient layout does not match wavedec output.")

    x_hat_flat = np.empty_like(x_flat)
    coeffs_flat = np.empty((sum(coeff_lengths), M), dtype=x_flat.dtype)
    offsets = np.cumsum((0,) + tuple(coeff_lengths[:-1]))

    for col in range(M):
        col_coeffs = [coeffs_denoised[i][:, col] for i in range(n_levels)]
        x_hat_flat[:, col] = pywt.waverec(col_coeffs, wavelet, mode=mode)[:N]
        for i, (offset, length) in enumerate(zip(offsets, coeff_lengths)):
            coeffs_flat[offset:offset + length, col] = coeffs_denoised[i][:, col]

    dxdt_hat_flat = Dphi @ coeffs_flat

    x_hat = np.moveaxis(x_hat_flat.reshape(shape), 0, axis)
    dxdt_hat = np.moveaxis(dxdt_hat_flat.reshape(shape), 0, axis)

    return x_hat, dxdt_hat
