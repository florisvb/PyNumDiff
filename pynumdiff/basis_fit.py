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


def waveletdiff(x, dt, wavelet='db8', level=None, threshold=1.0, axis=0, mode='periodization'):
    """Smooth and differentiate noisy data in a wavelet basis.

    Three steps: (1) decompose x with the DWT and soft-threshold the detail
    coefficients to denoise (Donoho-Johnstone universal threshold), reconstructing
    a smoothed x_hat; (2) extend x_hat antisymmetrically so the periodic derivative
    operator stays accurate at the edges; (3) recover the wavelet scaling
    coefficients of x_hat and apply the analytic derivative of the wavelet basis.

    The derivative differentiates the basis functions themselves rather than
    finite-differencing the signal. PyWavelets treats the samples as finest-level
    scaling coefficients, so x_hat is the interpolant x(t) = sum_n a_n phi(t/dt - n)
    for the scaling function phi. Sampling x and its analytic derivative on the grid
    gives two convolutions against phi and phi' evaluated at *integers*,

        x_hat = Phi @ a     and     x' = Phi_prime @ a,

    so x' = Phi_prime @ Phi^-1 @ x_hat, exact for signals the basis can represent.
    The integer samples phi(p), phi'(p) are the eigenvalue-1 and eigenvalue-1/2
    eigenvectors of the refinement relation phi(t) = sqrt2 sum_k h_k phi(2t - k)
    (the "connection coefficients"), normalized to reproduce constants and ramps.
    This is the wavelet-basis representation of the derivative operator from
    Beylkin (1992); the connection coefficients follow Latto, Resnikoff &
    Tenenbaum (1991), for Daubechies' compactly supported wavelets (1988).

    Because the DWT requires uniform spacing, this method only accepts a scalar
    time step dt (not a vector of sample times). For non-uniformly sampled data,
    use :func:`rbfdiff` or :func:`splinediff` instead.

    References:
        G. Beylkin, "On the representation of operators in bases of compactly
        supported wavelets," SIAM J. Numer. Anal. 29(6):1716-1740, 1992.
        A. Latto, H. L. Resnikoff & E. Tenenbaum, "The evaluation of connection
        coefficients of compactly supported wavelets," Proc. French-USA Workshop
        on Wavelets and Turbulence, 1991.

    :param np.array x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: uniform time step between samples.
    :param str wavelet: PyWavelets wavelet name. Must have a differentiable scaling
        function, so smoother wavelets give better derivatives: 'db8' (default) and
        'sym8' are best for noisy data; 'db4', 'sym4', and 'coif2' also work well.
    :param int level: decomposition depth. None (default) resolves to
        min(pywt.dwt_max_level(N, wavelet), 5) to avoid over-decomposing short signals.
    :param float threshold: soft-thresholding scale factor in [0, inf).
    :param int axis: axis along which to differentiate (default 0).
    :param str mode: PyWavelets signal extension mode for the denoising transform.
        'periodization' keeps coefficient arrays compact. The derivative operator is
        periodic, so x_hat is antisymmetrically extended before it is applied (see below).
    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if not np.isscalar(dt):
        raise ValueError("`dt` must be a scalar. The DWT requires uniformly sampled data. "
            "For variable step sizes, use rbfdiff or splinediff instead.")

    # The Haar scaling function is a step, so it has no pointwise derivative and the
    # connection-coefficient operator below is undefined for it. Haar/db1 is the only
    # orthonormal wavelet with a 2-tap filter, so dec_len identifies it.
    if pywt.Wavelet(wavelet).dec_len == 2:
        raise ValueError("The Haar/db1 wavelet has a discontinuous (piecewise-constant) scaling "
            "function with no derivative, so it cannot be used to differentiate. Pick a smoother "
            "wavelet such as 'db4', 'sym4', or 'coif2'.")

    N = x.shape[axis]
    x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0)) # differentiation axis to front
    shape = x_work.shape                                   # remember it to restore the input's dimensionality
    x_flat = x_work.reshape(N, -1)                         # rest of the dims flattened into columns
    Ne = 3 * N - 2                                         # length after the antisymmetric extension in step 2

    # Build the wavelet-basis derivative operator (depends only on the grid and wavelet).
    # Sampling the refinement relation phi(t) = sqrt2 sum_k h_k phi(2t - k) at integers makes
    # phi(p) the eigenvalue-1 and phi'(p) the eigenvalue-1/2 eigenvector of T[p,q] = sqrt2 h_{2p-q}.
    h = np.array(pywt.Wavelet(wavelet).rec_lo); h = h / h.sum() * np.sqrt(2) # refinement filter, integral of phi = 1
    L = len(h); p = np.arange(L)                            # phi is supported on the integers [0, L-1]
    shift = 2 * p[:, None] - p[None, :]
    T = np.where((shift >= 0) & (shift < L), np.sqrt(2) * h[np.clip(shift, 0, L - 1)], 0.0)
    evals, evecs = np.linalg.eig(T)
    phi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); phi /= phi.sum()           # sum_p phi(p) = 1
    dphi = np.real(evecs[:, np.argmin(np.abs(evals - 0.5))]); dphi /= np.dot(p, dphi)*-1 # sum_p p*phi'(p) = -1
    # Phi and Phi_prime hold circulant samples of phi and phi'/dt on the extended grid; both
    # share a common shift that cancels in Phi_prime @ Phi^-1, so the offset choice is cosmetic.
    rows, cols, phi_vals, dphi_vals = [], [], [], []
    m = np.arange(Ne)
    for offset, phi_p, dphi_p in zip(p, phi, dphi / dt):
        rows.extend(m); cols.extend((m - offset) % Ne); phi_vals.extend([phi_p]*Ne); dphi_vals.extend([dphi_p]*Ne)
    Phi = sparse.csr_matrix((phi_vals, (rows, cols)), shape=(Ne, Ne)).tocsc()           # to invert
    Phi_prime = sparse.csr_matrix((dphi_vals, (rows, cols)), shape=(Ne, Ne))            # to apply

    if level is None:
        level = min(pywt.dwt_max_level(N, wavelet), 5)

    # 1. Denoise: DWT all columns at once, then soft-threshold the detail bands. The
    # noise level is estimated robustly per column from the finest details (coeffs[-1]).
    coeffs = pywt.wavedec(x_flat, wavelet, level=level, mode=mode, axis=0)
    sigma = np.maximum(np.median(np.abs(coeffs[-1]), axis=0) / 0.6745, 1e-10)
    thresh = threshold * sigma * np.sqrt(2 * np.log(N))
    coeffs = [coeffs[0]] + [pywt.threshold(c, thresh[np.newaxis, :], mode='soft') for c in coeffs[1:]]
    x_hat = pywt.waverec(coeffs, wavelet, mode=mode, axis=0)[:N]

    # 2. The derivative operator is periodic, but x_hat usually isn't. Extend it
    # antisymmetrically (reflect through each endpoint: x[-1-k] -> 2*x[0]-x[1+k]) so the
    # periodic wrap is continuous in both value and slope, which keeps the derivative
    # accurate at the edges instead of spiking there. This is the odd-symmetry analog of
    # spectraldiff's even extension; a ramp extends to a ramp, so slopes survive exactly.
    left = 2 * x_hat[0] - x_hat[1:][::-1]
    right = 2 * x_hat[-1] - x_hat[:-1][::-1]
    x_ext = np.concatenate([left, x_hat, right], axis=0)  # length 3N-2, original at [N-1:2N-1]

    # 3. Differentiate the basis: recover the scaling coefficients a = Phi^-1 @ x_ext, then
    # apply the analytic basis derivative dxdt = Phi_prime @ a, and crop back to the original.
    a = sparse.linalg.spsolve(Phi, x_ext)
    dxdt_flat = (Phi_prime @ a.reshape(Ne, -1))[N - 1:2 * N - 1]

    x_hat = np.moveaxis(x_hat.reshape(shape), 0, axis)
    dxdt_hat = np.moveaxis(dxdt_flat.reshape(shape), 0, axis)
    return x_hat, dxdt_hat
