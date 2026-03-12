"""Methods based on fitting basis functions to data"""
from warnings import warn
import numpy as np
from scipy import sparse

from pynumdiff.utils import utility

def spectraldiff(x, dt, params=None, options=None, high_freq_cutoff=None,
                 even_extension=True, pad_to_zero_dxdt=True, axis=0):
    """Take a derivative in the Fourier domain, with high frequency attentuation.

    :param np.array[float] x: data to differentiate
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

    x = np.asarray(x)
    x0 = np.moveaxis(x, axis, 0) # move time axis to the front of the array
    # Now x0 dims are (number of data points, number of signals)
    L = x0.shape[0]

    # Make derivative go to zero at the ends (optional):
    if pad_to_zero_dxdt:
        padding = 100
        pre = np.repeat(x0[0:1], padding, axis=0)
        post = np.repeat(x0[-1:], padding, axis=0)
        x0 = np.concatenate((pre, x0, post), axis=0) # np.concatenate works along any axis, unlike hstack/vstack
    else:
        padding = 0

    # Do even extension (optional)
    if even_extension is True:
        x0 = np.concatenate((x0, x0[::-1, ...]), axis=0)

    # Form wavenumbers
    N = x0.shape[0]
    k = np.concatenate((np.arange(N//2 + 1), np.arange(-N//2 + 1, 0)))
    if N % 2 == 0: k[N//2] = 0 # odd derivatives get the Nyquist element zeroed out

    # Filter to zero out higher wavenumbers
    discrete_cutoff = int(high_freq_cutoff * N / 2) # Nyquist is at N/2 location, and we're cutting off as a fraction of that
    
    filt = np.ones(k.shape)  # start with all frequencies passing
    filt[discrete_cutoff:-discrete_cutoff] = 0  # zero out high-frequency components
    filt = filt.reshape((N,) + (1,)*(x0.ndim-1)) # reshape for broadcasting over extra dimensions

    # Smoothed signal
    X = np.fft.fft(x0, axis=0)

    x_hat0 = np.real(np.fft.ifft(filt * X, axis=0))
    x_hat0 = x_hat0[padding:L+padding]

    # Derivative = 90 deg phase shift
    omega = 2*np.pi/(dt*N) # factor of 2pi/T turns wavenumbers into frequencies in radians/s
    k0 = k.reshape((N,) + (1,)*(x0.ndim-1))
    dxdt0 = np.real(np.fft.ifft(1j * k0 * omega * filt * X, axis=0))
    dxdt0 = dxdt0[padding:L+padding]
    # move back to original axis position
    x_hat = np.moveaxis(x_hat0, 0, axis)
    dxdt_hat = np.moveaxis(dxdt0, 0, axis)
    
    return x_hat, dxdt_hat


def rbfdiff(x, dt_or_t, sigma=1, lmbd=0.01):
    """Find smoothed function and derivative estimates by fitting noisy data with radial-basis-functions. Naively,
    fill a matrix with basis function samples and solve a linear inverse problem against the data, but truncate tiny
    values to make columns sparse. Each basis function "hill" is topped with a "tower" of height :code:`lmbd` to reach
    noisy data samples, and the final smoothed reconstruction is found by razing these and only keeping the hills.

    :param np.array[float] x: data to differentiate
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param float sigma: controls width of radial basis functions
    :param float lmbd: controls smoothness

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.isscalar(dt_or_t):
        t = np.arange(len(x))*dt_or_t
    else: # support variable step size for this function
        if len(x) != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        t = dt_or_t

    # The below does the approximate equivalent of this code, but sparsely in O(N sigma^2), since the rbf falls off rapidly
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

    rbf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(t), len(t))) # Build sparse kernels, O(N sigma) entries
    drbfdt = sparse.csr_matrix((dvals, (rows, cols)), shape=(len(t), len(t)))
    rbf_regularized = rbf + lmbd*sparse.eye(len(t), format="csr") # identity matrix gives a little extra height at the centers
    alpha = sparse.linalg.spsolve(rbf_regularized, x) # solve sparse system targeting the noisy data, O(N sigma^2)

    return rbf @ alpha, drbfdt @ alpha # find samples of reconstructions using the smooth bases
