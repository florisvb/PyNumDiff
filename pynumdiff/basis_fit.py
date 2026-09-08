"""Methods based on fitting basis functions to data"""
import numpy as np
from scipy import sparse
import pywt

from pynumdiff.utils import utility

def spectraldiff(x, dt, cutoff_freq, extension='odd', pad_to_flat=False, axis=0):
    """Take a derivative in the Fourier domain, with high frequency attentuation.

    The FFT treats data as periodic, so a signal whose ends do not meet has a discontinuity across the wrap, and
    Gibbs ringing contaminates the estimate. `extension` picks the remedy: 'detrend' subtracts the line through the
    endpoints, 'even' mirrors the signal, and 'odd' detrends and then reflects through the last endpoint.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param float cutoff_freq: The high frequency cutoff as a multiple of the Nyquist frequency: Should be between 0
        and 1. Frequencies below this threshold will be kept, and at and above will be zeroed.
    :param str extension: how to make the data periodic: :code:`None`, :code:`'even'`, :code:`'detrend'`, or :code:`'odd'`.
        None is right only for genuinely periodic signals.
    :param bool pad_to_flat: if True, extend the edges with smoothed repeats of the end values, giving the Gibbs
        ringing somewhere to go that gets discarded.
    :param int axis: data dimension along which to differentiate

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values spread through the FFT to make the whole spectrum NaN.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The FFT assumes uniformly sampled data.")
    if extension not in (None, 'even', 'detrend', 'odd'): raise ValueError("`extension` must be None, 'even', 'detrend', or 'odd'.")

    x = np.moveaxis(x, axis, 0)
    N = len(x)
    y = x.reshape(N, -1) # flat 2D of all the vectors to differentiate

    pad = 0
    if pad_to_flat: # repeat the end values outward, smooth the joins, then restore the original in the middle
        pad = 100
        padded = np.concatenate((np.repeat(y[:1], pad, axis=0), y, np.repeat(y[-1:], pad, axis=0)))
        smoothed = utility.convolutional_smoother(padded, utility.uniform_kernel(pad//2), axis=0)
        smoothed[pad:pad+N] = y
        y = smoothed

    P = len(y); t = np.arange(P)[:, None]*dt # P for "potentially padded"
    if extension in ('detrend', 'odd') and cutoff_freq > 0: # the line through the endpoints; its derivative is a constant added back below
        slope = (y[-1] - y[0])/((P-1)*dt)
        y = y - slope*t # reassign so not in place if y is still a view on x
    else: slope = 0

    if extension == 'odd': y = np.concatenate((y, 2*y[-1] - y[-2:0:-1])) # reflect across endpoint
    elif extension == 'even': y = np.concatenate((y, y[::-1])) # mirror 

    M = len(y)
    k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))[:, None]

    # Smoothed signal, with the high wavenumbers zeroed out. Nyquist is at wavenumber M/2, and we're cutting off as a fraction of that.
    X = np.fft.fft(y, axis=0) * (np.abs(k) < cutoff_freq * M/2)
    x_hat = (np.real(np.fft.ifft(X, axis=0))[:P] + slope*t)[pad:pad+N] # de-extend, put the trend back, then crop the padding

    # Derivative = 90 deg phase shift
    if M % 2 == 0: k[M//2] = 0 # odd derivatives get the Nyquist element zeroed out, see https://pavelkomarov.com/spectral-derivatives/math.pdf section 3.1
    omega = 2*np.pi/(dt*M) # factor of 2pi/T turns wavenumbers into frequencies in radians/s
    dxdt_hat = (np.real(np.fft.ifft(1j * k * omega * X, axis=0))[:P] + slope)[pad:pad+N] # and the trend's constant slope

    return np.moveaxis(x_hat.reshape(x.shape), 0, axis), np.moveaxis(dxdt_hat.reshape(x.shape), 0, axis)


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
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values cause the interpolation to return NaN everywhere.")

    N = x.shape[axis]
    x = np.moveaxis(x, axis, 0) # bring axis of differentiation to front so each N repeats comprise vector
    plump = x.shape
    x_flattened = x.reshape(N, -1) # (N, M) matrix where each column is a vector along the original axis

    if np.isscalar(dt_or_t):
        t = np.arange(N)*dt_or_t
    else: # support variable step size for this function
        if N != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        if np.any(np.diff(dt_or_t) <= 0): raise ValueError("`dt_or_t` must be strictly increasing. Out-of-order or repeated sample locations make neighbor differences and windows meaningless.")
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


def waveletdiff(x, dt, wavelet='db8', level=None, threshold=1.0, axis=0, mode='symmetric'):
    """Smooth and differentiate noisy data in a wavelet basis.

    Three steps: (1) decompose x with the DWT and soft-threshold the detail coefficients to denoise (Donoho-Johnstone
    universal threshold), reconstructing a smoothed x_hat; (2) extend x_hat antisymmetrically so the periodic derivative
    operator stays accurate at the edges; (3) recover the wavelet scaling coefficients of x_hat and apply the analytic
    derivative of the wavelet basis.

    The derivative differentiates the basis functions themselves rather than finite-differencing the signal. PyWavelets
    treats the samples as finest-level scaling coefficients, so x_hat is the interpolant x(t) = sum_n a_n phi(t/dt - n)
    for the scaling function phi. Sampling x and its analytic derivative on the grid gives two convolutions against phi
    and phi' evaluated at *integers*,

        x_hat = Phi @ a     and     x' = Phi_prime @ a,

    so x' = Phi_prime @ Phi^-1 @ x_hat, exact for signals the basis can represent. Both matrices need only the values of
    phi and phi' at the integers, which evaluating the refinement relation phi(t) = sqrt2 sum_k h_k phi(2t - k) at t = p
    turns into a small eigenproblem: phi(p) is the eigenvalue-1 eigenvector of T[p,q] = sqrt2 h_{2p-q} and phi'(p) the
    eigenvalue-1/2 one, each normalized to reproduce constants and ramps. These are point values, not the connection
    coefficients int phi'(x) phi(x-l) dx that give the derivative's matrix elements in the basis. Representing d/dt
    exactly in a compactly supported wavelet basis is due to Beylkin (1992), who does it via those integrals on V_0;
    here the operator is instead assembled from point samples of phi and phi', which the circulant structure lets us
    apply as a transfer function. The eigenvector route to lattice values is standard (Daubechies 1992, ch. 6), and
    the wavelets themselves are Daubechies' (1988).

    References:
        G. Beylkin, "On the representation of operators in bases of compactly supported wavelets," SIAM J. Numer.
        Anal. 29(6):1716-1740, 1992.
        I. Daubechies, "Ten Lectures on Wavelets," CBMS-NSF Regional Conference Series in Applied Mathematics 61,
        SIAM, 1992.

    :param np.array x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: uniform time step between samples.
    :param str wavelet: PyWavelets wavelet name. Must have a differentiable scaling function, so smoother wavelets give
        better derivatives: 'db8' (default) and 'sym8' are best for noisy data; 'db4', 'sym4', and 'coif2' also work well.
    :param int level: decomposition depth. None (default) resolves to min(pywt.dwt_max_level(N, wavelet), 5) to avoid
        over-decomposing short signals.
    :param float threshold: soft-thresholding scale factor in [0, inf).
    :param int axis: axis along which to differentiate (default 0).
    :param str mode: PyWavelets signal extension mode for the denoising transform. 'symmetric' mirrors the signal, which
        performs better on aperiodic data than the wrapping modes ('periodization', 'periodic'), which cause discontinuity
        at beginning and end.
    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values spread through the DWT to make every coefficient NaN.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The DWT requires uniformly sampled data.")
    if pywt.Wavelet(wavelet).dec_len == 2: # 2-tap filter identifies Haar under any of its names: haar, db1, bior1.1, rbio1.1.
        raise ValueError("The Haar/db1 wavelet has a discontinuous (piecewise-constant) scaling function with no derivative,"
            " so it cannot be used to differentiate. Pick a smoother wavelet such as 'db4', 'sym4', or 'coif2'.")

    N = x.shape[axis]
    x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0)) # differentiation axis to front
    shape = x_work.shape                                   # remember it to restore the input's dimensionality
    x_flat = x_work.reshape(N, -1)                         # rest of the dims flattened into columns
    Ne = 3 * N - 2                                         # length after the antisymmetric extension in step 2
    if level is None: level = min(pywt.dwt_max_level(N, wavelet), 5)

    # Build the wavelet-basis derivative operator (depends only on the grid and wavelet). Sampling the refinement
    # relation phi(t) = sqrt2 sum_k h_k phi(2t - k) at integers makes phi(p) the eigenvalue-1 and phi'(p) the
    # eigenvalue-1/2 eigenvector of T[p,q] = sqrt2 h_{2p-q}.
    h = np.array(pywt.Wavelet(wavelet).rec_lo); h = h / h.sum() * np.sqrt(2) # refinement filter, integral of phi = 1
    L = len(h); p = np.arange(L)                            # phi is supported on the integers [0, L-1]
    shift = 2 * p[:, None] - p[None, :]
    T = np.where((shift >= 0) & (shift < L), np.sqrt(2) * h[np.clip(shift, 0, L - 1)], 0.0)
    evals, evecs = np.linalg.eig(T)
    phi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); phi /= phi.sum()           # sum_p phi(p) = 1
    dphi = np.real(evecs[:, np.argmin(np.abs(evals - 0.5))]); dphi /= np.dot(p, dphi)*-1 # sum_p p*phi'(p) = -1
    # Phi and Phi_prime hold circulant samples of phi and phi'/dt on the extended grid; both
    # share a common shift that cancels in Phi_prime @ Phi^-1, so the offset choice is cosmetic.
    # Circulant means each is a convolution by its own first column, so the DFT diagonalizes both, and the composition
    # Phi_prime @ Phi^-1 collapses to multiplication by one transfer function H = fft(phi')/fft(phi). Storing H is the
    # same operator as the two matrices, just in the basis where they are diagonal, so step 3 becomes an FFT pair
    # instead of a sparse solve on a (3N-2)-square matrix. Dividing is safe here because phi's integer samples are
    # concentrated and nearly flat in frequency: |fft(phi)| stays within a factor of ~2 of 1 for every wavelet allowed above.
    c_phi = np.zeros(Ne); c_phi[:L] = phi            # first column of Phi
    c_dphi = np.zeros(Ne); c_dphi[:L] = dphi/dt      # first column of Phi_prime
    H = (np.fft.fft(c_dphi) / np.fft.fft(c_phi))[:, np.newaxis]

    # 1. Denoise: DWT all columns at once, then soft-threshold the detail bands. The noise level is estimated
    # robustly per column from the finest details (coeffs[-1]).
    coeffs = pywt.wavedec(x_flat, wavelet, level=level, mode=mode, axis=0)
    sigma = np.maximum(utility.robust_data_scale(coeffs[-1], center=False, keepdims=True), 1e-10) # uncentered, as in Donoho-Johnstone, max to guard https://github.com/PyWavelets/pywt/issues/866, TODO remove later
    thresh = threshold * sigma * np.sqrt(2 * np.log(N))
    # coeffs[0] is the coarse approximation and doesn't need to be thresholded. At threshold 0 soft thresholding is the
    # identity, but pywt hands back NaN for coefficients that are exactly 0 (PyWavelets/pywt#866), so skip the no-op call.
    if threshold > 0: coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
    x_hat = pywt.waverec(coeffs, wavelet, mode=mode, axis=0)[:N]

    # 2. The derivative operator is periodic, but x_hat usually isn't. Extend it antisymmetrically (reflect
    # through each endpoint: x[-1-k] -> 2*x[0]-x[1+k]) so the periodic wrap is continuous in both value and slope,
    # which keeps the derivative accurate at the edges instead of spiking there. This is the odd-symmetry analog of
    # spectraldiff's even extension; a ramp extends to a ramp, so slopes survive exactly.
    left = 2 * x_hat[0] - x_hat[1:][::-1]
    right = 2 * x_hat[-1] - x_hat[:-1][::-1]
    x_ext = np.concatenate([left, x_hat, right], axis=0)  # length 3N-2, original at [N-1:2N-1]

    # 3. Differentiate the basis: multiplying by H both recovers the scaling coefficients a = Phi^-1 @ x_ext and
    # applies the analytic basis derivative dxdt = Phi_prime @ a, in one pass. Crop back to the original.
    dxdt_flat = np.real(np.fft.ifft(H * np.fft.fft(x_ext, axis=0), axis=0))[N - 1:2 * N - 1]

    x_hat = np.moveaxis(x_hat.reshape(shape), 0, axis)
    dxdt_hat = np.moveaxis(dxdt_flat.reshape(shape), 0, axis)
    return x_hat, dxdt_hat
