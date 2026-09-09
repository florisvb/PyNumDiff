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
    :param str extension: how to make the data periodic: :code:`None`, :code:`'even'`, :code:`'detrend'`, or :code:`'odd'`,
        where the latter also detrends before extension to avoid discontinuity. None is for genuinely periodic signals.
    :param bool pad_to_flat: if True, extend the edges with smoothed-in repeats of the end values, giving Gibbs ringing
        somewhere to go that gets discarded.
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


_DERIV_TAPS = {} # untrimmed derivative-operator taps per wavelet; independent of the data, dt, and N

# Orthogonal wavelets whose scaling function's spectrum dips near zero, making it ill-conditioned to invert: the
# derivative operator then spreads over 469 to 3113 taps, against 35 to 199 for the other 67 pywt offers, and stops
# being the local filter this method is built around. Found by sweeping min|fft(phi)| over all of them; the cut at
# 0.3 is where the trade stops paying, since a stricter floor rejects more families without shortening the worst
# operator any further.
_ILL_CONDITIONED = frozenset(('db5', 'db14', 'db18', 'db27', 'db36', 'sym11', 'sym13'))

def waveletdiff(x, dt, wavelet='db8', level=None, threshold=1.0, axis=0, mode='symmetric', num_shifts=None):
    """Smooth and differentiate noisy data in a wavelet basis.

    Three steps: (1) decompose x with the DWT and soft-threshold the detail coefficients to denoise (Donoho-Johnstone
    universal threshold), reconstructing a smoothed x_hat; (2) detrend x_hat and reflect it through its last endpoint so it is
    genuinely periodic and the derivative operator stays accurate at the edges; (3) recover the wavelet scaling coefficients of x_hat and apply the analytic
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
    :param int num_shifts: how many starting offsets of the transform grid to average the denoised estimate over.
        None (default) resolves to :code:`2**level`, one full period of the cascade's alignment ambiguity, which
        scales down on its own for short signals. 1 recovers a single transform. Cost is proportional.
    :param str mode: PyWavelets signal extension mode, governing the denoising transform only; the derivative's
        own edges are handled by reflection regardless. 'symmetric' mirrors the signal, which
        performs better on aperiodic data than the wrapping modes ('periodization', 'periodic'), which cause discontinuity
        at beginning and end.
    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values spread through the DWT to make every coefficient NaN.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The DWT requires uniformly sampled data.")
    W = pywt.Wavelet(wavelet) # itself rejects continuous wavelets, which have no filter bank to build phi from
    if W.name in _ILL_CONDITIONED: raise ValueError(f"'{wavelet}' is orthogonal but too ill-conditioned here: its "
        "scaling function's spectrum dips near zero, so inverting it spreads the derivative operator over hundreds "
        "of taps and it stops being local. Pick another wavelet, such as 'db8', 'sym8', 'coif2', or 'dmey'.")
    if not W.orthogonal or W.dec_len == 2: # a 2-tap filter is Haar under any of its names: haar, db1, bior1.1, rbio1.1
        raise ValueError(f"The '{wavelet}' family can't differentiate. That needs an orthogonal wavelet with a "
            "differentiable scaling function, but biorthogonal ones (bior, rbio) put zeros in fft(phi) for the "
            "derivative operator to divide by, and Haar's phi is a step. Try 'db8', 'sym8', or 'coif2'.")

    N = x.shape[axis]
    x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0)) # differentiation axis to front
    shape = x_work.shape                                   # remember it to restore the input's dimensionality
    x_flat = x_work.reshape(N, -1)                         # rest of the dims flattened into columns
    if N < 2: raise ValueError(f"`x` is only {N} long along axis {axis}; nothing to differentiate.")
    if level is None: level = min(pywt.dwt_max_level(N, wavelet), 5)
    if num_shifts is None: num_shifts = 2**level # the cascade decimates by 2 per level, so alignments repeat here

    # Build the derivative operator, once per wavelet. Phi and Phi_prime hold samples of phi and phi' on the
    # integer grid, so Phi^-1 turns samples into the coefficients of the interpolant they define and Phi_prime
    # differentiates it. Both are circulant, so the DFT diagonalizes them and the composition Phi_prime @ Phi^-1
    # collapses to one transfer function fft(phi')/fft(phi). Transforming that back reveals the operator is
    # *local* -- a few dozen taps -- so we keep the taps and convolve, and the data never gets transformed.
    # Sampling the refinement relation phi(t) = sqrt2 sum_k h_k phi(2t - k) at integers makes phi(p) the
    # eigenvalue-1 and phi'(p) the eigenvalue-1/2 eigenvector of T[p,q] = sqrt2 h_{2p-q}.
    if wavelet not in _DERIV_TAPS:
        h = np.array(W.rec_lo); h = h / h.sum() * np.sqrt(2)    # refinement filter, integral of phi = 1
        L = len(h); p = np.arange(L)                            # phi is supported on the integers [0, L-1]
        shift = 2 * p[:, None] - p[None, :]
        T = np.where((shift >= 0) & (shift < L), np.sqrt(2) * h[np.clip(shift, 0, L - 1)], 0.0)
        evals, evecs = np.linalg.eig(T)
        phi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); phi /= phi.sum()            # sum_p phi(p) = 1
        dphi = np.real(evecs[:, np.argmin(np.abs(evals - 0.5))]); dphi /= np.dot(p, dphi)*-1 # sum_p p*phi'(p) = -1
        # M must exceed twice the impulse response's reach, so the inverse transform cannot wrap onto itself.
        # Among the wavelets the guard admits, the widest reach is db35's 99, but roundoff at the trim level
        # can push db23 and db32 to the window edge at 256, so 512 is the size that is clean for all of them,
        # and it dwarfs the longest admitted filter (coif17, 102 taps).
        M = 512; c_phi = np.zeros(M); c_phi[:L] = phi; c_dphi = np.zeros(M); c_dphi[:L] = dphi
        k = np.fft.fftshift(np.real(np.fft.ifft(np.fft.fft(c_dphi) / np.fft.fft(c_phi)))) # lag 0 now in the middle
        c = M//2; reach = int(np.max(np.nonzero(np.abs(k) > 1e-12*np.abs(k).max())[0]) - c) # trim the tiny tail
        # Trimming perturbs two moments that must hold exactly: a derivative kills constants and returns 1 on a
        # ramp, i.e. sum(k) = 0 and sum(m*k) = -1 for lag m. Restore both here, once, or constants come back at
        # 1e-10 instead of 1e-14.
        k = k[c-reach:c+reach+1]; m = np.arange(-reach, reach+1)
        k = k - k.mean(); _DERIV_TAPS[wavelet] = k/-np.dot(m, k)

    taps = _DERIV_TAPS[wavelet]
    half = len(taps)//2
    if half > N - 1: # a short signal cannot reflect the filter's full reach, so shorten the filter to suit
        half = N - 1; taps = taps[len(taps)//2 - half : len(taps)//2 + half + 1]
        # Shortening costs little on its own, but it moves sum(m*k) away from -1, and rescaling by a moment that
        # has drifted spreads the loss across every tap as a gain error -- 0.1% of truncation became 24% that way.
        # So the moment doubles as the check on whether the shortening was harmless.
        m = np.arange(-half, half + 1)
        taps = taps - taps.mean(); ramp = -np.dot(m, taps)
        if abs(ramp - 1) > 1e-3: raise ValueError(f"'{wavelet}' needs a {len(_DERIV_TAPS[wavelet])}-tap derivative "
            f"operator, too wide to fit {N} samples along axis {axis}. Use a shorter wavelet, such as 'sym8' or "
            f"'coif1', or supply more data.")
        taps = taps/ramp
    taps = taps/dt

    # 1. Denoise: DWT all columns at once, then soft-threshold the detail bands. The noise level is estimated
    # robustly per column from the finest details (coeffs[-1]). The transform decimates by two at every level, so
    # which samples get paired depends on where the grid starts, and thresholding is nonlinear enough that the
    # answer does too -- ringing appears around sharp features at one alignment and not the next. Averaging over
    # `num_shifts` starting offsets removes that arbitrariness (Coifman and Donoho's translation-invariant
    # denoising, 1995), the same ensembling the sliding-window methods get for free. Offsets are made by
    # prepending reflected samples rather than rotating, which would wrap the far end of an aperiodic signal
    # around onto the near one.
    x_hat = 0
    for shift in range(num_shifts):
        shifted = np.concatenate([2*x_flat[0] - x_flat[shift:0:-1], x_flat]) if shift else x_flat
        coeffs = pywt.wavedec(shifted, wavelet, level=level, mode=mode, axis=0)
        sigma = np.maximum(utility.robust_data_scale(coeffs[-1], center=False, keepdims=True), 1e-10) # uncentered, as in Donoho-Johnstone, max to guard https://github.com/PyWavelets/pywt/issues/866, TODO remove later
        thresh = threshold * sigma * np.sqrt(2 * np.log(N))
        # coeffs[0] is the coarse approximation and doesn't need to be thresholded. At threshold 0 soft thresholding is the
        # identity, but pywt hands back NaN for coefficients that are exactly 0 (PyWavelets/pywt#866), so skip the no-op call.
        if threshold > 0: coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
        x_hat = x_hat + pywt.waverec(coeffs, wavelet, mode=mode, axis=0)[shift:shift+N]
    x_hat = x_hat/num_shifts

    # 2. The filter reaches `half` samples past each end, so extend by that much and no more. Reflect through the
    # endpoints, x[-1-k] -> 2*x[0]-x[1+k], which continues value and slope alike -- a ramp extends to a ramp -- so
    # the edge estimates stay accurate instead of spiking. Being local, the operator needs no periodicity and no
    # detrending; only these few samples.
    left = 2 * x_hat[0] - x_hat[half:0:-1]
    right = 2 * x_hat[-1] - x_hat[-2:-half-2:-1]
    x_ext = np.concatenate([left, x_hat, right], axis=0)

    # 3. Differentiate the basis: convolving with `taps` recovers the scaling coefficients a = Phi^-1 @ x_ext and
    # applies the analytic basis derivative dxdt = Phi_prime @ a in one pass. Crop the reflected margin back off.
    # 'valid' keeps only the outputs whose kernel sat entirely inside x_ext, which is exactly the original N once
    # the reflected margin is accounted for -- no boundary rule to choose and no wasted outputs to slice away.
    dxdt_flat = np.stack([np.convolve(col, taps, mode='valid') for col in x_ext.T], axis=1)

    x_hat = np.moveaxis(x_hat.reshape(shape), 0, axis)
    dxdt_hat = np.moveaxis(dxdt_flat.reshape(shape), 0, axis)
    return x_hat, dxdt_hat
