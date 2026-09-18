"""Methods based on fitting basis functions to data"""
import numpy as np
import pywt
from scipy import sparse
from scipy.linalg import convolution_matrix
from warnings import warn

from pynumdiff.utils import utility


def spectraldiff(x, dt, cutoff_freq, extension='odd', pad_to_flat=False, axis=0):
    """Take a derivative in the Fourier domain, with high frequency attentuation.

    The FFT treats data as periodic, so a signal whose ends do not meet has a discontinuity across the wrap, and
    Gibbs ringing contaminates the estimate. `extension` picks the remedy: 'detrend' subtracts the line through the
    endpoints, 'even' mirrors the signal, and 'odd' detrends and then reflects through the last endpoint.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param float cutoff_freq: high frequency cutoff as a multiple of the Nyquist frequency: Should be between 0
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

    N = x.shape[axis]
    x = np.moveaxis(x, axis, 0)
    x_flat = x.reshape(N, -1) # flat 2D of all the vectors to differentiate

    pad = 0
    if pad_to_flat: # repeat the end values outward, smooth the joins, then restore the original in the middle
        pad = 100
        padded = np.concatenate((np.repeat(x_flat[:1], pad, axis=0), x_flat, np.repeat(x_flat[-1:], pad, axis=0))) # [:1] rather than [0] so 2D shape
        smoothed = utility.convolutional_smoother(padded, utility.uniform_kernel(pad//2), axis=0)
        smoothed[pad:pad+N] = x_flat
        x_flat = smoothed

    P = len(x_flat) # P for "potentially padded"
    if extension in ('detrend', 'odd') and cutoff_freq > 0: # frequency guard so at 0 return 0, not the trend
        slope = (x_flat[-1] - x_flat[0])/(P-1) # technically missing a /dt, but the trend would just *dt it back out 
        trend = slope * np.arange(P)[:, np.newaxis] # the line through the endpoints, slope * time
        x_flat = x_flat - trend # reassign so not in place if x_flat is still a view on x
    else: slope = 0; trend = 0

    if extension == 'odd': x_flat = np.concatenate((x_flat, 2*x_flat[-1] - x_flat[-2:0:-1])) # reflect across endpoint
    elif extension == 'even': x_flat = np.concatenate((x_flat, x_flat[::-1])) # mirror

    M = len(x_flat) # could be N, or P, or 2N - 2, or 2P - 2
    k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))[:, np.newaxis]

    # Smoothed signal, with the high wavenumbers zeroed out. Nyquist is at wavenumber M/2, and we're cutting off as a fraction of that.
    X = np.fft.fft(x_flat, axis=0) * (np.abs(k) < cutoff_freq * M/2)
    x_hat = (np.real(np.fft.ifft(X, axis=0))[:P] + trend)[pad:pad+N] # de-extend, put the trend back, then crop the padding

    # Derivative = 90 deg phase shift
    if M % 2 == 0: k[M//2] = 0 # odd derivatives get the Nyquist element zeroed out, see https://pavelkomarov.com/spectral-derivatives/math.pdf section 3.1
    omega = 2*np.pi/(dt*M) # factor of 2pi/T turns wavenumbers into frequencies in radians/s
    dxdt_hat = (np.real(np.fft.ifft(1j * k * omega * X, axis=0))[:P] + slope/dt)[pad:pad+N] # add the trend's constant slope

    return np.moveaxis(x_hat.reshape(x.shape), 0, axis), np.moveaxis(dxdt_hat.reshape(x.shape), 0, axis)


def rbfdiff(x, dt_or_t, sigma=1, lmbd=0.01, axis=0):
    """Find smoothed function and derivative estimates by fitting noisy data with radial-basis-functions. Naively,
    fill a matrix with basis function samples and solve a linear inverse problem against the data, but truncate tiny
    values to make columns sparse. Each basis function "hill" is topped with a "tower" of height :code:`lmbd` to reach
    toward noisy data samples, and the final smoothed reconstruction is found by razing these and only keeping the hills.

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
    x_flat = x.reshape(N, -1) # (N, M) matrix where each column is a vector along the original axis

    if np.isscalar(dt_or_t):
        t = np.arange(N)*dt_or_t
    else: # support variable step size for this function
        if N != len(dt_or_t): raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        if np.any(np.diff(dt_or_t) <= 0): raise ValueError("`dt_or_t` must be strictly increasing. Out-of-order or repeated sample locations make neighbor differences and windows meaningless.")
        t = dt_or_t

    # For each vector along the axis of differentiation, the below does the approximate equivalent of this code,
    # but sparsely in O(N sigma^2), since the rbf falls off rapidly. Since A is Toeplitz for uniform spacing, it
    # could also be done via convolution in O(N sigma), but that would give up irregular grid support.
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
    alpha = sparse.linalg.spsolve(rbf_regularized, x_flat) # solve sparse system targeting the noisy data,
                                                           # can take matrix target, O(N sigma^2) for each vector
    x_hat_flat = rbf @ alpha # find samples of reconstructions using the smooth bases
    dxdt_hat_flat = drbfdt @ alpha

    return np.moveaxis(x_hat_flat.reshape(x.shape), 0, axis), np.moveaxis(dxdt_hat_flat.reshape(x.shape), 0, axis)


FIR = {} # wavelet -> (φ⁻¹, φ, φ') operators as FIR filters

def waveletdiff(x, dt, wavelet='db8', mode='symmetric', threshold=2.0, level=None, num_shifts=None, axis=0):
    """Put data in a scaling function basis, wavelet transform to separate noise into a wavelet basis, threshold,
    and differentiate in the reassembled scaling fuction basis.

    :param np.array x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: uniform step between samples
    :param str wavelet: type name from PyWavelets. Must have a differentiable scaling function.
    :param str mode: signal extension mode for the Discrete Wavelet Transform decomposition, :code:`pywt.wavedec`
    :param float threshold: scale factor in [0, inf) multiplying the Donoho-Johnstone universal hard threshold.
    :param int level: decomposition depth. None detects level automatically from the data, descending until a
        detail band's energy per coefficient is quadruple that of the finest-resolution details.
    :param int num_shifts: how many starting offsets to average for cycle-spinning. None uses :code:`2**level`,
        one full period of the cascade's alignment.
    :param int axis: data dimension along which to differentiate

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. Missing values spread through the DWT to make every coefficient NaN.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The DWT requires uniformly sampled data.")
    # Reject wavelet types where fft(phi) has entries at or near zero, where inverting becomes impossible or turns
    # into an obscenely long FIR filter. Also reject biorthogonal wavelets that don't preserve noise scale well, because
    # we need isometry-ish for automatic level selection and the equal-scale assumption behind universal Donoho-Johnstone.
    W = pywt.Wavelet(wavelet) # rejects continuous wavelets, which have no filter bank to build phi from
    if (W.name in ('db5', 'db14', 'db18', 'db27', 'db36', 'sym11', 'sym13') or W.dec_len == 2 # a 2-tap filter is Haar under one of several names
        or (not W.orthogonal and W.name not in ('bior4.4', 'bior6.8', 'rbio4.4', 'rbio6.8'))):
        raise ValueError(f"'{wavelet}' can't be used to differentiate. The scaling function must be continuous, have "
            "a spectrum that stays away from zero, and be orthonormal or nearly so to its own integer shifts.")

    N = x.shape[axis]
    x = np.moveaxis(x, axis, 0)
    x_flat = x.reshape(N, -1)

    # Step 0: Lock down adaptive parameters
    max_level = pywt.dwt_max_level(N, wavelet) # structural ceiling: how many halvings the filter still fits in
    if max_level < 1: raise ValueError(f"`x` is only {N} long along axis {axis}, too short for '{wavelet}'.")
    if level is None: # Descending and thresholding an additional (lower frequency) detail band helps if that band
        # is noise and hurts if it is signal. White noise should have the same scale in every band due to isometry
        # of orthonormal transform, so find where a robust scale starts to run into signal (grow).
        bands = pywt.wavedec(x_flat[:, 0], wavelet, level=max_level, mode=mode) # fully decompose a single vector to probe
        finest_scale = max(utility.robust_data_scale(bands[-1], center=False), 1e-12) # guard with a tiny value in case of super smooth data
        level = 1 # finest scale I ever met in my whole life ♪ ♫ ♬ Details are spiritually centered, like a guru, so feel their raw energy, their aura
        while level < max_level and utility.robust_data_scale(bands[-(1+level)], center=False) <= 2*finest_scale: level += 1
    if num_shifts is None: num_shifts = 2**level # the deepest level's functions span 2^level samples, so alignments repeat here

    # Step 1: Build the three operators for this wavelet
    if wavelet not in FIR:
        # Step i: Form T. pywt stores filters as possibly 0-padded lists, so trim. Use the *reconstruction* filter, because
        # correlation (+k indexer) disassembles and convolution (-k indexer) reassembles. Renormalize so 1 in eig (T).
        h = np.array(W.rec_lo); h = np.trim_zeros(h * (np.abs(h) > 1e-12)); h = h/h.sum()*np.sqrt(2)
        T = np.sqrt(2) * convolution_matrix(h, len(h))[::2] # T[n,k] = √2 h_{2n-k}
        l, V = np.linalg.eig(T) # φ is eigenvector corresponding to eigenvalue 1, and φ' is vec with val 1/2
        # Step ii: Get φ. Match against λ, because eig doesn't guarantee order; real() because T has other, complex λ
        phi = np.real(V[:, np.argmin(np.abs(l - 1))]) # normalize so scaling function has ∫ = 1
        phi = np.trim_zeros(phi * (np.abs(phi) > 1e-12), trim='b'); phi /= np.sum(phi) # 'b' to trim only backside
        # Step iii: Get φ'. Why dphi normalizer: Σₖ φ'[k]·f[n−k] with f[m] = a·m + b should come out to a. With Σₖ φ'[k] = 0:
        dphi = np.real(V[:, np.argmin(np.abs(l - 0.5))]) # Σₖ φ'[k]·(a(n−k) + b) = −a·Σₖ k φ'[k] -> −Σₖ k φ'[k] = 1
        dphi = np.trim_zeros(dphi * (np.abs(dphi) > 1e-12), trim='b')
        dphi -= dphi.mean(); dphi /= -np.dot(np.arange(len(dphi)), dphi) # subtract mean because Σₖ φ'[k] = 0
        # Step iv: Solve for truncated φ⁻¹. FFT in 512 buckets, large enough the inverse transform can spread out comfortably
        phi_inv = np.fft.fftshift(np.real(np.fft.ifft(1/np.fft.fft(phi, 512)))) # fftshift to put 0 in the middle, because non-causal filter
        r = np.max(np.abs(np.nonzero(np.abs(phi_inv) > 1e-12)[0] - 256)) # radius from center to furthest index carrying >threshold 
        phi_inv = phi_inv[256-r:256+r+1]; phi_inv /= np.sum(phi_inv) # renormalize post trim

        FIR[wavelet] = phi_inv, phi, dphi

    phi_inv, phi, dphi = FIR[wavelet] # pull filters from the cache
    r = (len(phi_inv) - 1)//2 # radius of the prefilter
    if r >= N - 1: raise ValueError(f"'{wavelet}' uses a {len(phi_inv)}-tap pre-filter, too wide to fit {N} samples along axis {axis}.")
    if num_shifts > N - 1 - r: num_shifts = N - 1 - r; warn(f"Only {num_shifts} of the requested alignments fit in {N} samples. Spinning fewer.")

    # Step 2: Pre-filter, samples -> finest scaling coefficients, because feeding the DWT raw samples is the "wavelet crime"
    # (Strang and Nguyen, 1996), harmless for denoising, but defines a curve in the basis that does not pass through the
    # samples. Odd extend the ends, reflecting across the average of 2 endpoints, to continue both value and slope.
    x_ext = np.concatenate([x_flat[0] + x_flat[1] - x_flat[r+num_shifts:1:-1], # Extend so the non-causal filter can fit with its center
        x_flat, x_flat[-1] + x_flat[-2] - x_flat[-3:-2-(r+num_shifts):-1]], axis=0) # at the data start + room for cycle-spinning
    c = np.stack([np.convolve(x_i, phi_inv, mode='valid') for x_i in x_ext.T], axis=1) # length N + 2(num_shifts - 1)

    # Step 3: Decompose, denoise by hard-thresholding detail bands (rather than soft because soft shrinks survivors
    # and biases answers), with noise scale estimated robustly, and reconstruct. Cycle-spin (Coifman and Donoho, 1995)
    c_hat = 0 # over the margin to average out offset sensitivity.
    djuth = threshold * np.sqrt(2 * np.log(N)) # Donoho-Johnstone universal threshold
    for shift in range(num_shifts):
        cddd = pywt.wavedec(c[shift:shift+num_shifts-1+N], wavelet, level=level, mode=mode, axis=0) # length N + num_shifts - 1
        sigma_hat = utility.robust_data_scale(cddd[-1], center=False, keepdims=True) # uncentered, as in Donoho-Johnstone
        cddd = [cddd[0]] + [pywt.threshold(d, sigma_hat * djuth, mode='hard') for d in cddd[1:]]
        c_hat += pywt.waverec(cddd, wavelet, mode=mode, axis=0)[num_shifts-1-shift:num_shifts-1+N-shift] # length N slice corresponding
    c_hat /= num_shifts                                                                         # to middle of c, between extensions

    # Step 4: Recover smooth estimate and derivative. Odd extend ĉ, because slopes are preserved in coefficient space too:
    # x[n] = Σₖ φ[k]·c[n-k] with affine coefficient sequence c[m] = a·m + b -> x[n] = Σₖ φ[k]·(a(n-k) + b) = b·Σₖφ[k] +
    # a·n·Σₖφ[k] - a·Σₖ k φ[k] = a·n + b − a·μ using Σφ = 1 and μ = Σ k φ(k). Same slope, intercept shifted.
    c_hat_ext = np.concatenate([2*c_hat[0] - c_hat[len(dphi)-1:0:-1], c_hat], axis=0) # filters are causal, so only extend one-sided; dphi is longer
    x_hat_flat = np.stack([np.convolve(c_hat_i, phi, mode='valid') for c_hat_i in c_hat_ext[-N-len(phi)+1:].T], axis=1)
    dxdt_hat_flat = np.stack([np.convolve(c_hat_i, dphi/dt, mode='valid') for c_hat_i in c_hat_ext.T], axis=1)

    return np.moveaxis(x_hat_flat.reshape(x.shape), 0, axis), np.moveaxis(dxdt_hat_flat.reshape(x.shape), 0, axis)
