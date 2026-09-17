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
        slope = (x_flat[-1] - x_flat[0])/((P-1)*dt) 
        trend = slope * np.arange(P)[:, np.newaxis]*dt # the line through the endpoints
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
    dxdt_hat = (np.real(np.fft.ifft(1j * k * omega * X, axis=0))[:P] + slope)[pad:pad+N] # add the trend's constant slope

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
        detail band's energy is quadruple that of the finest-resolution details, per coefficient.
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
    max_level = pywt.dwt_max_level(N, wavelet) # structural ceiling: how many halvings the filter still fits in
    if max_level < 1: raise ValueError(f"`x` is only {N} long along axis {axis}, too short for '{wavelet}'.")

    N = x.shape[axis]
    x = np.moveaxis(x, axis, 0)
    x_flat = x.reshape(N, -1)

    if level is None: # Descending and thresholding an additional (lower frequency) detail band helps if that band
        # is noise and hurts if it is signal. White noise should have the same scale in every band due to isometry
        # of orthonormal transform, so find where a robust scale starts to run into signal (grow).
        bands = pywt.wavedec(x_flat[:, 0], wavelet, level=max_level, mode=mode) # all of them
        # finest scale I ever met in my whole life ♪ ♫ ♬ Details are spiritually centered, like a guru, so feel their raw energy, also like a guru
        finest_scale = max(utility.robust_data_scale(bands[-1], center=False), 1e-10) # guard with a tiny value in case of super smooth data
        level = 1; while level < max_level and utility.robust_data_scale(bands[-(1+level)], center=False) <= 2*finest_scale: level += 1
    if num_shifts is None: num_shifts = 2**level # the deepest level's functions span 2^level samples, so alignments repeat here

    if wavelet not in FIR: # then build the three operators for this wavelet
        h = np.array(W.rec_lo) # pywt zero-pads biorthogonal filters out to a common length, so trim
        nz = np.nonzero(np.abs(h) > 1e-12)[0]
        h = h[nz[0]:nz[-1]+1]
        h = h/h.sum()*np.sqrt(2)        # pins sum(h) = sqrt2, which the eigenvalue ladder rests on
        L = len(h); p = np.arange(L)    # phi is supported on the integers [0, L-1]
        # T[n,k] = sqrt2 h_{2n-k}, with h zero outside [0, L-1]. Giving h zero margins wide enough for the whole
        # index range, -(L-1) to 2(L-1), lets the out-of-range entries land in them and needs no masking.
        hp = np.zeros(3*L); hp[L:2*L] = h
        # Sampling phi(t) = sqrt2 sum_k h_k phi(2t-k) at integers makes phi(p) the eigenvalue-1 and phi'(p) the
        # eigenvalue-1/2 eigenvector of T[p,q] = sqrt2 h_{2p-q}.
        T = np.sqrt(2) * hp[2*p[:, None] - p[None, :] + L]
        evals, evecs = np.linalg.eig(T)
        # The sum rules put 1 and 1/2 in the spectrum exactly, but not as the two largest: db2, sym2, coif1 and
        # dmey each carry a stray eigenvalue between them. So match on value rather than on rank.
        phi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); phi /= phi.sum()          # sum_p phi(p) = 1
        dphi = np.real(evecs[:, np.argmin(np.abs(evals - 0.5))]); dphi /= -np.dot(p, dphi) # sum_p p*phi'(p) = -1
        # phi and phi' are already finite tap sequences; only A^-1 needs a transform, because inverting the
        # Toeplitz A means inverting its symbol and coming back, and those taps have no closed form. M must
        # exceed twice their reach so the inverse transform cannot wrap onto itself; 512 is clean for every
        # wavelet the guard admits and dwarfs the longest of them (coif17, 102 taps).
        # Trim at 1e-12. A looser 1e-9 saves a dozen taps but costs real accuracy wherever fft(phi) is nearly
        # flat and so A^-1 is nearly a delta: it cuts sym8's A^-1 to 7 taps and doubles its error on noisy
        # data, and holds sym6, coif2 and the biorthogonals short of the degree p-1 exactness the sum rules
        # buy them. The moments restored in _norm pin constants and ramps at any trim; this buys the rest.
        M = 512; c = M//2
        k = np.fft.fftshift(np.real(np.fft.ifft(1/np.fft.fft(phi, M)))) # lag 0 now in the middle
        r_inv = max(1, int(np.max(np.nonzero(np.abs(k) > 1e-12*np.abs(k).max())[0]) - c))
        def _center(v): # a causal sequence becomes a kernel centred on lag 0, cut at its last live tap
            r = max(1, int(np.max(np.nonzero(np.abs(v) > 1e-12*np.abs(v).max())[0])))
            return np.concatenate([np.zeros(r), v[:r+1]]), r
        def _norm(k, r, derivative):
            if not derivative: return k/k.sum()         # reproduces constants
            k = k - k.mean()                            # kills constants,
            return k/-np.dot(np.arange(-r, r+1), k)     # returns 1 on a ramp
        FIR[wavelet] = (_norm(k[c-r_inv:c+r_inv+1], r_inv, False),
                               _norm(*_center(dphi), True), _norm(*_center(phi), False))

    phi_inv, dphi, phi = FIR[wavelet]
    reach = (len(phi_inv) - 1)//2
    if reach + num_shifts - 1 > N - 1: num_shifts = max(1, N - reach) # short signal, so spin fewer alignments
    margin = reach + num_shifts - 1
    if margin > N - 1: raise ValueError(f"'{wavelet}' needs a {len(inv_taps)}-tap pre-filter, too wide to fit "
        f"{N} samples along axis {axis}. Use a shorter wavelet, such as 'sym8' or 'coif1', or supply more data.")

    # 1. Pre-filter: samples -> the finest scaling coefficients, which is what the cascade is defined on. Feeding
    # it the samples instead is Strang and Nguyen's "wavelet crime" (1996, eq. 7.29), harmless for denoising but
    # not here, since the curve through the samples is not the curve the samples' own coefficients describe.
    # One extension covers this filter's reach and every cycle-spin offset below. The rule is antireflection,
    # x[-1-k] -> 2*x[0]-x[1+k]: the signal rotated 180 degrees about its endpoint, which is exact for an affine
    # sequence and so continues value and slope alike. It is not exact for curvature -- a parabola gets reflected
    # the wrong way -- which is the one place this boundary rule visibly loses. Step 3 uses the same rule for the
    # reason given there. The alternative, mirroring (x[-1-k] -> x[k]), wins on long noisy records because it
    # reuses real samples so the margin carries the right noise, and loses on short or clean ones because it
    # plants a kink; across N from 31 to 401 and noise over two orders it averages out slightly behind, and
    # switching between them on an estimate of which regime you are in recovers about 2% for a threshold nobody
    # can defend, so one rule serves both ends of the method.
    ext = np.concatenate([2*x_flat[0] - x_flat[margin:0:-1], x_flat, 2*x_flat[-1] - x_flat[-2:-margin-2:-1]], axis=0)
    a = np.stack([np.convolve(col, inv_taps, mode='valid') for col in ext.T], axis=1)

    # 2. Denoise in the basis: hard-threshold the detail bands, noise scale estimated robustly per column from
    # the finest of them. The transform decimates by two per level, so which coefficients pair up depends on
    # where the grid starts, and thresholding is nonlinear enough that the answer does too -- ringing appears at
    # one alignment and not the next. Averaging over `num_shifts` offsets removes that (Coifman and Donoho's
    # translation-invariant denoising, 1995); the offsets are windows slid over margin that already exists, so
    # nothing is invented per shift. Hard rather than soft, both Donoho-Johnstone at the same universal
    # threshold: soft shrinks every survivor by lambda, a bias proportional to the signal that shows up directly
    # as error correlation, where hard leaves survivors alone and measures better on both counts.
    window = len(a) - (num_shifts - 1)
    a_hat = 0
    for shift in range(num_shifts):
        coeffs = pywt.wavedec(a[shift:shift+window], wavelet, level=level, mode=mode, axis=0)
        sigma = utility.robust_data_scale(coeffs[-1], center=False, keepdims=True) # uncentered, as in Donoho-Johnstone
        coeffs = [coeffs[0]] + [pywt.threshold(c, threshold * sigma * np.sqrt(2 * np.log(N)), mode='hard') for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet, mode=mode, axis=0)[:window]
        a_hat = a_hat + rec[(num_shifts-1)-shift : window-shift]
    a_hat = a_hat/num_shifts # denoised coefficients, back to length N

    # 3. Differentiate the basis: convolve the coefficients with phi' for the derivative and phi for the smoothed
    # signal, both exact for anything the basis represents. This extends *coefficients* rather than samples, and
    # the same antireflection is right for them because A is Toeplitz: every entry depends only on n-m, so A
    # commutes with shifts and therefore carries affine sequences to affine sequences. Concretely, a_m = c + b*m
    # gives x_n = sum_m a_m phi(n-m) = c + b*(n - mu) using sum(phi) = 1 and the centroid mu = sum_p p*phi(p) --
    # same slope, intercept moved by mu -- and A^-1 carries it back the same way. So "locally affine at the
    # boundary" means the same thing in both domains, and a rule exact on affine sequences is exact in both.
    # The limitation transfers too: curvature is reflected wrongly here exactly as it would be on samples.
    # Mirroring here instead costs 18%. 'valid' crops the margin back off exactly.
    def _apply(seq, taps):
        half = len(taps)//2
        wide = np.concatenate([2*seq[0] - seq[half:0:-1], seq, 2*seq[-1] - seq[-2:-half-2:-1]], axis=0)
        return np.stack([np.convolve(col, taps, mode='valid') for col in wide.T], axis=1)
    return np.moveaxis(_apply(a_hat, phi).reshape(shape), 0, axis), np.moveaxis(_apply(a_hat, dphi/dt).reshape(shape), 0, axis)
