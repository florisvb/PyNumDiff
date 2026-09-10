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
        padded = np.concatenate((np.repeat(y[:1], pad, axis=0), y, np.repeat(y[-1:], pad, axis=0))) # y[:1] rather than y[0] so 2D shape
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


_OPERATORS = {} # (A^-1, phi', phi) taps per wavelet; independent of the data, dt, and N

def waveletdiff(x, dt, wavelet='db8', level=None, threshold=2.0, axis=0, mode='symmetric', num_shifts=None):
    """Smooth and differentiate noisy data in a wavelet basis.

    Three steps: (1) pre-filter the samples into finest-scale scaling coefficients, which is what the cascade is
    defined on; (2) decompose with the DWT and hard-threshold the detail coefficients to denoise (Donoho-Johnstone
    universal threshold), averaged over every alignment of the transform grid; (3) apply the analytic derivative of
    the basis to the denoised coefficients.

    The derivative differentiates the basis functions themselves rather than finite-differencing the signal. PyWavelets
    treats the samples as finest-level scaling coefficients, so x_hat is the interpolant x(t) = sum_n a_n phi(t/dt - n)
    for the scaling function phi. Sampling x and its analytic derivative on the grid gives two convolutions against phi
    and phi' evaluated at *integers*,

        x_hat = A @ a     and     x' = A_prime @ a,

    so x' = A_prime @ A^-1 @ x_hat, exact for signals the basis can represent. Both matrices need only the values of
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
    :param int level: decomposition depth. None (default) picks it from the data, descending while each new detail
        band still looks like noise and stopping at the first that stands above the noise floor.
    :param float threshold: hard-thresholding scale factor in [0, inf), multiplying the Donoho-Johnstone universal
        threshold. Hard and soft shrinkage do not share a scale: both zero the same coefficients, but hard keeps
        the survivors whole where soft shrinks them by lambda, so hard smooths less at equal multiplier and wants
        about twice the value. 2 is the median best across the benchmark signals, against 1.5 for soft.
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
    # Refused families, all for the same reason: fft(phi) dips to or through zero, so inverting it is either
    # impossible or spreads the operator over hundreds of taps and it stops being the local filter this method is
    # built around. The named seven are the orthogonal wavelets whose min|fft(phi)| falls under 0.3, found by
    # sweeping all 67 pywt offers; below that cut the operator runs 469 to 3113 taps against 35 to 199 above it.
    if (W.name in ('db5', 'db14', 'db18', 'db27', 'db36', 'sym11', 'sym13') or not W.orthogonal
            or W.dec_len == 2): # a 2-tap filter is Haar under any of its names: haar, db1, bior1.1, rbio1.1
        raise ValueError(f"'{wavelet}' can't differentiate here: it needs an orthogonal wavelet whose scaling "
            "function is differentiable and well conditioned to invert. Try 'db8', 'sym8', or 'coif2'.")

    N = x.shape[axis]
    x_work = np.ascontiguousarray(np.moveaxis(x, axis, 0)) # differentiation axis to front
    shape = x_work.shape                                   # remember it to restore the input's dimensionality
    x_flat = x_work.reshape(N, -1)                         # rest of the dims flattened into columns
    if N < 2: raise ValueError(f"`x` is only {N} long along axis {axis}; nothing to differentiate.")
    max_level = pywt.dwt_max_level(N, wavelet) # structural ceiling: how many halvings the filter still fits in
    if max_level < 1: raise ValueError(f"`x` is only {N} long along axis {axis}, too short for '{wavelet}'.")
    if level is None:
        # Let the data pick the depth. Descending one level thresholds one more detail band, which helps if that
        # band is noise and hurts if it is signal; white noise carries the same scale in every band, so keep
        # going while a band's robust scale stays near the finest band's and stop at the first one above it.
        # Within 6% of the depth chosen knowing the truth, where a fixed cap of 5 is 23% off. Flat over 1.5-3.
        bands = pywt.wavedec(x_flat[:, 0], wavelet, level=max_level, mode=mode)
        scales = [utility.robust_data_scale(c, center=False) for c in bands[:0:-1]] # finest band first
        level = max(1, next((j for j, sc in enumerate(scales, 1) if sc > 2.0*max(scales[0], 1e-10)), max_level+1) - 1)
    if num_shifts is None: num_shifts = 2**level # the cascade decimates by 2 per level, so alignments repeat here

    # Three operators, built once per wavelet on one grid so their delays compose. A and A_prime hold phi and
    # phi' on the integer grid, so A^-1 turns samples into the coefficients of the interpolant they define and
    # A_prime differentiates it; all are circulant, so each is a short filter and the data is never transformed.
    # Sampling phi(t) = sqrt2 sum_k h_k phi(2t-k) at integers makes phi(p) the eigenvalue-1 and phi'(p) the
    # eigenvalue-1/2 eigenvector of T[p,q] = sqrt2 h_{2p-q}.
    if wavelet not in _OPERATORS:
        h = np.array(W.rec_lo); h = h / h.sum() * np.sqrt(2)    # refinement filter, integral of phi = 1
        L = len(h); p = np.arange(L)                            # phi is supported on the integers [0, L-1]
        shift = 2 * p[:, None] - p[None, :]
        T = np.where((shift >= 0) & (shift < L), np.sqrt(2) * h[np.clip(shift, 0, L - 1)], 0.0)
        evals, evecs = np.linalg.eig(T)
        phi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); phi /= phi.sum()            # sum_p phi(p) = 1
        dphi = np.real(evecs[:, np.argmin(np.abs(evals - 0.5))]); dphi /= np.dot(p, dphi)*-1 # sum_p p*phi'(p) = -1
        # M must exceed twice the impulse response's reach so the inverse transform cannot wrap onto itself; 512
        # is clean for every wavelet the guard admits and dwarfs the longest of them (coif17, 102 taps).
        M = 512; c_phi = np.zeros(M); c_phi[:L] = phi; c_dphi = np.zeros(M); c_dphi[:L] = dphi
        def _taps(spectrum, derivative):
            k = np.fft.fftshift(np.real(np.fft.ifft(spectrum))); c = M//2 # lag 0 now in the middle
            # Trim at 1e-9: db8's A^-1 goes 57 taps to 43 at no measurable cost on noisy data. Looser still would
            # halve it again, but only by giving up exactness -- the moments restored just below pin constants and
            # ramps whatever the trim, while t^2 degrades from 1e-12 to 1e-7 between 1e-9 and 1e-6, and this
            # method's claim is that it differentiates the fitted curve exactly rather than differencing it.
            reach = max(1, int(np.max(np.nonzero(np.abs(k) > 1e-9*np.abs(k).max())[0]) - c))
            k = k[c-reach:c+reach+1]; m = np.arange(-reach, reach+1)
            if derivative: k = k - k.mean(); return k/-np.dot(m, k)
            return k/k.sum()
        _OPERATORS[wavelet] = (_taps(1/np.fft.fft(c_phi), False), _taps(np.fft.fft(c_dphi), True),
                               _taps(np.fft.fft(c_phi), False))

    inv_taps, dphi_taps, phi_taps = _OPERATORS[wavelet]
    reach = (len(inv_taps) - 1)//2
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
        sigma = np.maximum(utility.robust_data_scale(coeffs[-1], center=False, keepdims=True), 1e-10) # uncentered, as in Donoho-Johnstone, max to guard https://github.com/PyWavelets/pywt/issues/866, TODO remove later
        # coeffs[0] is the coarse approximation and isn't thresholded. At threshold 0 this is the identity, but
        # pywt hands back NaN for coefficients that are exactly 0 (PyWavelets/pywt#866), so skip the no-op call.
        if threshold > 0:
            thresh = threshold * sigma * np.sqrt(2 * np.log(N))
            coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='hard') for c in coeffs[1:]]
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
    dxdt_hat = np.moveaxis(_apply(a_hat, dphi_taps/dt).reshape(shape), 0, axis)
    x_hat = np.moveaxis(_apply(a_hat, phi_taps).reshape(shape), 0, axis)
    return x_hat, dxdt_hat
