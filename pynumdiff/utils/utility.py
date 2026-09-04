"""Simple, short, reusable helper functions"""
from itertools import chain
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import huber
from scipy.stats import norm
from scipy.ndimage import convolve1d


def huber_const(M):
    """Scale that makes :code:`sum(huber())` interpolate :math:`\\sqrt{2}\\|\\cdot\\|_1` and :math:`\\frac{1}{2}\\|\\cdot\\|_2^2`,
    from https://jmlr.org/papers/volume14/aravkin13a/aravkin13a.pdf, with correction for missing sqrt. Here :code:`huber`
    refers to `scipy.special.huber <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html>`_.

    :param float M: Huber parameter, where the function turns from quadratic to linear
    :return: (float) -- appropriate scale factor to normalize the Huber function
    """
    a = 2*np.exp(-M**2 / 2)/M
    b = np.sqrt(2*np.pi)*(2*norm.cdf(M) - 1)
    return np.sqrt((2*a*(1 + M**2)/M**2 + b)/(a + b))


def robust_data_scale(x, axis=0, center=True, keepdims=False):
    """A robust stand-in for :code:`np.std`, normalized so it recovers :math:`\\sigma` on Gaussian data but cannot be
    inflated by a handful of outliers the way a standard deviation can.

    :param np.array[float] x: data whose scale to measure. NaNs are ignored rather than propagated.
    :param int axis: data dimension along which to measure
    :param bool center: whether to subtract the median first. :code:`False` gives the uncentered Donoho-Johnstone
        form, which is correct when the input is already known to be zero-mean, as for wavelet coefficients
    :param bool keepdims: whether to leave the reduced axis in place with length 1, so the result broadcasts
        back against the data it came from

    :return: **scale** (float or np.array[float]) -- robust scatter, in the units of :code:`x`
    """
    # These 3 lines are `scipy.stats.median_abs_deviation(x, scale='normal', nan_policy='omit')`, but faster on inputs < ~10^5 long
    med = np.median if not np.any(np.isnan(x)) else np.nanmedian # pay for skipping NaNs only when there are any to skip
    c = med(x, axis=axis, keepdims=True) if center else 0 # center=False is the uncentered form waveletdiff wants
    return med(np.abs(x - c), axis=axis, keepdims=keepdims)/0.6744897501960817 # divide by Phi^-1(3/4), i.e. scale='normal'

def robust_noise_scale(x, axis=0):
    """Estimate the standard deviation of the *noise* in :code:`x`, as opposed to the scale of :code:`x` itself.
    Second differencing annihilates constants and linear trends outright, and turns a quadratic into a constant that
    the centering inside :code:`robust_data_scale` then removes, so what survives is essentially noise. Its
    :math:`(1, -2, 1)` stencil inflates variance by :math:`(1^2 + (-2)^2 + 1^2)\\hat\\sigma^2 = 6\\hat\\sigma^2`,
    hence the :math:`\\sqrt{6}`.

    :param np.array[float] x: noisy data. NaNs are ignored rather than propagated.
    :param int axis: data dimension along which to measure

    :return: **sigma_hat** (float or np.array[float]) -- robust estimate of the noise standard deviation
    """
    return robust_data_scale(np.diff(x, 2, axis=axis), axis=axis)/np.sqrt(6)


def integrate_dxdt_hat(dxdt_hat, dt_or_t, axis=0):
    """Wrapper for scipy.integrate.cumulative_trapezoid. Use 0 as first value so lengths match, see #88.

    :param np.array[float] dxdt_hat: estimate derivative of timeseries
    :param float dt_or_t: step size if given as a scalar or a vector of sample locations
    :param int axis: data dimension along which to integrate

    :return: **x_hat** (np.array[float]) -- integral of dxdt_hat
    """
    return cumulative_trapezoid(dxdt_hat, initial=0, axis=axis)*dt_or_t if np.isscalar(dt_or_t) \
            else cumulative_trapezoid(dxdt_hat, x=dt_or_t, initial=0, axis=axis)

def estimate_integration_constant(x, x_hat, M=6, axis=0):
    """Integration leaves an unknown integration constant. This function finds a best fit integration
    constant to correct the DC of :code:`x_hat` (the integral of dxdt_hat) by optimizing
    :math:`\\min_c J(\\hat{x} + c - x)`, where :math:`J` is the Huber loss function or the :math:`\\ell_1`
    or :math:`\\ell_2` norm.

    :param np.array[float] x: timeseries of measurements
    :param np.array[float] x_hat: smoothed estimate of x
    :param float M: constant estimation is robustified with the Huber loss. :code:`M` here is in units of scaled
        mean absolute deviation of residuals, so scatter can be calculated and used to normalize without being
        thrown off by outliers. The default is intended to capture the idea of "six sigma": Assuming Gaussian
        :code:`x - xhat` errors, the portion of inliers beyond the Huber loss' transition is only about 1.97e-9.
    :param int axis: data dimension along which integration was performed

    :return: **integration constant** (float or np.array[float]) -- initial condition(s) to best align
             :math:`\\mathbf{\\hat{x}}` with :math:`\\mathbf{x}`
    """
    if M == float('inf'): # large M looks like l2 loss
        return np.mean(x - x_hat, axis=axis, keepdims=True) # Solves the l2 minimization, argmin_c ||x_hat + c - x||_2^2
    elif M < 1e-3: # small M looks like l1 loss, and Huber gets too flat to work well
        return np.median(x - x_hat, axis=axis, keepdims=True) # Solves the l1 minimization, argmin_c ||x_hat + c - x||_1
    else: # Huber case, no closed form, so use optimizer
        r = np.moveaxis(x - x_hat, axis, 0); s = r.shape; r = r.reshape(s[0], -1) # residual vectors, unrolled into a flat matrix
        sigma = robust_data_scale(r, axis=0, keepdims=True) # keep the axis so it broadcasts against the residuals
        sigma[sigma == 0] = 1 # avert divide-by-zero below; a σ == 0 entry means the corresponding vector in x - x_hat == some C everywhere
            # -> cost fn has argmin of exactly C in the corresponding entry of the c vector, regardless of scale -> choose scale 1 so
            # initial guess using median residuals captures these exactly, because optimization might otherwise ignore small offsets
        z = r/sigma # compute once to avoid rework during optimization. The residual is normalized rather than scaling M so the cumulative
            # (sum below) square (from inside huber) doesn't overflow. huber(M*σ, r) \propto huber(M, r/σ), so the argmin is unchanged. See #217
        # Solve for the constant w = c/σ in units of σ rather than data units to counteract normalization, which has made cost only 1/σ times
        # as sensitive to c. (SLSQP's fixed step and tolerance don't natively adapt to data scale.)
        c = sigma*minimize(lambda w: np.sum(huber(M, w - z)), # (x_hat + c - x)/σ = w - z
            np.median(z, axis=0), # seed with the l1 solution, exactly the Cs when residuals are constant
            method='SLSQP', jac=lambda w: np.sum(np.clip(w - z, -M, M), axis=0)).x # d/dw sum(huber) is sum(clip(., -M, M)) per index; provide for speed
        return np.moveaxis(c.reshape((1,) + s[1:]), 0, axis) # re-reorder axes, length 1 along the integration axis


def uniform_kernel(u):
    """A uniform boxcar of total integral 1

    :param int or np.array[float] u: a window size, or positions
    :return: **kernel** (np.array[float]) -- weights summing to 1
    """
    return np.ones(u)/u if np.isscalar(u) else np.ones_like(u, dtype=float)/len(u)

def gaussian_kernel(u):
    """A gaussian truncated at 2.7 sigma, leaving edges 2.6e-2 times smaller than the peak.

    :param int or np.array[float] u: a window size, or positions scaled to [-1, 1]
    :return: **kernel** (np.array[float]) -- weights summing to 1
    """
    if np.isscalar(u): u = np.linspace(-1, 1, u)
    ker = np.exp(-(2.7*np.asarray(u, dtype=float))**2/2)
    return ker/np.sum(ker) # always normalized

def friedrichs_kernel(u):
    """A bump function, natural support (-1, 1), going flat against zero at the edges. Inputs squeezed by 0.9 to
    make edges 1.4e-2 smaller than the peak, so window width is comparable with `gaussian_kernel`.

    :param int or np.array[float] u: a window size, or positions scaled to [-1, 1]
    :return: **kernel** (np.array[float]) -- weights summing to 1
    """
    if np.isscalar(u): u = np.linspace(-1, 1, u)
    ker = np.exp(-1/(1 - (0.9*u)**2))
    return ker/np.sum(ker) # always normalized

def convolutional_smoother(x, kernel, num_iterations=1, axis=0):
    """Perform smoothing by convolving x with a kernel.

    :param np.array[float] x: 1D data
    :param np.array[float] kernel: kernel to use in convolution
    :param int num_iterations: number of iterations, >=1
    :param int axis: data dimension along which convolution is performed
    
    :return: **x_hat** (np.array[float]) -- smoothed x
    """
    x_hat = x

    for i in range(num_iterations):
        x_hat = convolve1d(x_hat, kernel, axis=axis, mode='reflect') # 'reflect' pads the signal with repeats

    return x_hat


def slide_function(func, x, dt_or_t, kernel, window_size, stride, min_samples=1, pass_weights=False, **kwargs):
    """Slide a smoothing derivative function across a timeseries with specified window size, and
    combine the results according to kernel weights.

    :param callable func: name of the function to slide
    :param np.array[float] x: data to differentiate
    :param float or np.array[float] dt_or_t: constant step size (scalar) or array of sample locations (same length as x)
    :param callable kernel: kernel function on [-1, 1], internally stretech to window widths to weight samples
    :param int window_size: window width in units of the (average) spacing between samples, i.e. the (intended) number of
        samples in a window
    :param int stride: step size for slide (e.g. 1 means slide by 1 index location)
    :param int min_samples: fewest samples a window may hold before it is widened, only used with sparse sections of
        irregularly-spaced data, to ensure enough samples to fit
    :param bool pass_weights: whether weights should be passed to func via update to kwargs
    :param dict kwargs: passed to :code:`func`

    :return: - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    half_size = (window_size - 1)//2
    equispaced = np.isscalar(dt_or_t)
    if equispaced: kernel = kernel(window_size) # kernel is now a vector of samples rather than a function
    else: half_size *= (dt_or_t[-1] - dt_or_t[0])/(len(x) - 1) # multiply in the average gap

    x_hat = np.zeros(x.shape) # zeros rather than empty, because we'll add solutions as we go
    dxdt_hat = np.zeros(x.shape)
    weight_sum = np.zeros(x.shape)

    # iterate strided-out window midpoints, plus the last index when the final window doesn't reach the tail of the array.
    for midpoint in chain(range(0, len(x), stride), () if (len(x)-1) % stride <= (window_size - 1)//2 else (len(x)-1,)):
        # find which samples the window holds, taking care at the array's ends
        if equispaced: # half_size is in units of samples
            start = max(0, midpoint - half_size)
            end = min(len(x), midpoint + half_size + 1) # +1 because slicing is exclusive of end
            w = kernel[start - midpoint + half_size:end - midpoint + half_size]
            if end - start < window_size: w = w/np.sum(w) # renormalize the kernel slice if necessary
        else: # half_size is in units of average dt
            start = np.searchsorted(dt_or_t, dt_or_t[midpoint] - half_size, 'left')
            end = np.searchsorted(dt_or_t, dt_or_t[midpoint] + half_size, 'right')
            while end - start < min_samples and (start > 0 or end < len(x)): # not enough samples, so widen
                start = max(0, start - 1); end = min(len(x), end + 1)
            stretch = max(half_size, dt_or_t[midpoint] - dt_or_t[start], dt_or_t[end-1] - dt_or_t[midpoint]) # in case the while widened
            w = kernel((dt_or_t[start:end] - dt_or_t[midpoint])/stretch) # weights for irregularly-spaced samples

        window = slice(start, end)
        if pass_weights: kwargs['weights'] = w

        # Run the function on the window and add weighted results to cumulative answers. If not equispaced, pass times for window.
        x_window_hat, dxdt_window_hat = func(x[window], dt_or_t if equispaced else dt_or_t[window], **kwargs)
        x_hat[window] += w * x_window_hat
        dxdt_hat[window] += w * dxdt_window_hat
        weight_sum[window] += w # save sum of weights for normalization at the end

    return x_hat/weight_sum, dxdt_hat/weight_sum


def peakdet(x, delta, t=None):
    """Find peaks and valleys of 1D array. A point is considered a maximum peak if it has the maximal
    value, and was preceded (to the left) by a value lower by delta. Converted from MATLAB script at
    http://billauer.co.il/peakdet.html Eli Billauer, 3.4.05 (Explicitly not copyrighted). This function
    is released to the public domain; Any use is allowed.

    :param np.array[float] x: array for which to find peaks and valleys
    :param float delta: threshold for finding peaks and valleys. A point is considered a maximum peak
        if it has the maximal value, and was preceded (to the left) by a value lower by delta.
    :param np.array[float] t: optional domain points where data comes from, to make indices into locations

    :return: - **maxtab** -- indices or locations (column 1) and values (column 2) of maxima
             - **mintab** -- indices or locations (column 1) and values (column 2) of minima
    """
    maxtab = []
    mintab = []
    if t is None:
        t = np.arange(len(x))
    elif len(x) != len(t):
        raise ValueError('Input vectors x and t must have same length')
    if not (np.isscalar(delta) and delta > 0):
        raise ValueError('Input argument delta must be a positive scalar')

    mn, mx = np.inf, -1*np.inf
    mnpos, mxpos = np.nan, np.nan
    lookformax = True
    for i in np.arange(len(x)):
        this = x[i]
        if this > mx:
            mx = this
            mxpos = t[i]
        if this < mn:
            mn = this
            mnpos = t[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = t[i]
                lookformax = False # now searching for a min
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = t[i]
                lookformax = True # now searching for a max

    return np.array(maxtab), np.array(mintab)
