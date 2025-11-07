import os, sys, copy
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation


def hankel_matrix(x, num_delays, pad=False): # fixed delay step of 1
    """Unused throughout the repo

    :param np.array[float] x: data
    :param int num_delays: number of times to 1-step shift data
    :param bool pad: if True, return width is len(x), else width is len(x) - num_delays + 1

    :return: a Hankel Matrix `m`. For example, if `x = [a, b, c, d, e]` and `num_delays = 3`:\n
        With `pad = False`::\n
            m = [[a, b, c],
                 [b, c, d],
                 [c, d, e]]\n
        With `pad = True`::\n
            m = [[a, b, c, d, e],
                 [b, c, d, e, 0],
                 [c, d, e, 0, 0]]
    """
    m = x.copy()
    for d in range(1, num_delays):
        xi = x[d:]
        xi = np.pad(xi, (0, len(x)-len(xi)), 'constant', constant_values=0)
        m = np.vstack((m, xi))
    if not pad:
        return m[:, 0:-1*num_delays+1]
    return m


def matrix_inv(X, max_sigma=1e-16):
    """Stable (pseudo) matrix inversion using singular value decomposition. Unused throughout the repo.

    :param np.array X: matrix to invert
    :param float max_sigma: smallest singular values to take into account. matrix will be truncated prior to inversion based on this value.

    :return: (np.array) -- pseudo inverse
    """
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = Sigma**-1
    Sigma_inv[np.where(Sigma < max_sigma)[0]] = 0  # helps reduce instabilities
    return V.T.dot(np.diag(Sigma_inv)).dot(U.T)


def peakdet(x, delta, t=None):
    """Find peaks and valleys of 1D array. A point is considered a maximum peak if it has the maximal
    value, and was preceded (to the left) by a value lower by delta. Converted from MATLAB script at
    http://billauer.co.il/peakdet.html Eli Billauer, 3.4.05 (Explicitly not copyrighted). This function
    is released to the public domain; Any use is allowed.

    :param np.array[float] x: array for which to find peaks and valleys
    :param float delta: threshold for finding peaks and valleys. A point is considered a maximum peak
        if it has the maximal value, and was preceded (to the left) by a value lower by delta.
    :param np.array[float] t: optional domain points where data comes from, to make indices into locations

    :return: tuple[np.array, np.array] of\n
             - **maxtab** -- indices or locations (column 1) and values (column 2) of maxima
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


def huber(x, M):
    """Huber loss function, for outlier-robust applications, `see here <https://www.cvxpy.org/api_reference/cvxpy.atoms.elementwise.html#huber>`_"""
    absx = np.abs(x)
    return np.where(absx <= M, 0.5*x**2, M*(absx - 0.5*M))


def integrate_dxdt_hat(dxdt_hat, _t):
    """Wrapper for scipy.integrate.cumulative_trapezoid. Use 0 as first value so lengths match, see #88.

    :param np.array[float] dxdt_hat: estimate derivative of timeseries
    :param float _t: step size if given as a scalar or a vector of sample locations

    :return: **x_hat** (np.array[float]) -- integral of dxdt_hat
    """
    return cumulative_trapezoid(dxdt_hat, initial=0)*_t if np.isscalar(_t) \
            else cumulative_trapezoid(dxdt_hat, x=_t, initial=0)


def estimate_integration_constant(x, x_hat, M=6):
    """Integration leaves an unknown integration constant. This function finds a best fit integration
    constant given x and x_hat (the integral of dxdt_hat) by optimizing :math:`\\min_c ||x - \\hat{x} + c||_2`.

    :param np.array[float] x: timeseries of measurements
    :param np.array[float] x_hat: smoothed estimate of x
    :param float M: robustifies constant estimation using Huber loss. The default is intended to capture the idea
        of "six sigma": Assuming Gaussian inliers and M in units of standard deviation, the portion of inliers
        beyond the Huber loss' transition is only about 1.97e-9. M here is in units of scaled mean absolute deviation,
        so scatter can be calculated and used to normalize data without being thrown off by outliers.

    :return: **integration constant** (float) -- initial condition that best aligns x_hat with x
    """
    if M == float('inf'): # calculates the constant to be mean(diff(x, x_hat)), equivalent to argmin_{x0} ||x_hat + x0 - x||_2^2
        return np.mean(x - x_hat) # Solves the L2 distance minimization
    elif M < 0.1: # small M looks like L1 loss, and Huber gets too flat to work well
        return np.median(x - x_hat) # Solves the L1 distance minimization
    else:
        sigma = median_abs_deviation(x - x_hat, scale='normal') # M is in units of this robust scatter metric
        if sigma < 1e-6: sigma = 1 # guard against divide by zero
        return minimize(lambda x0: np.mean(huber((x - (x_hat+x0))/sigma, M)), # fn to minimize in 1st argument
            0, method='SLSQP').x[0] # result is a vector, even if initial guess is just a scalar


def mean_kernel(window_size):
    """A uniform boxcar of total integral 1"""
    return np.ones(window_size)/window_size

def gaussian_kernel(window_size):
    """A truncated gaussian"""
    sigma = window_size / 6.
    t = np.linspace(-2.7*sigma, 2.7*sigma, window_size)
    ker = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(t**2)/(2*sigma**2)) # gaussian function itself
    return ker / np.sum(ker)

def friedrichs_kernel(window_size):
    """A bump function"""
    x = np.linspace(-0.999, 0.999, window_size)
    ker = np.exp(-1/(1-x**2))
    return ker / np.sum(ker)


def convolutional_smoother(x, kernel, num_iterations=1):
    """Perform smoothing by convolving x with a kernel.

    :param np.array[float] x: 1D data
    :param np.array[float] kernel: kernel to use in convolution
    :param int num_iterations: number of iterations, >=1
    
    :return: **x_hat** (np.array[float]) -- smoothed x
    """
    pad_width = len(kernel)//2
    x_hat = x

    for i in range(num_iterations):
        x_padded = np.pad(x_hat, pad_width, mode='symmetric') # pad with repetition of the edges
        x_hat = np.convolve(x_padded, kernel, 'valid')[:len(x)] # 'valid' slices out only full-overlap spots

    return x_hat


def slide_function(func, x, dt, kernel, *args, stride=1, pass_weights=False, **kwargs):
    """Slide a smoothing derivative function across a timeseries with specified window size, and
    combine the results according to kernel weights.

    :param callable func: name of the function to slide
    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param np.array[float] kernel: values to weight the sliding window
    :param list args: passed to func
    :param int stride: step size for slide (e.g. 1 means slide by 1 step)
    :param bool pass_weights: whether weights should be passed to func via update to kwargs
    :param dict kwargs: passed to func

    :return: tuple[np.array, np.array] of\n
             - **x_hat** -- estimated (smoothed) x
             - **dxdt_hat** -- estimated derivative of x
    """
    if len(kernel) % 2 == 0: raise ValueError("Kernel window size should be odd.")
    half_window_size = (len(kernel) - 1)//2 # int because len(kernel) is always odd

    x_hat = np.zeros(x.shape)
    dxdt_hat = np.zeros(x.shape)
    weight_sum = np.zeros(x.shape)

    for i,midpoint in enumerate(range(0, len(x), stride)): # iterate window midpoints
        # find where to index data and kernel, taking care at edges
        start = max(0, midpoint - half_window_size)
        end = min(len(x), midpoint + half_window_size + 1) # +1 because slicing is exclusive of end
        window = slice(start, end)

        kstart = max(0, half_window_size - midpoint)
        kend = kstart + (end - start)
        kslice = slice(kstart, kend)

        w = kernel if (end-start) == len(kernel) else kernel[kslice]/np.sum(kernel[kslice])
        if pass_weights: kwargs['weights'] = w

        # run the function on the window and add weighted results to cumulative answers
        x_window_hat, dxdt_window_hat = func(x[window], dt, *args, **kwargs)
        x_hat[window] += w * x_window_hat
        dxdt_hat[window] += w * dxdt_window_hat
        weight_sum[window] += w # save sum of weights for normalization at the end

    return x_hat/weight_sum, dxdt_hat/weight_sum
