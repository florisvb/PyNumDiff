import os, sys, copy, scipy
import numpy as np


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


# Trapazoidal integration, with 0 first value so that the lengths match. See #88.
def integrate_dxdt_hat(dxdt_hat, dt):
    """Wrapper for scipy.integrate.cumulative_trapezoid to integrate dxdt_hat that ensures the integral has the same length

    :param np.array[float] dxdt_hat: estimate derivative of timeseries
    :param float dt: time step in seconds

    :return: **x_hat** (np.array[float]) -- integral of dxdt_hat
    """
    return np.hstack((0, scipy.integrate.cumulative_trapezoid(dxdt_hat)))*dt


# Optimization routine to estimate the integration constant.
def estimate_initial_condition(x, x_hat):
    """Integration leaves an unknown integration constant. This function finds a best fit integration constant given x and
    x_hat (the integral of dxdt_hat) by optimizing :math:`\\min_c ||x - \\hat{x} + c||_2`.

    :param np.array[float] x: timeseries of measurements
    :param np.array[float] x_hat: smoothed estiamte of x, for the purpose of this function this should have been determined
        by integrate_dxdt_hat

    :return: **integration constant** (float) -- initial condition that best aligns x_hat with x
    """
    return scipy.optimize.minimize(lambda x0, x, xhat: np.linalg.norm(x - (x_hat+x0)), # fn to minimize in 1st argument
        0, args=(x, x_hat), method='SLSQP').x[0] # result is a vector, even if initial guess is just a scalar


# kernels
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

def convolutional_smoother(x, kernel, iterations=1):
    """Perform smoothing by convolving x with a kernel.

    :param np.array[float] x: 1D data
    :param np.array[float] kernel: kernel to use in convolution
    :param int iterations: number of iterations, >=1
    
    :return: **x_hat** (np.array[float]) -- smoothed x
    """
    x_hat = np.hstack((x[::-1], x, x[::-1])) # pad
    w = np.linspace(0, 1, len(x_hat)) # weights

    for _ in range(iterations):
        x_hat_f = np.convolve(x_hat, kernel, 'same')
        x_hat_b = np.convolve(x_hat[::-1], kernel, 'same')[::-1]
        
        x_hat = x_hat_f*w + x_hat_b*(1-w)

    return x_hat[len(x):len(x)*2]


def slide_function(func, x, dt, kernel, *args, stride=1, pass_weights=False, **kwargs):
    """Slide a smoothing derivative function across a timeseries with specified window size.

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

    weights = np.zeros((int(np.ceil(len(x)/stride)), len(x)))
    x_hats = np.zeros(weights.shape)
    dxdt_hats = np.zeros(weights.shape)

    for i,midpoint in enumerate(range(0, len(x), stride)): # iterate window midpoints
        # find where to index data and kernel, taking care at edges
        window = slice(max(0, midpoint - half_window_size),
                        min(len(x), midpoint + half_window_size + 1)) # +1 because slicing works [,)
        kslice = slice(max(0, half_window_size - midpoint),
                        min(len(kernel), len(kernel) - (midpoint + half_window_size + 1 - len(x))))

        # weights need to be renormalized if running off an edge
        weights[i, window] = kernel if kslice.stop - kslice.stop == len(kernel) else kernel[kslice]/np.sum(kernel[kslice])
        if pass_weights: kwargs['weights'] = weights[i, window]

        # run the function on the window and save results
        x_hats[i,window], dxdt_hats[i,window] = func(x[window], dt, *args, **kwargs)

    weights /= weights.sum(axis=0, keepdims=True) # normalize the weights
    x_hat = np.sum(weights*x_hats, axis=0)
    dxdt_hat = np.sum(weights*dxdt_hats, axis=0)

    return x_hat, dxdt_hat
