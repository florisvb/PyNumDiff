"""
All kinds of utilities
"""
import os
import sys
import copy
import numpy as np
import scipy


def get_filenames(path, contains, does_not_contain=('~', '.pyc')):
    """
    Create list of files found in given path that contain or do not contain certain strings.

    :param path: path in which to look for files
    :type path: string (directory path)

    :param contains: string that filenames must contain
    :type contains: string

    :param does_not_contain: list of strings that filenames must not contain, optional
    :type does_not_contain: list of strings

    :return: list of filenames
    :rtype: list of strings
    """
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    all_filelist.remove('')
    filelist = []
    for _, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append(os.path.join(path, filename))
    return filelist


def is_odd(num):
    """
    tell if it is an odd number

    :param num: (integer) the number we need to tell
    :return: (boolean) true or false
    """
    return num & 0x1


def isnotebook():
    """
    Checks to see if the environment is an interactive notebook or not.
    :return: True or False
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        # elif shell == 'TerminalInteractiveShell':
        #     return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def hankel_matrix(x, num_delays, pad=False):  # fixed delay step of 1
    """
    :param x: numpy array or matrix
    :param num_delays: int, number of times to 1-step shift data
    :param pad:
    :return: a Hankel Matrix m

            e.g.  if
                    x = [a, b, c, d, e] and num_delays = 3
            then with pad = False:
                    m = [['a', 'b', 'c'],
                         ['b', 'c', 'd'],
                         ['c', 'd', 'e']]
            or pad = True:
                    m = [['a', 'b', 'c', 'd', 'e'],
                         ['b', 'c', 'd', 'e',  0],
                         ['c', 'd', 'e',  0,   0]]
    """

    m = copy.copy(x)
    for d in range(1, num_delays):
        xi = x[:, d:]
        xi = np.pad(xi, ((0, 0), (0, x.shape[1]-xi.shape[1])), 'constant', constant_values=0)
        m = np.vstack((m, xi))
    if not pad:
        return m[:, 0:-1*num_delays]
    return m


def matrix_inv(X, max_sigma=1e-16):
    """
    Stable (pseudo) matrix inversion using singular value decomposition

    :param X: matrix to invert
    :type X: np.matrix or np.array

    :param max_sigma: smallest singular values to take into account. matrix will be truncated prior to inversion based on this value.
    :type max_sigma: float

    :return: matrix pseudo inverse
    :rtype: np.array or np.matrix
    """
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = Sigma**-1
    Sigma_inv[np.where(Sigma < max_sigma)[0]] = 0  # helps reduce instabilities
    return V.T.dot(np.diag(Sigma_inv)).dot(U.T)


def total_variation(x):
    """
    Calculate the total variation of an array

    :param x: timeseries
    :type x: np.array

    :return: total variation
    :rtype: float

    """
    if np.isnan(x).any():
        return np.nan
    x1 = np.ravel(x)[0:-1]
    x2 = np.ravel(x)[1:]
    return np.sum(np.abs(x2-x1))/len(x1)  # mostly equivalent to cvxpy.tv(x2-x1).value


def peakdet(v, delta, x=None):
    """
    Find peaks and valleys of 1D array. A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.

    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    :param v: array for which to find peaks and valleys
    :typpe v: np.array

    :param delta: threshold for finding peaks and valleys. A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.
    :type delta: float

    :return: tuple of min and max locations and values:
            - maxtab: array with locations (column 1) and values of maxima (column 2)
            - mintab: array with locations (column 1) and values of minima (column 2)
    :rtype: tuple -> (np.array, np.array)

    """
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -1*np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


# simple finite difference
def finite_difference(x, dt):
    """
    :param x: (np.array of floats, 1xN) time series to differentiate
    :param dt: (float) time step
    :param params: (list): (int, optional) number of iterations ignored if 'iterate' not in options
    :param options: (bool) iterate the finite difference method (smooths the estimates)
    :return: x_hat: smoothed x; dxdt_hat: derivative of x
    """
    # Calculate the finite difference
    dxdt_hat = np.diff(x)/dt
    # Pad the data
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1]))
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0)

    return x, dxdt_hat


# Trapazoidal integration, with interpolated final point so that the lengths match.
def integrate_dxdt_hat(dxdt_hat, dt):
    """
    Wrapper for scipy.integrate.cumulative_trapezoid to integrate dxdt_hat that ensures the integral has the same length

    :param dxdt_hat: estimate derivative of timeseries
    :type dxdt_hat: np.array

    :param dt: time step in seconds
    :type dt: float

    :return: integral of dxdt_hat
    :rtype: np.array
    """
    x = scipy.integrate.cumulative_trapezoid(dxdt_hat)
    first_value = x[0] - np.mean(dxdt_hat[0:1])
    x = np.hstack((first_value, x))*dt
    return x


# Optimization routine to estimate the integration constant.
def estimate_initial_condition(x, x_hat):
    """
    Integration leaves an unknown integration constant. This function finds a best fit integration constant given x, and x_hat (the integral of dxdt_hat)

    :param x: timeseries of measurements
    :type x: np.array

    :param x_hat: smoothed estiamte of x, for the purpose of this function this should have been determined by integrate_dxdt_hat
    :type x_hat: np.array

    :return: integration constant (i.e. initial condition) that best aligns x_hat with x
    :rtype: float
    """
    def f(x0, *args):
        x, x_hat = args[0]
        error = np.linalg.norm(x - (x_hat+x0))
        return error
    result = scipy.optimize.minimize(f, [0], args=[x, x_hat], method='SLSQP')
    return result.x


# kernels
def __mean_kernel__(window_size):
    """
    :param window_size:
    :return:
    """
    return np.ones(window_size)/window_size


def __gaussian_kernel__(window_size):
    """
    :param window_size:
    :return:
    """
    sigma = window_size / 6.
    t = np.linspace(-2.7*sigma, 2.7*sigma, window_size)
    gaussian_func = lambda t, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(t**2)/(2*sigma**2))
    ker = gaussian_func(t, sigma)
    return ker / np.sum(ker)


def __friedrichs_kernel__(window_size):
    """
    :param window_size:
    :return:
    """
    x = np.linspace(-0.999, 0.999, window_size)
    ker = np.exp(-1/(1-x**2))
    return ker / np.sum(ker)
