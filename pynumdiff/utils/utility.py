import numpy as np
import copy
import sys
import scipy
import os

def get_filenames(path, contains, does_not_contain=['~', '.pyc']):
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    filelist = []
    for i, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append( os.path.join(path, filename) )
    return filelist

def is_odd(num):
    return num & 0x1


def isnotebook():
    '''
    Checks to see if the environment is an interactive notebook or not.

    Returns
    -------

    True or False
    
    '''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def hankel_matrix(x, num_delays, pad=False): # fixed delay step of 1
    '''
    
    Parameters
    ----------

    x : numpy array or matrix

    num_delays : Int
                 Number of times to 1-step shift data

    Returns
    -------

    m : Hankel Matrix
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
    '''
    m = copy.copy(x)
    for d in range(1,num_delays):
        xi = x[:, d:]
        xi = np.pad(xi, ((0,0), (0, x.shape[1]-xi.shape[1])), 'constant', constant_values=(0))
        #xi = xi + np.random.normal(0, 0.001, xi.shape[1])
        m = np.vstack((m, xi))
    if pad is False:
        return m[:,0:-1*num_delays]
    else:
        return m

def matrix_inv(X, max_sigma=1e-16):
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = Sigma**-1
    Sigma_inv[np.where(Sigma<max_sigma)[0]] = 0 # helps reduce instabilities
    return V.T.dot(np.diag(Sigma_inv)).dot(U.T)

def total_variation(x):
    if np.isnan(x).any():
        return np.nan
    x1 = np.ravel(x)[0:-1]
    x2 = np.ravel(x)[1:]
    return np.sum(np.abs(x2-x1))/len(x1) # mostly equivalent to cvxpy.tv(x2-x1).value

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
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
def finite_difference(x, dt, params=[], options=None):
    '''
    First order centered finite difference.
    
    Parameters
    ----------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step
    params  : (list)  [iterations] : (int, optional) number of iterations
                                                     ignored if 'iterate' not in options
    options : (dict) {'iterate'}   : (bool)          iterate the finite difference method (smooths the estimates)

    Returns
    -------
    x_hat : smoothed x
    dxdt_hat     : derivative of x

    '''

    # Calculate the finite difference
    dxdt_hat = np.diff(x)/dt
    # Pad the data
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1]))
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0)

    return x, dxdt_hat

    
# Trapazoidal integration, with interpolated final point so that the lengths match.
def integrate_dxdt_hat(dxdt_hat, dt):
    x = scipy.integrate.cumtrapz(dxdt_hat)
    first_value = x[0] - np.mean(dxdt_hat[0:1]) 
    x = np.hstack((first_value, x))*dt
    return x

# Optimization routine to estimate the integration constant.
def estimate_initial_condition(x, x_hat):
    def f(x0, *args):
        x, x_hat = args[0]
        error = np.linalg.norm(x- (x_hat+x0) )
        return error 
    result = scipy.optimize.minimize(f, [0], args=[x, x_hat], method='SLSQP')
    return result.x

### kernels
def __mean_kernel__(window_size):
    return np.ones(window_size)/window_size

def __gaussian_kernel__(window_size):
    sigma = window_size / 6.
    t = np.linspace(-2.7*sigma, 2.7*sigma, window_size)
    gaussian_func = lambda t, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(t**2)/(2*sigma**2))
    ker = gaussian_func(t, sigma)
    return ker / np.sum(ker)

def __friedrichs_kernel__(window_size):
    x = np.linspace(-0.999,0.999,window_size)
    ker = np.exp(-1/(1-x**2))
    return ker / np.sum(ker)