import numpy as np 
import time
import copy
import math

import scipy.signal

# included code
from pynumdiff.finite_difference import first_order as finite_difference
from pynumdiff.utils import utility as utility
__friedrichs_kernel__ = utility.__friedrichs_kernel__
__gaussian_kernel__ = utility.__gaussian_kernel__
__mean_kernel__ = utility.__mean_kernel__

####################################################################################################################################################
# Smoothing methods
####################################################################################################################################################

def __median_smooth__(x, window_size):
    assert window_size%2 == 1 # is odd
    x_hat = scipy.signal.medfilt(x, window_size)
    return x_hat

### convolve kernels
def __convolutional_smoother__(x, kernel, iterations):
    '''
    Perform mean smoothing by convolving mean kernel with x
    followed by first order finite difference

    Inputs
    ------
    x          : (np.array of floats, 1xN) time series to differentiate
    kernel     : (np.array of floats, 1xwindow_size) kernel to use in convolution
    
    Parameters
    ----------
    iterations : (int) number of iterations, >=1

    Returns
    -------
    x_hat : smoothed x
    
    '''
    
    # pad
    x_hat = np.hstack((x[::-1], x, x[::-1]))
    for i in range(iterations):
        x_hat_f = np.convolve(x_hat, kernel, 'same')
        x_hat_b = np.convolve(x_hat[::-1], kernel, 'same')[::-1]

        w = np.arange(0,len(x_hat_f),1)
        w = w/np.max(w)
        
        x_hat = x_hat_f*w + x_hat_b*(1-w)

    return x_hat[len(x):len(x)*2]

####################################################################################################################################################
# Smoothing finite differences
####################################################################################################################################################

def mediandiff(x, dt, params, options={'iterate': False}):
    '''
    Perform median smoothing using scipy.signal.medfilt
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [filter_window_size] 
                     or if 'iterate' in options: [filter_window_size, num_iterations]
    options : (dict) {}
                     or {'iterate': True} to run multiple iterations of the smoother. 
                                          Note: iterate does nothing for median smoother.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        window_size, iterations = params
    else:
        iterations = 1
        if type(params) is list:
            window_size = params[0]
        else:
            window_size = params

    if not window_size%2:
        window_size += 1

    x_hat = x
    for i in range(iterations):
        x_hat = __median_smooth__(x_hat, window_size)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


def meandiff(x, dt, params, options={'iterate': False}):
    '''
    Perform mean smoothing by convolving mean kernel with x
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [filter_window_size] 
                     or if 'iterate' in options: [filter_window_size, num_iterations]
    options : (dict) {}
                     or {'iterate': True} to run multiple iterations of the smoother

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x
    
    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        window_size, iterations = params
    else:
        iterations = 1
        if type(params) is list:
            window_size = params[0]
        else:
            window_size = params

    kernel = __mean_kernel__(window_size)
    x_hat = __convolutional_smoother__(x, kernel, iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat

def gaussiandiff(x, dt, params, options={'iterate': False}):
    '''
    Perform gaussian smoothing by convolving gaussian kernel with x
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [filter_window_size] 
                     or if 'iterate' in options: [filter_window_size, num_iterations]
    options : (dict) {}
                     or {'iterate': True} to run multiple iterations of the smoother

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x
    
    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        window_size, iterations = params
    else:
        iterations = 1
        if type(params) is list:
            window_size = params[0]
        else:
            window_size = params

    kernel = __gaussian_kernel__(window_size)
    x_hat = __convolutional_smoother__(x, kernel, iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat

def friedrichsdiff(x, dt, params, options={'iterate': False}):
    '''
    Perform friedrichs smoothing by convolving friedrichs kernel with x
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [filter_window_size] 
                     or if 'iterate' in options: [filter_window_size, num_iterations]
    options : (dict) {}
                     or {'iterate': True} to run multiple iterations of the smoother

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x
    
    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        window_size, iterations = params
    else:
        iterations = 1
        if type(params) is list:
            window_size = params[0]
        else:
            window_size = params

    kernel = __friedrichs_kernel__(window_size)
    x_hat = __convolutional_smoother__(x, kernel, iterations)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat

def butterdiff(x, dt, params, options={'iterate': False}):
    '''
    Perform butterworth smoothing on x with scipy.signal.filtfilt
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [n, wn], n  = order of the filter; 
                              wn = Cutoff frequency. For a discrete timeseries, 
                                   the value is normalized to the range 0-1, 
                                   where 1 is the Nyquist frequency.
                     or if 'iterate' in options: [n, wn, num_iterations]
    options : (dict) {}
                     or {'iterate'  : True,      to run multiple iterations of the smoother
                         'padmethod': "pad"}     "pad" or "gust", see scipy.signal.filtfilt

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x
    
    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        n, wn, iterations = params
    else:
        iterations = 1
        n, wn = params

    b, a = scipy.signal.butter(n, wn)

    x_hat = x
    for i in range(iterations):
        if len(x) < 9:
            x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad", padlen=len(x)-1)
        else:
            x_hat = scipy.signal.filtfilt(b, a, x_hat, method="pad")

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    offset = np.mean(x) - np.mean(x_hat)
    x_hat = x_hat + offset

    return x_hat, dxdt_hat

def splinediff(x, dt, params, options={'iterate': False}):
    '''
    Perform spline smoothing on x with scipy.interpolate.UnivariateSpline
    followed by first order finite difference

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [k, s], k: Order of the spline. A kth order spline can be differentiated k times.
                             s: Positive smoothing factor used to choose the number of knots. 
                                Number of knots will be increased until the smoothing condition is satisfied: 
                                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
                     or if 'iterate' in options: [k, s, num_iterations]
    options : (dict) {}
                     or {'iterate': True} to run multiple iterations of the smoother

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x
    
    '''
    if 'iterate' in options.keys() and options['iterate'] is True:
        k, s, iterations = params
    else:
        iterations = 1
        k, s = params

    t = np.arange(0, len(x)*dt, dt)

    x_hat = x
    for i in range(iterations):
        spline = scipy.interpolate.UnivariateSpline(t, x_hat, k=k, s=s)
        x_hat = spline(t)

    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat
