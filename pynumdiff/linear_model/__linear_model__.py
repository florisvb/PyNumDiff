import numpy as np 
import time
import copy
import math
import scipy
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# included modules
from pynumdiff.finite_difference import first_order as finite_difference 
import pynumdiff.smooth_finite_difference
from pynumdiff.utils import utility as utility
from pynumdiff import smooth_finite_difference
__friedrichs_kernel__ = utility.__friedrichs_kernel__
__gaussian_kernel__ = utility.__gaussian_kernel__
KERNELS = {'friedrichs': __friedrichs_kernel__,
           'gaussian': __gaussian_kernel__}

# optional packages
warned = False
try:
    import pychebfun
except:
    logging.info('Import Error\nCould not import pychebfun.\nInstall pychebfun (https://github.com/pychebfun/pychebfun/) to use chebfun derivatives.\n')
    warned = True
try:
    import pydmd.dmdc
except:
    logging.info('Import Error\nCould not import pydmd.\nInstall pydmd (florisvb fork: https://github.com/florisvb/PyDMD) to use dmd derivatives.\n')
    warned = True
try:
    import cvxpy
except:
    logging.info('Import Error\nCould not import cvxpy.\nInstall cvxpy (http://www.cvxpy.org/install/index.html) to use lineardiff.\nRecommended solver: MOSEK, free academic license available: https://www.mosek.com/products/academic-licenses/ \n')
    warned = True

if warned == True:
    logging.info('Import Error\nDespite these import errors, you can still use many of the methods without additional installations.\n')

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def __slide_function__(func, x, dt, params, window_size, step_size, kernel_name):
    '''
    Slide a smoothing derivative function across a timeseries with specified window size.
    
    Inputs
    ------
    func    : (function) name of the function to slide
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) see func for requirements
    window_size : (int) size of the sliding window
    step_size   : (int) step size for slide (e.g. 1 means slide by 1 step) 
    kernel_name : (string) name of the smoothing kernel 
                  (e.g. 'friedrichs' or 'gaussian')

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    # get smoothing kernel
    if not window_size%2: # then make odd
        window_size += 1
    ker = KERNELS[kernel_name](window_size)

    x_hat_list = []
    dxdt_hat_list = []
    weight_list = []

    for p in range(0, len(x), step_size):
        # deal with end points
        start = p- int((window_size-1)/2)
        end = p+ int((window_size-1)/2)+1
        
        ker_start = 0
        ker_end = window_size
        ker_middle = int((window_size-1)/2)
        
        if start < 0:
            ker_start = np.abs(start)
            ker_middle = ker_middle - np.abs(start)
            start = 0
        if end > len(x):
            ker_end = window_size - (end-len(x))
            end = len(x)

        # weights
        w = ker[ker_start:ker_end]
        w = w/np.sum(w)

        # run the function on the window
        _x = x[start:end]
        x_hat, dxdt_hat = func(_x, dt, params, options={'weights': w})

        # stack results
        z_x_hat = np.zeros([len(x)])
        z_x_hat[start:end] = x_hat
        x_hat_list.append(z_x_hat)

        z_dxdt_hat = np.zeros([len(x)])
        z_dxdt_hat[start:end] = dxdt_hat
        dxdt_hat_list.append(z_dxdt_hat)

        z_weights = np.zeros([len(x)])
        z_weights[start:end] = w
        weight_list.append(z_weights)

    # column norm weights
    weights = np.vstack(weight_list)
    for col in range(weights.shape[1]):
        weights[:, col] = weights[:, col] / np.sum(weights[:, col])

    # stack and weight x_hat and dxdt_hat
    x_hat = np.vstack(x_hat_list)
    dxdt_hat = np.vstack(dxdt_hat_list)

    x_hat = np.sum(weights*x_hat, axis=0)
    dxdt_hat = np.sum(weights*dxdt_hat, axis=0)

    return x_hat, dxdt_hat

####################################################################################################################################################
# Savitzky-Golay filter
####################################################################################################################################################

def savgoldiff(x, dt, params, options={'smooth': True}):
    '''
    Use the Savitzky-Golay to smooth the data and calculate the first derivative.
    Uses scipy.signal.savgol_filter
    The Savitzky-Golay is very similar to the sliding polynomial fit, but slightly noisier, and much faster.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [N,              : (int)    order of the polynomial
                       window_size,    : (int)    size of the sliding window, must be odd (if not, 1 is added)
                       smoothing_win]  : (int)    size of the window used for gaussian smoothing, a good default is = window_size, but smaller for high freq data

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    N, window_size, smoothing_win = params

    if window_size > len(x)-1:
        window_size = len(x)-1

    if smoothing_win > len(x)-1:
        smoothing_win = len(x)-1

    if window_size <= N:
        window_size = N+1   
        
    if not window_size%2: # then make odd
        window_size += 1

    dxdt_hat = scipy.signal.savgol_filter(x, window_size, N, deriv=1) / dt

    if 1: #options['smooth']:
        kernel = __gaussian_kernel__(smoothing_win)
        dxdt_hat = pynumdiff.smooth_finite_difference.__convolutional_smoother__(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat

####################################################################################################################################################
# Polynomial fitting
####################################################################################################################################################

def __polydiff__(x, dt, params, options={}):
    '''
    Fit polynomials to the timeseries, and differentiate the polynomials.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [N]        : (int)                order of the polynomial
    options : (dict) {'weights'} : (np.array, optional) weights applied to each point in calculating the polynomial fit. Defaults to 1s if missing.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    if 'weights' in options.keys():
        w = options['weights']
    else:
        w = np.ones_like(x)

    if type(params) is list:
        order = params[0]
    else:
        order = params
    
    t = np.arange(1, len(x)+1)*dt

    #  polyfit
    r = np.polyfit(t, x, order, w=w)[::-1]
    
    # derivative coefficients
    dr = copy.copy(r[1:])
    for i in range(len(dr)):
        dr[i] = dr[i]*(i+1)
        
    ## evaluate dxdt_hat
    dxdt_hat = 0
    for i in range(len(dr)):
        dxdt_hat += dr[i]*t**(i)

    ## evaluate smooth x
    x_hat = 0
    for i in range(len(r)):
        x_hat += r[i]*t**(i)
        
    return x_hat, dxdt_hat

def polydiff(x, dt, params, options={'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'}):
    '''
    Fit polynomials to the timeseries, and differentiate the polynomials.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [N,            : (int)    order of the polynomial
                       window_size], : (int)    size of the sliding window (ignored if not sliding)
    options : (dict) {'sliding'      : (bool)   slide the method (True or False)
                      'step_size'    : (int)    step size for sliding (smaller value is more accurate and more time consuming)
                      'kernel_name'} : (string) kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    if 'sliding' in options.keys() and options['sliding'] is True:
        window_size = copy.copy(params[-1])
        if window_size < params[0]*3:
            window_size = params[0]*3+1
            params[1] = window_size
        return __slide_function__(__polydiff__, x, dt, params, window_size, options['step_size'], options['kernel_name'])

    else:
        return __polydiff__(x, dt, params, options={})

####################################################################################################################################################
# Chebychev
####################################################################################################################################################


def __chebydiff__(x, dt, params, options={}):
    '''
    Fit the timeseries with chebyshev polynomials, and differentiate this model.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [N] : (int) order of the polynomial
    options : (dict) {}

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    if type(params) is list:
        N = params[0]
    else:
        N = params

    mean = np.mean(x)
    x = x - mean

    def f(y):
        t = np.linspace(-1, 1, len(x))
        return np.interp(y, t, x)

    # Chebychev polynomial
    poly = pychebfun.chebfun(f, N=N, domain=[-1, 1])
    ts = np.linspace(poly.domain()[0], poly.domain()[-1], len(x) )

    x_hat = poly(ts) + mean
    dxdt_hat = poly.differentiate()(ts)*(2/len(x))/dt

    return x_hat, dxdt_hat

def chebydiff(x, dt, params, options={'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'}):
    '''
    Slide a smoothing derivative function across a timeseries with specified window size.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [N,            : (int)    order of the polynomial
                       window_size], : (int)    size of the sliding window (ignored if not sliding)
    options : (dict) {'sliding'      : (bool)   slide the method (True or False)
                      'step_size'    : (int)    step size for sliding (smaller value is more accurate and more time consuming)
                      'kernel_name'} : (string) kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    if 'sliding' in options.keys() and options['sliding'] is True:
        window_size = copy.copy(params[-1])
        if window_size < params[0]*2:
            window_size = params[0]*2+1
            params[1] = window_size
        return __slide_function__(__chebydiff__, x, dt, params, window_size, options['step_size'], options['kernel_name'])

    else:
        return __chebydiff__(x, dt, params, options={})
    

####################################################################################################################################################
# Dynamic Mode Decomposition
####################################################################################################################################################

def __dmddiff__(X, dt, params, options={}):
    '''
    Fit the timeseries with optimized dynamic mode decomposition model.
    
    Inputs
    ------
    X       : (np.matrix of floats, NxM) hankel matrix of time delay embedded time series to differentiate. 
                                         N = time steps, M = time delay embedding
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [delay_embedding, : (int)  amount of delay_embedding 
                       svd_rank]        : (int)  rank trunkcation
    options : (dict) {}

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    delay_embedding = params[0]
    svd_rank = params[1]

    if len(params) == 2: # not sliding:
        L = X.shape[0] + delay_embedding
    else: # sliding
        L = X.shape[0]

    X = X.T

    Ue = np.zeros_like(X)

    # DMD
    dmdc = pydmd.dmdc.DMDc(svd_rank=svd_rank, opt=True)
    dmdc.fit(np.array(X), np.array(Ue[:,1:]), B=np.eye(X.shape[0]))

    # DMD reconstruction
    x0 = X[:, 0]
    Xr = dmdc.forward_backward_reconstruct(x0, L, 0, overlap=2)

    integral_x_hat = np.real(np.ravel( Xr[0,:] ))
    integral_x_hat, x_hat = finite_difference(integral_x_hat, dt)
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat 


def dmddiff(x, dt, params, options={'sliding': True, 'step_size': 10, 'kernel_name': 'gaussian'}):
    '''
    Fit the timeseries with optimized dynamic mode decomposition model.
    The DMD runs on the itnegral of x, to reduce effects of noise.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [delay_embedding, : (int)  amount of delay_embedding 
                       svd_rank,        : (int)  rank trunkcation
                       window_size],    : (int)  size of the sliding window (ignored if not sliding)
    options : (dict) {'sliding'         : (bool)   slide the method (True or False)
                      'step_size'       : (int)    step size for sliding (smaller value is more accurate and more time consuming)
                      'kernel_name'}    : (string) kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    
    
    delay_embedding = params[0]
    svd_rank = params[1]
    
    if delay_embedding >= len(x-1):
        delay_embedding = len(x-1)
    if delay_embedding < svd_rank:
        delay_embedding = svd_rank

    if 'sliding' in options.keys() and options['sliding'] is True:
        window_size = copy.copy(params[2])
        if window_size <= svd_rank + 1:
            window_size = svd_rank + 1

    # forward
    integral_x = utility.integrate_dxdt_hat(x, dt)
    X = utility.hankel_matrix(np.matrix(integral_x), delay_embedding).T
    if 'sliding' in options.keys() and options['sliding'] is True:
        x_hat_forward, dxdt_hat_forward = __slide_function__(__dmddiff__, X, dt, params, window_size, options['step_size'], options['kernel_name'])
    else:
        x_hat_forward, dxdt_hat_forward = __dmddiff__(X, dt, params, options={})

    # backward
    integral_x = utility.integrate_dxdt_hat(x[::-1], dt)
    X = utility.hankel_matrix(np.matrix(integral_x), delay_embedding).T
    if 'sliding' in options.keys() and options['sliding'] is True:
        x_hat_backward, dxdt_hat_backward = __slide_function__(__dmddiff__, X, dt, params, window_size, options['step_size'], options['kernel_name'])
    else:
        x_hat_backward, dxdt_hat_backward = __dmddiff__(X, dt, params, options={})

    # weights
    w = np.arange(1,len(x_hat_forward)+1,1)[::-1]
    w = np.pad(w, [0, len(x)-len(w)], mode='constant')
    wfb = np.vstack((w, w[::-1]))
    norm = np.sum(wfb, axis=0)

    # orient and pad
    x_hat_forward = np.pad(x_hat_forward, [0, len(x)-len(x_hat_forward)], mode='constant')
    x_hat_backward = np.pad(x_hat_backward[::-1], [len(x)-len(x_hat_backward), 0], mode='constant')

    # merge
    x_hat = x_hat_forward*w/norm + x_hat_backward*w[::-1]/norm
    x_hat, dxdt_hat = finite_difference(x_hat, dt)

    return x_hat, dxdt_hat


####################################################################################################################################################
# Integral formulation
####################################################################################################################################################

class DeWhiten(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def dewhiten(self, m):
        return (m+self.mean)*self.std
    
class Whiten(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def whiten(self, m):
        return (m-self.mean)/self.std

def whiten_library(library):
    white_library = []
    dewhiten = []
    whiten = []
    for m in library:
        m += 2*(np.random.random(len(m))-0.5)*1e-16 # in case we have a pure constant
        
        std_m = np.std(m)
        mean_m = np.mean(m)
        
        w = Whiten(mean_m, std_m)
        dw = DeWhiten(mean_m, std_m)
        
        white_library.append(w.whiten(m))
        whiten.append(w)
        dewhiten.append(dw)
        
    return white_library, whiten, dewhiten

def __solve_for_A_and_C_given_X_and_Xdot__(X, Xdot, num_integrations, dt, gammaC=1e-1, gammaA=1e-6, solver='MOSEK', A_known=None, epsilon=1e-6, rows_of_interest='all'): 
    assert type(X) == np.matrix 
    assert type(Xdot) == np.matrix

    if rows_of_interest == 'all':
        rows_of_interest = np.arange(0, X.shape[0])

    # Set up the variables
    A = cvxpy.Variable((X.shape[0], X.shape[0]))
    C = cvxpy.Variable((X.shape[0], num_integrations))

    # Integrate the integration constants
    Csum = 0
    t = np.arange(0, X.shape[1])*dt
    for n in range(num_integrations):
        C_subscript = n
        t_exponent = num_integrations - n -1
        den = math.factorial(t_exponent)
        Cn = cvxpy.vstack((1/den*C[i, C_subscript]*t**t_exponent for i in range(X.shape[0])))
        Csum = Csum + Cn

    # Define the objective function
    error = cvxpy.sum_squares(Xdot[rows_of_interest, :] - (A*X + Csum)[rows_of_interest, :]) 
    C_regularization = gammaC*cvxpy.sum(cvxpy.abs(C))
    A_regularization = gammaA*cvxpy.sum(cvxpy.abs(A))
    obj = cvxpy.Minimize(error + C_regularization + A_regularization) 

    # constraints
    constraints = []
    if A_known is not None:
        for i in range(A_known.shape[0]):
            for j in range(A_known.shape[1]):
                if not np.isnan(A_known[i,j]):
                    constraint_lo = A[i,j] >= A_known[i,j]-epsilon
                    constraint_hi = A[i,j] <= A_known[i,j]+epsilon 
                    constraints.extend([constraint_lo, constraint_hi])

    # Solve the problem
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver=solver) # MOSEK does not take max_iters
    
    A = np.matrix(A.value)
    return A, np.matrix(C.value)

def __integrate_dxdt_hat_matrix__(dxdt_hat, dt):
    assert type(dxdt_hat) == np.matrix
    x = np.matrix(scipy.integrate.cumtrapz(dxdt_hat, axis=1))
    first_value = x[:,0] - np.mean(dxdt_hat[:, 0:1], axis=1) 
    x = np.hstack((first_value, x))*dt
    return x  

def __lineardiff__(x, dt, params, options={}):
    '''
    Estimate the parameters for a system xdot = Ax, and use that to calculate the derivative
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [N,     : (int, >1) order (e.g. 2: velocity; 3: acceleration)
                      gamma] : (float) regularization term
    options : (dict) {}

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    N, gamma = params
    mean = np.mean(x)
    x = x - mean
    
    # Generate the matrix of integrals of x
    X = [x]
    for n in range(1,N):
        X.append(utility.integrate_dxdt_hat(X[-1], dt) )
    X = np.matrix(np.vstack(X[::-1]))
    integral_Xdot = X
    integral_X = __integrate_dxdt_hat_matrix__(X, dt)
    
    # Solve for A and the integration constants
    A, C = __solve_for_A_and_C_given_X_and_Xdot__(integral_X, integral_Xdot, N, dt, gamma)
    
    # Add the integration constants
    Csum = 0
    t = np.arange(0, X.shape[1])*dt
    for n in range(0, N-1):
        C_subscript = n
        t_exponent = N - n - 2
        den = math.factorial(t_exponent)
        Cn = np.vstack((1/den*C[i, C_subscript]*t**t_exponent for i in range(X.shape[0])))
        Csum = Csum + Cn
    Csum = np.matrix(Csum)
    
    # Use A and C to calculate the derivative
    Xdot_reconstructed = (A*X + Csum)
    dxdt_hat = np.ravel(Xdot_reconstructed[-1, :])

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x_hat = x_hat +utility.estimate_initial_condition(x+mean, x_hat)

    return x_hat, dxdt_hat

def lineardiff(x, dt, params, options={'sliding': True, 'step_size': 10, 'kernel_name': 'friedrichs'}):
    '''
    Slide a smoothing derivative function across a timeseries with specified window size.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [N,            : (int)    order of the dynamics (e.g. 3 means rank 3 A matrix, with states = {x, xdot, xddot})
                       gamma,        : (float) regularization term
                       window_size], : (int)    size of the sliding window (ignored if not sliding)
    options : (dict) {'sliding'      : (bool)   slide the method (True or False)
                      'step_size'    : (int)    step size for sliding (smaller value is more accurate and more time consuming)
                      'kernel_name'} : (string) kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    if 'sliding' in options.keys() and options['sliding'] is True:
        window_size = copy.copy(params[-1])
        params = params[0:-1]

        # forward and backward
        x_hat_forward, dxdt_hat_forward = __slide_function__(__lineardiff__, x, dt, params, window_size, options['step_size'], options['kernel_name'])
        x_hat_backward, dxdt_hat_backward = __slide_function__(__lineardiff__, x[::-1], dt, params, window_size, options['step_size'], options['kernel_name'])

        # weights
        w = np.arange(1,len(x_hat_forward)+1,1)[::-1]
        w = np.pad(w, [0, len(x)-len(w)], mode='constant')
        wfb = np.vstack((w, w[::-1]))
        norm = np.sum(wfb, axis=0)

        # orient and pad
        x_hat_forward = np.pad(x_hat_forward, [0, len(x)-len(x_hat_forward)], mode='constant')
        x_hat_backward = np.pad(x_hat_backward[::-1], [len(x)-len(x_hat_backward), 0], mode='constant')

        # merge
        x_hat = x_hat_forward*w/norm + x_hat_backward*w[::-1]/norm
        x_hat, dxdt_hat = finite_difference(x_hat, dt)

        return x_hat, dxdt_hat

    else:
        return __lineardiff__(x, dt, params, options={})

####################################################################################################################################################
# Spectral derivative
####################################################################################################################################################

def spectraldiff(x, dt, params, options={'even_extension': True, 'pad_to_zero_dxdt': True}): 
    '''
    Take a derivative in the fourier domain, with high frequency attentuation.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [wn]: (float) the high frequency cut off 
    options : (dict) {'even_extension',:   (bool) if True, extend the time series with
                                                  an even extension so signal 
                                                  starts and ends at the same value.
                      'pad_to_zero_dxdt'}: (bool) if True, extend the time series with
                                                  extensions that smoothly force the derivative
                                                  to zero. This allows the spectral derivative
                                                  to fit data which does not start and end 
                                                  with derivatives equal to zero.


    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if type(params) is list:
        wn = params[0]
    else:
        wn = params

    original_L = len(x)

    # make derivative go to zero at ends (optional)
    if options['pad_to_zero_dxdt']:
        padding = 100
        pre = x[0]*np.ones(padding)
        post = x[-1]*np.ones(padding)
        x = np.hstack((pre, x, post))
        x_hat, xdot_hat = smooth_finite_difference.meandiff(x, dt, [int(padding/2)], options={'iterate': False})
        x_hat[padding:-padding] = x[padding:-padding]
        x = x_hat
    else:
        padding = 0

    # Do even extension (optional)
    if options['even_extension'] is True:
        x = np.hstack((x, x[::-1])) 

    # If odd, make N even, and pad x
    L = len(x)
    if L % 2 != 0:
        N = L+1
        x = np.hstack((x, x[-1] + dt*(x[-1]-x[-1])))
    else:
        N = L
        
    # Define the frequency range. 
    k = np.asarray(list(range(0, int(N/2) )) + [0] + list(range( int(-N/2) + 1,0)))
    k = k*2*np.pi/(dt*N)

    # Frequency based smoothing: remove signals with a frequency higher than wn
    discrete_wn = int(wn*N)
    k[discrete_wn:N-discrete_wn] = 0

    # Derivative = 90 deg phase shift
    dxdt_hat = np.real(np.fft.ifft(1.0j * k * np.fft.fft(x)))
    dxdt_hat = dxdt_hat[padding:original_L+padding]

    # Integrate to get x_hat
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt) 
    x0 = utility.estimate_initial_condition(x[padding:original_L+padding], x_hat)
    x_hat = x_hat + x0
    
    return x_hat, dxdt_hat 
