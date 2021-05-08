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
from pynumdiff.linear_model import __integrate_dxdt_hat_matrix__, __solve_for_A_and_C_given_X_and_Xdot__
import pynumdiff.smooth_finite_difference
from pynumdiff.utils import utility as utility
from pynumdiff import smooth_finite_difference
__gaussian_kernel__ = utility.__gaussian_kernel__

# optional packages
try:
    import pydmd.dmdc
except:
    logging.info('Import Error.\nCould not import pydmd. Install pydmd (florisvb fork: https://github.com/florisvb/PyDMD) to use dmd derivatives.\n')
try:
    import cvxpy
except:
    logging.info('Import Error.\nCould not import cvxpy. Install cvxpy (http://www.cvxpy.org/install/index.html) to use linearmodel and nonlinearmodel.\nRecommended solver: MOSEK, free academic license available: https://www.mosek.com/products/academic-licenses/\n')

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def __integrate__(x, dt):
    y = scipy.integrate.cumtrapz( x )*dt
    y = np.hstack((y, y[-1]-x[-1]*dt))
    return y

def integrate_library(library, dt):
    library = [__integrate__(l, dt) for l in library]
    return library

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

####################################################################################################################################################
# Discover a linear model
####################################################################################################################################################

def linearmodel(x, data, dt, params, options={'smooth': True}):
    '''
    Estimate the parameters for a system xdot = Ax + Bu, and use that to calculate the derivative
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    data    : (list of np.array of floats like x) additional time series data that may be relevant to modeling xdot = Ax + Bu
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [N,          : (int, >1) order (e.g. 2: velocity; 3: acceleration)
                      gammaA      : (float) regularization term for A (try 1e-6)
                      gammaC      : (float) regularization term for integration constants (try 1e-1)
                      window_size : (int) if options['smooth'] == True, window_size determines size over which gaussian smoothing is applied
    options : (dict)  {'smooth',}    : (bool)   if True, apply gaussian smoothing to the result with the same window size

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    try:
        N, gammaA, gammaC, window_size = params
    except:
        N, gammaA, gammaC = params

    mean = np.mean(x)
    x = x - mean
    
    # Generate the matrix of integrals of x
    X = [x]
    for n in range(1,N):
        X.append(utility.integrate_dxdt_hat(X[-1], dt) )
    X = X[::-1]
    for d in data:
        for n in range(1,N-1):
            d = utility.integrate_dxdt_hat(d, dt)
        X.append(d)
    X = np.matrix(np.vstack(X))
        
    integral_Xdot = X
    integral_X = __integrate_dxdt_hat_matrix__(X, dt)
    
    # Solve for A and the integration constants
    A, C = __solve_for_A_and_C_given_X_and_Xdot__(integral_X, integral_Xdot, N, dt, gammaC=gammaC, gammaA=gammaA, solver='MOSEK', A_known=None, epsilon=1e-6, rows_of_interest=[N-1])
    
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
    dxdt_hat = np.ravel(Xdot_reconstructed[N-1, :])

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x_hat = x_hat +utility.estimate_initial_condition(x+mean, x_hat)

    if options['smooth']:
        kernel = __gaussian_kernel__(window_size)
        dxdt_hat = pynumdiff.smooth_finite_difference.__convolutional_smoother__(dxdt_hat, kernel, 1)

    return x_hat, dxdt_hat

####################################################################################################################################################
# Integral SINDy (nonlinear model)
####################################################################################################################################################

def nonlinearmodel(x, library, dt, params, options={'smooth': True, 'solver': 'MOSEK'}):
    '''
    Use the integral form of SINDy to find a sparse dynamical system model for the output, x, given a library of features.
    Then take the derivative of that model to estimate the derivative.  
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    library : (list of 1D arrays) list of features to use for building the model
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [gamma,        : (int)    sparsity knob (higher = more sparse model)
                       window_size], : (int)    if option smooth, this determines the smoothing window
    options : (dict)  {'smooth',     : (bool)   if True, apply gaussian smoothing to the result with the same window size
                       'solver'}     : (str)    solver to use with cvxpy, MOSEK is default

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''

    # Features
    int_library = integrate_library(library, dt)
    w_int_library, w_int_library_func, dw_int_library_func = whiten_library(int_library)

    # Whitened states
    w_state, w_state_func, dw_state_func = whiten_library([x])
    w_x_hat, = w_state

    # dewhiten integral library coefficients
    integrated_library_std = []
    integrated_library_mean = []
    for d in dw_int_library_func:
        integrated_library_std.append(d.std)
        integrated_library_mean.append(d.mean)
    integrated_library_std = np.array(integrated_library_std)
    integrated_library_mean = np.array(integrated_library_mean)

    # dewhiten state coefficients
    state_std = []
    state_mean = []
    for d in dw_state_func:
        state_std.append(d.std)
        state_mean.append(d.mean)
    state_std = np.array(state_std)
    state_mean = np.array(state_mean)

    # Define loss function
    var = cvxpy.Variable( (1, len(library)) )
    sum_squared_error_x = cvxpy.sum_squares( w_x_hat[1:-1] - (w_int_library*var[0,:])[1:-1] )
    sum_squared_error = cvxpy.sum([sum_squared_error_x])
    
    # Solve convex optimization problem
    gamma = params[0]
    solver = options['solver']
    L = cvxpy.sum( sum_squared_error + gamma*cvxpy.norm1(var) )
    obj = cvxpy.Minimize(L)
    prob = cvxpy.Problem(obj)
    r = prob.solve(solver=solver)
    sindy_coefficients = var.value

    integrated_library_offset = np.matrix(sindy_coefficients[0,:]/integrated_library_std)*np.matrix(integrated_library_mean).T
    estimated_coefficients = sindy_coefficients[0,:]/integrated_library_std*np.tile(state_std[0], [len(int_library), 1]).T
    offset = -1*(state_std[0]*np.ravel(integrated_library_offset)) + state_mean

    # estimate derivative
    dxdt_hat = np.ravel(np.matrix(estimated_coefficients)*np.matrix(library))

    if options['smooth']:
        window_size = params[1]
        kernel = __gaussian_kernel__(window_size)
        dxdt_hat = pynumdiff.smooth_finite_difference.__convolutional_smoother__(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat

