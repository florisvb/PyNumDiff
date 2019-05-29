import numpy as np 
import scipy.integrate
import cvxpy


# included modules
from pynumdiff.finite_difference import first_order as finite_difference 
import pynumdiff.smooth_finite_difference
from pynumdiff.utils import utility as utility
from pynumdiff import smooth_finite_difference
__gaussian_kernel__ = utility.__gaussian_kernel__

from pynumdiff.model_based import __ukf_sqrt__ as __ukf_sqrt__

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def integrate(x, dt):
    y = scipy.integrate.cumtrapz( x )*dt
    y = np.hstack((y, y[-1]-x[-1]*dt))
    return y

def integrate_library(library, dt):
    library = [integrate(l, dt) for l in library]
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
# Integral SINDy
####################################################################################################################################################

def sindy(x, library, dt, params, options={'smooth': True, 'solver': 'MOSEK'}):
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

####################################################################################################################################################
# Unscented Kalman Filter
####################################################################################################################################################

def ukf(m, u, s0, f, h, Q, R, alpha=0.001, beta=2):
    '''
    Use an Unscecnted Kalman Filter to estimate the derivatives. 
    
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

    if u is None:
        u = np.matrix([[None]*m.shape[1]])

    x, P, s = __ukf_sqrt__.ukf_sqrt(m, u, s0, f, h, Q, R, alpha=0.01, beta=2)

    return x