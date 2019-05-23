import numpy as np 

import cvxpy


# included modules
from pynumdiff.finite_difference import first_order as finite_difference 
import pynumdiff.smooth_finite_difference
from pynumdiff.utils import utility as utility
from pynumdiff import smooth_finite_difference
__gaussian_kernel__ = utility.__gaussian_kernel__

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def integrate_library(library, dt):
    library = [np.cumsum(l)*dt for l in library]
    return library

class DeWhiten(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    def dewhiten(self, m):
        return (m+self.mean)*self.var

def whiten_library(library):
    white_library = []
    dewhiten = []
    for m in library:
        m += 2*(np.random.random(len(m))-0.5)*1e-8 # in case we have a pure constant
        
        var_m = np.std(m)**2
        mean_m = np.mean(m)
        
        white_m = (m-mean_m)/var_m
        white_library.append(white_m)

        dewhiten.append(DeWhiten(mean_m, var_m))
    return white_library, dewhiten

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
    integrated_library = integrate_library(library, dt)
    white_integrated_library, dewhiten_integrated_library = whiten_library(integrated_library)

    # Whitened states
    white_states, dewhiten_states = whiten_library([x])
    white_x = white_states[0]

    # Setup convex optimization problem
    var = cvxpy.Variable( (1, len(integrated_library)) )
    #return white_x, white_integrated_library, var
    sum_squared_error_x = cvxpy.sum_squares( white_x - white_integrated_library*var[0,:] )
    sum_squared_error = cvxpy.sum([sum_squared_error_x])

    # Solve convex optimization problem
    gamma = params[0]
    solver = options['solver']
    L = cvxpy.sum( sum_squared_error + gamma*cvxpy.norm1(var) )
    obj = cvxpy.Minimize(L)
    prob = cvxpy.Problem(obj)
    r = prob.solve(solver=solver)
    sindy_coefficients = var.value

    # dewhiten library coefficients
    integrated_library_var = []
    integrated_library_mean = []
    for d in dewhiten_integrated_library:
        integrated_library_var.append(d.var)
        integrated_library_mean.append(d.mean)
    integrated_library_var = np.array(integrated_library_var)
    integrated_library_mean = np.array(integrated_library_mean)

    # dewhiten state coefficients
    state_var = []
    state_mean = []
    for d in dewhiten_states:
        state_var.append(d.var)
        state_mean.append(d.mean)
    state_var = np.array(state_var)
    state_mean = np.array(state_mean)

    integrated_library_offset = np.matrix(sindy_coefficients/integrated_library_var)*np.matrix(integrated_library_mean).T
    estimated_coefficients = sindy_coefficients/integrated_library_var*np.tile(state_var, [len(integrated_library), 1]).T
    offset = -1*(state_var*np.ravel(integrated_library_offset)) + state_mean

    # estimate derivative
    dxdt_hat = np.ravel(np.matrix(estimated_coefficients[0,:])*np.matrix(library))

    if options['smooth']:
        window_size = params[1]
        kernel = __gaussian_kernel__(window_size)
        dxdt_hat = pynumdiff.smooth_finite_difference.__convolutional_smoother__(dxdt_hat, kernel, 1)

    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    x_hat = x_hat + x0

    return x_hat, dxdt_hat
