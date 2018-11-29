import scipy.optimize
import numpy as np

from pynumdiff.utils import utility as utility
from pynumdiff.utils import evaluate as evaluate
import pynumdiff.smooth_finite_difference

from pynumdiff.optimize.__optimize__ import __objective_function__
from pynumdiff.optimize.__optimize__ import __optimize__

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def __kerneldiff__(function, x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    # initial condition
    if params is None:
        if options['iterate'] is False:
            params = [[5], [15], [30], [50]]
        else:
            window_sizes = [5, 15, 30, 50]
            iterations = [1, 5, 10]
            params = []
            for window_size in window_sizes:
                for iteration in iterations:
                    params.append([window_size, iteration])

    # param types and bounds
    if options['iterate'] is False:
        params_types = [int,]
        params_low = [1,]
        params_high = [1e6,]
    else:
        params_types = [int, int]
        params_low = [1, 1]
        params_high = [1e6, 1e2]

    # optimize
    args = [function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding]
    opt_params, opt_val = __optimize__(params, args) 

    return opt_params, opt_val



####################################################################################################################################################
# Optimize functions
####################################################################################################################################################

def mediandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    '''
    Optimize the parameters for smooth_finite_difference.mediandiff 
    
    See pynumdiff.optimize.smooth_finite_difference.docstring for detailed documentation.
    '''
    function = pynumdiff.smooth_finite_difference.mediandiff
    opt_params, opt_val = __kerneldiff__(function, x, dt, params, options, dxdt_truth, tvgamma, padding) 
    return opt_params, opt_val

def meandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    '''
    Optimize the parameters for smooth_finite_difference.meandiff 
    
    See pynumdiff.optimize.smooth_finite_difference.docstring for detailed documentation.
    '''
    function = pynumdiff.smooth_finite_difference.meandiff
    opt_params, opt_val = __kerneldiff__(function, x, dt, params, options, dxdt_truth, tvgamma, padding) 
    return opt_params, opt_val

def gaussiandiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    '''
    Optimize the parameters for smooth_finite_difference.gaussiandiff 
    
    See pynumdiff.optimize.smooth_finite_difference.docstring for detailed documentation.
    '''
    function = pynumdiff.smooth_finite_difference.gaussiandiff
    opt_params, opt_val = __kerneldiff__(function, x, dt, params, options, dxdt_truth, tvgamma, padding) 
    return opt_params, opt_val

def friedrichsdiff(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    '''
    Optimize the parameters for smooth_finite_difference.friedrichsdiff 
    
    See pynumdiff.optimize.smooth_finite_difference.docstring for detailed documentation.
    '''
    function = pynumdiff.smooth_finite_difference.friedrichsdiff
    opt_params, opt_val = __kerneldiff__(function, x, dt, params, options, dxdt_truth, tvgamma, padding) 
    return opt_params, opt_val


